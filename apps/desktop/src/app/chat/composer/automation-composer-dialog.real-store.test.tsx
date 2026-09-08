import { JsonRpcGatewayError } from '@hermes/shared'
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n/context'
import {
  $automationComposer,
  openAutomationComposerForEdit
} from '@/store/automation-composer'
import { $gateway } from '@/store/gateway'
import { resetBackgroundPollingGuard } from '@/store/runtime-gone'
import {
  $sessionControlBySession,
  applySessionControlSnapshot
} from '@/store/session-control'

import { AutomationComposerDialog } from './automation-composer-dialog'

// Keep the REAL session-control store (this is the point of the suite: the
// existing dialog test mocks the whole store, so it cannot observe a busy
// rejection mis-classified as unavailability). Only the pieces a real store
// test must not drag in are stubbed.
const { refreshLegacyGoal } = vi.hoisted(() => ({ refreshLegacyGoal: vi.fn() }))

vi.mock('@/store/goals', async importOriginal => ({
  ...(await importOriginal()),
  refreshSessionGoal: refreshLegacyGoal
}))

vi.mock('@/store/session', async importOriginal => {
  const actual = (await importOriginal()) as Record<string, unknown>

  return {
    ...actual,
    $activeSessionId: { get: vi.fn(() => 'session-123'), listen: vi.fn(() => () => {}) }
  }
})

vi.mock('@/lib/haptics', () => ({ triggerHaptic: () => {} }))
vi.mock('@/app/session/hooks/use-prompt-actions/queue-if-busy', () => ({
  queueKickoffIfSessionBusy: vi.fn(() => 'idle')
}))

const BUSY_MESSAGE =
  'Goal is busy with a live session. Pause the Goal and wait for this turn to finish, ' +
  'or stop the session, then save your edit.'

const UNAVAILABLE_MESSAGE = 'Automation controls are unavailable. Check the connection and refresh.'

const SNAPSHOT = {
  goal: {
    created_at: 1_700_000_000,
    contract: {
      boundaries: 'desktop store only',
      constraints: 'do not lose draft edits',
      outcome: 'the edit retries without being torn down',
      stop_when: 'a human decision is required',
      verification: 'real-store component tests pass'
    },
    gates: [{ attempts: 0, command: 'npm test', last_exit_code: null, max_retries: 2, timeout_seconds: 60 }],
    max_turns: 12,
    status: 'active',
    subgoals: ['Keep the tests green'],
    title: 'Original objective',
    turns_used: 3,
    updated_at: 1_700_000_100
  },
  heartbeat: null,
  loop: null,
  revision: 'revision-1',
  updated_at: 1_700_000_100
}

const EXEC_DISPATCH = { display: null, message: null, notice: '', output: '✓ Goal updated', type: 'exec' }

/** Session-bound gateway RPC mock. Reads always succeed with `SNAPSHOT`; the
 *  provided `actionImpl` governs the first `session.control` mutation call. */
function useGateway(request: (_method: string, params: Record<string, unknown>) => Promise<unknown>): void {
  $gateway.set({ request } as never)
}

/** A healthily-hydrated editable goal for the captured session. */
function seedEditableGoal(): void {
  applySessionControlSnapshot('session-123', SNAPSHOT)
}

async function renderForEdit(onSubmitText = vi.fn().mockResolvedValue(true)) {
  let result!: ReturnType<typeof render>
  await act(async () => {
    result = render(
      <I18nProvider configClient={{ getConfig: async () => ({}), saveConfig: async () => ({ ok: true }) }}>
        <AutomationComposerDialog conversationTitle="Current chat" onOpenCron={vi.fn()} onSubmitText={onSubmitText} />
      </I18nProvider>
    )
  })

  act(() => openAutomationComposerForEdit('goal', 'session-123'))

  return result
}

function promptBox(): HTMLTextAreaElement {
  return screen.getByLabelText(/goal prompt/i) as HTMLTextAreaElement
}

const saveButton = (): HTMLButtonElement => screen.getByRole('button', { name: 'Save goal' }) as HTMLButtonElement

beforeEach(() => {
  refreshLegacyGoal.mockReset()
  useGateway(vi.fn(async () => ({ control: SNAPSHOT })))
})

afterEach(() => {
  $automationComposer.set({ open: false, sessionId: null, type: 'goal', mode: 'create', submitting: false, error: null })
  $sessionControlBySession.set({})
  $gateway.set(null as never)
  resetBackgroundPollingGuard()
  vi.clearAllMocks()
})

describe('AutomationComposerDialog (real store, edit retry)', () => {
  it('keeps an editable goal editable across a busy goal.update rejection, then a retry succeeds without losing the draft or injecting a prompt', async () => {
    const onSubmitText = vi.fn().mockResolvedValue(true)
    let mutationCount = 0

    // A real gateway routes reads and mutations through the same request fn;
    // only goal.update is busy on its first try.
    const gatewayRequest = vi
      .fn((_method: string, _params: Record<string, unknown>) => {
        if (_method === 'session.control.read') {
          return Promise.resolve({ control: SNAPSHOT })
        }

        mutationCount += 1

        if (mutationCount === 1) {
          return Promise.reject(new JsonRpcGatewayError(BUSY_MESSAGE, { code: 4004 }))
        }

        return Promise.resolve({
          control: { ...SNAPSHOT, revision: 'after-update' },
          dispatch: EXEC_DISPATCH
        })
      })

    useGateway(gatewayRequest)
    seedEditableGoal()
    await renderForEdit(onSubmitText)

    await screen.findByLabelText(/goal prompt/i)
    fireEvent.change(promptBox(), { target: { value: 'My unsaved objective' } })
    expect(saveButton().disabled).toBe(false)

    // First save is rejected because the goal is busy.
    fireEvent.click(saveButton())
    await waitFor(() => expect(screen.getByText(/busy with a live session/i)).toBeTruthy())

    // The busy rejection is retryable, NOT an outage: the generic unavailable
    // banner must not appear and Save must stay enabled for a retry.
    expect(screen.queryByText(UNAVAILABLE_MESSAGE)).toBeNull()
    await waitFor(() => expect(saveButton().disabled).toBe(false))

    // The draft the user typed is preserved across the rejection.
    expect(promptBox().value).toBe('My unsaved objective')

    // Classify the rejection where it belongs: the session is still supported
    // and healthy — only the action was declined.
    expect($sessionControlBySession.get()['session-123']).toMatchObject({
      capability: 'supported',
      error: null,
      actionError: BUSY_MESSAGE
    })

    // Retry later (goal is free): the same draft saves and the dialog closes.
    fireEvent.click(saveButton())
    await waitFor(() => expect($automationComposer.get().open).toBe(false))

    // One hydration read + exactly the two goal.update dispatches (busy reject,
    // then the successful retry) — the retry never re-fires and never double-sends.
    expect(mutationCount).toBe(2)
    expect(gatewayRequest).toHaveBeenCalledTimes(3)
    // An edit dispatch is `exec`, never a `send` — no continuation prompt is
    // injected into the session on save or retry.
    expect(onSubmitText).not.toHaveBeenCalled()
  })

  it('disables Save and shows the outage banner when the read/connection genuinely fails', async () => {
    useGateway(
      vi.fn(async () => {
        throw new Error('connection reset')
      })
    )
    seedEditableGoal()

    await renderForEdit()
    await screen.findByLabelText(/goal prompt/i)

    // The dialog's foreground refresh fails → entry.error → unavailable.
    await waitFor(() => expect(screen.getByText(UNAVAILABLE_MESSAGE)).toBeTruthy())
    await waitFor(() => expect(saveButton().disabled).toBe(true))

    expect($sessionControlBySession.get()['session-123']).toMatchObject({
      capability: 'supported',
      error: 'connection reset',
      actionError: null
    })
  })

  it('disables Save when the backend does not support the control RPC', async () => {
    useGateway(
      vi.fn(async () => {
        throw new JsonRpcGatewayError('method not found', { code: -32601 })
      })
    )
    seedEditableGoal()

    await renderForEdit()
    await screen.findByLabelText(/goal prompt/i)

    await waitFor(() => expect(screen.getByText(UNAVAILABLE_MESSAGE)).toBeTruthy())
    await waitFor(() => expect(saveButton().disabled).toBe(true))
    expect(refreshLegacyGoal).toHaveBeenCalledWith('session-123')
    expect($sessionControlBySession.get()['session-123']!.capability).toBe('unsupported')
  })

  it('does not double-submit a pending save', async () => {
    let release!: (value: unknown) => void
    const portfolio = { calls: 0 }

    const gatewayRequest = vi.fn((method: string) => {
      if (method === 'session.control.read') {
        return Promise.resolve({ control: SNAPSHOT })
      }

      portfolio.calls += 1

      return new Promise(resolve => {
        release = resolve
      })
    })

    useGateway(gatewayRequest)
    await renderForEdit()

    fireEvent.change(promptBox(), { target: { value: 'My unsaved objective' } })
    fireEvent.click(saveButton())

    // While the save is in flight the store is `submitting`; a redundant click
    // must not dispatch a second mutation.
    await waitFor(() => expect($automationComposer.get().submitting).toBe(true))
    expect(portfolio.calls).toBe(1)
    fireEvent.click(saveButton())
    expect(portfolio.calls).toBe(1)

    act(() => release({ control: { ...SNAPSHOT, revision: 'saved' }, dispatch: EXEC_DISPATCH }))
    await waitFor(() => expect($automationComposer.get().open).toBe(false))
    expect(portfolio.calls).toBe(1)
  })
})