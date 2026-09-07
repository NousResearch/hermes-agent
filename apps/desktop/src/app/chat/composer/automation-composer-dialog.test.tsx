import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n/context'
import {
  $automationComposer,
  type AutomationType,
  invalidateOnSessionSwitch,
  openAutomationComposer
} from '@/store/automation-composer'
import { $activeSessionId } from '@/store/session'
import { $sessionControlBySession, runSessionControlAction } from '@/store/session-control'

import { AutomationComposerDialog } from './automation-composer-dialog'

vi.mock('@/store/session', async importOriginal => {
  const actual = (await importOriginal()) as Record<string, unknown>

  return {
    ...actual,
    $activeSessionId: { get: vi.fn(() => 'session-123'), listen: vi.fn(() => () => {}) }
  }
})

vi.mock('@/store/session-control', async () => {
  const { atom } = await import('nanostores')

  return {
    $sessionControlBySession: atom({}),
    refreshSessionControl: vi.fn(),
    runSessionControlAction: vi.fn()
  }
})

vi.mock('@/app/session/hooks/use-prompt-actions/queue-if-busy', () => ({
  queueKickoffIfSessionBusy: vi.fn(() => 'idle')
}))

vi.mock('@/lib/haptics', () => ({ triggerHaptic: () => {} }))

async function renderDialog(onSubmitText = vi.fn().mockResolvedValue(true)) {
  let result!: ReturnType<typeof render>
  await act(async () => {
    result = render(
      <I18nProvider configClient={{ getConfig: async () => ({}), saveConfig: async () => ({ ok: true }) }}>
        <AutomationComposerDialog conversationTitle="Current chat" onOpenCron={vi.fn()} onSubmitText={onSubmitText} />
      </I18nProvider>
    )
  })

  return result
}

const openAs = (type: AutomationType = 'goal') => {
  openAutomationComposer(type, 'session-123')
}

afterEach(() => {
  $automationComposer.set({ open: false, sessionId: null, type: 'goal', submitting: false, error: null })
  $sessionControlBySession.set({})
  vi.clearAllMocks()
})

describe('AutomationComposerDialog', () => {
  it('surfaces a failed goal kickoff rather than closing silently', async () => {
    vi.mocked(runSessionControlAction).mockResolvedValueOnce({ type: 'send', message: 'kickoff', display: 'Goal' } as never)
    await renderDialog(vi.fn().mockResolvedValue(false))
    act(() => openAs('goal'))
    fireEvent.change(await screen.findByLabelText('Goal prompt'), { target: { value: 'Goal' } })
    fireEvent.click(screen.getByRole('button', { name: 'Start goal' }))
    await waitFor(() => expect($automationComposer.get().error).toBeTruthy())
    expect(screen.getByRole('dialog')).toBeTruthy()
  })
  it('offers management instead of replacing an existing goal', async () => {
    $sessionControlBySession.set({ 'session-123': { capability: 'supported', snapshot: { goal: { title: 'Existing goal', status: 'active' } } } } as never)
    await renderDialog()
    act(() => openAs('goal'))
    expect(await screen.findByRole('button', { name: 'Manage existing' })).toBeTruthy()
    expect(screen.getByText('Existing goal')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Manage existing' }))
    expect(runSessionControlAction).not.toHaveBeenCalled()
  })
  it('explains that an empty draft needs a conversation', async () => {
    await renderDialog()
    act(() => openAutomationComposer('goal', null))
    expect(await screen.findByText(/start a conversation first/i)).toBeTruthy()
    expect((screen.getByRole('button', { name: 'Start goal' }) as HTMLButtonElement).disabled).toBe(true)
  })

  it('renders nothing when closed', async () => {
    await renderDialog()
    expect(screen.queryByRole('dialog')).toBeNull()
  })

  it('renders the dialog with type tabs when open', async () => {
    await renderDialog()
    act(() => openAs('goal'))
    expect(await screen.findByRole('dialog')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Goal' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Loop' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Heartbeat' })).toBeTruthy()
  })

  it('shows goal fields by default with first-run consequence', async () => {
    await renderDialog()
    act(() => openAs('goal'))
    await screen.findByRole('dialog')
    expect(screen.getByLabelText(/completion criteria/i)).toBeTruthy()
    expect(screen.getByText(/work starts immediately/i)).toBeTruthy()
  })

  it('switching to loop reveals interval, run limit and stop condition fields', async () => {
    await renderDialog()
    act(() => openAs('goal'))
    await screen.findByRole('dialog')
    fireEvent.click(screen.getByRole('button', { name: 'Loop' }))
    expect(await screen.findByLabelText(/interval/i)).toBeTruthy()
    expect(screen.getByLabelText(/run limit/i)).toBeTruthy()
    expect(screen.getByLabelText(/stop condition/i)).toBeTruthy()
  })

  it('switching to heartbeat reveals interval and idle behavior', async () => {
    await renderDialog()
    act(() => openAs('goal'))
    await screen.findByRole('dialog')
    fireEvent.click(screen.getByRole('button', { name: 'Heartbeat' }))
    expect(await screen.findByLabelText(/interval/i)).toBeTruthy()
    expect(screen.getByText(/idle/i)).toBeTruthy()
  })

  it('submits goal.create and closes on success', async () => {
    vi.mocked(runSessionControlAction).mockResolvedValueOnce({
      type: 'send',
      display: 'Fix the bug',
      message: '[continuing] Fix the bug',
      notice: '✓ Goal set',
      output: '✓ Goal set'
    } as never)
    await renderDialog()
    act(() => openAs('goal'))
    await screen.findByRole('dialog')
    fireEvent.change(screen.getByLabelText(/goal prompt/i), { target: { value: 'Fix the bug' } })
    fireEvent.click(screen.getByRole('button', { name: /start goal/i }))
    await waitFor(() => expect(runSessionControlAction).toHaveBeenCalledWith('session-123', 'goal.create', expect.anything()))
    await waitFor(() => expect($automationComposer.get().open).toBe(false))
  })

  it('submits loop.create with interval and run limit', async () => {
    vi.mocked(runSessionControlAction).mockResolvedValueOnce({
      type: 'exec',
      display: null,
      message: null,
      notice: '',
      output: ''
    } as never)
    await renderDialog()
    act(() => openAs('loop'))
    await screen.findByRole('dialog')
    fireEvent.change(screen.getByLabelText(/loop prompt/i), { target: { value: 'Poll CI' } })
    fireEvent.change(screen.getByLabelText(/interval/i), { target: { value: '300' } })
    fireEvent.change(screen.getByLabelText(/run limit/i), { target: { value: '5' } })
    fireEvent.click(screen.getByRole('button', { name: /start loop/i }))
    await waitFor(() =>
      expect(runSessionControlAction).toHaveBeenCalledWith('session-123', 'loop.create', {
        prompt: 'Poll CI',
        interval_seconds: 300,
        run_limit: 5
      })
    )
    await waitFor(() => expect($automationComposer.get().open).toBe(false))
  })

  it('submits heartbeat.create with interval', async () => {
    vi.mocked(runSessionControlAction).mockResolvedValueOnce({
      type: 'exec',
      display: null,
      message: null,
      notice: '',
      output: ''
    } as never)
    await renderDialog()
    act(() => openAs('heartbeat'))
    await screen.findByRole('dialog')
    fireEvent.change(screen.getByLabelText(/heartbeat prompt/i), { target: { value: 'Health check' } })
    fireEvent.change(screen.getByLabelText(/interval/i), { target: { value: '600' } })
    fireEvent.click(screen.getByRole('button', { name: /create heartbeat/i }))
    await waitFor(() =>
      expect(runSessionControlAction).toHaveBeenCalledWith('session-123', 'heartbeat.create', {
        prompt: 'Health check',
        interval_seconds: 600
      })
    )
    await waitFor(() => expect($automationComposer.get().open).toBe(false))
  })

  it('keeps the dialog open with an error on backend rejection', async () => {
    vi.mocked(runSessionControlAction).mockRejectedValueOnce(
      new Error('A goal already exists for this session. Clear it first with goal.clear.')
    )
    await renderDialog()
    act(() => openAs('goal'))
    await screen.findByRole('dialog')
    fireEvent.change(screen.getByLabelText(/goal prompt/i), { target: { value: 'Fix the bug' } })
    fireEvent.click(screen.getByRole('button', { name: /start goal/i }))
    await waitFor(() => expect($automationComposer.get().open).toBe(true))
    expect(await screen.findByText(/already exists/i)).toBeTruthy()
  })

  it('disables the submit button while submitting', async () => {
    let release!: () => void
    vi.mocked(runSessionControlAction).mockImplementationOnce(
      () =>
        new Promise(resolve => {
          release = () => resolve({ type: 'exec' } as never)
        })
    )
    await renderDialog()
    act(() => openAs('goal'))
    await screen.findByRole('dialog')
    fireEvent.change(screen.getByLabelText(/goal prompt/i), { target: { value: 'Fix the bug' } })
    fireEvent.click(screen.getByRole('button', { name: /start goal/i }))
    await waitFor(() => expect($automationComposer.get().submitting).toBe(true))
    expect(screen.getByRole('button', { name: /start goal/i }).getAttribute('disabled')).not.toBeNull()
    act(() => release())
    await waitFor(() => expect($automationComposer.get().submitting).toBe(false))
  })

  it('does not submit an empty prompt', async () => {
    await renderDialog()
    act(() => openAs('goal'))
    await screen.findByRole('dialog')
    fireEvent.click(screen.getByRole('button', { name: /start goal/i }))
    expect(runSessionControlAction).not.toHaveBeenCalled()
    expect($automationComposer.get().open).toBe(true)
  })

  it('links to the cron UI from the dialog', async () => {
    await renderDialog()
    act(() => openAs('goal'))
    await screen.findByRole('dialog')
    expect(screen.getByText(/scheduled job instead/i)).toBeTruthy()
  })

  it('closes when the captured session is switched away', async () => {
    await renderDialog()
    act(() => openAs('goal'))
    await screen.findByRole('dialog')
    vi.mocked($activeSessionId.get).mockReturnValueOnce('session-other')
    act(() => invalidateOnSessionSwitch())
    expect($automationComposer.get().open).toBe(false)
  })
})