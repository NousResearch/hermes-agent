import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as GoalsModule from '@/store/goals'
import type * as SessionControlModule from '@/store/session-control'

const { mockRefreshSessionControl, mockRunSessionControlAction, mockRefreshSessionGoal } = vi.hoisted(() => ({
  mockRefreshSessionControl: vi.fn(),
  mockRunSessionControlAction: vi.fn(),
  mockRefreshSessionGoal: vi.fn()
}))

vi.mock('@/store/session-control', async importOriginal => {
  const actual = await importOriginal<typeof SessionControlModule>()

  return {
    ...actual,
    refreshSessionControl: mockRefreshSessionControl,
    runSessionControlAction: mockRunSessionControlAction
  }
})

vi.mock('@/store/goals', async importOriginal => {
  const actual = await importOriginal<typeof GoalsModule>()

  return {
    ...actual,
    refreshSessionGoal: mockRefreshSessionGoal
  }
})

import { I18nProvider } from '@/i18n'
import { clearQueuedPrompts, getQueuedPrompts } from '@/store/composer-queue'
import { $goalsBySession } from '@/store/goals'
import {
  $sessionControlBySession,
  type SessionControlEntry,
  type SessionControlGoal,
  type SessionControlHeartbeat,
  type SessionControlLoop,
  type SessionControlSnapshot
} from '@/store/session-control'
import { $sessionStates } from '@/store/session-states'
import { $todosBySession } from '@/store/todos'

import { ComposerStatusStack } from './index'

class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
vi.stubGlobal('ResizeObserver', ResizeObserverStub)

const SID = 'sess-ctrl-1'

const sampleGoal = (overrides?: Partial<SessionControlGoal>): SessionControlGoal => ({
  contract: {
    boundaries: 'test boundaries',
    constraints: 'test constraints',
    outcome: 'test outcome',
    stop_when: 'test stop condition',
    verification: 'test verification'
  },
  gates: [
    {
      attempts: 1,
      command: 'npm test',
      last_exit_code: 0,
      max_retries: 3,
      timeout_seconds: 30
    }
  ],
  max_turns: 20,
  status: 'active',
  subgoals: ['First criterion', 'Second criterion'],
  title: 'Execute complete work order',
  turns_used: 3,
  ...overrides
})

const sampleLoop = (overrides?: Partial<SessionControlLoop>): SessionControlLoop => ({
  awaiting_response: false,
  created_at: 1700000000,
  current_delay: 120,
  deferred_by_goal: false,
  interval_seconds: 120,
  last_fired_at: 1700000000,
  max_ticks: 10,
  mode: 'interval',
  next_due_at: 1700000120,
  prompt: 'Check pending reviews',
  status: 'active',
  ticks_fired: 3,
  times: 10,
  until: '',
  ...overrides
})

const sampleHeartbeat = (overrides?: Partial<SessionControlHeartbeat>): SessionControlHeartbeat => ({
  created_at: 1700000000,
  fire_count: 4,
  interval_seconds: 1800,
  last_fired_at: 1700000000,
  prompt: 'System health check',
  status: 'active',
  ...overrides
})

const sampleSnapshot = (overrides?: Partial<SessionControlSnapshot>): SessionControlSnapshot => ({
  goal: sampleGoal(),
  heartbeat: null,
  loop: null,
  revision: 'rev-goal-1',
  updated_at: 1700000000,
  ...overrides
})

const mockEntry = (overrides?: Partial<SessionControlEntry>): SessionControlEntry => ({
  capability: 'supported',
  error: null,
  loading: false,
  pendingAction: null,
  snapshot: sampleSnapshot(),
  ...overrides
})

function renderStack(sessionId: null | string = SID, props: Record<string, unknown> = {}) {
  return render(
    <MemoryRouter>
      <I18nProvider configClient={null} initialLocale="en">
        <ComposerStatusStack queue={null} sessionId={sessionId} {...props} />
      </I18nProvider>
    </MemoryRouter>
  )
}

describe('ComposerStatusStack session-control UI', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    $goalsBySession.set({})
    $sessionControlBySession.set({})
    $todosBySession.set({})
    mockRefreshSessionControl.mockResolvedValue(undefined)
    mockRefreshSessionGoal.mockResolvedValue(undefined)
  })

  afterEach(() => {
    cleanup()
    $goalsBySession.set({})
    $sessionControlBySession.set({})
    $todosBySession.set({})
    $sessionStates.set({})
    clearQueuedPrompts('stored-1')
  })

  // 1. initial structured hydration on mount; old-gateway/legacy goal remains until supported
  // 1. initial structured hydration on mount; old-gateway/legacy goal remains until supported
  it('calls refreshSessionControl on mount and keeps legacy goal while capability is unknown/unsupported', () => {
    $goalsBySession.set({
      [SID]: { status: 'active', title: 'Legacy Goal Title', updatedAt: Date.now() }
    })
    $sessionControlBySession.set({
      [SID]: { capability: 'unknown', error: null, loading: false, pendingAction: null, snapshot: null }
    })

    renderStack()

    expect(mockRefreshSessionControl).toHaveBeenCalledWith(SID)
    expect(mockRefreshSessionGoal).not.toHaveBeenCalled()
    // Legacy goal remains available while capability is unknown.
    fireEvent.click(screen.getByRole('button', { name: /Goal active/ }))
    expect(screen.getByText('Legacy Goal Title')).toBeTruthy()
  })

  // 2. supported structured goal replaces, not duplicates, legacy goal
  it('replaces legacy goal when structured session control is supported without duplication', () => {
    $goalsBySession.set({
      [SID]: { status: 'active', title: 'Legacy Goal Title', updatedAt: Date.now() }
    })
    $sessionControlBySession.set({
      [SID]: mockEntry({
        capability: 'supported',
        snapshot: sampleSnapshot({ goal: sampleGoal({ title: 'Structured Goal Title' }) })
      })
    })

    renderStack()

    fireEvent.click(screen.getByRole('button', { name: /Goal active/ }))
    expect(screen.getByText('Structured Goal Title')).toBeTruthy()
    expect(screen.queryByText('Legacy Goal Title')).toBeNull()
  })

  // 11. loop pause/resume/stop calls exact action
  it('calls loop.pause when pause loop is clicked', async () => {
    mockRunSessionControlAction.mockResolvedValue({
      type: 'exec',
      output: 'Paused loop',
      notice: null,
      message: null,
      display: null
    })

    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({
          goal: null,
          loop: sampleLoop({ status: 'active' })
        })
      })
    })

    renderStack()

    const loopMenuTrigger = screen.getByRole('button', { name: /loop actions/i })
    fireEvent.click(loopMenuTrigger)

    const pauseItem = await screen.findByRole('menuitem', { name: /pause loop/i })
    fireEvent.click(pauseItem)

    await waitFor(() => {
      expect(mockRunSessionControlAction).toHaveBeenCalledWith(SID, 'loop.pause', undefined)
    })
  })

  // 12. heartbeat interval/next/fire count and controls
  it("shows a precise live countdown and makes an overdue heartbeat's idle wait explicit", () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date(1_700_000_030_000))

    try {
      $sessionControlBySession.set({
        [SID]: mockEntry({
          snapshot: sampleSnapshot({
            goal: null,
            heartbeat: sampleHeartbeat({
              created_at: 1_700_000_000,
              fire_count: 0,
              interval_seconds: 90,
              last_fired_at: 0,
              status: 'active'
            })
          })
        })
      })

      renderStack()

      expect(screen.getByText(/Heartbeat active · every 90s · next 00:01:00/)).toBeTruthy()

      act(() => {
        vi.advanceTimersByTime(60_000)
      })

      expect(screen.getByText(/Heartbeat active · every 90s · due — waiting for idle/)).toBeTruthy()
    } finally {
      vi.useRealTimers()
    }
  })

  // 13. action rejection produces alert/live feedback
  it('displays an alert role when action rejects', async () => {
    mockRunSessionControlAction.mockRejectedValue(new Error('Gateway rejected mutation'))

    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({
          goal: sampleGoal({ status: 'active' })
        })
      })
    })

    renderStack()

    const menuTrigger = screen.getByRole('button', { name: /goal actions/i })
    fireEvent.click(menuTrigger)

    const pauseItem = await screen.findByRole('menuitem', { name: /pause goal/i })
    fireEvent.click(pauseItem)

    const alert = await screen.findByRole('alert')
    expect(alert.textContent).toContain('Gateway rejected mutation')
  })

  // 14. send dispatch submits exact hidden continuation via ChatBar callback and never calls a text slash path
  it('submits hidden continuation on send dispatch through onSubmit callback', async () => {
    const onSubmit = vi.fn().mockResolvedValue(true)

    mockRunSessionControlAction.mockResolvedValue({
      type: 'send',
      message: 'Continue toward goal: verify tests',
      output: null,
      notice: null,
      display: null
    })

    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({
          goal: sampleGoal({ status: 'paused' })
        })
      })
    })

    renderStack(SID, { onSubmit })

    const menuTrigger = screen.getByRole('button', { name: /goal actions/i })
    fireEvent.click(menuTrigger)

    const resumeItem = await screen.findByRole('menuitem', { name: /resume goal/i })
    fireEvent.click(resumeItem)

    await waitFor(() => {
      expect(onSubmit).toHaveBeenCalledWith('Continue toward goal: verify tests', {
        displayKind: 'hidden',
        sessionId: SID
      })
    })
  })

  // 21. continuation failure when onSubmit returns false
  it('reports continuation failure when onSubmit returns false or is missing', async () => {
    const onSubmit = vi.fn().mockResolvedValue(false)

    mockRunSessionControlAction.mockResolvedValue({
      display: null,
      message: 'Continue toward goal: run checks',
      notice: null,
      output: null,
      type: 'send'
    })

    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({ goal: sampleGoal({ status: 'paused' }) })
      })
    })

    renderStack(SID, { onSubmit })

    const menuTrigger = screen.getByRole('button', { name: /goal actions/i })
    fireEvent.click(menuTrigger)

    const resumeItem = await screen.findByRole('menuitem', { name: /resume goal/i })
    fireEvent.click(resumeItem)

    await waitFor(() => {
      const alert = screen.getByRole('alert')
      expect(alert.textContent).toContain('Failed to submit goal continuation')
    })
  })

  it('queues the resume continuation instead of failing when the target session is busy', async () => {
    const onSubmit = vi.fn().mockResolvedValue(false)
    $sessionStates.set({ [SID]: { busy: true, storedSessionId: 'stored-1' } as never })

    mockRunSessionControlAction.mockResolvedValue({
      display: '/goal resume',
      message: 'Continue toward goal: verify tests',
      notice: null,
      output: null,
      type: 'send'
    })
    $sessionControlBySession.set({
      [SID]: mockEntry({ snapshot: sampleSnapshot({ goal: sampleGoal({ status: 'paused' }) }) })
    })

    renderStack(SID, { onSubmit })
    fireEvent.click(screen.getByRole('button', { name: /goal actions/i }))
    fireEvent.click(await screen.findByRole('menuitem', { name: /resume goal/i }))

    await waitFor(() => {
      expect(getQueuedPrompts('stored-1')).toMatchObject([
        { displayText: '/goal resume', text: 'Continue toward goal: verify tests' }
      ])
    })
    expect(onSubmit).not.toHaveBeenCalled()
    expect(screen.queryByRole('alert')).toBeNull()
    expect(screen.getByText(/continuation queued/i)).toBeTruthy()
  })
})
