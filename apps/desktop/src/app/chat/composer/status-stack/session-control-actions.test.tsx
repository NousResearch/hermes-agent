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
import { clearQueuedPrompts } from '@/store/composer-queue'
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

const SID = 'sess-ctrl-act-1'
const SID_OTHER = 'sess-ctrl-act-other'

const sampleGoal = (overrides?: Partial<SessionControlGoal>): SessionControlGoal => ({
  contract: {
    boundaries: '',
    constraints: '',
    outcome: '',
    stop_when: '',
    verification: ''
  },
  gates: [],
  max_turns: 20,
  status: 'active',
  subgoals: ['First criterion', 'Second criterion'],
  title: 'Test goal',
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
  revision: 'rev-1',
  updated_at: 1700000000,
  ...overrides
})

const mockEntry = (overrides?: Partial<SessionControlEntry>): SessionControlEntry => ({
  capability: 'supported',
  error: null,
  loading: false,
  pendingAction: null,
  actionError: null,
  snapshot: sampleSnapshot(),
  ...overrides
})

const execDispatch = {
  type: 'exec' as const,
  output: 'ok',
  notice: null,
  message: null,
  display: null
}

function renderStack(
  sessionId: null | string = SID,
  props: Record<string, unknown> = {}
) {
  return render(
    <MemoryRouter>
      <I18nProvider configClient={null} initialLocale="en">
        <ComposerStatusStack queue={null} sessionId={sessionId} {...props} />
      </I18nProvider>
    </MemoryRouter>
  )
}

function resetStores() {
  $goalsBySession.set({})
  $sessionControlBySession.set({})
  $todosBySession.set({})
}

function clearStoresAndMocks() {
  vi.clearAllMocks()
  resetStores()
  mockRefreshSessionControl.mockResolvedValue(undefined)
  mockRefreshSessionGoal.mockResolvedValue(undefined)
}

function openGoalMenu() {
  fireEvent.click(screen.getByRole('button', { name: /goal actions/i }))
}

function openLoopMenu() {
  fireEvent.click(screen.getByRole('button', { name: /loop actions/i }))
}

function openHeartbeatMenu() {
  fireEvent.click(screen.getByRole('button', { name: /heartbeat actions/i }))
}

describe('Session control mouse actions — goal', () => {
  beforeEach(clearStoresAndMocks)

  afterEach(() => {
    cleanup()
    resetStores()
    $sessionStates.set({})
    clearQueuedPrompts(SID)
  })

  it('dispatches goal.pause with correct session id when Pause goal is clicked', async () => {
    mockRunSessionControlAction.mockResolvedValue(execDispatch)
    const onSubmit = vi.fn()

    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({ goal: sampleGoal({ status: 'active' }) })
      })
    })

    renderStack(SID, { onSubmit })

    openGoalMenu()
    fireEvent.click(await screen.findByRole('menuitem', { name: /pause goal/i }))

    await waitFor(() => {
      expect(mockRunSessionControlAction).toHaveBeenCalledWith(SID, 'goal.pause', undefined)
    })
    expect(mockRunSessionControlAction).toHaveBeenCalledTimes(1)
    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('dispatches goal.resume with correct session id when Resume goal is clicked', async () => {
    mockRunSessionControlAction.mockResolvedValue(execDispatch)
    const onSubmit = vi.fn()

    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({ goal: sampleGoal({ status: 'paused' }) })
      })
    })

    renderStack(SID, { onSubmit })

    openGoalMenu()
    fireEvent.click(await screen.findByRole('menuitem', { name: /resume goal/i }))

    await waitFor(() => {
      expect(mockRunSessionControlAction).toHaveBeenCalledWith(SID, 'goal.resume', undefined)
    })
    expect(mockRunSessionControlAction).toHaveBeenCalledTimes(1)
    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('dispatches goal.unwait when Resume now is clicked on a waiting goal', async () => {
    mockRunSessionControlAction.mockResolvedValue(execDispatch)

    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({
          goal: sampleGoal({
            status: 'active',
            wait_barrier: { reason: 'waiting for upstream', type: 'until', until_at: 1700009999 }
          })
        })
      })
    })

    renderStack()

    openGoalMenu()
    fireEvent.click(await screen.findByRole('menuitem', { name: /resume now/i }))

    await waitFor(() => {
      expect(mockRunSessionControlAction).toHaveBeenCalledWith(SID, 'goal.unwait', undefined)
    })
    expect(mockRunSessionControlAction).toHaveBeenCalledTimes(1)
  })

  it('dispatches goal.pause from waiting goal menu (Pause goal item is visible alongside Resume now)', async () => {
    mockRunSessionControlAction.mockResolvedValue(execDispatch)

    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({
          goal: sampleGoal({
            status: 'active',
            wait_barrier: { reason: '', type: 'until', until_at: 1700009999 }
          })
        })
      })
    })

    renderStack()

    openGoalMenu()
    fireEvent.click(await screen.findByRole('menuitem', { name: /pause goal/i }))

    await waitFor(() => {
      expect(mockRunSessionControlAction).toHaveBeenCalledWith(SID, 'goal.pause', undefined)
    })
    expect(mockRunSessionControlAction).toHaveBeenCalledTimes(1)
  })

  it('does not dispatch goal.resume when goal is active (menu shows Pause, not Resume)', () => {
    mockRunSessionControlAction.mockResolvedValue(execDispatch)
    const onSubmit = vi.fn()

    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({ goal: sampleGoal({ status: 'active' }) })
      })
    })

    renderStack(SID, { onSubmit })

    openGoalMenu()

    expect(screen.queryByRole('menuitem', { name: /resume goal/i })).toBeNull()
    expect(mockRunSessionControlAction).not.toHaveBeenCalled()
    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('does not dispatch goal.pause when goal is paused (menu shows Resume, not Pause)', () => {
    mockRunSessionControlAction.mockResolvedValue(execDispatch)
    const onSubmit = vi.fn()

    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({ goal: sampleGoal({ status: 'paused' }) })
      })
    })

    renderStack(SID, { onSubmit })

    openGoalMenu()

    expect(screen.queryByRole('menuitem', { name: /pause goal/i })).toBeNull()
    expect(mockRunSessionControlAction).not.toHaveBeenCalled()
    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('does not dispatch goal.unwait when goal is not waiting', () => {
    mockRunSessionControlAction.mockResolvedValue(execDispatch)
    const onSubmit = vi.fn()

    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({ goal: sampleGoal({ status: 'active' }) })
      })
    })

    renderStack(SID, { onSubmit })

    openGoalMenu()

    expect(screen.queryByRole('menuitem', { name: /resume now/i })).toBeNull()
    expect(mockRunSessionControlAction).not.toHaveBeenCalled()
    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('shows Clear goal confirmation dialog and dispatches goal.clear on confirm', async () => {
    mockRunSessionControlAction.mockResolvedValue(execDispatch)
    const onSubmit = vi.fn()

    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({ goal: sampleGoal({ status: 'active' }) })
      })
    })

    renderStack(SID, { onSubmit })

    openGoalMenu()
    fireEvent.click(await screen.findByRole('menuitem', { name: /clear goal/i }))

    const dialog = await screen.findByRole('dialog')
    expect(dialog.textContent).toContain('Clear goal?')
    expect(dialog.textContent).toContain('Are you sure you want to clear the active goal?')
    expect(mockRunSessionControlAction).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: /clear goal/i }))

    await waitFor(() => {
      expect(mockRunSessionControlAction).toHaveBeenCalledWith(SID, 'goal.clear', undefined)
    })
    expect(mockRunSessionControlAction).toHaveBeenCalledTimes(1)
    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('dismisses Clear goal dialog on cancel without dispatching', async () => {
    mockRunSessionControlAction.mockResolvedValue(execDispatch)
    const onSubmit = vi.fn()

    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({ goal: sampleGoal({ status: 'active' }) })
      })
    })

    renderStack(SID, { onSubmit })

    openGoalMenu()
    fireEvent.click(await screen.findByRole('menuitem', { name: /clear goal/i }))

    const dialog = await screen.findByRole('dialog')
    expect(mockRunSessionControlAction).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: /^cancel$/i }))

    await waitFor(() => {
      expect(screen.queryByRole('dialog')).toBeNull()
    })
    expect(mockRunSessionControlAction).not.toHaveBeenCalled()
    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('confirm after session switch: dispatches only to original SID, never to the new one', async () => {
    mockRunSessionControlAction.mockResolvedValue(execDispatch)

    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({ goal: sampleGoal({ status: 'active', title: 'Session A goal' }) })
      }),
      [SID_OTHER]: mockEntry({
        snapshot: sampleSnapshot({ goal: sampleGoal({ status: 'active', title: 'Session B goal' }) })
      })
    })

    const { rerender } = renderStack(SID)

    openGoalMenu()
    fireEvent.click(await screen.findByRole('menuitem', { name: /clear goal/i }))

    const dialog = await screen.findByRole('dialog')
    expect(dialog.textContent).toContain('Clear goal?')
    expect(mockRunSessionControlAction).not.toHaveBeenCalled()

    // Session changes while dialog is open — rerender with both sessions populated
    await act(async () => {
      rerender(
        <MemoryRouter>
          <I18nProvider configClient={null} initialLocale="en">
            <ComposerStatusStack queue={null} sessionId={SID_OTHER} />
          </I18nProvider>
        </MemoryRouter>
      )
    })

    const dialogAfterSwitch = screen.queryByRole('dialog')

    if (!dialogAfterSwitch) {
      // Dialog auto-dismissed on session switch — zero dispatch is the invariant
      expect(mockRunSessionControlAction).not.toHaveBeenCalled()

      return
    }

    // Dialog persists: attempt Confirm — the captured onConfirm closure closes
    // over handleAction from the original SID render, so dispatch targets SID.
    fireEvent.click(screen.getByRole('button', { name: /clear goal/i }))

    await waitFor(() => {
      expect(mockRunSessionControlAction).toHaveBeenCalled()
    })

    // Exactly one dispatch, to the original session
    expect(mockRunSessionControlAction).toHaveBeenCalledTimes(1)
    expect(mockRunSessionControlAction).toHaveBeenCalledWith(SID, 'goal.clear', undefined)
    // Must never dispatch to the session that was switched to
    expect(mockRunSessionControlAction).not.toHaveBeenCalledWith(SID_OTHER, expect.anything(), expect.anything())
  })
})

describe('Session control mouse actions — criteria', () => {
  beforeEach(clearStoresAndMocks)

  afterEach(() => {
    cleanup()
    resetStores()
    $sessionStates.set({})
    clearQueuedPrompts(SID)
  })

  it('dispatches subgoal.remove with correct index on Remove criterion confirm', async () => {
    mockRunSessionControlAction.mockResolvedValue(execDispatch)

    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({
          goal: sampleGoal({ subgoals: ['Alpha', 'Beta', 'Gamma'] })
        })
      })
    })

    renderStack()

    const removeBtn = screen.getByRole('button', { name: /remove criterion 2/i })
    fireEvent.click(removeBtn)

    const dialog = await screen.findByRole('dialog')
    expect(dialog.textContent).toContain('Remove criterion 2?')
    expect(dialog.textContent).toContain('Are you sure you want to remove criterion 2?')
    expect(mockRunSessionControlAction).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: /remove criterion 2/i }))

    await waitFor(() => {
      expect(mockRunSessionControlAction).toHaveBeenCalledWith(SID, 'subgoal.remove', { index: 2 })
    })
    expect(mockRunSessionControlAction).toHaveBeenCalledTimes(1)
  })

  it('dismisses Remove criterion dialog on cancel without dispatching', async () => {
    mockRunSessionControlAction.mockResolvedValue(execDispatch)

    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({
          goal: sampleGoal({ subgoals: ['Alpha'] })
        })
      })
    })

    renderStack()

    fireEvent.click(screen.getByRole('button', { name: /remove criterion 1/i }))
    const dialog = await screen.findByRole('dialog')
    expect(mockRunSessionControlAction).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: /^cancel$/i }))

    await waitFor(() => {
      expect(screen.queryByRole('dialog')).toBeNull()
    })
    expect(mockRunSessionControlAction).not.toHaveBeenCalled()
  })

  it('dispatches subgoal.clear on Clear all criteria confirm', async () => {
    mockRunSessionControlAction.mockResolvedValue(execDispatch)

    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({
          goal: sampleGoal({ subgoals: ['A', 'B'] })
        })
      })
    })

    renderStack()

    fireEvent.click(screen.getByRole('button', { name: /clear all criteria/i }))

    const dialog = await screen.findByRole('dialog')
    expect(dialog.textContent).toContain('Clear all criteria?')
    expect(dialog.textContent).toContain('Are you sure you want to remove all criteria from this goal?')
    expect(mockRunSessionControlAction).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: /clear all criteria/i }))

    await waitFor(() => {
      expect(mockRunSessionControlAction).toHaveBeenCalledWith(SID, 'subgoal.clear', undefined)
    })
    expect(mockRunSessionControlAction).toHaveBeenCalledTimes(1)
  })

  it('dismisses Clear all criteria dialog on cancel without dispatching', async () => {
    mockRunSessionControlAction.mockResolvedValue(execDispatch)

    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({
          goal: sampleGoal({ subgoals: ['A'] })
        })
      })
    })

    renderStack()

    fireEvent.click(screen.getByRole('button', { name: /clear all criteria/i }))
    const dialog = await screen.findByRole('dialog')
    expect(mockRunSessionControlAction).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: /^cancel$/i }))

    await waitFor(() => {
      expect(screen.queryByRole('dialog')).toBeNull()
    })
    expect(mockRunSessionControlAction).not.toHaveBeenCalled()
  })

  it('dispatches subgoal.add with text when Add criterion is submitted', async () => {
    mockRunSessionControlAction.mockResolvedValue(execDispatch)

    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({
          goal: sampleGoal({ subgoals: [] })
        })
      })
    })

    renderStack()

    fireEvent.click(screen.getByRole('button', { name: /add criterion/i }))

    const dialog = await screen.findByRole('dialog')
    const textarea = dialog.querySelector('textarea')!
    fireEvent.change(textarea, { target: { value: 'New criterion text' } })

    expect(mockRunSessionControlAction).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: /^add$/i }))

    await waitFor(() => {
      expect(mockRunSessionControlAction).toHaveBeenCalledWith(SID, 'subgoal.add', { text: 'New criterion text' })
    })
    expect(mockRunSessionControlAction).toHaveBeenCalledTimes(1)
  })

  it('cancels Add criterion after typing without dispatching', async () => {
    mockRunSessionControlAction.mockResolvedValue(execDispatch)

    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({
          goal: sampleGoal({ subgoals: [] })
        })
      })
    })

    renderStack()

    fireEvent.click(screen.getByRole('button', { name: /add criterion/i }))

    const dialog = await screen.findByRole('dialog')
    const textarea = dialog.querySelector('textarea')!
    fireEvent.change(textarea, { target: { value: 'Typed but will be discarded' } })

    expect(mockRunSessionControlAction).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: /^cancel$/i }))

    await waitFor(() => {
      expect(screen.queryByRole('dialog')).toBeNull()
    })
    expect(mockRunSessionControlAction).not.toHaveBeenCalled()
  })

  it('does not show Clear all criteria button when no criteria exist', () => {
    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({
          goal: sampleGoal({ subgoals: [] })
        })
      })
    })

    renderStack()

    expect(screen.queryByRole('button', { name: /clear all criteria/i })).toBeNull()
  })
})

describe('Session control mouse actions — loop', () => {
  beforeEach(clearStoresAndMocks)

  afterEach(() => {
    cleanup()
    resetStores()
    $sessionStates.set({})
    clearQueuedPrompts(SID)
  })

  it('dispatches loop.pause when Pause loop is clicked', async () => {
    mockRunSessionControlAction.mockResolvedValue(execDispatch)

    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({
          goal: null,
          loop: sampleLoop({ status: 'active' })
        })
      })
    })

    renderStack()

    openLoopMenu()
    fireEvent.click(await screen.findByRole('menuitem', { name: /pause loop/i }))

    await waitFor(() => {
      expect(mockRunSessionControlAction).toHaveBeenCalledWith(SID, 'loop.pause', undefined)
    })
    expect(mockRunSessionControlAction).toHaveBeenCalledTimes(1)
  })

  it('dispatches loop.resume when Resume loop is clicked', async () => {
    mockRunSessionControlAction.mockResolvedValue(execDispatch)

    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({
          goal: null,
          loop: sampleLoop({ status: 'paused' })
        })
      })
    })

    renderStack()

    openLoopMenu()
    fireEvent.click(await screen.findByRole('menuitem', { name: /resume loop/i }))

    await waitFor(() => {
      expect(mockRunSessionControlAction).toHaveBeenCalledWith(SID, 'loop.resume', undefined)
    })
    expect(mockRunSessionControlAction).toHaveBeenCalledTimes(1)
  })

  it('shows Stop loop confirmation and dispatches loop.stop on confirm', async () => {
    mockRunSessionControlAction.mockResolvedValue(execDispatch)

    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({
          goal: null,
          loop: sampleLoop({ status: 'active' })
        })
      })
    })

    renderStack()

    openLoopMenu()
    fireEvent.click(await screen.findByRole('menuitem', { name: /stop loop/i }))

    const dialog = await screen.findByRole('dialog')
    expect(dialog.textContent).toContain('Stop loop?')
    expect(dialog.textContent).toContain('Are you sure you want to stop this loop?')
    expect(mockRunSessionControlAction).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: /stop loop/i }))

    await waitFor(() => {
      expect(mockRunSessionControlAction).toHaveBeenCalledWith(SID, 'loop.stop', undefined)
    })
    expect(mockRunSessionControlAction).toHaveBeenCalledTimes(1)
  })

  it('dismisses Stop loop dialog on cancel without dispatching', async () => {
    mockRunSessionControlAction.mockResolvedValue(execDispatch)

    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({
          goal: null,
          loop: sampleLoop({ status: 'active' })
        })
      })
    })

    renderStack()

    openLoopMenu()
    fireEvent.click(await screen.findByRole('menuitem', { name: /stop loop/i }))
    const dialog = await screen.findByRole('dialog')
    expect(mockRunSessionControlAction).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: /^cancel$/i }))

    await waitFor(() => {
      expect(screen.queryByRole('dialog')).toBeNull()
    })
    expect(mockRunSessionControlAction).not.toHaveBeenCalled()
  })

  it('dispatches loop.stop immediately without confirmation when loop is done', async () => {
    mockRunSessionControlAction.mockResolvedValue(execDispatch)

    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({
          goal: null,
          loop: sampleLoop({ status: 'done' })
        })
      })
    })

    renderStack()

    openLoopMenu()
    fireEvent.click(await screen.findByRole('menuitem', { name: /dismiss loop/i }))

    await waitFor(() => {
      expect(mockRunSessionControlAction).toHaveBeenCalledWith(SID, 'loop.stop', undefined)
    })
    expect(mockRunSessionControlAction).toHaveBeenCalledTimes(1)
    expect(screen.queryByRole('dialog')).toBeNull()
  })

  it('does not show Pause loop when loop is paused (shows Resume instead)', () => {
    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({
          goal: null,
          loop: sampleLoop({ status: 'paused' })
        })
      })
    })

    renderStack()

    openLoopMenu()

    expect(screen.queryByRole('menuitem', { name: /pause loop/i })).toBeNull()
  })

  it('does not show Resume loop when loop is active (shows Pause instead)', () => {
    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({
          goal: null,
          loop: sampleLoop({ status: 'active' })
        })
      })
    })

    renderStack()

    openLoopMenu()

    expect(screen.queryByRole('menuitem', { name: /resume loop/i })).toBeNull()
  })
})

describe('Session control mouse actions — heartbeat', () => {
  beforeEach(clearStoresAndMocks)

  afterEach(() => {
    cleanup()
    resetStores()
    $sessionStates.set({})
    clearQueuedPrompts(SID)
  })

  it('dispatches heartbeat.pause when Pause heartbeat is clicked', async () => {
    mockRunSessionControlAction.mockResolvedValue(execDispatch)

    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({
          goal: null,
          heartbeat: sampleHeartbeat({ status: 'active' })
        })
      })
    })

    renderStack()

    openHeartbeatMenu()
    fireEvent.click(await screen.findByRole('menuitem', { name: /pause heartbeat/i }))

    await waitFor(() => {
      expect(mockRunSessionControlAction).toHaveBeenCalledWith(SID, 'heartbeat.pause', undefined)
    })
    expect(mockRunSessionControlAction).toHaveBeenCalledTimes(1)
  })

  it('dispatches heartbeat.resume when Resume heartbeat is clicked', async () => {
    mockRunSessionControlAction.mockResolvedValue(execDispatch)

    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({
          goal: null,
          heartbeat: sampleHeartbeat({ status: 'paused' })
        })
      })
    })

    renderStack()

    openHeartbeatMenu()
    fireEvent.click(await screen.findByRole('menuitem', { name: /resume heartbeat/i }))

    await waitFor(() => {
      expect(mockRunSessionControlAction).toHaveBeenCalledWith(SID, 'heartbeat.resume', undefined)
    })
    expect(mockRunSessionControlAction).toHaveBeenCalledTimes(1)
  })

  it('shows Clear heartbeat confirmation and dispatches heartbeat.clear on confirm', async () => {
    mockRunSessionControlAction.mockResolvedValue(execDispatch)

    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({
          goal: null,
          heartbeat: sampleHeartbeat({ status: 'active' })
        })
      })
    })

    renderStack()

    openHeartbeatMenu()
    fireEvent.click(await screen.findByRole('menuitem', { name: /clear heartbeat/i }))

    const dialog = await screen.findByRole('dialog')
    expect(dialog.textContent).toContain('Clear heartbeat?')
    expect(dialog.textContent).toContain('Are you sure you want to clear this heartbeat?')
    expect(mockRunSessionControlAction).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: /clear heartbeat/i }))

    await waitFor(() => {
      expect(mockRunSessionControlAction).toHaveBeenCalledWith(SID, 'heartbeat.clear', undefined)
    })
    expect(mockRunSessionControlAction).toHaveBeenCalledTimes(1)
  })

  it('dismisses Clear heartbeat dialog on cancel without dispatching', async () => {
    mockRunSessionControlAction.mockResolvedValue(execDispatch)

    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({
          goal: null,
          heartbeat: sampleHeartbeat({ status: 'active' })
        })
      })
    })

    renderStack()

    openHeartbeatMenu()
    fireEvent.click(await screen.findByRole('menuitem', { name: /clear heartbeat/i }))
    const dialog = await screen.findByRole('dialog')
    expect(mockRunSessionControlAction).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: /^cancel$/i }))

    await waitFor(() => {
      expect(screen.queryByRole('dialog')).toBeNull()
    })
    expect(mockRunSessionControlAction).not.toHaveBeenCalled()
  })

  it('does not show Pause heartbeat when heartbeat is paused (shows Resume instead)', () => {
    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({
          goal: null,
          heartbeat: sampleHeartbeat({ status: 'paused' })
        })
      })
    })

    renderStack()

    openHeartbeatMenu()

    expect(screen.queryByRole('menuitem', { name: /pause heartbeat/i })).toBeNull()
  })

  it('does not show Resume heartbeat when heartbeat is active (shows Pause instead)', () => {
    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({
          goal: null,
          heartbeat: sampleHeartbeat({ status: 'active' })
        })
      })
    })

    renderStack()

    openHeartbeatMenu()

    expect(screen.queryByRole('menuitem', { name: /resume heartbeat/i })).toBeNull()
  })
})
