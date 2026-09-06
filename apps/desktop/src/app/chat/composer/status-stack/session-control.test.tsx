import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { Profiler } from 'react'
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
import { $goalsBySession } from '@/store/goals'
import {
  $sessionControlBySession,
  type SessionControlEntry,
  type SessionControlGoal,
  type SessionControlHeartbeat,
  type SessionControlLoop,
  type SessionControlSnapshot
} from '@/store/session-control'
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
  })

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
    // Legacy goal is rendered while capability is unknown
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

    expect(screen.getByText('Structured Goal Title')).toBeTruthy()
    expect(screen.queryByText('Legacy Goal Title')).toBeNull()
  })

  // 3. goal header state/turn metadata and expanded title
  it('displays goal header state/turn metadata and expanded title', () => {
    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({
          goal: sampleGoal({
            status: 'active',
            title: 'My Important Goal',
            turns_used: 3,
            max_turns: 20
          })
        })
      })
    })

    renderStack()

    expect(screen.getByText(/Goal active · Turn 3\/20/)).toBeTruthy()
    expect(screen.getByText('My Important Goal')).toBeTruthy()
  })

  // 4. real criteria render separately from Tasks and have no checkbox role
  it('renders criteria as neutral numbered rows without checkbox roles, distinct from Tasks', () => {
    $todosBySession.set({
      [SID]: [{ id: 'todo-1', content: 'Todo item 1', status: 'in_progress' }]
    })
    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({
          goal: sampleGoal({
            subgoals: ['First criterion', 'Second criterion']
          })
        })
      })
    })

    renderStack()

    // Criteria are visible
    expect(screen.getByText('First criterion')).toBeTruthy()
    expect(screen.getByText('Second criterion')).toBeTruthy()
    // Criteria do not have checkbox roles
    expect(screen.queryAllByRole('checkbox')).toHaveLength(0)

    // Tasks are also rendered
    expect(screen.getByText('Todo item 1')).toBeTruthy()
  })

  // 5. active/paused/waiting/done menus expose only valid actions
  it('exposes only valid actions for active goal menu', async () => {
    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({
          goal: sampleGoal({ status: 'active' })
        })
      })
    })

    renderStack()

    const menuTrigger = screen.getByRole('button', { name: /goal actions/i })
    fireEvent.pointerDown(menuTrigger, { button: 0, pointerType: 'mouse' })

    await waitFor(() => {
      expect(screen.getByRole('menuitem', { name: /view details/i })).toBeTruthy()
      expect(screen.getByRole('menuitem', { name: /add criterion/i })).toBeTruthy()
      expect(screen.getByRole('menuitem', { name: /pause goal/i })).toBeTruthy()
      expect(screen.getByRole('menuitem', { name: /clear goal/i })).toBeTruthy()
      expect(screen.queryByRole('menuitem', { name: /resume/i })).toBeNull()
      expect(screen.queryByRole('menuitem', { name: /mark done/i })).toBeNull()
    })
  })

  // 6. visible accessible menu trigger, left-click, Shift+F10, and contextmenu
  it('opens goal actions menu via click, Shift+F10, and contextmenu', async () => {
    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({
          goal: sampleGoal({ status: 'active' })
        })
      })
    })

    renderStack()

    const menuTrigger = screen.getByRole('button', { name: /goal actions/i })
    expect(menuTrigger.getAttribute('aria-haspopup')).toBe('menu')

    // Context menu on goal row
    const goalRow =
      screen.getByText('Execute complete work order').closest('[data-slot="status-section"]') || menuTrigger

    fireEvent.contextMenu(goalRow)

    await waitFor(() => {
      expect(screen.getByRole('menuitem', { name: /pause goal/i })).toBeTruthy()
    })
  })

  // 7. add criterion dialog submits exact action args
  it('opens add criterion dialog and submits exact subgoal.add action args', async () => {
    mockRunSessionControlAction.mockResolvedValue({
      type: 'exec',
      output: 'Added subgoal',
      notice: null,
      message: null,
      display: null
    })

    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({
          goal: sampleGoal({ status: 'active' })
        })
      })
    })

    renderStack()

    const addBtn = screen.getByRole('button', { name: /add criterion/i })
    fireEvent.click(addBtn)

    const textarea = await screen.findByRole('textbox', { name: /criterion/i })
    fireEvent.change(textarea, { target: { value: 'New verification rule' } })

    const submitBtn = screen.getByRole('button', { name: /^add$/i })
    fireEvent.click(submitBtn)

    await waitFor(() => {
      expect(mockRunSessionControlAction).toHaveBeenCalledWith(SID, 'subgoal.add', {
        text: 'New verification rule'
      })
    })
  })

  // 8. remove/clear/goal-clear destructive confirmations block until confirmed
  it('blocks clear goal until confirmed in ConfirmDialog', async () => {
    mockRunSessionControlAction.mockResolvedValue({
      type: 'exec',
      output: 'Cleared goal',
      notice: null,
      message: null,
      display: null
    })

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

    const clearItem = await screen.findByRole('menuitem', { name: /clear goal/i })
    fireEvent.click(clearItem)

    // ConfirmDialog opens
    const confirmBtn = await screen.findByRole('button', { name: /confirm|clear/i })
    expect(mockRunSessionControlAction).not.toHaveBeenCalled()

    fireEvent.click(confirmBtn)

    await waitFor(() => {
      expect(mockRunSessionControlAction).toHaveBeenCalledWith(SID, 'goal.clear', undefined)
    })
  })

  // 9. goal details render structured contract/wait/gates
  it('renders structured contract, wait barrier, and quality gates in goal details dialog', async () => {
    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({
          goal: sampleGoal({
            contract: {
              boundaries: 'Stay in worktree',
              constraints: 'No global installs',
              outcome: 'Clean PR',
              stop_when: 'All green',
              verification: 'Run full suite'
            },
            gates: [
              {
                attempts: 2,
                command: 'npm run test:ui',
                last_exit_code: 0,
                max_retries: 3,
                timeout_seconds: 45
              }
            ],
            wait_barrier: {
              reason: 'Waiting for worker process',
              target: 'proc-123',
              type: 'session'
            }
          })
        })
      })
    })

    renderStack()

    const viewDetailsBtn = screen.getByRole('button', { name: /view details/i })
    fireEvent.click(viewDetailsBtn)

    await waitFor(() => {
      expect(screen.getByText('Clean PR')).toBeTruthy()
      expect(screen.getByText('Run full suite')).toBeTruthy()
      expect(screen.getByText('No global installs')).toBeTruthy()
      expect(screen.getByText('Stay in worktree')).toBeTruthy()
      expect(screen.getByText('All green')).toBeTruthy()
      expect(screen.getByText('npm run test:ui')).toBeTruthy()
      expect(screen.getByText(/waiting for worker process/i)).toBeTruthy()
    })
  })

  // 10. loop cycle/cadence/next/deferred/awaiting/done states
  it('displays loop cadence, run count, and deferred state', () => {
    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({
          goal: null,
          loop: sampleLoop({
            status: 'active',
            times: 10,
            ticks_fired: 3,
            deferred_by_goal: true
          })
        })
      })
    })

    renderStack()

    expect(screen.getByText(/Loop deferred · Run 3\/10/)).toBeTruthy()
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

  it('colors the heartbeat icon for active, due, and paused states', () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date(1_700_000_030_000))

    try {
      $sessionControlBySession.set({
        [SID]: mockEntry({
          snapshot: sampleSnapshot({
            goal: null,
            heartbeat: sampleHeartbeat({
              created_at: 1_700_000_000,
              interval_seconds: 90,
              last_fired_at: 0,
              status: 'active'
            })
          })
        })
      })

      renderStack()

      const icon = () => window.document.querySelector('[data-slot="session-control-heartbeat"] .codicon-pulse')

      expect(icon()?.className).toContain('text-emerald-500')

      act(() => {
        vi.advanceTimersByTime(60_000)
      })

      expect(icon()?.className).toContain('text-amber-500')

      act(() => {
        $sessionControlBySession.set({
          [SID]: mockEntry({
            snapshot: sampleSnapshot({ heartbeat: sampleHeartbeat({ status: 'paused' }) })
          })
        })
      })

      expect(icon()?.className).toContain('text-red-500')
    } finally {
      vi.useRealTimers()
    }
  })

  it('colors the loop, goal, and criteria markers from their verified state', () => {
    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({
          goal: sampleGoal({ status: 'active', last_verdict: 'continue' }),
          loop: sampleLoop({ status: 'active' })
        })
      })
    })

    renderStack()

    const loopIcon = () => window.document.querySelector('[data-slot="session-control-loop"] .codicon-sync')
    const goalIcon = () => window.document.querySelector('[data-slot="session-control-goal"] .codicon-target')
    const criteriaMarker = () => window.document.querySelector('[data-slot="session-control-goal"] [data-slot="criteria-state-marker"]')

    expect(loopIcon()?.className).toContain('text-emerald-500')
    expect(goalIcon()?.className).toContain('text-emerald-500')
    expect(criteriaMarker()?.className).toContain('text-emerald-500')

    act(() => {
      $sessionControlBySession.set({
        [SID]: mockEntry({
          snapshot: sampleSnapshot({
            goal: sampleGoal({ status: 'paused', paused_reason: 'Paused for review' }),
            loop: sampleLoop({ status: 'paused', paused_reason: 'Paused for review' })
          })
        })
      })
    })

    expect(loopIcon()?.className).toContain('text-amber-500')
    expect(goalIcon()?.className).toContain('text-amber-500')
    expect(criteriaMarker()?.className).toContain('text-amber-500')

    act(() => {
      $sessionControlBySession.set({
        [SID]: mockEntry({
          snapshot: sampleSnapshot({
            goal: sampleGoal({ last_verdict: 'blocked', status: 'active' }),
            loop: sampleLoop({ status: 'done', last_stop_reason: 'Reached configured run count' })
          })
        })
      })
    })

    expect(screen.getByText(/Goal blocked · Turn 3\/20/)).toBeTruthy()
    expect(goalIcon()?.className).toContain('text-red-500')
    expect(criteriaMarker()?.className).toContain('text-red-500')
    expect(loopIcon()?.className).toContain('text-muted-foreground/70')
  })

  it('renders heartbeat and triggers heartbeat.pause', async () => {
    mockRunSessionControlAction.mockResolvedValue({
      type: 'exec',
      output: 'Paused heartbeat',
      notice: null,
      message: null,
      display: null
    })

    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({
          goal: null,
          heartbeat: sampleHeartbeat({ status: 'active', interval_seconds: 1800 })
        })
      })
    })

    renderStack()

    expect(screen.getByText(/Heartbeat active · every 30m/)).toBeTruthy()

    const hbMenuTrigger = screen.getByRole('button', { name: /heartbeat actions/i })
    fireEvent.click(hbMenuTrigger)

    const pauseItem = await screen.findByRole('menuitem', { name: /pause heartbeat/i })
    fireEvent.click(pauseItem)

    await waitFor(() => {
      expect(mockRunSessionControlAction).toHaveBeenCalledWith(SID, 'heartbeat.pause', undefined)
    })
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

  // 15. render-count / cross-session isolation
  it('does not re-render visible control section when an unrelated session updates', () => {
    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({ goal: sampleGoal({ title: 'My Stable Goal' }) })
      })
    })

    let commits = 0

    render(
      <Profiler id="session-control" onRender={() => void (commits += 1)}>
        <MemoryRouter>
          <I18nProvider configClient={null} initialLocale="en">
            <ComposerStatusStack queue={null} sessionId={SID} />
          </I18nProvider>
        </MemoryRouter>
      </Profiler>
    )

    const initialCommits = commits

    expect(initialCommits).toBeGreaterThan(0)

    // Update another session in the store
    act(() => {
      $sessionControlBySession.set({
        ...$sessionControlBySession.get(),
        'other-session': mockEntry({
          snapshot: sampleSnapshot({ goal: sampleGoal({ title: 'Other Goal' }) })
        })
      })
    })

    expect(commits).toBe(initialCommits)
  })

  // 16. menu trigger keyboard Shift+F10 and contextmenu
  it('opens goal actions menu via Shift+F10 keydown on the trigger', async () => {
    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({ goal: sampleGoal({ status: 'active' }) })
      })
    })

    renderStack()

    const menuTrigger = screen.getByRole('button', { name: /goal actions/i })
    fireEvent.keyDown(menuTrigger, { key: 'F10', shiftKey: true })

    await waitFor(() => {
      expect(screen.getByRole('menuitem', { name: /pause goal/i })).toBeTruthy()
    })
  })

  it('opens goal context menu via contextmenu event on the goal card', async () => {
    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({ goal: sampleGoal({ status: 'active' }) })
      })
    })

    const { container } = renderStack()

    const goalSection = container.querySelector('[data-slot="session-control-goal"]')
    expect(goalSection).toBeTruthy()
    fireEvent.contextMenu(goalSection!)

    await waitFor(() => {
      expect(screen.getByRole('menuitem', { name: /pause goal/i })).toBeTruthy()
    })
  })

  // 17. touch targets and distinguishable criterion accessible names
  it('provides distinguishable copy and remove button names for each criterion and 24px minimum sizing', () => {
    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({
          goal: sampleGoal({
            subgoals: ['First criterion', 'Second criterion']
          })
        })
      })
    })

    renderStack()

    const copyBtn1 = screen.getByRole('button', { name: /copy criterion 1/i })
    const copyBtn2 = screen.getByRole('button', { name: /copy criterion 2/i })
    const removeBtn1 = screen.getByRole('button', { name: /remove criterion 1/i })
    const removeBtn2 = screen.getByRole('button', { name: /remove criterion 2/i })

    expect(copyBtn1).toBeTruthy()
    expect(copyBtn2).toBeTruthy()
    expect(removeBtn1).toBeTruthy()
    expect(removeBtn2).toBeTruthy()

    const menuTrigger = screen.getByRole('button', { name: /goal actions/i })
    expect(menuTrigger.className).toMatch(/size-6|min-h-\[24px\]|h-6 w-6/)
    expect(copyBtn1.className).toMatch(/size-6|min-h-\[24px\]|h-6 w-6/)
  })

  // 18. localized feedback only; raw CLI output suppressed
  it('announces localized success and suppresses raw CLI dispatch output', async () => {
    mockRunSessionControlAction.mockResolvedValue({
      display: null,
      message: null,
      notice: 'Raw backend notice text #99',
      output: 'Raw backend CLI output text #42',
      type: 'exec'
    })

    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({ goal: sampleGoal({ status: 'active' }) })
      })
    })

    renderStack()

    const menuTrigger = screen.getByRole('button', { name: /goal actions/i })
    fireEvent.click(menuTrigger)

    const pauseItem = await screen.findByRole('menuitem', { name: /pause goal/i })
    fireEvent.click(pauseItem)

    await waitFor(() => {
      const liveRegion = screen.getByText(/action succeeded|action completed/i)
      expect(liveRegion).toBeTruthy()
      expect(liveRegion.closest('[aria-live="polite"]')).toBeTruthy()
      expect(screen.queryByText(/Raw backend CLI output text/i)).toBeNull()
      expect(screen.queryByText(/Raw backend notice text/i)).toBeNull()
    })
  })

  // 19. error dismissal works and reappears on changed store error
  it('dismisses store error on click and reveals a newly changed error', async () => {
    $sessionControlBySession.set({
      [SID]: mockEntry({
        error: 'Initial store error',
        snapshot: sampleSnapshot()
      })
    })

    renderStack()

    const alert = screen.getByRole('alert')
    expect(alert.textContent).toContain('Action failed: Initial store error')

    const dismissBtn = screen.getByRole('button', { name: /dismiss error/i })
    fireEvent.click(dismissBtn)

    await waitFor(() => {
      expect(screen.queryByRole('alert')).toBeNull()
    })

    act(() => {
      $sessionControlBySession.set({
        [SID]: mockEntry({
          error: 'Subsequent new store error',
          snapshot: sampleSnapshot()
        })
      })
    })

    await waitFor(() => {
      const newAlert = screen.getByRole('alert')
      expect(newAlert.textContent).toContain('Subsequent new store error')
    })
  })

  // 20. truthful async dialogs / add criterion remains open on failure
  it('keeps add criterion dialog open with preserved text when action rejects', async () => {
    mockRunSessionControlAction.mockRejectedValue(new Error('Validation failed'))

    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({ goal: sampleGoal({ status: 'active' }) })
      })
    })

    renderStack()

    const addBtn = screen.getByRole('button', { name: /add criterion/i })
    fireEvent.click(addBtn)

    const textarea = await screen.findByRole('textbox', { name: /criterion/i })
    fireEvent.change(textarea, { target: { value: 'Keep this text' } })

    const submitBtn = screen.getByRole('button', { name: /^add$/i })
    fireEvent.click(submitBtn)

    await waitFor(() => {
      expect(screen.getByRole('alert')).toBeTruthy()
      expect(screen.getByRole('dialog')).toBeTruthy()
      expect((screen.getByRole('textbox', { name: /criterion/i }) as HTMLTextAreaElement).value).toBe('Keep this text')
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

  it.each([null, ''])('reports continuation failure when a send dispatch has an empty message', async message => {
    const onSubmit = vi.fn().mockResolvedValue(true)

    mockRunSessionControlAction.mockResolvedValue({
      display: null,
      message,
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

    fireEvent.click(screen.getByRole('button', { name: /goal actions/i }))
    fireEvent.click(await screen.findByRole('menuitem', { name: /resume goal/i }))

    await waitFor(() => {
      expect(screen.getByRole('alert').textContent).toContain('Failed to submit goal continuation')
    })
    expect(onSubmit).not.toHaveBeenCalled()
    expect(screen.queryByText(/action succeeded|action completed/i)).toBeNull()
  })

  // 22. destructive confirmations for remove, clear all, stop loop, clear heartbeat
  it('blocks remove criterion until confirmed in ConfirmDialog with 1-based index', async () => {
    mockRunSessionControlAction.mockResolvedValue({
      display: null,
      message: null,
      notice: null,
      output: null,
      type: 'exec'
    })

    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({
          goal: sampleGoal({ subgoals: ['First subgoal'] })
        })
      })
    })

    renderStack()

    const removeBtn = screen.getByRole('button', { name: /remove criterion 1/i })
    fireEvent.click(removeBtn)

    const confirmBtn = await screen.findByRole('button', { name: /confirm|remove/i })
    expect(mockRunSessionControlAction).not.toHaveBeenCalled()

    fireEvent.click(confirmBtn)

    await waitFor(() => {
      expect(mockRunSessionControlAction).toHaveBeenCalledWith(SID, 'subgoal.remove', { index: 1 })
    })
  })

  it('blocks clear all criteria until confirmed in ConfirmDialog', async () => {
    mockRunSessionControlAction.mockResolvedValue({
      display: null,
      message: null,
      notice: null,
      output: null,
      type: 'exec'
    })

    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({
          goal: sampleGoal({ subgoals: ['Subgoal A', 'Subgoal B'] })
        })
      })
    })

    renderStack()

    const clearAllBtn = screen.getByRole('button', { name: /clear all criteria/i })
    fireEvent.click(clearAllBtn)

    const confirmBtn = await screen.findByRole('button', { name: /confirm|clear/i })
    expect(mockRunSessionControlAction).not.toHaveBeenCalled()

    fireEvent.click(confirmBtn)

    await waitFor(() => {
      expect(mockRunSessionControlAction).toHaveBeenCalledWith(SID, 'subgoal.clear', undefined)
    })
  })

  it('blocks stop active loop until confirmed in ConfirmDialog', async () => {
    mockRunSessionControlAction.mockResolvedValue({
      display: null,
      message: null,
      notice: null,
      output: null,
      type: 'exec'
    })

    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({ goal: null, loop: sampleLoop({ status: 'active' }) })
      })
    })

    renderStack()

    const loopMenuTrigger = screen.getByRole('button', { name: /loop actions/i })
    fireEvent.click(loopMenuTrigger)

    const stopItem = await screen.findByRole('menuitem', { name: /stop loop/i })
    fireEvent.click(stopItem)

    const confirmBtn = await screen.findByRole('button', { name: /confirm|stop/i })
    expect(mockRunSessionControlAction).not.toHaveBeenCalled()

    fireEvent.click(confirmBtn)

    await waitFor(() => {
      expect(mockRunSessionControlAction).toHaveBeenCalledWith(SID, 'loop.stop', undefined)
    })
  })

  it('blocks clear heartbeat until confirmed in ConfirmDialog', async () => {
    mockRunSessionControlAction.mockResolvedValue({
      display: null,
      message: null,
      notice: null,
      output: null,
      type: 'exec'
    })

    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({ goal: null, heartbeat: sampleHeartbeat({ status: 'active' }) })
      })
    })

    renderStack()

    const hbMenuTrigger = screen.getByRole('button', { name: /heartbeat actions/i })
    fireEvent.click(hbMenuTrigger)

    const clearItem = await screen.findByRole('menuitem', { name: /clear heartbeat/i })
    fireEvent.click(clearItem)

    const confirmBtn = await screen.findByRole('button', { name: /confirm|clear/i })
    expect(mockRunSessionControlAction).not.toHaveBeenCalled()

    fireEvent.click(confirmBtn)

    await waitFor(() => {
      expect(mockRunSessionControlAction).toHaveBeenCalledWith(SID, 'heartbeat.clear', undefined)
    })
  })

  // 23. state-aware menus for goal and loop
  it('exposes state-aware menu items for paused, waiting, and done goals', async () => {
    // Paused goal
    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({ goal: sampleGoal({ status: 'paused' }) })
      })
    })

    const { unmount } = renderStack()
    const pausedTrigger = screen.getByRole('button', { name: /goal actions/i })
    fireEvent.click(pausedTrigger)

    await waitFor(() => {
      expect(screen.getByRole('menuitem', { name: /resume goal/i })).toBeTruthy()
      expect(screen.queryByRole('menuitem', { name: /pause goal/i })).toBeNull()
    })
    unmount()

    // Waiting goal (has wait barrier)
    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({
          goal: sampleGoal({
            status: 'active',
            wait_barrier: { reason: 'waiting', target: 'worker', type: 'session' }
          })
        })
      })
    })

    const { unmount: unmountWait } = renderStack()
    const waitTrigger = screen.getByRole('button', { name: /goal actions/i })
    fireEvent.click(waitTrigger)

    await waitFor(() => {
      expect(screen.getByRole('menuitem', { name: /resume now/i })).toBeTruthy()
      expect(screen.getByRole('menuitem', { name: /pause goal/i })).toBeTruthy()
      expect(screen.queryByRole('menuitem', { name: /resume goal/i })).toBeNull()
    })
    unmountWait()

    // Done goal
    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({ goal: sampleGoal({ status: 'done' }) })
      })
    })

    renderStack()
    const doneTrigger = screen.getByRole('button', { name: /goal actions/i })
    fireEvent.click(doneTrigger)

    await waitFor(() => {
      expect(screen.getByRole('menuitem', { name: /clear goal/i })).toBeTruthy()
      expect(screen.queryByRole('menuitem', { name: /pause goal/i })).toBeNull()
      expect(screen.queryByRole('menuitem', { name: /resume goal/i })).toBeNull()
      expect(screen.queryByRole('menuitem', { name: /add criterion/i })).toBeNull()
    })
  })

  it('exposes state-aware header and menus for loop states', async () => {
    // Loop active
    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({
          goal: null,
          loop: sampleLoop({ awaiting_response: false, status: 'active' })
        })
      })
    })

    const { unmount: unmountActive } = renderStack()
    expect(screen.getByText(/Loop active/i)).toBeTruthy()
    unmountActive()

    // Loop awaiting response
    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({
          goal: null,
          loop: sampleLoop({ awaiting_response: true, status: 'active' })
        })
      })
    })

    const { unmount: unmountWaiting } = renderStack()
    expect(screen.getByText(/Awaiting response/i)).toBeTruthy()
    unmountWaiting()

    // Loop deferred
    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({
          goal: null,
          loop: sampleLoop({ deferred_by_goal: true, status: 'active' })
        })
      })
    })

    const { unmount: unmountDeferred } = renderStack()
    expect(screen.getByText(/Loop deferred/i)).toBeTruthy()
    unmountDeferred()

    // Loop paused
    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({
          goal: null,
          loop: sampleLoop({ status: 'paused' })
        })
      })
    })

    const { unmount: unmountPaused } = renderStack()
    expect(screen.getByText(/Loop paused/i)).toBeTruthy()
    unmountPaused()

    // Loop done
    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({
          goal: null,
          loop: sampleLoop({ status: 'done' })
        })
      })
    })

    renderStack()
    expect(screen.getByText(/Loop finished/i)).toBeTruthy()
    const loopMenuTrigger = screen.getByRole('button', { name: /loop actions/i })
    fireEvent.click(loopMenuTrigger)

    await waitFor(() => {
      expect(screen.getByRole('menuitem', { name: /dismiss loop/i })).toBeTruthy()
      expect(screen.queryByRole('menuitem', { name: /pause loop/i })).toBeNull()
      expect(screen.queryByRole('menuitem', { name: /resume loop/i })).toBeNull()
    })
  })

  // 24. quality gates middle dot and hiding empty details
  it('renders styled middle dot in quality gates metadata and hides details when empty', async () => {
    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({
          goal: sampleGoal({
            contract: { boundaries: '', constraints: '', outcome: '', stop_when: '', verification: '' },
            gates: [{ attempts: 1, command: 'gate-cmd', last_exit_code: 0, max_retries: 2, timeout_seconds: 30 }]
          })
        })
      })
    })

    const { unmount } = renderStack()
    const viewDetailsBtn = screen.getByRole('button', { name: /view details/i })
    fireEvent.click(viewDetailsBtn)

    await waitFor(() => {
      expect(screen.getByText(/1\/2 attempts/i)).toBeTruthy()
      expect(screen.queryByText('?')).toBeNull()
    })
    unmount()

    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({
          goal: sampleGoal({
            contract: { boundaries: '', constraints: '', outcome: '', stop_when: '', verification: '' },
            gates: []
          })
        })
      })
    })

    renderStack()
    expect(screen.queryByRole('button', { name: /view details/i })).toBeNull()
  })

  it('does not render a leading menu separator for a done goal without details', async () => {
    $sessionControlBySession.set({
      [SID]: mockEntry({
        snapshot: sampleSnapshot({
          goal: sampleGoal({
            contract: { boundaries: '', constraints: '', outcome: '', stop_when: '', verification: '' },
            gates: [],
            status: 'done'
          })
        })
      })
    })

    renderStack()

    fireEvent.click(screen.getByRole('button', { name: /goal actions/i }))

    await screen.findByRole('menuitem', { name: /clear goal/i })
    expect(screen.queryByRole('separator')).toBeNull()
  })
})
