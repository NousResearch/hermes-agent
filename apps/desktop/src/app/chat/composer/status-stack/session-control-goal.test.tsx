import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
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

const SID = 'sess-goal-interrupted'
const INTERRUPTED_AT = 1_700_000_000

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
  subgoals: [],
  title: 'Ship the crash-resume flow',
  turns_used: 3,
  ...overrides
})

const mockEntry = (goal: SessionControlGoal, overrides?: Partial<SessionControlEntry>): SessionControlEntry => {
  const snapshot: SessionControlSnapshot = {
    goal,
    heartbeat: null,
    loop: null,
    revision: 'rev-1',
    updated_at: 1_700_000_000
  }

  return {
    capability: 'supported',
    error: null,
    loading: false,
    pendingAction: null,
    snapshot,
    ...overrides
  }
}

function renderStack(props: Record<string, unknown> = {}) {
  return render(
    <MemoryRouter>
      <I18nProvider configClient={null} initialLocale="en">
        <ComposerStatusStack queue={null} sessionId={SID} {...props} />
      </I18nProvider>
    </MemoryRouter>
  )
}

describe('goal card interrupted state', () => {
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
    vi.useRealTimers()
    $goalsBySession.set({})
    $sessionControlBySession.set({})
    $todosBySession.set({})
    $sessionStates.set({})
  })

  it('labels an active goal whose turn was killed as interrupted and offers a one-click resume', () => {
    $sessionControlBySession.set({
      [SID]: mockEntry(sampleGoal({ interrupted_at: INTERRUPTED_AT }))
    })

    renderStack()

    expect(screen.getByRole('button', { name: /Interrupted · Turn 3\/20/ })).toBeTruthy()
    expect(screen.queryByRole('button', { name: /Goal active/ })).toBeNull()
    expect(screen.getByRole('button', { name: 'Resume goal' })).toBeTruthy()
  })

  it('continues the goal without resetting the turn budget and submits the continuation hidden', async () => {
    const onSubmit = vi.fn().mockResolvedValue(true)

    mockRunSessionControlAction.mockResolvedValue({
      display: '/goal continue',
      message: 'Continue toward goal: Ship the crash-resume flow',
      notice: null,
      output: null,
      type: 'send'
    })

    $sessionControlBySession.set({
      [SID]: mockEntry(sampleGoal({ interrupted_at: INTERRUPTED_AT }))
    })

    renderStack({ onSubmit })

    fireEvent.click(screen.getByRole('button', { name: 'Resume goal' }))

    await waitFor(() => {
      expect(mockRunSessionControlAction).toHaveBeenCalledWith(SID, 'goal.continue', undefined)
    })

    expect(onSubmit).toHaveBeenCalledWith('Continue toward goal: Ship the crash-resume flow', {
      displayKind: 'hidden',
      sessionId: SID
    })
  })

  it('offers the same continue action from the goal menu', async () => {
    mockRunSessionControlAction.mockResolvedValue({
      display: '/goal continue',
      message: 'Continue toward goal: Ship the crash-resume flow',
      notice: null,
      output: null,
      type: 'send'
    })

    $sessionControlBySession.set({
      [SID]: mockEntry(sampleGoal({ interrupted_at: INTERRUPTED_AT }))
    })

    renderStack({ onSubmit: vi.fn().mockResolvedValue(true) })

    fireEvent.click(screen.getByRole('button', { name: /goal actions/i }))
    fireEvent.click(await screen.findByRole('menuitem', { name: /resume goal/i }))

    await waitFor(() => {
      expect(mockRunSessionControlAction).toHaveBeenCalledWith(SID, 'goal.continue', undefined)
    })
  })

  it('hides the interrupted affordance once the continuation turn is running', () => {
    $sessionControlBySession.set({
      [SID]: mockEntry(sampleGoal({ interrupted_at: INTERRUPTED_AT }), { pendingAction: 'goal.continue' })
    })

    renderStack()

    expect(screen.getByRole('button', { name: /Goal active · Turn 3\/20/ })).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Resume goal' })).toBeNull()
  })

  it('renders exactly as before when the goal is not interrupted', () => {
    $sessionControlBySession.set({ [SID]: mockEntry(sampleGoal()) })

    renderStack()

    expect(screen.getByRole('button', { name: /Goal active · Turn 3\/20/ })).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Resume goal' })).toBeNull()
  })
})

describe('goal card elapsed clock', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    $sessionControlBySession.set({})
    mockRefreshSessionControl.mockResolvedValue(undefined)
    mockRefreshSessionGoal.mockResolvedValue(undefined)
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    $sessionControlBySession.set({})
  })

  it('measures elapsed from the persisted created_at, so a restart does not reset it', () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date(1_700_003_661_000))

    $sessionControlBySession.set({
      // Goal born 1h 1m 1s before this (freshly reloaded) render.
      [SID]: mockEntry(sampleGoal({ created_at: 1_700_000_000 }))
    })

    renderStack()

    expect(screen.getByRole('button', { name: /Goal active · Turn 3\/20 · 1:01:01/ })).toBeTruthy()
  })

  it('omits the elapsed clock when the snapshot carries no created_at', () => {
    $sessionControlBySession.set({ [SID]: mockEntry(sampleGoal()) })

    renderStack()

    expect(screen.getByRole('button', { name: /Goal active · Turn 3\/20$/ })).toBeTruthy()
  })
})
