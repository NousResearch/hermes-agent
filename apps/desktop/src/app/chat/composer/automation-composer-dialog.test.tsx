import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n/context'
import {
  $automationComposer,
  type AutomationType,
  invalidateOnSessionSwitch,
  openAutomationComposer,
  openAutomationComposerForEdit
} from '@/store/automation-composer'
import { $activeSessionId } from '@/store/session'
import { $sessionControlBySession, refreshSessionControl, runSessionControlAction } from '@/store/session-control'

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
  $automationComposer.set({ open: false, sessionId: null, type: 'goal', mode: 'create', submitting: false, error: null })
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

  it('prefills an existing goal once without overwriting a typed draft on refresh', async () => {
    $sessionControlBySession.set({
      'session-123': {
        capability: 'supported',
        snapshot: {
          goal: {
            title: 'Original objective',
            status: 'active',
            subgoals: ['Keep the tests green'],
            max_turns: 12
          }
        }
      }
    } as never)
    await renderDialog()

    act(() => openAutomationComposerForEdit('goal', 'session-123'))

    const prompt = await screen.findByLabelText(/goal prompt/i)
    expect((prompt as HTMLTextAreaElement).value).toBe('Original objective')
    fireEvent.click(screen.getByText('Advanced'))
    expect((screen.getByLabelText(/max continuation turns/i) as HTMLInputElement).value).toBe('12')
    expect(screen.getByText('Keep the tests green')).toBeTruthy()

    fireEvent.change(prompt, { target: { value: 'My unsaved objective' } })
    act(() => {
      $sessionControlBySession.set({
        'session-123': {
          capability: 'supported',
          snapshot: {
            goal: {
              title: 'Server refresh objective',
              status: 'active',
              subgoals: ['Server refresh criterion'],
              max_turns: 20
            }
          }
        }
      } as never)
    })

    expect((screen.getByLabelText(/goal prompt/i) as HTMLTextAreaElement).value).toBe('My unsaved objective')
  })

  it('keeps an editable goal available while its control snapshot refreshes', async () => {
    $sessionControlBySession.set({
      'session-123': {
        capability: 'supported',
        error: null,
        actionError: null,
        loading: true,
        pendingAction: null,
        snapshot: { goal: { title: 'Original objective', status: 'active', subgoals: [], max_turns: 12 } }
      }
    } as never)
    await renderDialog()
    act(() => openAutomationComposerForEdit('goal', 'session-123'))
    await screen.findByLabelText(/goal prompt/i)
    expect(screen.queryByText('Automation controls are unavailable. Check the connection and refresh.')).toBeNull()
    expect((screen.getByRole('button', { name: 'Save goal' }) as HTMLButtonElement).disabled).toBe(false)
  })

  it('keeps Save enabled after a transient busy action rejection', async () => {
    $sessionControlBySession.set({
      'session-123': {
        capability: 'supported',
        error: null,
        actionError: 'Session is busy',
        loading: false,
        pendingAction: null,
        snapshot: { goal: { title: 'Original objective', status: 'active', subgoals: [], max_turns: 12 } }
      }
    } as never)
    await renderDialog()
    act(() => openAutomationComposerForEdit('goal', 'session-123'))
    await screen.findByLabelText(/goal prompt/i)
    expect(screen.queryByText('Automation controls are unavailable. Check the connection and refresh.')).toBeNull()
    expect((screen.getByRole('button', { name: 'Save goal' }) as HTMLButtonElement).disabled).toBe(false)
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

  it('shows loop min interval error and disables Start when interval is below backend minimum', async () => {
    $sessionControlBySession.set({
      'session-123': {
        capability: 'supported',
        snapshot: {
          goal: null,
          loop: null,
          heartbeat: null,
          loop_min_interval_seconds: 30,
          revision: 'r1',
          updated_at: 1
        }
      }
    } as never)
    await renderDialog()
    act(() => openAs('loop'))
    await screen.findByRole('dialog')
    fireEvent.change(screen.getByLabelText(/loop prompt/i), { target: { value: 'Poll CI' } })
    fireEvent.change(screen.getByLabelText(/interval/i), { target: { value: '10' } })
    expect(await screen.findByText(/at least 30 seconds/i)).toBeTruthy()
    expect((screen.getByRole('button', { name: /start loop/i }) as HTMLButtonElement).disabled).toBe(true)
  })

  it('keeps loop dialog open with entered value and visible error on invalid interval', async () => {
    $sessionControlBySession.set({
      'session-123': {
        capability: 'supported',
        snapshot: {
          goal: null,
          loop: null,
          heartbeat: null,
          loop_min_interval_seconds: 30,
          revision: 'r1',
          updated_at: 1
        }
      }
    } as never)
    await renderDialog()
    act(() => openAs('loop'))
    await screen.findByRole('dialog')
    fireEvent.change(screen.getByLabelText(/loop prompt/i), { target: { value: 'Poll CI' } })
    fireEvent.change(screen.getByLabelText(/interval/i), { target: { value: '10' } })
    await screen.findByText(/at least 30 seconds/i)
    fireEvent.click(screen.getByRole('button', { name: /start loop/i }))
    expect(runSessionControlAction).not.toHaveBeenCalled()
    expect($automationComposer.get().open).toBe(true)
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

  it('omits run_limit from loop.create when field is blank', async () => {
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
    // Leave run limit blank — create mode should omit it entirely
    fireEvent.click(screen.getByRole('button', { name: /start loop/i }))
    await waitFor(() =>
      expect(runSessionControlAction).toHaveBeenCalledWith('session-123', 'loop.create', {
        prompt: 'Poll CI',
        interval_seconds: 300
      })
    )
  })

  it('submits loop.update with run_limit=0 when edit field is blank (clears cap)', async () => {
    vi.mocked(runSessionControlAction).mockResolvedValueOnce({
      type: 'exec',
      display: null,
      message: null,
      notice: '',
      output: ''
    } as never)
    $sessionControlBySession.set({
      'session-123': {
        capability: 'supported',
        snapshot: {
          goal: null,
          loop: { prompt: 'Poll CI', interval_seconds: 300, times: 10, until: '', status: 'active', mode: 'interval', ticks_fired: 3, max_ticks: 10, current_delay: 0, next_due_at: 0, last_fired_at: 0, created_at: 0, awaiting_response: false, deferred_by_goal: false },
          heartbeat: null
        }
      }
    } as never)
    await renderDialog()
    act(() => openAutomationComposerForEdit('loop', 'session-123'))
    await screen.findByLabelText(/run limit/i)
    // Run limit prefilled from snapshot; clear it to blank
    fireEvent.change(screen.getByLabelText(/run limit/i), { target: { value: '' } })
    fireEvent.click(screen.getByRole('button', { name: /save loop/i }))
    await waitFor(() =>
      expect(runSessionControlAction).toHaveBeenCalledWith('session-123', 'loop.update', {
        prompt: 'Poll CI',
        interval_seconds: 300,
        run_limit: 0
      })
    )
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

  it('allows interval below 30 when snapshot has no loop_min_interval_seconds (legacy backend)', async () => {
    $sessionControlBySession.set({
      'session-123': {
        capability: 'supported',
        snapshot: {
          goal: null,
          loop: null,
          heartbeat: null,
          revision: 'r1',
          updated_at: 1
        }
      }
    } as never)
    await renderDialog()
    act(() => openAs('loop'))
    await screen.findByRole('dialog')
    fireEvent.change(screen.getByLabelText(/loop prompt/i), { target: { value: 'Poll CI' } })
    fireEvent.change(screen.getByLabelText(/interval/i), { target: { value: '10' } })
    expect(screen.queryByText(/at least/)).toBeNull()
    expect((screen.getByRole('button', { name: /start loop/i }) as HTMLButtonElement).disabled).toBe(false)
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

  describe('edit-mode Pause button', () => {
    const PAUSE_DISPATCH = { display: null, message: null, notice: '', output: '', type: 'exec' }

    const GOAL_SNAPSHOT = {
      capability: 'supported',
      snapshot: {
        goal: { title: 'Existing goal', status: 'active', subgoals: [], max_turns: 5 },
        loop: null,
        heartbeat: null
      }
    }

    const LOOP_SNAPSHOT = {
      capability: 'supported',
      snapshot: {
        goal: null,
        loop: { prompt: 'Poll CI', interval_seconds: 300, times: 5, until: '', status: 'active', mode: 'interval', ticks_fired: 0, max_ticks: 5, current_delay: 0, next_due_at: 0, last_fired_at: 0, created_at: 0, awaiting_response: false, deferred_by_goal: false },
        heartbeat: null
      }
    }

    const HB_SNAPSHOT = {
      capability: 'supported',
      snapshot: {
        goal: null,
        loop: null,
        heartbeat: { prompt: 'Health check', interval_seconds: 300, status: 'active', fire_count: 0, last_fired_at: 0, created_at: 0 }
      }
    }

    it('shows a Pause button for goal in edit mode and dispatches goal.pause', async () => {
      vi.mocked(runSessionControlAction).mockResolvedValueOnce(PAUSE_DISPATCH as never)
      $sessionControlBySession.set({ 'session-123': GOAL_SNAPSHOT } as never)
      await renderDialog()
      act(() => openAutomationComposerForEdit('goal', 'session-123'))

      const pauseBtn = await screen.findByRole('button', { name: /pause/i })
      expect(pauseBtn).toBeTruthy()
      expect((pauseBtn as HTMLButtonElement).disabled).toBe(false)

      fireEvent.click(pauseBtn)
      await waitFor(() =>
        expect(runSessionControlAction).toHaveBeenCalledWith('session-123', 'goal.pause')
      )
      expect($automationComposer.get().open).toBe(true)
    })

    it('shows a Pause button for loop in edit mode and dispatches loop.pause', async () => {
      vi.mocked(runSessionControlAction).mockResolvedValueOnce(PAUSE_DISPATCH as never)
      $sessionControlBySession.set({ 'session-123': LOOP_SNAPSHOT } as never)
      await renderDialog()
      act(() => openAutomationComposerForEdit('loop', 'session-123'))

      const pauseBtn = await screen.findByRole('button', { name: /pause/i })
      expect(pauseBtn).toBeTruthy()

      fireEvent.click(pauseBtn)
      await waitFor(() =>
        expect(runSessionControlAction).toHaveBeenCalledWith('session-123', 'loop.pause')
      )
      expect($automationComposer.get().open).toBe(true)
    })

    it('shows a Pause button for heartbeat in edit mode and dispatches heartbeat.pause', async () => {
      vi.mocked(runSessionControlAction).mockResolvedValueOnce(PAUSE_DISPATCH as never)
      $sessionControlBySession.set({ 'session-123': HB_SNAPSHOT } as never)
      await renderDialog()
      act(() => openAutomationComposerForEdit('heartbeat', 'session-123'))

      const pauseBtn = await screen.findByRole('button', { name: /pause/i })
      expect(pauseBtn).toBeTruthy()

      fireEvent.click(pauseBtn)
      await waitFor(() =>
        expect(runSessionControlAction).toHaveBeenCalledWith('session-123', 'heartbeat.pause')
      )
      expect($automationComposer.get().open).toBe(true)
    })

    it('keeps typed draft intact after a successful goal pause', async () => {
      vi.mocked(runSessionControlAction).mockResolvedValueOnce(PAUSE_DISPATCH as never)
      $sessionControlBySession.set({ 'session-123': GOAL_SNAPSHOT } as never)
      await renderDialog()
      act(() => openAutomationComposerForEdit('goal', 'session-123'))

      const prompt = await screen.findByLabelText(/goal prompt/i)
      fireEvent.change(prompt, { target: { value: 'My unsaved objective' } })
      fireEvent.change(screen.getByPlaceholderText('A testable condition for done'), { target: { value: 'My unsaved criterion' } })
      fireEvent.click(screen.getByRole('button', { name: 'Add criterion' }))

      fireEvent.click(screen.getByRole('button', { name: /pause/i }))
      await waitFor(() => expect(runSessionControlAction).toHaveBeenCalled())

      expect((screen.getByLabelText(/goal prompt/i) as HTMLTextAreaElement).value).toBe('My unsaved objective')
      expect(screen.getByText('My unsaved criterion')).toBeTruthy()
      expect($automationComposer.get().open).toBe(true)
    })

    it('refreshes the session-control snapshot after a successful pause', async () => {
      vi.mocked(runSessionControlAction).mockResolvedValueOnce(PAUSE_DISPATCH as never)
      $sessionControlBySession.set({ 'session-123': GOAL_SNAPSHOT } as never)
      await renderDialog()
      act(() => openAutomationComposerForEdit('goal', 'session-123'))
      vi.mocked(refreshSessionControl).mockClear()

      fireEvent.click(screen.getByRole('button', { name: /pause/i }))
      await waitFor(() => expect(runSessionControlAction).toHaveBeenCalled())

      expect(refreshSessionControl).toHaveBeenCalledWith('session-123')
    })

    it('disables Save and blocks submit while Pause is pending', async () => {
      let releasePause!: () => void
      vi.mocked(runSessionControlAction).mockImplementationOnce(
        () => new Promise(resolve => { releasePause = () => resolve({ type: 'exec', display: null, message: null, notice: '', output: '' } as never) })
      )
      $sessionControlBySession.set({ 'session-123': GOAL_SNAPSHOT } as never)
      await renderDialog()
      act(() => openAutomationComposerForEdit('goal', 'session-123'))

      const prompt = await screen.findByLabelText(/goal prompt/i)
      fireEvent.change(prompt, { target: { value: 'Draft while pausing' } })

      fireEvent.click(screen.getByRole('button', { name: /pause/i }))

      // Pause is in flight — Save must be disabled.
      await waitFor(() => expect((screen.getByRole('button', { name: 'Save goal' }) as HTMLButtonElement).disabled).toBe(true))

      // A submit click while pausing must not dispatch an update action.
      fireEvent.click(screen.getByRole('button', { name: 'Save goal' }))
      expect(runSessionControlAction).toHaveBeenCalledTimes(1)
      expect(runSessionControlAction).toHaveBeenCalledWith('session-123', 'goal.pause')

      // Release the pending Pause; Save re-enables.
      act(() => releasePause())
      await waitFor(() => expect((screen.getByRole('button', { name: 'Save goal' }) as HTMLButtonElement).disabled).toBe(false))
    })

    it('shows a visible error when Pause fails, keeps the dialog open, and preserves the draft', async () => {
      vi.mocked(runSessionControlAction).mockRejectedValueOnce(
        new Error('Session is busy with a live turn. Pause after it finishes.')
      )
      $sessionControlBySession.set({ 'session-123': GOAL_SNAPSHOT } as never)
      await renderDialog()
      act(() => openAutomationComposerForEdit('goal', 'session-123'))

      const prompt = await screen.findByLabelText(/goal prompt/i)
      fireEvent.change(prompt, { target: { value: 'My unsaved objective' } })
      fireEvent.change(screen.getByPlaceholderText('A testable condition for done'), { target: { value: 'My unsaved criterion' } })
      fireEvent.click(screen.getByRole('button', { name: 'Add criterion' }))

      fireEvent.click(screen.getByRole('button', { name: /pause/i }))
      await waitFor(() => expect(runSessionControlAction).toHaveBeenCalled())

      // The session-control store surfaces the rejection as actionError; the
      // component must render it inline so the user sees the failure.
      act(() => {
        $sessionControlBySession.set({
          'session-123': {
            ...GOAL_SNAPSHOT,
            actionError: 'Session is busy with a live turn. Pause after it finishes.'
          }
        } as never)
      })

      expect(await screen.findByText(/busy with a live turn/i)).toBeTruthy()
      expect(screen.getByRole('dialog')).toBeTruthy()
      expect((screen.getByLabelText(/goal prompt/i) as HTMLTextAreaElement).value).toBe('My unsaved objective')
      expect(screen.getByText('My unsaved criterion')).toBeTruthy()
    })
  })
})