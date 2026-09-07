import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  $automationComposer,
  closeAutomationComposer,
  invalidateOnSessionSwitch,
  openAutomationComposer,
  setAutomationComposerType,
  submitAutomation
} from './automation-composer'
import { $gateway } from './gateway'
import { $activeSessionId } from './session'
import { runSessionControlAction } from './session-control'

vi.mock('./session', () => ({
  $activeSessionId: { get: vi.fn(() => 'session-123') }
}))

vi.mock('./gateway', async () => ({ $gateway: (await import('nanostores')).atom(null) }))

vi.mock('./session-control', () => ({
  runSessionControlAction: vi.fn()
}))

const resetStore = () => {
  $automationComposer.set({
    open: false,
    sessionId: null,
    type: 'goal',
    submitting: false,
    error: null
  })
}

afterEach(() => {
  resetStore()
  vi.clearAllMocks()
})

describe('automation-composer store', () => {
  describe('openAutomationComposer', () => {
    it('opens with the given type and captured session', () => {
      openAutomationComposer('loop', 'session-456')
      const state = $automationComposer.get()
      expect(state.open).toBe(true)
      expect(state.type).toBe('loop')
      expect(state.sessionId).toBe('session-456')
      expect(state.submitting).toBe(false)
      expect(state.error).toBe(null)
    })

    it('captures the active session when none is passed', () => {
      openAutomationComposer('goal')
      expect($automationComposer.get().sessionId).toBe('session-123')
    })

    it('does not reopen while submitting', () => {
      $automationComposer.set({
        open: true,
        sessionId: 'session-1',
        type: 'goal',
        submitting: true,
        error: null
      })
      openAutomationComposer('loop', 'session-2')
      const state = $automationComposer.get()
      expect(state.type).toBe('goal')
      expect(state.sessionId).toBe('session-1')
    })
  })

  describe('closeAutomationComposer', () => {
    it('closes and resets', () => {
      openAutomationComposer('loop', 'session-456')
      closeAutomationComposer()
      expect($automationComposer.get()).toEqual({
        open: false,
        sessionId: null,
        type: 'goal',
        submitting: false,
        error: null
      })
    })

    it('refuses to close while submitting', () => {
      $automationComposer.set({
        open: true,
        sessionId: 'session-1',
        type: 'goal',
        submitting: true,
        error: null
      })
      closeAutomationComposer()
      expect($automationComposer.get().open).toBe(true)
    })
  })

  describe('setAutomationComposerType', () => {
    it('switches type and clears a stale error', () => {
      openAutomationComposer('goal', 'session-1')
      $automationComposer.set({ ...$automationComposer.get(), error: 'boom' })
      setAutomationComposerType('heartbeat')
      const state = $automationComposer.get()
      expect(state.type).toBe('heartbeat')
      expect(state.error).toBe(null)
    })

    it('does not switch while submitting', () => {
      $automationComposer.set({
        open: true,
        sessionId: 'session-1',
        type: 'goal',
        submitting: true,
        error: null
      })
      setAutomationComposerType('loop')
      expect($automationComposer.get().type).toBe('goal')
    })
  })

  describe('invalidateOnSessionSwitch', () => {
    it('closes when the active session changed', () => {
      openAutomationComposer('goal', 'session-1')
      vi.mocked($activeSessionId.get).mockReturnValueOnce('session-2')
      invalidateOnSessionSwitch()
      expect($automationComposer.get().open).toBe(false)
    })

    it('keeps open when active session matches', () => {
      openAutomationComposer('goal', 'session-1')
      vi.mocked($activeSessionId.get).mockReturnValueOnce('session-1')
      invalidateOnSessionSwitch()
      expect($automationComposer.get().open).toBe(true)
    })

    it('does nothing when closed', () => {
      vi.mocked($activeSessionId.get).mockReturnValueOnce('session-2')
      invalidateOnSessionSwitch()
      expect($automationComposer.get().open).toBe(false)
    })

    it('ignores an unset captured session', () => {
      $automationComposer.set({ open: true, sessionId: null, type: 'goal', submitting: false, error: null })
      invalidateOnSessionSwitch()
      expect($automationComposer.get().open).toBe(true)
    })
  })

  describe('submitAutomation', () => {
    it('invalidates the dialog when its gateway changes', () => {
      openAutomationComposer('goal', 'session-1')
      $gateway.set({} as never)
      expect($automationComposer.get().open).toBe(false)
    })

    it('rejects a second submission before sending another request', async () => {
      let release!: () => void
      vi.mocked(runSessionControlAction).mockImplementationOnce(() => new Promise(resolve => { release = () => resolve({ type: 'exec' } as never) }))
      openAutomationComposer('goal', 'session-1')
      const first = submitAutomation('goal', { prompt: 'first' })
      await expect(submitAutomation('goal', { prompt: 'second' })).rejects.toThrow()
      expect(runSessionControlAction).toHaveBeenCalledTimes(1)
      release()
      await first
    })

    it('does not close a new dialog or return a kickoff from an old request', async () => {
      let release!: () => void
      vi.mocked(runSessionControlAction).mockImplementationOnce(() => new Promise(resolve => { release = () => resolve({ type: 'send', message: 'old kickoff' } as never) }))
      openAutomationComposer('goal', 'session-1')
      const first = submitAutomation('goal', { prompt: 'first' })
      vi.mocked($activeSessionId.get).mockReturnValueOnce('session-2')
      invalidateOnSessionSwitch()
      openAutomationComposer('heartbeat', 'session-2')
      release()
      await expect(first).rejects.toThrow()
      expect($automationComposer.get().open).toBe(true)
      expect($automationComposer.get().sessionId).toBe('session-2')
      expect($automationComposer.get().error).toBeNull()
    })
    it('throws and surfaces an error when no captured session exists', async () => {
      openAutomationComposer('goal')
      $automationComposer.set({ ...$automationComposer.get(), sessionId: null })
      await expect(submitAutomation('goal', { prompt: 'x' })).rejects.toThrow('No active session')
      expect($automationComposer.get().open).toBe(true)
    })

    it('marks submitting while in flight and closes on success', async () => {
      let release!: () => void
      const gate = new Promise<void>(resolve => { release = resolve })
      vi.mocked(runSessionControlAction).mockImplementationOnce(
        () => new Promise(resolve => gate.then(() => resolve({ type: 'exec' } as never)))
      )
      openAutomationComposer('goal', 'session-1')
      const promise = submitAutomation('goal', { prompt: 'x' })
      expect($automationComposer.get().submitting).toBe(true)
      release()
      await promise
      expect($automationComposer.get().open).toBe(false)
      expect($automationComposer.get().submitting).toBe(false)
    })

    it('dispatches the typed create action for the captured session', async () => {
      vi.mocked(runSessionControlAction).mockResolvedValueOnce({ type: 'exec' } as never)
      openAutomationComposer('loop', 'session-7')
      await submitAutomation('loop', { prompt: 'check', interval_seconds: 300, run_limit: 5 })
      expect(runSessionControlAction).toHaveBeenCalledWith('session-7', 'loop.create', {
        prompt: 'check',
        interval_seconds: 300,
        run_limit: 5
      })
    })

    it('keeps the dialog open with a bounded error on failure', async () => {
      vi.mocked(runSessionControlAction).mockRejectedValueOnce(
        new Error('A goal already exists for this session. Clear it first with goal.clear.')
      )
      openAutomationComposer('goal', 'session-1')
      await expect(submitAutomation('goal', { prompt: 'x' })).rejects.toThrow('already exists')
      const state = $automationComposer.get()
      expect(state.open).toBe(true)
      expect(state.submitting).toBe(false)
      expect(state.error).toContain('already exists')
    })

    it('truncates overly long error messages', async () => {
      vi.mocked(runSessionControlAction).mockRejectedValueOnce(new Error('x'.repeat(500)))
      openAutomationComposer('goal', 'session-1')
      await expect(submitAutomation('goal', { prompt: 'x' })).rejects.toThrow()
      expect($automationComposer.get().error!.length).toBeLessThanOrEqual(240)
    })
  })
})
