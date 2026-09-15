import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  $activeComputerUse,
  $computerUseBySession,
  cleanAppLabel,
  clearAllComputerUseStates,
  clearComputerUseState,
  COMPLETED_DISMISS_DELAY_MS,
  ERROR_DISMISS_DELAY_MS,
  extractComputerUseArgs,
  formatComputerUseTarget,
  sessionComputerUse,
  setComputerUseCompleted,
  setComputerUseDrafting,
  setComputerUseError,
  setComputerUseRunning
} from './computer-use'
import { $activeSessionId } from './session'

describe('computer-use store', () => {
  const SID_A = 'session-alpha'
  const SID_B = 'session-bravo'

  beforeEach(() => {
    vi.useFakeTimers()
    clearAllComputerUseStates()
    $activeSessionId.set(SID_A)
  })

  afterEach(() => {
    vi.clearAllTimers()
    vi.useRealTimers()
  })

  it('normalizes application names with cleanAppLabel', () => {
    expect(cleanAppLabel('chrome.exe')).toBe('Chrome')
    expect(cleanAppLabel('C:\\Windows\\System32\\notepad.exe')).toBe('Notepad')
    expect(cleanAppLabel('/Applications/Safari.app')).toBe('Safari')
    expect(cleanAppLabel('screen')).toBe('Screen')
    expect(cleanAppLabel('desktop')).toBe('Desktop')
    expect(cleanAppLabel('Visual Studio Code')).toBe('Visual Studio Code')
    expect(cleanAppLabel(undefined)).toBe('')
  })

  it('formats target summary accurately for various actions, coordinates, and apps', () => {
    expect(formatComputerUseTarget({ action: 'capture', app: 'chrome.exe', mode: 'som' })).toBe(
      'Chrome · SOM Capture'
    )
    expect(formatComputerUseTarget({ action: 'click', app: 'Notepad', element: 12 })).toBe(
      'Notepad · Click #12'
    )
    expect(formatComputerUseTarget({ action: 'click', coordinate: [450, 320] })).toBe(
      'Click (450, 320)'
    )
    expect(formatComputerUseTarget({ action: 'double_click', app: 'VS Code', coordinate: [100, 200] })).toBe(
      'VS Code · Double Click (100, 200)'
    )
    expect(formatComputerUseTarget({ action: 'type', app: 'Discord', text: 'Hello World' })).toBe(
      'Discord · Type "Hello World"'
    )
    expect(formatComputerUseTarget({ action: 'type', text: 'a very long message text here' })).toBe(
      'Type "a very long …"'
    )
    expect(formatComputerUseTarget({ action: 'key', keys: 'Enter' })).toBe('Key (Enter)')
    expect(formatComputerUseTarget({ action: 'scroll', app: 'screen' })).toBe('Screen · Scroll')
    expect(formatComputerUseTarget({ action: 'capture', app: 'desktop', mode: 'vision' })).toBe(
      'Desktop · VISION Capture'
    )
    expect(formatComputerUseTarget({ action: 'focus_app', app: 'Spotify' })).toBe(
      'Spotify · Focus App'
    )
    expect(formatComputerUseTarget({})).toBe('Active')
  })

  it('extracts arguments without polluting undefined values into state', () => {
    // Nested object args
    const payload1 = {
      name: 'computer_use',
      args: { action: 'click', app: 'Terminal', element: 5 }
    }

    expect(extractComputerUseArgs(payload1)).toEqual({
      action: 'click',
      app: 'Terminal',
      element: 5
    })

    // Stringified JSON args
    const payload2 = {
      name: 'computer_use',
      args: JSON.stringify({ action: 'type', text: 'hello', window_title: 'Firefox' })
    }

    expect(extractComputerUseArgs(payload2)).toEqual({
      action: 'type',
      app: 'Firefox',
      text: 'hello'
    })

    // Coordinate array
    const payload3 = {
      args: { action: 'click', coordinate: [100, 200] }
    }

    expect(extractComputerUseArgs(payload3)).toEqual({
      action: 'click',
      coordinate: [100, 200]
    })
  })

  it('preserves app name across progressive tool updates without clobbering', () => {
    setComputerUseRunning(SID_A, { action: 'type', app: 'Notepad' })
    expect($computerUseBySession.get()[SID_A]?.targetSummary).toBe('Notepad · Type')

    // Progress update without re-passing app
    setComputerUseRunning(SID_A, { text: 'hello world' })
    const state = $computerUseBySession.get()[SID_A]
    expect(state?.app).toBe('Notepad')
    expect(state?.action).toBe('type')
    expect(state?.targetSummary).toBe('Notepad · Type "hello world"')
  })

  it('tracks drafting, running, and completed lifecycle with auto-dismiss at 2.5s', () => {
    // Drafting
    setComputerUseDrafting(SID_A)
    let state = $computerUseBySession.get()[SID_A]
    expect(state).toBeDefined()
    expect(state?.phase).toBe('drafting')
    expect(state?.targetSummary).toBe('Preparing…')

    // Running
    setComputerUseRunning(SID_A, { action: 'capture', app: 'Chrome', mode: 'som' })
    state = $computerUseBySession.get()[SID_A]
    expect(state?.phase).toBe('running')
    expect(state?.app).toBe('Chrome')
    expect(state?.action).toBe('capture')
    expect(state?.targetSummary).toBe('Chrome · SOM Capture')

    // Active session mirror
    expect($activeComputerUse.get()?.targetSummary).toBe('Chrome · SOM Capture')

    // Completed
    setComputerUseCompleted(SID_A, { durationSeconds: 0.42 })
    state = $computerUseBySession.get()[SID_A]
    expect(state?.phase).toBe('completed')
    expect(state?.durationSeconds).toBe(0.42)
    expect(state?.completedAt).toBeDefined()

    // Linger before auto-dismiss
    vi.advanceTimersByTime(1000)
    expect($computerUseBySession.get()[SID_A]).toBeDefined()

    // Dismisses after full 2500ms elapses
    vi.advanceTimersByTime(COMPLETED_DISMISS_DELAY_MS - 1000)
    expect($computerUseBySession.get()[SID_A]).toBeUndefined()
    expect($activeComputerUse.get()).toBeNull()
  })

  it('tracks error state and auto-dismisses at 5s (giving enough reading time)', () => {
    setComputerUseRunning(SID_A, { action: 'click', app: 'Explorer' })
    setComputerUseError(SID_A, 'Element #9 not found in window tree')

    const state = $computerUseBySession.get()[SID_A]
    expect(state?.phase).toBe('error')
    expect(state?.error).toBe('Element #9 not found in window tree')

    // At 2.5s, error should STILL be visible
    vi.advanceTimersByTime(2500)
    expect($computerUseBySession.get()[SID_A]).toBeDefined()

    // Dismisses after 5000ms elapses
    vi.advanceTimersByTime(ERROR_DISMISS_DELAY_MS - 2500)
    expect($computerUseBySession.get()[SID_A]).toBeUndefined()
  })

  it('clearComputerUseState with onlyIfUnfinished=true preserves completed states', () => {
    // Session 1: completed
    setComputerUseRunning(SID_A, { action: 'click', app: 'Chrome' })
    setComputerUseCompleted(SID_A)

    // Session 2: still running
    setComputerUseRunning(SID_B, { action: 'type', app: 'Slack' })

    // When turn completes, clearing with onlyIfUnfinished=true protects SID_A
    clearComputerUseState(SID_A, true)
    expect($computerUseBySession.get()[SID_A]?.phase).toBe('completed')

    // But clears unfinished SID_B
    clearComputerUseState(SID_B, true)
    expect($computerUseBySession.get()[SID_B]).toBeUndefined()

    // Unconditional clear removes completed as well
    clearComputerUseState(SID_A, false)
    expect($computerUseBySession.get()[SID_A]).toBeUndefined()
  })

  it('isolates state per session and updates $activeComputerUse correctly', () => {
    setComputerUseRunning(SID_A, { action: 'type', app: 'Slack' })
    setComputerUseRunning(SID_B, { action: 'capture', app: 'Chrome' })

    const sessionAState = sessionComputerUse(SID_A)
    const sessionBState = sessionComputerUse(SID_B)

    expect(sessionAState.get()?.app).toBe('Slack')
    expect(sessionBState.get()?.app).toBe('Chrome')

    // Active session is SID_A
    expect($activeComputerUse.get()?.app).toBe('Slack')

    // Switching active session updates $activeComputerUse
    $activeSessionId.set(SID_B)
    expect($activeComputerUse.get()?.app).toBe('Chrome')

    // Clear SID_A does not affect SID_B
    clearComputerUseState(SID_A)
    expect(sessionAState.get()).toBeNull()
    expect(sessionBState.get()?.app).toBe('Chrome')
  })

  it('safeguards against rapid progress events resurrecting settled states', () => {
    // Session completed
    setComputerUseRunning(SID_A, { action: 'click', app: 'Chrome' })
    setComputerUseCompleted(SID_A)
    expect($computerUseBySession.get()[SID_A]?.phase).toBe('completed')

    // Late progress event arriving after completion
    setComputerUseRunning(SID_A, { action: 'click', app: 'Chrome' }, true)
    // Must remain completed, not reset to running
    expect($computerUseBySession.get()[SID_A]?.phase).toBe('completed')

    // Session errored
    setComputerUseRunning(SID_B, { action: 'type', app: 'Notepad' })
    setComputerUseError(SID_B, 'Failed')
    expect($computerUseBySession.get()[SID_B]?.phase).toBe('error')

    // Late progress event arriving after error
    setComputerUseRunning(SID_B, { text: 'hello' }, true)
    // Must remain error, not reset to running
    expect($computerUseBySession.get()[SID_B]?.phase).toBe('error')
  })

  describe('extractToolErrorMessage edge-case error normalization', () => {
    it('handles direct string and object errors', async () => {
      const { extractToolErrorMessage } = await import('./computer-use')

      expect(extractToolErrorMessage({ error: 'Window minimized' })).toBe('Window minimized')
      expect(extractToolErrorMessage({ error: { message: 'Timeout waiting for HWND' } })).toBe(
        'Timeout waiting for HWND'
      )
      expect(extractToolErrorMessage({ error: { error: 'Access denied' } })).toBe('Access denied')
      expect(extractToolErrorMessage({ error: true })).toBe('Action failed')
    })

    it('handles JSON-stringified and object results with ok: false or error fields', async () => {
      const { extractToolErrorMessage } = await import('./computer-use')

      expect(
        extractToolErrorMessage({
          result: JSON.stringify({ error: { message: 'Invalid coordinate bounds' } })
        })
      ).toBe('Invalid coordinate bounds')

      expect(
        extractToolErrorMessage({
          result: JSON.stringify({ hint: 'Target window is occluded', ok: false })
        })
      ).toBe('Target window is occluded')

      expect(
        extractToolErrorMessage({
          result: { exit_code: 1, message: 'Named pipe connection broken' }
        })
      ).toBe('Named pipe connection broken')

      expect(
        extractToolErrorMessage({
          result: { is_error: true, message: 'Process exited abnormally' }
        })
      ).toBe('Process exited abnormally')
    })

    it('strips redundant error prefixes and cleans multi-line stack traces', async () => {
      const { cleanErrorMessage, extractToolErrorMessage } = await import('./computer-use')

      expect(cleanErrorMessage('Error: Failed to find element')).toBe('Failed to find element')
      expect(cleanErrorMessage('ToolError: Action aborted')).toBe('Action aborted')
      expect(
        cleanErrorMessage('Error: Win32Exception\n   at WindowsDriver.Click()\n   at Hermes.Run()')
      ).toBe('Win32Exception')

      expect(
        extractToolErrorMessage({
          error: 'Error: Connection reset by peer\nTraceback (most recent call last):'
        })
      ).toBe('Connection reset by peer')
    })

    it('returns undefined for non-error payloads', async () => {
      const { extractToolErrorMessage } = await import('./computer-use')

      expect(extractToolErrorMessage({ result: 'Success' })).toBeUndefined()
      expect(extractToolErrorMessage({ result: JSON.stringify({ ok: true, value: 42 }) })).toBeUndefined()
      expect(extractToolErrorMessage(undefined)).toBeUndefined()
      expect(extractToolErrorMessage(null)).toBeUndefined()
    })

    it('safely handles circular structures and Error instances without throwing', async () => {
      const { extractToolErrorMessage } = await import('./computer-use')

      // Error instance
      const err = new Error('Named pipe broken')
      expect(extractToolErrorMessage({ error: err })).toBe('Named pipe broken')

      // Circular reference object in error
      const circular: Record<string, unknown> = { message: 'Circular crash prevented' }
      circular.self = circular
      expect(extractToolErrorMessage({ error: circular })).toBe('Circular crash prevented')

      // Circular reference object in result
      const circularResult: Record<string, unknown> = { ok: false, message: 'Circular in result' }
      circularResult.self = circularResult
      expect(extractToolErrorMessage({ result: circularResult })).toBe('Circular in result')
    })
  })

  it('memoizes sessionComputerUse store instances per sessionId', () => {
    const store1 = sessionComputerUse(SID_A)
    const store2 = sessionComputerUse(SID_A)
    const storeB = sessionComputerUse(SID_B)

    expect(store1).toBe(store2)
    expect(store1).not.toBe(storeB)
  })

  it('extracts toolId from payload tool_id or id in extractComputerUseArgs', () => {
    expect(extractComputerUseArgs({ id: 'call-123', name: 'computer_use' })).toEqual({
      toolId: 'call-123'
    })
    expect(extractComputerUseArgs({ name: 'computer_use', tool_id: 'call-456' })).toEqual({
      toolId: 'call-456'
    })
  })
})
