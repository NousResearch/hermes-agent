import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { TRANSLATIONS } from '@/i18n/catalog'

import {
  $activeComputerUse,
  $computerUseBySession,
  cleanAppLabel,
  cleanErrorMessage,
  clearAllComputerUseStates,
  clearComputerUseState,
  COMPLETED_DISMISS_DELAY_MS,
  ERROR_DISMISS_DELAY_MS,
  extractComputerUseArgs,
  extractToolFailure,
  formatComputerUseTarget,
  sessionComputerUse,
  setComputerUseCompleted,
  setComputerUseDrafting,
  setComputerUseError,
  setComputerUseRunning
} from './computer-use'
import { $activeSessionId } from './session'

const labels = TRANSLATIONS.en.titlebar.computerUse

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
    expect(cleanAppLabel('Visual Studio Code')).toBe('Visual Studio Code')
    expect(cleanAppLabel(undefined)).toBe('')
  })

  it('formats target summary for various actions, coordinates, and apps', () => {
    expect(formatComputerUseTarget({ action: 'capture', app: 'chrome.exe', mode: 'som' }, labels)).toBe(
      'Chrome · Capture (SOM)'
    )
    expect(formatComputerUseTarget({ action: 'click', app: 'Notepad', element: 12 }, labels)).toBe(
      'Notepad · Click #12'
    )
    expect(formatComputerUseTarget({ action: 'click', coordinate: [450, 320] }, labels)).toBe('Click (450, 320)')
    expect(formatComputerUseTarget({ action: 'double_click', app: 'VS Code', coordinate: [100, 200] }, labels)).toBe(
      'VS Code · Double-click (100, 200)'
    )
    expect(formatComputerUseTarget({ action: 'key', keys: 'ctrl+s' }, labels)).toBe('Key (ctrl+s)')
    expect(formatComputerUseTarget({ action: 'scroll', app: 'screen' }, labels)).toBe('Screen · Scroll')
    expect(formatComputerUseTarget({ action: 'capture', app: 'desktop', mode: 'vision' }, labels)).toBe(
      'Desktop · Capture (VISION)'
    )
    expect(formatComputerUseTarget({ action: 'launch', app: 'Calculator' }, labels)).toBe('Calculator · Launch')
    expect(formatComputerUseTarget({ action: 'zoom_in' }, labels)).toBe('zoom in')
    expect(formatComputerUseTarget({ action: 'constructor' }, labels)).toBe('constructor')
    expect(formatComputerUseTarget({}, labels)).toBe('Operating desktop')
  })

  it('shows how much was typed, never what was typed', () => {
    expect(formatComputerUseTarget({ action: 'type', app: 'Discord', textLength: 11 }, labels)).toBe(
      'Discord · Type (11 characters)'
    )
    expect(formatComputerUseTarget({ action: 'type', textLength: 1 }, labels)).toBe('Type (1 character)')

    const call = extractComputerUseArgs({ args: { action: 'type', text: 'hunter2' }, name: 'computer_use' })

    expect(call).toEqual({ action: 'type', textLength: 7 })
    expect(JSON.stringify(call)).not.toContain('hunter2')
  })

  it('formats with the active locale', () => {
    expect(
      formatComputerUseTarget({ action: 'type', app: 'desktop', textLength: 3 }, TRANSLATIONS.fr.titlebar.computerUse)
    ).toBe('Bureau · Saisie (3 caractères)')
  })

  it('extracts arguments without polluting undefined values into state', () => {
    expect(
      extractComputerUseArgs({ args: { action: 'click', app: 'Terminal', element: 5 }, name: 'computer_use' })
    ).toEqual({ action: 'click', app: 'Terminal', element: 5 })

    expect(
      extractComputerUseArgs({
        args: JSON.stringify({ action: 'type', text: 'hello', window_title: 'Firefox' }),
        name: 'computer_use'
      })
    ).toEqual({ action: 'type', app: 'Firefox', textLength: 5 })

    expect(extractComputerUseArgs({ args: { action: 'click', coordinate: [100, 200] } })).toEqual({
      action: 'click',
      coordinate: [100, 200]
    })
  })

  it('extracts toolId from payload tool_id or id', () => {
    expect(extractComputerUseArgs({ id: 'call-123', name: 'computer_use' })).toEqual({ toolId: 'call-123' })
    expect(extractComputerUseArgs({ name: 'computer_use', tool_id: 'call-456' })).toEqual({ toolId: 'call-456' })
  })

  it("replaces the previous call's details when a new call starts", () => {
    setComputerUseRunning(SID_A, { action: 'click', app: 'Notepad', element: 12, toolId: 'call-a' })
    setComputerUseError(SID_A, 'Element #12 not found')

    setComputerUseRunning(SID_A, { action: 'click', coordinate: [100, 200], toolId: 'call-b' })

    const state = $computerUseBySession.get()[SID_A]

    expect(state).toMatchObject({ phase: 'running', toolId: 'call-b' })
    expect(state?.element).toBeUndefined()
    expect(state?.app).toBeUndefined()
    expect(state?.error).toBeUndefined()
    expect(formatComputerUseTarget(state!, labels)).toBe('Click (100, 200)')

    // The error's dismiss timer belonged to call A and must not retire call B.
    vi.advanceTimersByTime(ERROR_DISMISS_DELAY_MS)
    expect($computerUseBySession.get()[SID_A]?.phase).toBe('running')
  })

  it('tracks drafting, running, and completed lifecycle with auto-dismiss at 2.5s', () => {
    setComputerUseDrafting(SID_A)
    expect($computerUseBySession.get()[SID_A]?.phase).toBe('drafting')

    setComputerUseRunning(SID_A, { action: 'capture', app: 'Chrome', mode: 'som' })
    let state = $computerUseBySession.get()[SID_A]
    expect(state).toMatchObject({ action: 'capture', app: 'Chrome', phase: 'running' })
    expect($activeComputerUse.get()?.app).toBe('Chrome')

    setComputerUseCompleted(SID_A, { durationSeconds: 0.42 })
    state = $computerUseBySession.get()[SID_A]
    expect(state?.phase).toBe('completed')
    expect(state?.durationSeconds).toBe(0.42)
    expect(state?.completedAt).toBeDefined()

    vi.advanceTimersByTime(1000)
    expect($computerUseBySession.get()[SID_A]).toBeDefined()

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

    vi.advanceTimersByTime(2500)
    expect($computerUseBySession.get()[SID_A]).toBeDefined()

    vi.advanceTimersByTime(ERROR_DISMISS_DELAY_MS - 2500)
    expect($computerUseBySession.get()[SID_A]).toBeUndefined()
  })

  it('records a failure without a reason as an error with no message', () => {
    setComputerUseRunning(SID_A, { action: 'click' })
    setComputerUseError(SID_A)

    expect($computerUseBySession.get()[SID_A]).toMatchObject({ phase: 'error' })
    expect($computerUseBySession.get()[SID_A]?.error).toBeUndefined()
  })

  it('never settles a call that is no longer live', () => {
    // Retired (e.g. by Stop): a late outcome must not recreate the pill.
    setComputerUseError(SID_A, 'canceled')
    setComputerUseCompleted(SID_B)
    expect($computerUseBySession.get()).toEqual({})

    // Already settled: a duplicate outcome must not flip it.
    setComputerUseRunning(SID_A, { action: 'click' })
    setComputerUseCompleted(SID_A)
    setComputerUseError(SID_A, 'late failure')
    expect($computerUseBySession.get()[SID_A]?.phase).toBe('completed')
  })

  it('clearComputerUseState with onlyIfUnfinished=true preserves completed states', () => {
    setComputerUseRunning(SID_A, { action: 'click', app: 'Chrome' })
    setComputerUseCompleted(SID_A)

    setComputerUseRunning(SID_B, { action: 'type', app: 'Slack' })

    clearComputerUseState(SID_A, true)
    expect($computerUseBySession.get()[SID_A]?.phase).toBe('completed')

    clearComputerUseState(SID_B, true)
    expect($computerUseBySession.get()[SID_B]).toBeUndefined()

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
    expect($activeComputerUse.get()?.app).toBe('Slack')

    $activeSessionId.set(SID_B)
    expect($activeComputerUse.get()?.app).toBe('Chrome')

    clearComputerUseState(SID_A)
    expect(sessionAState.get()).toBeNull()
    expect(sessionBState.get()?.app).toBe('Chrome')
  })

  it('memoizes sessionComputerUse store instances per sessionId', () => {
    const store1 = sessionComputerUse(SID_A)
    const store2 = sessionComputerUse(SID_A)
    const storeB = sessionComputerUse(SID_B)

    expect(store1).toBe(store2)
    expect(store1).not.toBe(storeB)
  })

  describe('extractToolFailure edge-case error normalization', () => {
    it('handles direct string and object errors', () => {
      expect(extractToolFailure({ error: 'Window minimized' })).toEqual({ message: 'Window minimized' })
      expect(extractToolFailure({ error: { message: 'Timeout waiting for HWND' } })).toEqual({
        message: 'Timeout waiting for HWND'
      })
      expect(extractToolFailure({ error: { error: 'Access denied' } })).toEqual({ message: 'Access denied' })
      expect(extractToolFailure({ error: true })).toEqual({})
      expect(extractToolFailure({ error: '' })).toEqual({})
    })

    it('handles JSON-stringified and object results with ok: false or error fields', () => {
      expect(
        extractToolFailure({ result: JSON.stringify({ error: { message: 'Invalid coordinate bounds' } }) })
      ).toEqual({ message: 'Invalid coordinate bounds' })
      expect(extractToolFailure({ result: JSON.stringify({ hint: 'Target window is occluded', ok: false }) })).toEqual({
        message: 'Target window is occluded'
      })
      expect(extractToolFailure({ result: { exit_code: 1, message: 'Named pipe connection broken' } })).toEqual({
        message: 'Named pipe connection broken'
      })
      expect(extractToolFailure({ result: { is_error: true, message: 'Process exited abnormally' } })).toEqual({
        message: 'Process exited abnormally'
      })
      expect(extractToolFailure({ result: { ok: false } })).toEqual({})
    })

    it('strips redundant error prefixes and cleans multi-line stack traces', () => {
      expect(cleanErrorMessage('Error: Failed to find element')).toBe('Failed to find element')
      expect(cleanErrorMessage('ToolError: Action aborted')).toBe('Action aborted')
      expect(cleanErrorMessage('Error: Win32Exception\n   at WindowsDriver.Click()\n   at Hermes.Run()')).toBe(
        'Win32Exception'
      )
      expect(
        extractToolFailure({ error: 'Error: Connection reset by peer\nTraceback (most recent call last):' })
      ).toEqual({ message: 'Connection reset by peer' })
    })

    it('returns undefined for non-error payloads', () => {
      expect(extractToolFailure({ result: 'Success' })).toBeUndefined()
      expect(extractToolFailure({ result: JSON.stringify({ ok: true, value: 42 }) })).toBeUndefined()
      expect(extractToolFailure(undefined)).toBeUndefined()
      expect(extractToolFailure(null)).toBeUndefined()
    })

    it('safely handles circular structures and Error instances without throwing', () => {
      expect(extractToolFailure({ error: new Error('Named pipe broken') })).toEqual({ message: 'Named pipe broken' })

      const circular: Record<string, unknown> = { message: 'Circular crash prevented' }
      circular.self = circular
      expect(extractToolFailure({ error: circular })).toEqual({ message: 'Circular crash prevented' })

      const circularResult: Record<string, unknown> = { ok: false, message: 'Circular in result' }
      circularResult.self = circularResult
      expect(extractToolFailure({ result: circularResult })).toEqual({ message: 'Circular in result' })
    })
  })
})
