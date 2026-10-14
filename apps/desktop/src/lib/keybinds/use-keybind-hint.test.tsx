import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { $busyInputMode, setBusyInputModeFromConfig } from '@/app/chat/composer/busy-input-mode'

import { formatCombo } from './combo'
import { useKeybindHint } from './use-keybind-hint'

afterEach(() => {
  cleanup()
  setBusyInputModeFromConfig('interrupt')
})

describe('useKeybindHint and display.busy_input_mode', () => {
  it('updates the Queue hint live when switching between Queue and Steer', () => {
    setBusyInputModeFromConfig('queue')
    const { result } = renderHook(() => useKeybindHint('composer.queue'))

    expect(result.current).toBe(formatCombo('enter'))

    act(() => setBusyInputModeFromConfig('steer'))
    expect(result.current).toBe(formatCombo('mod+enter'))
  })

  it('updates the Steer hint live without advertising it in Interrupt mode', () => {
    setBusyInputModeFromConfig('interrupt')
    const { result } = renderHook(() => useKeybindHint('composer.steer'))

    expect(result.current).toBeNull()

    act(() => setBusyInputModeFromConfig('queue'))
    expect(result.current).toBe(formatCombo('mod+enter'))

    act(() => setBusyInputModeFromConfig('steer'))
    expect(result.current).toBe(formatCombo('enter'))
  })

  it('keeps the default Enter-to-send hint unchanged', () => {
    setBusyInputModeFromConfig('queue')
    const { result } = renderHook(() => useKeybindHint('composer.send'))

    expect(result.current).toBe(formatCombo('enter'))
  })
})
