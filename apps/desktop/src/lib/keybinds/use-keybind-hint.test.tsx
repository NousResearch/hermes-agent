import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { $busyInputConfig, busyInputOwnerKey, type BusyInputMode } from '@/store/busy-input-mode'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection } from '@/store/session'

import { formatCombo } from './combo'
import { useKeybindHint } from './use-keybind-hint'

function setMode(mode: BusyInputMode) {
  const connection = $connection.get()
  $busyInputConfig.set({
    owner: busyInputOwnerKey(connection?.connectionId, $activeGatewayProfile.get()),
    connection,
    mode
  })
}

afterEach(() => {
  cleanup()
  $busyInputConfig.set(null)
})

describe('useKeybindHint and display.busy_input_mode', () => {
  it('updates the Queue hint live when switching between Queue and Steer', () => {
    setMode('queue')
    const { result } = renderHook(() => useKeybindHint('composer.queue'))

    expect(result.current).toBe(formatCombo('enter'))

    act(() => setMode('steer'))
    expect(result.current).toBe(formatCombo('mod+enter'))
  })

  it('updates the Steer hint live without advertising it in Interrupt mode', () => {
    setMode('interrupt')
    const { result } = renderHook(() => useKeybindHint('composer.steer'))

    expect(result.current).toBeNull()

    act(() => setMode('queue'))
    expect(result.current).toBe(formatCombo('mod+enter'))

    act(() => setMode('steer'))
    expect(result.current).toBe(formatCombo('enter'))
  })

  it('does not leak hints from another gateway profile', () => {
    setMode('queue')
    const { result } = renderHook(() => useKeybindHint('composer.queue'))
    expect(result.current).toBe(formatCombo('enter'))

    act(() => $busyInputConfig.set({
      owner: busyInputOwnerKey('another-connection', 'another-profile'),
      connection: { connectionId: 'another-connection' },
      mode: 'queue'
    }))
    expect(result.current).toBe(formatCombo('mod+enter'))
  })

  it('keeps the default Enter-to-send hint unchanged', () => {
    setMode('queue')
    const { result } = renderHook(() => useKeybindHint('composer.send'))

    expect(result.current).toBe(formatCombo('enter'))
  })
})
