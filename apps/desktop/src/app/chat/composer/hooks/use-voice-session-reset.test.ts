import { cleanup, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { stopVoicePlayback } from '@/lib/voice-playback'

import { useEndVoiceOnSessionSwitch, useStopVoicePlaybackOnUnmount } from './use-voice-session-reset'

vi.mock('@/lib/voice-playback', () => ({
  stopVoicePlayback: vi.fn()
}))

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

function setup(initial: string | null | undefined) {
  const onSwitch = vi.fn()
  const view = renderHook(({ id }: { id: string | null | undefined }) => useEndVoiceOnSessionSwitch(id, onSwitch), {
    initialProps: { id: initial }
  })

  return { onSwitch, ...view }
}

describe('useEndVoiceOnSessionSwitch', () => {
  it('does not fire on initial mount', () => {
    const { onSwitch } = setup('session-a')

    expect(onSwitch).not.toHaveBeenCalled()
  })

  it('does not fire when the session id is unchanged', () => {
    const { onSwitch, rerender } = setup('session-a')

    rerender({ id: 'session-a' })
    rerender({ id: 'session-a' })

    expect(onSwitch).not.toHaveBeenCalled()
  })

  it('fires once for each switch between real sessions', () => {
    const { onSwitch, rerender } = setup('session-a')

    rerender({ id: 'session-b' })
    rerender({ id: 'session-c' })
    rerender({ id: 'session-a' })

    expect(onSwitch).toHaveBeenCalledTimes(3)
  })

  it('does not fire on the first no-session to session persist', () => {
    const { onSwitch, rerender } = setup(null)

    rerender({ id: 'session-a' })

    expect(onSwitch).not.toHaveBeenCalled()
  })

  it('treats undefined and null as the same no-session state', () => {
    const { onSwitch, rerender } = setup(undefined)

    rerender({ id: null })
    rerender({ id: 'session-a' })

    expect(onSwitch).not.toHaveBeenCalled()
  })

  it('fires when leaving a session', () => {
    const { onSwitch, rerender } = setup('session-a')

    rerender({ id: null })

    expect(onSwitch).toHaveBeenCalledTimes(1)
  })

  it('uses the latest switch callback', () => {
    const first = vi.fn()
    const second = vi.fn()
    const { rerender } = renderHook(
      ({ id, cb }: { cb: () => void; id: string | null }) => useEndVoiceOnSessionSwitch(id, cb),
      { initialProps: { cb: first, id: 'session-a' as string | null } }
    )

    rerender({ cb: second, id: 'session-b' })

    expect(first).not.toHaveBeenCalled()
    expect(second).toHaveBeenCalledTimes(1)
  })
})

describe('useStopVoicePlaybackOnUnmount', () => {
  it('does not stop playback while mounted', () => {
    renderHook(() => useStopVoicePlaybackOnUnmount())

    expect(stopVoicePlayback).not.toHaveBeenCalled()
  })

  it('stops global voice playback on unmount', () => {
    const { unmount } = renderHook(() => useStopVoicePlaybackOnUnmount())

    unmount()

    expect(stopVoicePlayback).toHaveBeenCalledTimes(1)
  })
})
