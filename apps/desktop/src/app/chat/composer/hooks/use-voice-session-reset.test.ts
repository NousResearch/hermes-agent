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

function setup(initial: string | null) {
  const onSwitch = vi.fn()

  const view = renderHook(
    ({ id }: { id: string | null }) => useEndVoiceOnSessionSwitch(id, onSwitch),
    { initialProps: { id: initial } }
  )

  return { onSwitch, ...view }
}

describe('useEndVoiceOnSessionSwitch', () => {
  it('does not fire on initial mount', () => {
    const { onSwitch } = setup('A')
    expect(onSwitch).not.toHaveBeenCalled()
  })

  it('does not fire when the session id is unchanged across re-renders', () => {
    const { onSwitch, rerender } = setup('A')
    rerender({ id: 'A' })
    rerender({ id: 'A' })
    expect(onSwitch).not.toHaveBeenCalled()
  })

  it('fires once per switch between two real sessions', () => {
    const { onSwitch, rerender } = setup('A')
    rerender({ id: 'B' })
    expect(onSwitch).toHaveBeenCalledTimes(1)
    rerender({ id: 'C' })
    expect(onSwitch).toHaveBeenCalledTimes(2)
    // Switching back is still a switch.
    rerender({ id: 'A' })
    expect(onSwitch).toHaveBeenCalledTimes(3)
  })

  it('does not fire on the null→id persist of the session already in view', () => {
    const { onSwitch, rerender } = setup(null)
    rerender({ id: 'A' })
    expect(onSwitch).not.toHaveBeenCalled()
  })

  it('treats undefined and null as the same "no session"', () => {
    const onSwitch = vi.fn()

    const { rerender } = renderHook(
      ({ id }: { id: string | null | undefined }) => useEndVoiceOnSessionSwitch(id, onSwitch),
      { initialProps: { id: undefined as string | null | undefined } }
    )

    // undefined → null is not a switch (both are "no session")…
    rerender({ id: null })
    expect(onSwitch).not.toHaveBeenCalled()
    // …and undefined-seeded → first real id is the initial persist, not a switch.
    rerender({ id: 'A' })
    expect(onSwitch).not.toHaveBeenCalled()
  })

  it('fires when leaving a session (id→null)', () => {
    const { onSwitch, rerender } = setup('A')
    rerender({ id: null })
    expect(onSwitch).toHaveBeenCalledTimes(1)
  })

  it('uses the latest onSwitch callback', () => {
    const first = vi.fn()
    const second = vi.fn()

    const { rerender } = renderHook(
      ({ id, cb }: { id: string | null; cb: () => void }) => useEndVoiceOnSessionSwitch(id, cb),
      { initialProps: { id: 'A' as string | null, cb: first } }
    )

    rerender({ id: 'B', cb: second })
    expect(first).not.toHaveBeenCalled()
    expect(second).toHaveBeenCalledTimes(1)
  })
})

describe('useStopVoicePlaybackOnUnmount', () => {
  it('does not stop playback while mounted', () => {
    renderHook(() => useStopVoicePlaybackOnUnmount())
    expect(stopVoicePlayback).not.toHaveBeenCalled()
  })

  it('stops global read-aloud playback once on unmount (cold-switch teardown)', () => {
    const { unmount } = renderHook(() => useStopVoicePlaybackOnUnmount())
    expect(stopVoicePlayback).not.toHaveBeenCalled()
    unmount()
    expect(stopVoicePlayback).toHaveBeenCalledTimes(1)
  })
})
