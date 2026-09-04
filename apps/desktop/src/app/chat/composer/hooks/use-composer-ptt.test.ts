import { act, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { useComposerPtt } from './use-composer-ptt'

const key = (type: 'keydown' | 'keyup', init: KeyboardEventInit = {}) =>
  window.dispatchEvent(new KeyboardEvent(type, { code: 'AltLeft', key: 'Alt', ...init }))

const options = (overrides: Partial<Parameters<typeof useComposerPtt>[0]> = {}) => ({
  active: () => true,
  blocked: false,
  cancel: vi.fn(),
  maxRecordingSeconds: 120,
  start: vi.fn().mockResolvedValue(true),
  stop: vi.fn().mockResolvedValue(' hello '),
  submit: vi.fn().mockResolvedValue(true),
  ...overrides
})

describe('useComposerPtt', () => {
  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('starts on left Alt and submits exactly once on release without consuming the OS key', async () => {
    const state = options()
    renderHook(() => useComposerPtt(state))
    const down = new KeyboardEvent('keydown', { cancelable: true, code: 'AltLeft', key: 'Alt' })

    await act(async () => window.dispatchEvent(down))
    await act(async () => key('keydown', { repeat: true }))
    await act(async () => key('keyup'))
    await act(async () => key('keyup'))

    expect(down.defaultPrevented).toBe(false)
    expect(state.start).toHaveBeenCalledTimes(1)
    expect(state.stop).toHaveBeenCalledTimes(1)
    expect(state.submit).toHaveBeenCalledTimes(1)
    expect(state.submit).toHaveBeenCalledWith('hello')
  })

  it('ignores modifiers, blocked state, inactive composers, and keyup without a start', async () => {
    const state = options({ blocked: true })

    const { rerender } = renderHook(({ blocked, active }) => useComposerPtt({ ...state, blocked, active }), {
      initialProps: { active: () => true, blocked: true }
    })

    await act(async () => key('keydown', { ctrlKey: true }))
    await act(async () => key('keyup'))
    rerender({ active: () => false, blocked: false })
    await act(async () => key('keydown'))
    await act(async () => key('keyup'))

    expect(state.start).not.toHaveBeenCalled()
    expect(state.stop).not.toHaveBeenCalled()
    expect(state.submit).not.toHaveBeenCalled()
  })

  it('cancels a start that resolves after the composer becomes blocked', async () => {
    let resolveStart: (started: boolean) => void = () => undefined

    const state = options({
      start: vi.fn(
        () =>
          new Promise<boolean>(resolve => {
            resolveStart = resolve
          })
      )
    })

    const { rerender } = renderHook(({ blocked }) => useComposerPtt({ ...state, blocked }), {
      initialProps: { blocked: false }
    })

    await act(async () => key('keydown'))
    rerender({ blocked: true })
    await act(async () => resolveStart(true))

    expect(state.cancel).toHaveBeenCalledTimes(1)
    expect(state.stop).not.toHaveBeenCalled()
    expect(state.submit).not.toHaveBeenCalled()
  })

  it('cancels a start that resolves after the active composer changes', async () => {
    let active = true
    let resolveStart: (started: boolean) => void = () => undefined

    const state = options({
      active: () => active,
      start: vi.fn(
        () =>
          new Promise<boolean>(resolve => {
            resolveStart = resolve
          })
      )
    })

    renderHook(() => useComposerPtt(state))
    await act(async () => key('keydown'))
    active = false
    await act(async () => resolveStart(true))

    expect(state.cancel).toHaveBeenCalledTimes(1)
    expect(state.stop).not.toHaveBeenCalled()
    expect(state.submit).not.toHaveBeenCalled()
  })

  it('does not stop or submit when the asynchronous start fails', async () => {
    const state = options({ start: vi.fn().mockResolvedValue(false) })
    renderHook(() => useComposerPtt(state))

    await act(async () => key('keydown'))
    await act(async () => key('keyup'))

    expect(state.start).toHaveBeenCalledTimes(1)
    expect(state.cancel).not.toHaveBeenCalled()
    expect(state.stop).not.toHaveBeenCalled()
    expect(state.submit).not.toHaveBeenCalled()
  })

  it('cancels on window blur and unmount without submitting', async () => {
    const state = options()
    const { unmount } = renderHook(() => useComposerPtt(state))

    await act(async () => key('keydown'))
    await act(async () => window.dispatchEvent(new Event('blur')))
    unmount()

    expect(state.start).toHaveBeenCalledTimes(1)
    expect(state.cancel).toHaveBeenCalledTimes(1)
    expect(state.stop).not.toHaveBeenCalled()
    expect(state.submit).not.toHaveBeenCalled()
  })

  it('drops a transcript that resolves after composer ownership changes', async () => {
    let active = true
    let resolveStop: (text: string) => void = () => undefined

    const state = options({
      active: () => active,
      stop: vi.fn(
        () =>
          new Promise<string>(resolve => {
            resolveStop = resolve
          })
      )
    })

    renderHook(() => useComposerPtt(state))

    await act(async () => key('keydown'))
    await act(async () => key('keyup'))
    active = false
    await act(async () => resolveStop('stale'))

    expect(state.submit).not.toHaveBeenCalled()
  })

  it('cancels immediately when focus moves away from the owning composer', async () => {
    let active = true
    const state = options({ active: () => active })
    renderHook(() => useComposerPtt(state))

    await act(async () => key('keydown'))
    active = false
    await act(async () => document.dispatchEvent(new FocusEvent('focusin')))

    expect(state.cancel).toHaveBeenCalledTimes(1)
    expect(state.stop).not.toHaveBeenCalled()
    expect(state.submit).not.toHaveBeenCalled()
  })

  it('invalidates a pending transcript when the window blurs after release', async () => {
    let resolveStop: (text: string) => void = () => undefined

    const state = options({
      stop: vi.fn(
        () =>
          new Promise<string>(resolve => {
            resolveStop = resolve
          })
      )
    })

    renderHook(() => useComposerPtt(state))

    await act(async () => key('keydown'))
    await act(async () => key('keyup'))
    await act(async () => window.dispatchEvent(new Event('blur')))
    await act(async () => resolveStop('must not send'))

    expect(state.submit).not.toHaveBeenCalled()
  })

  it('abandons an active recording when the composer becomes blocked', async () => {
    const state = options()

    const { rerender } = renderHook(({ blocked }) => useComposerPtt({ ...state, blocked }), {
      initialProps: { blocked: false }
    })

    await act(async () => key('keydown'))
    rerender({ blocked: true })
    await act(async () => key('keyup'))

    expect(state.cancel).toHaveBeenCalledTimes(1)
    expect(state.stop).not.toHaveBeenCalled()
    expect(state.submit).not.toHaveBeenCalled()
  })

  it('abandons a finishing recording when the composer becomes blocked', async () => {
    let resolveStop: (text: string) => void = () => undefined

    const state = options({
      stop: vi.fn(
        () =>
          new Promise<string>(resolve => {
            resolveStop = resolve
          })
      )
    })

    const { rerender } = renderHook(({ blocked }) => useComposerPtt({ ...state, blocked }), {
      initialProps: { blocked: false }
    })

    await act(async () => key('keydown'))
    await act(async () => key('keyup'))
    rerender({ blocked: true })
    await act(async () => resolveStop('must not submit'))

    expect(state.cancel).toHaveBeenCalledTimes(1)
    expect(state.submit).not.toHaveBeenCalled()
  })

  it('auto-finishes through the same submit path at the recording cap', async () => {
    vi.useFakeTimers()
    const state = options({ maxRecordingSeconds: 1 })
    renderHook(() => useComposerPtt(state))

    await act(async () => key('keydown'))
    await act(async () => vi.advanceTimersByTimeAsync(1_000))
    await act(async () => key('keyup'))

    expect(state.stop).toHaveBeenCalledTimes(1)
    expect(state.submit).toHaveBeenCalledTimes(1)
  })
})
