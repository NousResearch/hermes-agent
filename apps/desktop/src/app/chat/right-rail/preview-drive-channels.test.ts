import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $rightRailActiveTabId } from '@/store/layout'
import { closeRightRail, openPreview } from '@/store/preview'
import { deferred } from '@/test/deferred'

import { actOnActivePreview } from './preview-act'
import { runPreviewGuestScript, usePreviewDriveChannels } from './preview-drive-channels'
import { activePreviewScriptRunner } from './preview-script-runner'

const originalDesktop = window.hermesDesktop
const begin = vi.fn(async () => 17)
const end = vi.fn(async (_token: number) => {})
const execute = vi.fn(async (_code: string): Promise<unknown> => 'answer')

const guest = () =>
  Object.assign(document.createElement('div'), {
    getWebContentsId: () => 42,
    executeJavaScript: execute
  })

beforeEach(() => {
  begin.mockReset().mockResolvedValue(17)
  end.mockReset().mockResolvedValue(undefined)
  execute.mockReset().mockResolvedValue('answer')
  window.hermesDesktop = {
    beginPreviewGuestActivity: begin,
    endPreviewGuestActivity: end,
    executePreviewGuestScript: (_token: number, code: string) => execute(code)
  } as unknown as Window['hermesDesktop']
})

afterEach(() => {
  window.hermesDesktop = originalDesktop
  vi.useRealTimers()
  closeRightRail()
})

describe('preview script wake scope', () => {
  it('does not dispatch guest code when the native scheduling capability is absent', async () => {
    window.hermesDesktop = {
      ...originalDesktop,
      beginPreviewGuestActivity: undefined,
      endPreviewGuestActivity: undefined
    }
    const view = guest()
    await expect(runPreviewGuestScript(view, 'must not run')).rejects.toThrow('unavailable')
    expect(view.executeJavaScript).not.toHaveBeenCalled()
  })

  it('holds the lease through the real script promise, releasing on both answers and failures', async () => {
    const view = guest()
    const result = deferred<unknown>()
    view.executeJavaScript.mockReturnValueOnce(result.promise)
    const running = runPreviewGuestScript(view, 'awaited script')
    await Promise.resolve()
    expect(begin).toHaveBeenCalledWith(42)
    expect(end).not.toHaveBeenCalled()
    result.resolve('finished')
    await expect(running).resolves.toBe('finished')
    expect(end).toHaveBeenLastCalledWith(17)
    view.executeJavaScript.mockRejectedValueOnce(new Error('page error'))
    await expect(runPreviewGuestScript(view, 'failing script')).rejects.toThrow('page error')
    expect(end).toHaveBeenCalledTimes(2)
  })

  it('cancels pending scripts and cancels before dispatch when acquisition races an abort', async () => {
    const view = guest()
    view.executeJavaScript.mockReturnValue(deferred().promise)
    const controller = new AbortController()
    const running = runPreviewGuestScript(view, 'pending script', controller.signal)
    const rejected = expect(running).rejects.toBe('interrupted')
    await vi.waitFor(() => expect(view.executeJavaScript).toHaveBeenCalledOnce())
    controller.abort('interrupted')
    await rejected
    expect(end).toHaveBeenCalledTimes(1)

    const acquisition = deferred<number>()
    begin.mockReturnValueOnce(acquisition.promise)
    const beforeDispatch = new AbortController()
    const next = runPreviewGuestScript(view, 'must not run', beforeDispatch.signal)
    const nextRejected = expect(next).rejects.toBe('interrupted')
    beforeDispatch.abort('interrupted')
    await nextRejected
    expect(end).toHaveBeenCalledTimes(1)
    acquisition.resolve(18)
    await vi.waitFor(() => expect(end).toHaveBeenCalledTimes(2))
    expect(view.executeJavaScript).toHaveBeenCalledTimes(1)
    expect(end).toHaveBeenLastCalledWith(18)
  })

  it('does not impose a new deadline on plugin scripts', async () => {
    vi.useFakeTimers()
    const view = guest()
    const result = deferred<unknown>()
    view.executeJavaScript.mockReturnValue(result.promise)
    const running = runPreviewGuestScript(view, 'long approved plugin script')
    await vi.advanceTimersByTimeAsync(120_000)
    expect(end).not.toHaveBeenCalled()
    result.resolve('finished')
    await expect(running).resolves.toBe('finished')
    expect(end).toHaveBeenCalledOnce()
  })

  it('releases a stalled action at its existing eight-second deadline', async () => {
    vi.useFakeTimers()
    openPreview({ kind: 'url', label: 'Browser', source: 'https://example.com', url: 'https://example.com' })
    const view = guest()
    view.executeJavaScript.mockReturnValue(deferred().promise)
    const { unmount } = renderHook(() => usePreviewDriveChannels({ current: view }, $rightRailActiveTabId.get()!, true))

    try {
      const running = actOnActivePreview({ kind: 'elements' })
      await vi.advanceTimersByTimeAsync(7_999)
      expect(end).not.toHaveBeenCalled()
      await vi.advanceTimersByTimeAsync(1)
      await running
      expect(end).toHaveBeenCalledTimes(2)
    } finally {
      unmount()
    }
  })

  it.each(['timeout', 'cancel'])(
    'ends %s while wake acquisition is pending, then releases a late token',
    async reason => {
      vi.useFakeTimers()
      openPreview({ kind: 'url', label: 'Browser', source: 'https://example.com', url: 'https://example.com' })
      const acquisition = deferred<number>()
      begin.mockReturnValueOnce(acquisition.promise)
      const view = guest()

      const { unmount } = renderHook(() =>
        usePreviewDriveChannels({ current: view }, $rightRailActiveTabId.get()!, true)
      )

      const controller = new AbortController()

      try {
        const running = actOnActivePreview({ kind: 'elements' }, controller.signal)

        if (reason === 'cancel') {
          controller.abort('interrupted')
        } else {
          await vi.advanceTimersByTimeAsync(8_000)
        }

        expect((await running).success).toBe(false)
        expect(end).not.toHaveBeenCalled()
        acquisition.resolve(23)
        await vi.advanceTimersByTimeAsync(0)
        expect(end).toHaveBeenCalledExactlyOnceWith(23)
        expect(view.executeJavaScript).not.toHaveBeenCalled()
      } finally {
        unmount()
      }
    }
  )

  it.each(['press', 'elements'])('releases cancelled %s work without dispatching keyboard input', async kind => {
    openPreview({ kind: 'url', label: 'Browser', source: 'https://example.com', url: 'https://example.com' })
    const view = Object.assign(guest(), { sendInputEvent: vi.fn() })
    view.executeJavaScript.mockReturnValue(deferred().promise)
    const { unmount } = renderHook(() => usePreviewDriveChannels({ current: view }, $rightRailActiveTabId.get()!, true))
    const controller = new AbortController()

    try {
      const running = actOnActivePreview({ kind, selector: 'input', key: 'Enter' }, controller.signal)
      await vi.waitFor(() => expect(begin).toHaveBeenCalledTimes(2))
      controller.abort('interrupted')
      expect((await running).success).toBe(false)
      expect(end).toHaveBeenCalledTimes(2)
      expect(view.sendInputEvent).not.toHaveBeenCalled()
    } finally {
      unmount()
    }
  })

  it('keeps typing awake between script trips and releases the full action after cancellation', async () => {
    vi.useFakeTimers()
    openPreview({ kind: 'url', label: 'Browser', source: 'https://example.com', url: 'https://example.com' })
    const live = new Set<number>()
    let token = 0
    begin.mockImplementation(async () => {
      live.add(++token)

      return token
    })
    end.mockImplementation(async id => {
      live.delete(id)
    })

    const view = Object.assign(guest(), {
      sendInputEvent: vi.fn(() => {
        expect(live.size).toBeGreaterThan(0)
      })
    })

    view.executeJavaScript.mockImplementation(async code =>
      JSON.stringify(
        code.includes('hermes-focus-probe')
          ? { success: true, focused: true, tag: 'INPUT' }
          : code.includes('document.activeElement') && !code.includes('"kind":"locate"')
            ? { success: true, field: true, len: 0 }
            : code.includes('"kind":"locate"')
              ? { success: true, point: { x: 20, y: 20 }, typable: true }
              : { success: true, hit: { tag: 'INPUT', trusted: true } }
      )
    )
    const { unmount } = renderHook(() => usePreviewDriveChannels({ current: view }, $rightRailActiveTabId.get()!, true))
    const controller = new AbortController()

    try {
      const running = actOnActivePreview(
        { kind: 'type', selector: 'input', text: 'x'.repeat(10_000) },
        controller.signal
      )
      await vi.advanceTimersByTimeAsync(9_000)
      expect(live.size).toBeGreaterThan(0)
      controller.abort('interrupted')
      await vi.advanceTimersByTimeAsync(1000)
      expect((await running).success).toBe(false)
      expect(view.sendInputEvent).toHaveBeenCalled()
      expect(live.size).toBe(0)
    } finally {
      unmount()
    }
  })

  it('releases pending channel work on unmount without destroying guest state', async () => {
    openPreview({ kind: 'url', label: 'Browser', source: 'https://example.com', url: 'https://example.com' })
    const view = guest()
    view.executeJavaScript.mockReturnValue(deferred().promise)
    const { unmount } = renderHook(() => usePreviewDriveChannels({ current: view }, $rightRailActiveTabId.get()!, true))
    const running = activePreviewScriptRunner()!('pending script')
    const rejected = expect(running).rejects.toMatchObject({ name: 'AbortError' })
    await act(async () => {
      await Promise.resolve()
    })
    unmount()
    await rejected
    expect(end).toHaveBeenCalledOnce()
    expect(activePreviewScriptRunner()).toBeNull()
  })
})
