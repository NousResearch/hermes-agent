import { afterEach, expect, it, vi } from 'vitest'

import { $sessionTiles } from '@/store/session-states'

import { createPluginContext } from './plugin'

const session = { runtimeSessionId: 'runtime', storedSessionId: 'stored', connectionId: 'local', profile: 'worker' }
afterEach(() => {
  Reflect.deleteProperty(window, 'hermesDesktop')
  $sessionTiles.set([])
  vi.clearAllTimers()
  vi.useRealTimers()
})
it('attributes viewer requests and closes them on disposal, rejecting stale closures', async () => {
  const disposers: Array<() => void> = []
  const ctx = createPluginContext('demo', d => disposers.push(d))
  expect(ctx.os.openViewer).toBeTypeOf('function')
  $sessionTiles.set([
    { storedSessionId: 'stored', runtimeId: 'runtime', ownerRoute: { connectionId: 'local', profile: 'worker' } }
  ])
  const openPluginViewer = vi.fn(async () => true)
  const closePluginViewer = vi.fn(async () => true)
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { openPluginViewer, closePluginViewer } })
  const input = { id: 'watch', url: 'http://127.0.0.1:9876/viewer?ticket=one', title: 'View', session }
  expect(await ctx.os.openViewer(input)).toBe(true)
  expect(openPluginViewer).toHaveBeenCalledWith('demo', { id: 'watch', url: input.url, title: 'View' })
  expect(await ctx.os.openViewer({ ...input, session: { ...session, profile: 'other' } })).toBe(false)
  disposers.forEach(d => d())
  expect(closePluginViewer).toHaveBeenCalledWith('demo')
  expect(await ctx.os.openViewer(input)).toBe(false)
  closePluginViewer.mockClear()
  expect(await ctx.os.closeViewer('watch')).toBe(false)
  expect(closePluginViewer).not.toHaveBeenCalled()
})

it('keeps callbacks in the originating renderer and retires them on native close, replacement and disposal', async () => {
  vi.useFakeTimers()
  const disposers: Array<() => void> = []
  const ctx = createPluginContext('demo', d => disposers.push(d))
  $sessionTiles.set([
    { storedSessionId: 'stored', runtimeId: 'runtime', ownerRoute: { connectionId: 'local', profile: 'worker' } }
  ])
  const openPluginViewer = vi.fn(async () => true)
  const closePluginViewer = vi.fn(async () => true)
  const isPluginViewerOpen = vi.fn(async () => true)
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: { openPluginViewer, closePluginViewer, isPluginViewerOpen }
  })
  const input = { id: 'watch', url: 'https://viewer.example/view#ticket=one', title: 'Viewer', session }
  const old = vi.fn(async () => {})
  expect(await ctx.os.openViewer({ ...input, onKeepAlive: old })).toBe(true)
  expect(openPluginViewer).toHaveBeenCalledWith('demo', { id: input.id, url: input.url, title: input.title })
  await vi.advanceTimersByTimeAsync(0)
  expect(old).toHaveBeenCalledOnce()
  $sessionTiles.set([])
  await vi.advanceTimersByTimeAsync(60_000)
  expect(old).toHaveBeenCalledTimes(2)
  expect(isPluginViewerOpen).toHaveBeenCalledWith('demo', input.id, input.url)
  isPluginViewerOpen.mockResolvedValue(false)
  await vi.advanceTimersByTimeAsync(60_000)
  isPluginViewerOpen.mockResolvedValue(true)
  await vi.advanceTimersByTimeAsync(60_000)
  expect(old).toHaveBeenCalledTimes(2)

  $sessionTiles.set([
    { storedSessionId: 'stored', runtimeId: 'runtime', ownerRoute: { connectionId: 'local', profile: 'worker' } }
  ])
  const replaced = vi.fn(async () => {})
  const current = vi.fn(async () => {})
  await ctx.os.openViewer({ ...input, onKeepAlive: replaced })
  await vi.advanceTimersByTimeAsync(0)
  await ctx.os.openViewer({ ...input, onKeepAlive: current })
  await vi.advanceTimersByTimeAsync(60_000)
  expect(replaced).toHaveBeenCalledOnce()
  expect(current).toHaveBeenCalledTimes(2)
  disposers.forEach(d => d())
  await vi.advanceTimersByTimeAsync(180_000)
  expect(current).toHaveBeenCalledTimes(2)
  expect(vi.getTimerCount()).toBe(0)
})

it('never starts a lease after failed, superseded, closed or disposed pending opens', async () => {
  vi.useFakeTimers()
  const disposers: Array<() => void> = []
  const ctx = createPluginContext('demo', d => disposers.push(d))
  $sessionTiles.set([
    { storedSessionId: 'stored', runtimeId: 'runtime', ownerRoute: { connectionId: 'local', profile: 'worker' } }
  ])
  const openPluginViewer = vi.fn(async () => false)
  const isPluginViewerOpen = vi.fn(async () => true)
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: { openPluginViewer, isPluginViewerOpen, closePluginViewer: vi.fn(async () => true) }
  })
  const onKeepAlive = vi.fn(async () => {})
  const input = { id: 'watch', url: 'https://viewer.example/view', title: 'Viewer', session, onKeepAlive }
  expect(await ctx.os.openViewer(input)).toBe(false)
  await vi.advanceTimersByTimeAsync(60_000)
  expect(onKeepAlive).not.toHaveBeenCalled()

  const bridge = window.hermesDesktop!
  const probe = bridge.isPluginViewerOpen
  delete bridge.isPluginViewerOpen
  openPluginViewer.mockResolvedValueOnce(true)
  expect(await ctx.os.openViewer(input)).toBe(true)
  await vi.advanceTimersByTimeAsync(60_000)
  expect(onKeepAlive).not.toHaveBeenCalled()
  bridge.isPluginViewerOpen = probe

  let finish!: (opened: boolean) => void
  openPluginViewer.mockImplementation(
    () =>
      new Promise<boolean>(resolve => {
        finish = resolve
      })
  )
  const pending = ctx.os.openViewer(input)
  await ctx.os.closeViewer(input.id)
  finish(true)
  expect(await pending).toBe(false)
  await vi.advanceTimersByTimeAsync(60_000)
  expect(onKeepAlive).not.toHaveBeenCalled()

  const obsolete = ctx.os.openViewer(input)
  const finishObsolete = finish
  openPluginViewer.mockResolvedValue(true)
  expect(await ctx.os.openViewer({ ...input, onKeepAlive: undefined })).toBe(true)
  finishObsolete(true)
  expect(await obsolete).toBe(false)
  await vi.advanceTimersByTimeAsync(60_000)
  expect(onKeepAlive).not.toHaveBeenCalled()

  openPluginViewer.mockImplementation(
    () =>
      new Promise<boolean>(resolve => {
        finish = resolve
      })
  )
  const disposed = ctx.os.openViewer(input)
  disposers.forEach(d => d())
  const closeAtDisposal = vi.mocked(window.hermesDesktop!.closePluginViewer!).mock.calls.length
  finish(true)
  expect(await disposed).toBe(false)
  // A late old context must not close a newly activated context's same-id window.
  expect(window.hermesDesktop!.closePluginViewer).toHaveBeenCalledTimes(closeAtDisposal)
  await vi.advanceTimersByTimeAsync(60_000)
  expect(onKeepAlive).not.toHaveBeenCalled()
  expect(vi.getTimerCount()).toBe(0)
})
