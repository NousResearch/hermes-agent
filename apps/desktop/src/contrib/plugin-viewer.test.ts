import { afterEach, expect, it, vi } from 'vitest'

import { $sessionTiles } from '@/store/session-states'

import { createPluginContext } from './plugin'

const session = { runtimeSessionId: 'runtime', storedSessionId: 'stored', connectionId: 'local', profile: 'worker' }
afterEach(() => {
  Reflect.deleteProperty(window, 'hermesDesktop')
  $sessionTiles.set([])
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
