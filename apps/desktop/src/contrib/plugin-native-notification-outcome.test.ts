import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import {
  clearPluginNotifyHandlers,
  invokePluginNotifyAction,
  invokePluginNotifyActivate,
  setNativeNotifyEnabled,
  setNativeNotifyKind
} from '@/store/native-notifications'
import { __resetNativeNotifyBaselineForTests, markNativeNotifyBaseline } from '@/store/notify-baseline'

import { createPluginContext } from './plugin'

const desktop = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const originalBridge = desktop.hermesDesktop
const notify = vi.fn()
let sequence = 0

const context = () => createPluginContext(`notification-outcome-${++sequence}`)

beforeEach(() => {
  notify.mockReset().mockResolvedValue(true)
  desktop.hermesDesktop = { notify } as unknown as Window['hermesDesktop']
  vi.spyOn(document, 'hasFocus').mockReturnValue(false)
  vi.spyOn(document, 'hidden', 'get').mockReturnValue(true)
  setNativeNotifyEnabled(true)
  setNativeNotifyKind('plugin', true)
  __resetNativeNotifyBaselineForTests()
})

afterEach(() => {
  clearPluginNotifyHandlers()
  desktop.hermesDesktop = originalBridge
  vi.restoreAllMocks()
})

it('reports accepted only after the native bridge affirms handoff', async () => {
  let accept!: (accepted: boolean) => void
  notify.mockImplementation(() => new Promise<boolean>(resolve => (accept = resolve)))
  let settled = false

  const outcome = context()
    .os.notifyWithResult({ title: 'Result' })
    .then(result => {
      settled = true

      return result
    })

  await Promise.resolve()
  expect(settled).toBe(false)
  expect(notify).toHaveBeenCalledWith(expect.objectContaining({ kind: 'plugin', title: 'Result' }))
  accept(true)
  await expect(outcome).resolves.toEqual({ status: 'accepted' })
})

it.each([
  ['master preference', () => setNativeNotifyEnabled(false)],
  ['plugin preference', () => setNativeNotifyKind('plugin', false)],
  ['startup baseline', () => markNativeNotifyBaseline()],
  [
    'foreground focus',
    () => {
      vi.mocked(document.hasFocus).mockReturnValue(true)
      vi.spyOn(document, 'hidden', 'get').mockReturnValue(false)
    }
  ]
])('reports suppressed without calling the bridge under %s', async (_label, suppress) => {
  suppress()

  await expect(context().os.notifyWithResult({ title: 'Guarded' })).resolves.toEqual({ status: 'suppressed' })
  expect(notify).not.toHaveBeenCalled()
})

it('shares per-plugin throttling with legacy notify without collapsing unrelated plugins', async () => {
  const ctx = context()

  expect(ctx.os.notify({ title: 'Legacy' })).toBeUndefined()
  await expect(ctx.os.notifyWithResult({ title: 'Repeat' })).resolves.toEqual({ status: 'suppressed' })
  await expect(context().os.notifyWithResult({ title: 'Other' })).resolves.toEqual({ status: 'accepted' })
  expect(notify).toHaveBeenCalledTimes(2)
})

it.each(['bridge absent', 'method absent'])('reports failed when the native %s', async unavailable => {
  desktop.hermesDesktop = unavailable === 'bridge absent' ? undefined : ({} as Window['hermesDesktop'])

  await expect(context().os.notifyWithResult({ title: 'Unavailable' })).resolves.toEqual({ status: 'failed' })
  expect(notify).not.toHaveBeenCalled()
})

it('reports failed only for a strict false bridge response', async () => {
  notify.mockResolvedValue(false)

  await expect(context().os.notifyWithResult({ title: 'Refused' })).resolves.toEqual({ status: 'failed' })
  expect(notify).toHaveBeenCalledTimes(1)
})

it.each([
  ['rejected', () => Promise.reject(new Error('Transport lost'))],
  [
    'synchronous throw',
    () => {
      throw new Error('Transport lost')
    }
  ],
  ['undefined', () => Promise.resolve(undefined)],
  ['ambiguous truthy', () => Promise.resolve('true')]
])('reports unknown when bridge invocation is %s', async (_label, invoke) => {
  notify.mockImplementation(invoke)

  await expect(context().os.notifyWithResult({ title: 'Uncertain' })).resolves.toEqual({ status: 'unknown' })
  expect(notify).toHaveBeenCalledTimes(1)
})

it('keeps legacy notify synchronous void and consumes bridge rejection', async () => {
  notify.mockRejectedValue(new Error('Transport lost'))

  expect(context().os.notify({ title: 'Legacy' })).toBeUndefined()
  await new Promise(resolve => setTimeout(resolve, 0))
  expect(notify).toHaveBeenCalledTimes(1)
})

it('retires legacy notification callbacks when the plugin unloads', () => {
  const disposers: Array<() => void> = []
  const ctx = createPluginContext(`legacy-${++sequence}`, dispose => disposers.push(dispose))
  const onActivate = vi.fn()

  expect(ctx.os.notify({ onActivate, title: 'Legacy' })).toBeUndefined()
  const { notifyId } = notify.mock.calls[0][0]
  invokePluginNotifyActivate(notifyId)
  expect(onActivate).toHaveBeenCalledTimes(1)

  disposers.forEach(dispose => dispose())
  invokePluginNotifyActivate(notifyId)
  expect(onActivate).toHaveBeenCalledTimes(1)
})

it('keeps callbacks through pending and unknown handoff, then retires them on unload', async () => {
  const disposers: Array<() => void> = []
  const ctx = createPluginContext(`uncertain-${++sequence}`, dispose => disposers.push(dispose))
  const onActivate = vi.fn()
  let reject!: (error: Error) => void
  notify.mockImplementation(() => new Promise((_resolve, fail) => (reject = fail)))

  const outcome = ctx.os.notifyWithResult({ onActivate, title: 'Uncertain' })
  const { notifyId } = notify.mock.calls[0][0]
  invokePluginNotifyActivate(notifyId)
  expect(onActivate).toHaveBeenCalledTimes(1)

  reject(new Error('Lost response'))
  await expect(outcome).resolves.toEqual({ status: 'unknown' })
  invokePluginNotifyActivate(notifyId)
  expect(onActivate).toHaveBeenCalledTimes(2)

  disposers.forEach(dispose => dispose())
  invokePluginNotifyActivate(notifyId)
  expect(onActivate).toHaveBeenCalledTimes(2)
})

it('retires callbacks on known failure', async () => {
  notify.mockResolvedValue(false)
  const onActivate = vi.fn()
  const onAction = vi.fn()

  await expect(
    context().os.notifyWithResult({
      actions: [{ id: 'open', label: 'Open', onAction }],
      onActivate,
      title: 'Refused'
    })
  ).resolves.toEqual({ status: 'failed' })

  const { notifyId } = notify.mock.calls[0][0]
  invokePluginNotifyActivate(notifyId)
  expect(invokePluginNotifyAction(notifyId, 'open')).toBe(false)
  expect(onActivate).not.toHaveBeenCalled()
  expect(onAction).not.toHaveBeenCalled()
})

it('plugin unload clears only callbacks owned by that plugin', async () => {
  const disposers: Array<() => void> = []
  const owned = createPluginContext(`owned-${++sequence}`, dispose => disposers.push(dispose))
  const onOwned = vi.fn()
  const onOther = vi.fn()

  await owned.os.notifyWithResult({ onActivate: onOwned, title: 'Owned' })
  const ownedId = notify.mock.calls[0][0].notifyId
  await context().os.notifyWithResult({ onActivate: onOther, title: 'Other' })
  const otherId = notify.mock.calls[1][0].notifyId

  disposers.forEach(dispose => dispose())
  invokePluginNotifyActivate(ownedId)
  invokePluginNotifyActivate(otherId)
  expect(onOwned).not.toHaveBeenCalled()
  expect(onOther).toHaveBeenCalledTimes(1)
})

it('does not resurrect callbacks when the plugin unloads during pending IPC', async () => {
  const disposers: Array<() => void> = []
  const ctx = createPluginContext(`pending-${++sequence}`, dispose => disposers.push(dispose))
  const onActivate = vi.fn()
  let accept!: (accepted: boolean) => void
  notify.mockImplementation(() => new Promise<boolean>(resolve => (accept = resolve)))

  const outcome = ctx.os.notifyWithResult({ onActivate, title: 'Pending' })
  const { notifyId } = notify.mock.calls[0][0]
  disposers.forEach(dispose => dispose())
  accept(true)

  await expect(outcome).resolves.toEqual({ status: 'accepted' })
  invokePluginNotifyActivate(notifyId)
  expect(onActivate).not.toHaveBeenCalled()
})

it('preserves rich notification presentation through the observable method', async () => {
  await expect(
    context().os.notifyWithResult({
      actions: [{ activate: { path: '/example/result/1' }, id: 'open', label: 'Open' }],
      activate: 'hermes://example/result/1',
      body: 'Body',
      icon: 'C:/icon.png',
      silent: true,
      title: 'Rich'
    })
  ).resolves.toEqual({ status: 'accepted' })

  expect(notify).toHaveBeenCalledWith(
    expect.objectContaining({
      actions: [{ activate: '/example/result/1', id: 'open', text: 'Open' }],
      activate: '/example/result/1',
      body: 'Body',
      icon: 'C:/icon.png',
      silent: true,
      title: 'Rich'
    })
  )
})
