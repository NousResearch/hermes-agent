import { EventEmitter } from 'node:events'

import type { Session, WebContents } from 'electron'
import { describe, expect, it, vi } from 'vitest'

import {
  applyPreviewGuestAttachPrefs,
  beginPreviewGuestActivity,
  endPreviewGuestActivity,
  executePreviewGuestScript
} from './preview-guest-throttle'
import { createStreamThrottle } from './stream-throttle'

describe('applyPreviewGuestAttachPrefs', () => {
  it('leaves every other webview alone', () => {
    const webPreferences: { backgroundThrottling?: boolean; preload?: string } = {}

    applyPreviewGuestAttachPrefs(webPreferences, 'persist:other', '/preload.js')

    expect(webPreferences).toEqual({})
  })
})

function contents(type: 'window' | 'webview', owner?: WebContents, session = {} as Session) {
  const emitter = new EventEmitter()
  let throttled = true
  let destroyed = false

  const guest = Object.assign(emitter, {
    getType: () => type,
    hostWebContents: owner,
    session,
    isDestroyed: () => destroyed,
    getBackgroundThrottling: () => throttled,
    getURL: () => 'https://preview.test',
    isLoadingMainFrame: vi.fn(() => false),
    executeJavaScript: vi.fn(async (_code: string): Promise<unknown> => 'answer'),
    setBackgroundThrottling: vi.fn((value: boolean) => {
      throttled = value
    }),
    destroy: () => {
      destroyed = true
      emitter.emit('destroyed')
    }
  })

  return guest as typeof guest & WebContents
}

function ownerFixture() {
  const owner = contents('window')
  const throttle = createStreamThrottle()
  const win = { isDestroyed: owner.isDestroyed, webContents: owner }
  throttle.register(win)

  return { owner, hold: () => throttle.holdWindow(win) }
}

describe('preview script activity', () => {
  it('releases the owner and listeners when the guest cannot be woken', () => {
    const { owner, hold } = ownerFixture()
    const guest = contents('webview', owner)
    guest.setBackgroundThrottling.mockImplementationOnce(() => {
      throw new Error('guest disappeared')
    })
    expect(() => beginPreviewGuestActivity(owner, guest, guest.session, hold)).toThrow('guest disappeared')
    expect(owner.getBackgroundThrottling()).toBe(true)
    expect(owner.listenerCount('destroyed')).toBe(0)
    const next = beginPreviewGuestActivity(owner, guest, guest.session, hold)
    expect(owner.getBackgroundThrottling()).toBe(false)
    endPreviewGuestActivity(owner, next)
    expect(owner.getBackgroundThrottling()).toBe(true)
  })

  it('keeps the owner awake until work in every guest finishes, including navigation', () => {
    const { owner, hold } = ownerFixture()
    const first = contents('webview', owner)
    const second = contents('webview', owner)
    const a = beginPreviewGuestActivity(owner, first, first.session, hold)
    const b = beginPreviewGuestActivity(owner, second, second.session, hold)
    first.emit('did-navigate', {}, 'https://new.test', 200, 'OK')
    endPreviewGuestActivity(owner, a)
    expect(first.getBackgroundThrottling()).toBe(true)
    expect(owner.getBackgroundThrottling()).toBe(false)
    expect(second.getBackgroundThrottling()).toBe(false)
    endPreviewGuestActivity(owner, b)
    expect(owner.getBackgroundThrottling()).toBe(true)
    expect(second.getBackgroundThrottling()).toBe(true)
  })

  it('restores idle throttling only after the last owned lease, ignoring foreign and duplicate releases', () => {
    const { owner, hold } = ownerFixture()
    const guest = contents('webview', owner)
    const first = beginPreviewGuestActivity(owner, guest, guest.session, hold)
    const second = beginPreviewGuestActivity(owner, guest, guest.session, hold)
    expect(guest.getBackgroundThrottling()).toBe(false)
    expect(owner.getBackgroundThrottling()).toBe(false)
    endPreviewGuestActivity(contents('window'), first)
    endPreviewGuestActivity(owner, first)
    endPreviewGuestActivity(owner, first)
    expect(guest.getBackgroundThrottling()).toBe(false)
    endPreviewGuestActivity(owner, second)
    expect(guest.getBackgroundThrottling()).toBe(true)
    expect(owner.getBackgroundThrottling()).toBe(true)
    expect(guest.setBackgroundThrottling.mock.calls).toEqual([[false], [true]])
    expect(owner.listenerCount('destroyed')).toBe(0)
    expect(guest.listenerCount('did-navigate')).toBe(0)
  })

  it('rejects other owners, partitions, and untrusted guest senders before changing scheduling', () => {
    const { owner, hold } = ownerFixture()
    const guest = contents('webview', owner)
    expect(() => beginPreviewGuestActivity(contents('window'), guest, guest.session, hold)).toThrow('belong')
    expect(() => beginPreviewGuestActivity(owner, guest, {} as Session, hold)).toThrow('belong')
    expect(() => beginPreviewGuestActivity(guest, guest, guest.session, hold)).toThrow('belong')
    expect(() => beginPreviewGuestActivity(owner, owner, owner.session, hold)).toThrow('belong')
    expect(guest.setBackgroundThrottling).not.toHaveBeenCalled()
    expect(owner.getBackgroundThrottling()).toBe(true)
  })

  it.each(['guest', 'owner'])('keeps overlapping work awake during cancelled provisional %s navigation', targetKind => {
    const { owner, hold } = ownerFixture()
    const guest = contents('webview', owner)
    const first = beginPreviewGuestActivity(owner, guest, guest.session, hold)
    const second = beginPreviewGuestActivity(owner, guest, guest.session, hold)
    const target = targetKind === 'owner' ? owner : guest

    target.emit('did-start-navigation', {}, 'https://slow.test', false, true)
    expect(guest.getBackgroundThrottling()).toBe(false)
    target.emit('did-fail-provisional-load', {}, -3, 'ERR_ABORTED', 'https://slow.test', true)
    target.emit('did-navigate-in-page', {}, 'https://old.test#section', true)
    target.emit('did-frame-navigate', {}, 'https://child.test', 200, 'OK', false, 1, 2)
    endPreviewGuestActivity(owner, first)
    expect(guest.getBackgroundThrottling()).toBe(false)
    endPreviewGuestActivity(owner, second)
    expect(guest.getBackgroundThrottling()).toBe(true)
    expect(guest.setBackgroundThrottling.mock.calls).toEqual([[false], [true]])
  })

  it.each(['owner-navigation', 'guest-crash', 'owner-crash', 'owner-destroyed', 'guest-destroyed'])(
    'retires overlapping leases on %s and ignores stale releases',
    event => {
      const { owner, hold } = ownerFixture()
      const guest = contents('webview', owner)
      const first = beginPreviewGuestActivity(owner, guest, guest.session, hold)
      const second = beginPreviewGuestActivity(owner, guest, guest.session, hold)
      const target = event.startsWith('owner') ? owner : guest

      if (event.endsWith('navigation')) {
        target.emit('did-navigate', {}, 'https://new.test', 200, 'OK')
      } else if (event.endsWith('crash')) {
        target.emit('render-process-gone', {}, {})
      } else {
        target.destroy()
      }

      expect(owner.listenerCount('destroyed')).toBe(0)
      expect(guest.listenerCount('destroyed')).toBe(0)

      if (!guest.isDestroyed()) {
        expect(guest.getBackgroundThrottling()).toBe(true)
      }

      const calls = guest.setBackgroundThrottling.mock.calls.length
      endPreviewGuestActivity(owner, first)
      endPreviewGuestActivity(owner, second)
      expect(guest.setBackgroundThrottling).toHaveBeenCalledTimes(calls)

      if (!owner.isDestroyed()) {
        expect(owner.getBackgroundThrottling()).toBe(true)
      }
    }
  )

  it('keeps queued work awake through navigation and dispatches only once the new document is loaded', async () => {
    const { owner, hold } = ownerFixture()
    const guest = contents('webview', owner)
    guest.isLoadingMainFrame.mockReturnValue(true)
    const token = beginPreviewGuestActivity(owner, guest, guest.session, hold)
    const running = executePreviewGuestScript(owner, token, 'await a frame')
    guest.emit('did-navigate', {}, 'https://new.test', 200, 'OK')
    expect(guest.executeJavaScript).not.toHaveBeenCalled()
    expect(guest.getBackgroundThrottling()).toBe(false)
    expect(owner.getBackgroundThrottling()).toBe(false)
    guest.emit('did-stop-loading')
    await expect(running).resolves.toBe('answer')
    expect(guest.getBackgroundThrottling()).toBe(true)
    expect(owner.getBackgroundThrottling()).toBe(true)
  })

  it('retires an abandoned document promise without retiring work waiting for its replacement', async () => {
    const { owner, hold } = ownerFixture()
    const guest = contents('webview', owner)
    guest.executeJavaScript.mockReturnValueOnce(new Promise(() => {}))
    const old = executePreviewGuestScript(owner, beginPreviewGuestActivity(owner, guest, guest.session, hold), 'old')
    const rejected = expect(old).rejects.toThrow('navigated')
    guest.isLoadingMainFrame.mockReturnValue(true)
    const next = executePreviewGuestScript(owner, beginPreviewGuestActivity(owner, guest, guest.session, hold), 'new')
    guest.emit('did-navigate', {}, 'https://new.test', 200, 'OK')
    await rejected
    expect(guest.getBackgroundThrottling()).toBe(false)
    guest.emit('did-stop-loading')
    await expect(next).resolves.toBe('answer')
    expect(guest.getBackgroundThrottling()).toBe(true)
    expect(owner.getBackgroundThrottling()).toBe(true)
  })

  it('cancels loading work without dispatch and rejects foreign execution without releasing the owner', async () => {
    const { owner, hold } = ownerFixture()
    const guest = contents('webview', owner)
    guest.isLoadingMainFrame.mockReturnValue(true)
    const token = beginPreviewGuestActivity(owner, guest, guest.session, hold)
    await expect(executePreviewGuestScript(contents('window'), token, 'foreign')).rejects.toThrow('belong')
    expect(owner.getBackgroundThrottling()).toBe(false)
    const running = executePreviewGuestScript(owner, token, 'cancelled')
    const rejected = expect(running).rejects.toThrow('ended')
    endPreviewGuestActivity(owner, token)
    await rejected
    guest.emit('did-stop-loading')
    expect(guest.executeJavaScript).not.toHaveBeenCalled()
    expect(owner.getBackgroundThrottling()).toBe(true)
  })
})
