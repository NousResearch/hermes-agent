/**
 * Unit tests for the pure find-in-page helpers. The IPC handlers in
 * main.ts are the only consumer — the helpers below must keep the wire
 * shape stable (match counter shape, defaults, no-throw-on-destroyed).
 */

import assert from 'node:assert/strict'
import { EventEmitter } from 'node:events'

import type { BrowserWindow } from 'electron'
import { describe, test } from 'vitest'

import {
  formatFoundInPage,
  installFindShortcut,
  installFoundInPageForwarder,
  performFind,
  performFindAfterIndexingStarted,
  stopFind
} from './find-in-page'

// Minimal webContents stub. The Electron.WebContents type is huge, so we
// model just the slice the helpers touch (`isDestroyed`, `findInPage`,
// `stopFindInPage`, `on`/`off`, `send`, `destroyed`, `emit`) and cast through
// `asWC()` at call sites.
interface FakeWebContents {
  calls: {
    find: Array<{ query: string; options: { forward: boolean; findNext: boolean } }>
    stop: Array<'clearSelection' | 'keepSelection' | 'activateSelection'>
    send: Array<{ channel: string; payload: unknown }>
  }
  isDestroyed: () => boolean
  destroy: () => void
  findInPage: (query: string, options: { forward: boolean; findNext: boolean }) => number
  stopFindInPage: (action: 'clearSelection' | 'keepSelection' | 'activateSelection') => void
  send: (channel: string, payload: unknown) => void
  on: typeof EventEmitter.prototype.on
  once: typeof EventEmitter.prototype.once
  off: typeof EventEmitter.prototype.off
  emit: (event: string | symbol, ...args: unknown[]) => boolean
}

function makeFakeWebContents(): FakeWebContents {
  const emitter = new EventEmitter()

  const calls = {
    find: [] as Array<{ query: string; options: { forward: boolean; findNext: boolean } }>,
    stop: [] as Array<'clearSelection' | 'keepSelection' | 'activateSelection'>,
    send: [] as Array<{ channel: string; payload: unknown }>
  }

  let destroyed = false

  return {
    calls,
    isDestroyed: () => destroyed,
    destroy() {
      destroyed = true
      emitter.emit('destroyed')
    },
    findInPage(query: string, options: { forward: boolean; findNext: boolean }) {
      calls.find.push({ query, options })

      return 17
    },
    stopFindInPage(action: 'clearSelection' | 'keepSelection' | 'activateSelection') {
      calls.stop.push(action)
    },
    send(channel: string, payload: unknown) {
      calls.send.push({ channel, payload })
    },
    on: emitter.on.bind(emitter),
    once: emitter.once.bind(emitter),
    off: emitter.off.bind(emitter),
    emit: emitter.emit.bind(emitter)
  }
}

function asWC(fake: FakeWebContents): Electron.WebContents {
  return fake as unknown as Electron.WebContents
}

describe('formatFoundInPage', () => {
  test('maps activeMatchOrdinal + matches onto the wire payload', () => {
    assert.deepEqual(formatFoundInPage({ activeMatchOrdinal: 3, matches: 12 }), {
      activeMatchOrdinal: 3,
      count: 12
    })
  })

  test('coerces missing fields to zero so the renderer never sees NaN', () => {
    assert.deepEqual(formatFoundInPage({}), { activeMatchOrdinal: 0, count: 0 })
    assert.deepEqual(formatFoundInPage({ activeMatchOrdinal: 0, matches: 0 }), {
      activeMatchOrdinal: 0,
      count: 0
    })
  })

  test('null / undefined inputs still produce a well-formed payload', () => {
    assert.deepEqual(formatFoundInPage(null as unknown as { activeMatchOrdinal?: number; matches?: number }), {
      activeMatchOrdinal: 0,
      count: 0
    })
    assert.deepEqual(formatFoundInPage(undefined), { activeMatchOrdinal: 0, count: 0 })
  })
})

describe('performFind', () => {
  test('forwards the query and options to webContents.findInPage', () => {
    const wc = makeFakeWebContents()
    performFind(asWC(wc), 'hello', { forward: true, findNext: false })
    assert.deepEqual(wc.calls.find, [{ query: 'hello', options: { forward: true, findNext: false } }])
  })

  test('defaults forward to true when omitted', () => {
    const wc = makeFakeWebContents()
    performFind(asWC(wc), 'x', { findNext: true })
    assert.deepEqual(wc.calls.find, [{ query: 'x', options: { forward: true, findNext: true } }])
  })

  test('defaults findNext to false when omitted', () => {
    const wc = makeFakeWebContents()
    performFind(asWC(wc), 'x', { forward: false })
    assert.deepEqual(wc.calls.find, [{ query: 'x', options: { forward: false, findNext: false } }])
  })

  test('treats null / non-object options as "all defaults"', () => {
    const wc = makeFakeWebContents()
    performFind(asWC(wc), 'x', null)
    assert.deepEqual(wc.calls.find, [{ query: 'x', options: { forward: true, findNext: false } }])
  })

  test('coerces a non-string query to string (defensive against bad renderer payloads)', () => {
    const wc = makeFakeWebContents()
    performFind(asWC(wc), 42 as unknown as string, null)
    assert.equal(wc.calls.find[0].query, '42')
  })

  test('is a no-op when webContents is null', () => {
    assert.doesNotThrow(() => performFind(null, 'q', null))
  })

  test('is a no-op when webContents is destroyed (does not throw across IPC)', () => {
    const wc = makeFakeWebContents()
    wc.destroy()
    performFind(asWC(wc), 'q', null)
    assert.equal(wc.calls.find.length, 0)
  })
})

describe('performFindAfterIndexingStarted', () => {
  test('resolves only after the matching request emits its first result', async () => {
    const wc = makeFakeWebContents()
    let resolved = false

    const pending = performFindAfterIndexingStarted(asWC(wc), 'needle', {
      forward: true,
      findNext: false
    }).then(() => {
      resolved = true
    })

    await Promise.resolve()
    assert.equal(resolved, false)

    wc.emit('found-in-page', {}, { requestId: 9, matches: 1 })
    await Promise.resolve()
    assert.equal(resolved, false)

    wc.emit('found-in-page', {}, { requestId: 17, matches: 1 })
    await pending
    assert.equal(resolved, true)
  })

  test('resolves safely if the webContents is destroyed before a result', async () => {
    const wc = makeFakeWebContents()
    const pending = performFindAfterIndexingStarted(asWC(wc), 'needle', null)

    wc.destroy()
    await pending
  })
})

describe('stopFind', () => {
  test('calls stopFindInPage with the default action (clearSelection)', () => {
    const wc = makeFakeWebContents()
    stopFind(asWC(wc))
    assert.deepEqual(wc.calls.stop, ['clearSelection'])
  })

  test('honors an explicit action argument', () => {
    const wc = makeFakeWebContents()
    stopFind(asWC(wc), 'keepSelection')
    assert.deepEqual(wc.calls.stop, ['keepSelection'])
  })

  test('is a no-op when webContents is null or destroyed', () => {
    assert.doesNotThrow(() => stopFind(null))
    const wc = makeFakeWebContents()
    wc.destroy()
    stopFind(asWC(wc))
    assert.equal(wc.calls.stop.length, 0)
  })
})

describe('installFoundInPageForwarder', () => {
  test('forwards found-in-page to the sender as a formatted payload', () => {
    const wc = makeFakeWebContents()
    installFoundInPageForwarder(asWC(wc))
    // Drive the fake's emit directly — this exercises the same code path
    // as Electron's actual `webContents.emit('found-in-page', …)`.
    wc.emit('found-in-page', {}, { activeMatchOrdinal: 2, matches: 5 })
    assert.deepEqual(wc.calls.send, [{ channel: 'hermes:found-in-page', payload: { activeMatchOrdinal: 2, count: 5 } }])
  })

  test('handles missing fields without throwing', () => {
    const wc = makeFakeWebContents()
    installFoundInPageForwarder(asWC(wc))
    wc.emit('found-in-page', {}, {})
    assert.deepEqual(wc.calls.send, [{ channel: 'hermes:found-in-page', payload: { activeMatchOrdinal: 0, count: 0 } }])
  })

  test('skips send when webContents is destroyed at fire time', () => {
    const wc = makeFakeWebContents()
    installFoundInPageForwarder(asWC(wc))
    wc.destroy()
    wc.emit('found-in-page', {}, { activeMatchOrdinal: 1, matches: 1 })
    assert.equal(wc.calls.send.length, 0, 'destroyed webContents must not be sent to')
  })

  test('returned uninstall removes the listener', () => {
    const wc = makeFakeWebContents()
    const uninstall = installFoundInPageForwarder(asWC(wc))
    uninstall()
    wc.emit('found-in-page', {}, { activeMatchOrdinal: 9, matches: 9 })
    assert.equal(wc.calls.send.length, 0, 'uninstalled listener must not fire')
  })

  test('returned uninstall on a null webContents is a safe no-op', () => {
    const uninstall = installFoundInPageForwarder(null)
    assert.doesNotThrow(() => uninstall())
  })

  test('returned uninstall on a destroyed webContents is a safe no-op', () => {
    const wc = makeFakeWebContents()
    wc.destroy()
    const uninstall = installFoundInPageForwarder(asWC(wc))
    assert.doesNotThrow(() => uninstall())
  })

  // Regression: the original PR scoped the forwarder to the global mainWindow,
  // so Cmd+F pressed in a secondary session window routed results back to the
  // primary. Pin that the helper does NOT close over any window other than the
  // webContents it was given — two forwarders installed on two distinct fakes
  // must each send only to their own sender.
  test('two forwarders installed on distinct webContents do not cross-fire', () => {
    const wcA = makeFakeWebContents()
    const wcB = makeFakeWebContents()
    installFoundInPageForwarder(asWC(wcA))
    installFoundInPageForwarder(asWC(wcB))
    wcA.emit('found-in-page', {}, { activeMatchOrdinal: 1, matches: 1 })
    assert.deepEqual(wcA.calls.send, [
      { channel: 'hermes:found-in-page', payload: { activeMatchOrdinal: 1, count: 1 } }
    ])
    assert.equal(wcB.calls.send.length, 0, 'wcB must not receive wcA results')
  })
})


describe('installFindShortcut', () => {
  // Minimal BrowserWindow stub: only `webContents` is touched.
  function makeFakeWindow(wc: FakeWebContents) {
    return { webContents: asWC(wc) } as unknown as BrowserWindow
  }

  test('sends hermes:open-find-bar on Ctrl+F (Linux/Windows) and prevents default', () => {
    const wc = makeFakeWebContents()
    const win = makeFakeWindow(wc)
    const uninstall = installFindShortcut(win)

    // Ctrl+F on Linux/Windows (no meta, no alt, no shift).
    const result = wc.emit('before-input-event', {}, {
      key: 'f',
      control: true,
      meta: false,
      alt: false,
      shift: false,
    })
    // The listener calls preventDefault on the event; the fake's emit returns
    // truthy because the event fired — what matters is the side effects.
    void result

    assert.deepEqual(wc.calls.send, [
      { channel: 'hermes:open-find-bar', payload: undefined }
    ])

    uninstall()
  })

  // macOS branch: inject `isMac: () => true` so we exercise the REAL
  // `meta` (Cmd) path — previously untested, because `process.platform` is
  // baked at import time and the old "Cmd+F" case actually sent Ctrl.
  test('sends hermes:open-find-bar on Cmd+F (meta) on macOS and prevents default', () => {
    const wc = makeFakeWebContents()
    const win = makeFakeWindow(wc)
    const uninstall = installFindShortcut(win, () => true)

    // Cmd+F on macOS: meta held, no control/alt/shift.
    wc.emit('before-input-event', {}, {
      key: 'f',
      control: false,
      meta: true,
      alt: false,
      shift: false,
    })

    assert.deepEqual(wc.calls.send, [
      { channel: 'hermes:open-find-bar', payload: undefined }
    ])

    uninstall()
  })

  // The design intentionally accepts literal Ctrl on macOS too (dual-channel)
  // so a non-macOS layout still works. Pin that behavior so the width of the
  // chord doesn't silently drift.
  test('accepts literal Ctrl+F on macOS (dual-channel with Cmd)', () => {
    const wc = makeFakeWebContents()
    const win = makeFakeWindow(wc)
    const uninstall = installFindShortcut(win, () => true)

    // Ctrl+F with no meta on macOS still opens the FindBar.
    wc.emit('before-input-event', {}, {
      key: 'F',
      control: true,
      meta: false,
      alt: false,
      shift: false,
    })

    assert.deepEqual(wc.calls.send, [
      { channel: 'hermes:open-find-bar', payload: undefined }
    ])

    uninstall()
  })

  // Cross-check: a bare Ctrl+F WITHOUT meta must NOT open on Linux/Windows,
  // where only `control` counts (the macOS `meta || control` widening must not
  // leak across the platform boundary).
  test('does NOT fire for Ctrl+F with meta only on Linux/Windows', () => {
    const wc = makeFakeWebContents()
    const win = makeFakeWindow(wc)
    const uninstall = installFindShortcut(win, () => false)

    wc.emit('before-input-event', {}, {
      key: 'f',
      control: false,
      meta: true,
      alt: false,
      shift: false,
    })

    assert.equal(wc.calls.send.length, 0, 'meta (Cmd) is not a valid chord on non-macOS')

    uninstall()
  })

  test('does NOT fire for plain F without Ctrl/Cmd', () => {
    const wc = makeFakeWebContents()
    const win = makeFakeWindow(wc)
    const uninstall = installFindShortcut(win)

    wc.emit('before-input-event', {}, {
      key: 'f',
      control: false,
      meta: false,
      alt: false,
      shift: false,
    })

    assert.equal(wc.calls.send.length, 0, 'plain F must not open the FindBar')

    uninstall()
  })

  test('does NOT fire for Ctrl+Shift+F (different chord)', () => {
    const wc = makeFakeWebContents()
    const win = makeFakeWindow(wc)
    const uninstall = installFindShortcut(win)

    wc.emit('before-input-event', {}, {
      key: 'f',
      control: true,
      meta: false,
      alt: false,
      shift: true,
    })

    assert.equal(wc.calls.send.length, 0, 'Ctrl+Shift+F is reserved (session.focusSearch)')

    uninstall()
  })

  test('does NOT fire for Ctrl+F with Alt held (combo change)', () => {
    const wc = makeFakeWebContents()
    const win = makeFakeWindow(wc)
    const uninstall = installFindShortcut(win)

    wc.emit('before-input-event', {}, {
      key: 'f',
      control: true,
      meta: false,
      alt: true,
      shift: false,
    })

    assert.equal(wc.calls.send.length, 0)

    uninstall()
  })

  test('uninstall detaches the listener', () => {
    const wc = makeFakeWebContents()
    const win = makeFakeWindow(wc)
    const uninstall = installFindShortcut(win)
    uninstall()

    wc.emit('before-input-event', {}, {
      key: 'f',
      control: true,
      meta: false,
      alt: false,
      shift: false,
    })

    assert.equal(wc.calls.send.length, 0, 'listener must be detached after uninstall()')
  })

  test('is a no-op on a destroyed webContents', () => {
    const wc = makeFakeWebContents()
    wc.destroy()
    const win = makeFakeWindow(wc)
    // Should not throw — uninstall is the no-op fn returned in this branch.
    const uninstall = installFindShortcut(win)
    assert.doesNotThrow(() => uninstall())
  })
})
