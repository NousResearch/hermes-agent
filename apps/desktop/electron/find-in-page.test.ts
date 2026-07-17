/**
 * Unit tests for the pure find-in-page helpers. The IPC handlers in
 * main.ts are the only consumer — the helpers below must keep the wire
 * shape stable (match counter shape, defaults, no-throw-on-destroyed).
 */

import assert from 'node:assert/strict'
import { EventEmitter } from 'node:events'

import { describe, test } from 'vitest'

import {
  formatFoundInPage,
  installFoundInPageForwarder,
  performFind,
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
  findInPage: (query: string, options: { forward: boolean; findNext: boolean }) => void
  stopFindInPage: (action: 'clearSelection' | 'keepSelection' | 'activateSelection') => void
  send: (channel: string, payload: unknown) => void
  on: typeof EventEmitter.prototype.on
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
    },
    stopFindInPage(action: 'clearSelection' | 'keepSelection' | 'activateSelection') {
      calls.stop.push(action)
    },
    send(channel: string, payload: unknown) {
      calls.send.push({ channel, payload })
    },
    on: emitter.on.bind(emitter),
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
    assert.deepEqual(
      formatFoundInPage(null as unknown as { activeMatchOrdinal?: number; matches?: number }),
      { activeMatchOrdinal: 0, count: 0 }
    )
    assert.deepEqual(formatFoundInPage(undefined), { activeMatchOrdinal: 0, count: 0 })
  })
})

describe('performFind', () => {
  test('forwards the query and options to webContents.findInPage', () => {
    const wc = makeFakeWebContents()
    performFind(asWC(wc), 'hello', { forward: true, findNext: false })
    assert.deepEqual(wc.calls.find, [
      { query: 'hello', options: { forward: true, findNext: false } }
    ])
  })

  test('defaults forward to true when omitted', () => {
    const wc = makeFakeWebContents()
    performFind(asWC(wc), 'x', { findNext: true })
    assert.deepEqual(wc.calls.find, [
      { query: 'x', options: { forward: true, findNext: true } }
    ])
  })

  test('defaults findNext to false when omitted', () => {
    const wc = makeFakeWebContents()
    performFind(asWC(wc), 'x', { forward: false })
    assert.deepEqual(wc.calls.find, [
      { query: 'x', options: { forward: false, findNext: false } }
    ])
  })

  test('treats null / non-object options as "all defaults"', () => {
    const wc = makeFakeWebContents()
    performFind(asWC(wc), 'x', null)
    assert.deepEqual(wc.calls.find, [
      { query: 'x', options: { forward: true, findNext: false } }
    ])
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
    assert.deepEqual(wc.calls.send, [
      { channel: 'hermes:found-in-page', payload: { activeMatchOrdinal: 2, count: 5 } }
    ])
  })

  test('handles missing fields without throwing', () => {
    const wc = makeFakeWebContents()
    installFoundInPageForwarder(asWC(wc))
    wc.emit('found-in-page', {}, {})
    assert.deepEqual(wc.calls.send, [
      { channel: 'hermes:found-in-page', payload: { activeMatchOrdinal: 0, count: 0 } }
    ])
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
