import test from 'node:test'
import assert from 'node:assert/strict'
import { CDP } from './cdp.mjs'

// A fake WebSocket that never responds to any sent message.
function silentWs() {
  const listeners = new Map()
  return {
    addEventListener: (ev, fn) => {
      if (!listeners.has(ev)) listeners.set(ev, [])
      listeners.get(ev).push(fn)
    },
    send() {},
    // test helper: emit an event to the CDP listeners
    _emit(ev, arg) {
      for (const fn of listeners.get(ev) ?? []) fn(arg)
    }
  }
}

test('BUG #5 (RED): send() rejects on timeout instead of hanging forever', async () => {
  const ws = silentWs()
  const cdp = new CDP(ws)

  await assert.rejects(
    () => cdp.send('Runtime.evaluate', {}, 50),
    /timed out/,
    'send must reject with a timeout error'
  )
})

test('BUG #5: send() still resolves when the response arrives in time', async () => {
  const ws = silentWs()
  const cdp = new CDP(ws)
  // Mimic what CDP.open() registers: a message listener that resolves pending.
  ws.addEventListener('message', ev => {
    const m = JSON.parse(typeof ev.data === 'string' ? ev.data : ev.data.toString('utf8'))
    if (m.id != null && cdp.pending.has(m.id)) {
      const { resolve, reject } = cdp.pending.get(m.id)
      cdp.pending.delete(m.id)
      m.error ? reject(new Error(m.error.message)) : resolve(m.result)
    }
  })
  // Capture the outgoing message id, then reply with a valid result.
  const origSend = ws.send.bind(ws)
  ws.send = raw => {
    const m = JSON.parse(raw)
    setTimeout(() => ws._emit('message', { data: JSON.stringify({ id: m.id, result: { ok: true } }) }), 5)
    origSend(raw)
  }

  const r = await cdp.send('Runtime.evaluate', {}, 1000)
  assert.deepEqual(r, { ok: true })
})

test('BUG #5: pending map is cleaned up after a timeout (no leak)', async () => {
  const ws = silentWs()
  const cdp = new CDP(ws)

  await assert.rejects(() => cdp.send('X', {}, 30), /timed out/)
  assert.equal(cdp.pending.size, 0, 'timed-out request must be removed from pending')
})

test('BUG #5: late response after timeout does not resolve a dead promise or crash', async () => {
  const ws = silentWs()
  const cdp = new CDP(ws)
  const origSend = ws.send.bind(ws)
  let reply = null
  ws.send = raw => {
    const m = JSON.parse(raw)
    reply = { id: m.id }
    origSend(raw)
  }

  await assert.rejects(() => cdp.send('X', {}, 30), /timed out/)
  // Response arrives after the timeout — must be ignored, not crash.
  assert.doesNotThrow(() => ws._emit('message', { data: JSON.stringify({ id: reply.id, result: {} }) }))
  assert.equal(cdp.pending.size, 0)
})

test('A5 (RED): eval tolerates a response without a result payload', async () => {
  const ws = silentWs()
  const cdp = new CDP(ws)
  ws.addEventListener('message', ev => {
    const m = JSON.parse(typeof ev.data === 'string' ? ev.data : ev.data.toString('utf8'))
    if (m.id != null && cdp.pending.has(m.id)) {
      const { resolve, reject } = cdp.pending.get(m.id)
      cdp.pending.delete(m.id)
      m.error ? reject(new Error(m.error.message)) : resolve(m.result)
    }
  })
  // Reply with a MALFORMED result: no `result.result` payload at all.
  const origSend = ws.send.bind(ws)
  ws.send = raw => {
    const m = JSON.parse(raw)
    setTimeout(() => ws._emit('message', { data: JSON.stringify({ id: m.id, result: {} }) }), 5)
    origSend(raw)
  }

  const v = await cdp.eval('1 + 1')
  assert.equal(v, undefined, 'malformed response must resolve undefined, not throw')
})
