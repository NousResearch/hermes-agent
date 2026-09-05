import test from 'node:test'
import assert from 'node:assert/strict'
import { actTools, handleAct } from './tools/act.mjs'

// In-memory CDP mock that records every send() call.
function mockCdp() {
  const sends = []
  const evals = []
  return {
    sends,
    evals,
    eval: async (expr) => {
      evals.push(expr)
      // centerOf / press read getBoundingClientRect → return the CENTER
      // (mirrors what the in-renderer eval computes: x + width/2).
      if (expr.includes('getBoundingClientRect')) return { x: 110, y: 55 }
      return true
    },
    send: async (method, params) => {
      sends.push({ method, params })
      return { ok: true }
    }
  }
}

const ctx = (cdp) => ({
  cdp,
  resolveSelector: (s) => s,
  evalBounded: async (expr) => 'eval-result'
})

test('ui_type uses Input.insertText (not dispatchKeyEvent char) — P0 regression fix', async () => {
  const cdp = mockCdp()
  const out = await handleAct('ui_type', { selector: '.composer', text: 'hello' }, ctx(cdp))
  assert.equal(out.typed, 5)
  const insert = cdp.sends.find((s) => s.method === 'Input.insertText')
  assert.ok(insert, 'expected Input.insertText to be called')
  assert.equal(insert.params.text, 'hello')
  const bad = cdp.sends.find((s) => s.method === 'Input.dispatchKeyEvent' && s.params.type === 'char')
  assert.equal(bad, undefined, 'must NOT use dispatchKeyEvent type:char (ignored by composer)')
})

test('ui_click sends real mousePressed + mouseReleased at element center', async () => {
  const cdp = mockCdp()
  // centerOf's eval returns the CENTER (mirrors in-renderer x + width/2).
  cdp.eval = async () => ({ x: 110, y: 55 })
  const out = await handleAct('ui_click', { selector: '.btn' }, ctx(cdp))
  assert.equal(out.clicked, true)
  assert.equal(out.at.x, 110)
  assert.equal(out.at.y, 55)
  const types = cdp.sends.filter((s) => s.method === 'Input.dispatchMouseEvent').map((s) => s.params.type)
  assert.deepEqual(types, ['mousePressed', 'mouseReleased'])
})

test('ui_press maps Enter to proper keyDown/keyUp with virtual key codes', async () => {
  const cdp = mockCdp()
  await handleAct('ui_press', { key: 'Enter' }, ctx(cdp))
  const kd = cdp.sends.find((s) => s.method === 'Input.dispatchKeyEvent' && s.params.type === 'keyDown')
  const ku = cdp.sends.find((s) => s.method === 'Input.dispatchKeyEvent' && s.params.type === 'keyUp')
  assert.ok(kd && ku, 'keyDown and keyUp must both be sent')
  assert.equal(kd.params.key, 'Enter')
  assert.equal(kd.params.windowsVirtualKeyCode, 13)
})

test('ui_press rejects unsupported keys', async () => {
  const cdp = mockCdp()
  await assert.rejects(() => handleAct('ui_press', { key: 'F17' }, ctx(cdp)), /unsupported key/)
})

test('ui_eval delegates to evalBounded', async () => {
  const cdp = mockCdp()
  const out = await handleAct('ui_eval', { expression: '1+1' }, ctx(cdp))
  assert.equal(out, 'eval-result')
})

test('handleAct throws on unknown act tool', async () => {
  const cdp = mockCdp()
  await assert.rejects(() => handleAct('ui_nope', {}, ctx(cdp)), /unknown act tool/)
})

test('actTools schema requires correct inputs', () => {
  const click = actTools.find((t) => t.name === 'ui_click')
  assert.deepEqual(click.inputSchema.required, ['selector'])
  const press = actTools.find((t) => t.name === 'ui_press')
  assert.deepEqual(press.inputSchema.required, ['key'])
})

// --- A3: act tools share the friendly empty-selector guard ---

test('A3 (RED): ui_click with empty selector rejects friendly BEFORE any CDP send', async () => {
  const cdp = mockCdp()
  await assert.rejects(
    () => handleAct('ui_click', { selector: '' }, ctx(cdp)),
    /selector required/
  )
  assert.equal(cdp.sends.length, 0, 'no CDP traffic may happen for a bad selector')
  assert.equal(cdp.evals.length, 0, 'no renderer eval may happen for a bad selector')
})

test('A3 (RED): ui_type with whitespace selector — same friendly guard', async () => {
  const cdp = mockCdp()
  await assert.rejects(
    () => handleAct('ui_type', { selector: '   ', text: 'x' }, ctx(cdp)),
    /selector required/
  )
  assert.equal(cdp.sends.length + cdp.evals.length, 0, 'no CDP traffic for a bad selector')
})
