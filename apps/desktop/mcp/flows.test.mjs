import test from 'node:test'
import assert from 'node:assert/strict'
import { flowTools, handleFlow } from './tools/flows.mjs'

// CDP mock with scriptable eval responses keyed by a marker in the expression.
function mockCdp(handlers = {}) {
  const sends = []
  return {
    sends,
    eval: async (expr) => {
      for (const [marker, value] of Object.entries(handlers)) {
        if (expr.includes(marker)) return typeof value === 'function' ? value(expr) : value
      }
      return true
    },
    send: async (method, params) => {
      sends.push({ method, params })
      return { ok: true }
    }
  }
}

test('ui_flow_edit: no user messages → edit-button-not-found outcome', async () => {
  const cdp = mockCdp({
    'aui_user-message-root': null // querySelectorAll returns empty
  })
  const out = await handleFlow('ui_flow_edit', {}, { cdp })
  assert.equal(out.outcome, 'edit-button-not-found')
})

test('ui_flow_edit: button present → types via insertText + presses Enter', async () => {
  const cdp = mockCdp({
    'aui_user-message-root': { hasMsg: true, hasBtn: true },
    'aui_edit-composer-root': true, // composer mounts after click
    'turnPair': 3 // countUserMessages baseline
  })
  const out = await handleFlow('ui_flow_edit', { newText: 'revised' }, { cdp })

  // typing must use Input.insertText (not dispatchKeyEvent char)
  const insert = cdp.sends.find((s) => s.method === 'Input.insertText')
  assert.ok(insert, 'expected Input.insertText during edit flow')
  assert.equal(insert.params.text, 'revised')

  // Enter must be dispatched as keyDown + keyUp
  const enters = cdp.sends.filter((s) => s.method === 'Input.dispatchKeyEvent' && s.params.key === 'Enter')
  assert.ok(enters.length >= 2, 'Enter keyDown+keyUp expected')

  // outcome computed from observations (composer closed + timeline grew)
  assert.ok(['accepted', 'stuck-open (likely silent-fail race)', 'closed-but-timeline-unclear'].includes(out.outcome))
})

test('ui_flow_model_switch installs a MutationObserver in thread content', async () => {
  const cdp = mockCdp({
    'threadContent': true
  })
  const out = await handleFlow('ui_flow_model_switch', {}, { cdp })
  assert.equal(out.observerInstalled, true)
  assert.ok(out.note.includes('__MCP_MUT__'))
})

test('handleFlow throws on unknown flow tool', async () => {
  const cdp = mockCdp()
  await assert.rejects(() => handleFlow('ui_flow_nope', {}, { cdp }), /unknown flow tool/)
})

test('flowTools are all gated', () => {
  for (const t of flowTools) assert.equal(t.gated, true, `${t.name} must be gated`)
})
