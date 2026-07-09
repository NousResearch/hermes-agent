/**
 * Tests for electron/ix-chat.ts — the native IX Agency chat's write gate.
 *
 * Run with: node --test electron/ix-chat.test.ts
 * (Wired into npm test:desktop:platforms in package.json.)
 *
 * Why this matters: the tool loop runs CLIENT-SIDE in the main process, so
 * this gate is the only thing standing between the model and a mutating
 * gateway call. It must (a) classify with the same token rules as the
 * portals (default-deny ambiguous, viz_render_* exempt, admin_call_mcp
 * classified structurally), (b) refuse to execute unapproved writes while
 * emitting a confirmation card, (c) execute ONLY after the UI's Confirm,
 * with the args FROZEN at request time, and (d) treat nonces as single-use
 * and expiring.
 */

import assert from 'node:assert/strict'
import test from 'node:test'

import {
  capToolOutput,
  classifyToolAccess,
  createWriteGate,
  executeGatedToolCall,
  gateResultForModel,
  type IxChatEvent,
  summarizeArgs
} from './ix-chat'

/* ── classifier ──────────────────────────────────────────────────────────── */

test('classifier: read-token tools are reads', () => {
  assert.equal(classifyToolAccess('nakama_list_users'), 'read')
  assert.equal(classifyToolAccess('grafana_get_dashboard'), 'read')
  assert.equal(classifyToolAccess('stripe_search_charges'), 'read')
  assert.equal(classifyToolAccess('nakama_health'), 'read')
})

test('classifier: write tokens win even when read tokens are present', () => {
  assert.equal(classifyToolAccess('stripe_create_refund'), 'write')
  assert.equal(classifyToolAccess('nakama_delete_user'), 'write')
  assert.equal(classifyToolAccess('n8n_update_workflow'), 'write')
  // both 'list' (read) and 'update' (write) — write must win
  assert.equal(classifyToolAccess('crm_list_and_update_contacts'), 'write')
})

test('classifier: ambiguous names are DEFAULT-DENIED as writes', () => {
  assert.equal(classifyToolAccess('mystery_thing'), 'write')
  assert.equal(classifyToolAccess(''), 'write')
  assert.equal(classifyToolAccess(undefined as unknown as string), 'write')
})

test('classifier: viz_render_* is exempt (creates new S3 artifacts only)', () => {
  assert.equal(classifyToolAccess('viz_render_chart'), 'read')
  assert.equal(classifyToolAccess('viz_render_video'), 'read')
})

test('classifier: connector prefixes are stripped before classification', () => {
  assert.equal(classifyToolAccess('posthog__list_events'), 'read')
  assert.equal(classifyToolAccess('stripe__create_charge'), 'write')
})

test('classifier: admin_call_mcp is classified structurally from its args', () => {
  assert.equal(classifyToolAccess('admin_call_mcp', { method: 'tools/list' }), 'read')
  assert.equal(classifyToolAccess('admin_call_mcp', { method: 'ping' }), 'read')
  assert.equal(classifyToolAccess('admin_call_mcp', { method: 'tools/call', tool: 'nakama_get_user' }), 'read')
  assert.equal(classifyToolAccess('admin_call_mcp', { method: 'tools/call', tool: 'nakama_ban_user' }), 'write')
  // missing/unknown inner tool or method — default deny
  assert.equal(classifyToolAccess('admin_call_mcp', { method: 'tools/call' }), 'write')
  assert.equal(classifyToolAccess('admin_call_mcp', { method: 'something/else' }), 'write')
  assert.equal(classifyToolAccess('admin_call_mcp', {}), 'write')
})

test('classifier: Twenty CRM execute_tool unwraps arguments.toolName', () => {
  assert.equal(
    classifyToolAccess('admin_call_mcp', {
      method: 'tools/call',
      tool: 'execute_tool',
      arguments: { toolName: 'find_many_companies' }
    }),
    'read'
  )
  assert.equal(
    classifyToolAccess('admin_call_mcp', {
      method: 'tools/call',
      tool: 'execute_tool',
      arguments: { toolName: 'create_company' }
    }),
    'write'
  )
})

/* ── gate primitives ─────────────────────────────────────────────────────── */

test('gate: nonces are single-use and unknown nonces are rejected', () => {
  const gate = createWriteGate()
  const nonce = gate.request('stripe_create_refund', { amount: 5 })

  assert.equal(gate.confirm('not-a-nonce'), false)
  assert.equal(gate.confirm(nonce), true)
  // double-confirm is rejected
  assert.equal(gate.confirm(nonce), false)

  assert.deepEqual(gate.redeem('stripe_create_refund'), { amount: 5 })
  // single-use: a second redeem finds nothing
  assert.equal(gate.redeem('stripe_create_refund'), null)
})

test('gate: deny drops the pending write', () => {
  const gate = createWriteGate()
  const nonce = gate.request('nakama_delete_user', { id: 'u1' })

  assert.equal(gate.deny(nonce), true)
  assert.equal(gate.confirm(nonce), false)
  assert.equal(gate.redeem('nakama_delete_user'), null)
})

test('gate: approvals expire', () => {
  let clock = 1_000

  const gate = createWriteGate(() => clock)
  const nonce = gate.request('nakama_delete_user', { id: 'u1' })

  assert.equal(gate.confirm(nonce), true)
  clock += 11 * 60_000 // past the 10-minute TTL
  assert.equal(gate.redeem('nakama_delete_user'), null)
})

test('gate: redeem is scoped to the requested tool', () => {
  const gate = createWriteGate()
  const nonce = gate.request('stripe_create_refund', { amount: 5 })

  gate.confirm(nonce)
  // approving a refund must not authorize a different write
  assert.equal(gate.redeem('nakama_delete_user'), null)
  assert.deepEqual(gate.redeem('stripe_create_refund'), { amount: 5 })
})

/* ── executor flow ───────────────────────────────────────────────────────── */

function collectEvents() {
  const events: IxChatEvent[] = []

  return { events, emit: (event: IxChatEvent) => events.push(event) }
}

test('executor: reads run straight through', async () => {
  const gate = createWriteGate()
  const { events, emit } = collectEvents()
  const calls: string[] = []

  const result = await executeGatedToolCall({
    name: 'nakama_list_users',
    args: { limit: 5 },
    gate,
    callGatewayTool: async name => {
      calls.push(name)

      return '{"users":[]}'
    },
    emit
  })

  assert.equal(result.status, 'ok')
  assert.equal(result.text, '{"users":[]}')
  assert.deepEqual(calls, ['nakama_list_users'])
  assert.equal(events.length, 0)
})

test('executor: unapproved writes are refused with a confirmation card', async () => {
  const gate = createWriteGate()
  const { events, emit } = collectEvents()
  let executed = 0

  const result = await executeGatedToolCall({
    name: 'stripe_create_refund',
    args: { charge: 'ch_1' },
    gate,
    callGatewayTool: async () => {
      executed++

      return 'refunded'
    },
    emit
  })

  assert.equal(executed, 0, 'the gateway must NOT be called')
  assert.equal(result.status, 'gated')
  assert.ok(result.nonce, 'renderer needs the nonce for the confirm card')
  // the model-facing text must flag the gate and never contain the nonce
  assert.match(result.text, /confirmation_required/)
  assert.ok(!result.text.includes(result.nonce as string), 'nonce must never reach the model')

  assert.equal(events.length, 1)
  assert.equal(events[0].type, 'confirmation-required')
  assert.equal(events[0].nonce, result.nonce)
})

test('executor: confirmed writes execute with the FROZEN args', async () => {
  const gate = createWriteGate()
  const { emit } = collectEvents()
  const executedArgs: Record<string, unknown>[] = []

  const callGatewayTool = async (_name: string, args: Record<string, unknown>) => {
    executedArgs.push(args)

    return 'done'
  }

  // 1. model tries the write — gated
  const first = await executeGatedToolCall({
    name: 'stripe_create_refund',
    args: { charge: 'ch_1', amount: 500 },
    gate,
    callGatewayTool,
    emit
  })

  // 2. user clicks Confirm in the UI
  assert.equal(gate.confirm(first.nonce as string), true)

  // 3. model re-issues the call — but with TAMPERED args
  const second = await executeGatedToolCall({
    name: 'stripe_create_refund',
    args: { charge: 'ch_1', amount: 500_000 },
    gate,
    callGatewayTool,
    emit
  })

  assert.equal(second.status, 'approved')
  assert.equal(executedArgs.length, 1)
  // execution used the args the USER approved, not the re-issued ones
  assert.deepEqual(executedArgs[0], { charge: 'ch_1', amount: 500 })
})

test('executor: denied writes stay gated on re-issue', async () => {
  const gate = createWriteGate()
  const { emit } = collectEvents()
  let executed = 0

  const callGatewayTool = async () => {
    executed++

    return 'done'
  }

  const first = await executeGatedToolCall({
    name: 'nakama_ban_user',
    args: { id: 'u1' },
    gate,
    callGatewayTool,
    emit
  })

  gate.deny(first.nonce as string)

  const second = await executeGatedToolCall({
    name: 'nakama_ban_user',
    args: { id: 'u1' },
    gate,
    callGatewayTool,
    emit
  })

  assert.equal(second.status, 'gated')
  assert.equal(executed, 0)
})

test('executor: gateway errors are reported, not thrown', async () => {
  const gate = createWriteGate()
  const { emit } = collectEvents()

  const result = await executeGatedToolCall({
    name: 'nakama_list_users',
    args: {},
    gate,
    callGatewayTool: async () => {
      throw new Error('gateway 502')
    },
    emit
  })

  assert.equal(result.status, 'error')
  assert.match(result.text, /gateway 502/)
})

/* ── helpers ─────────────────────────────────────────────────────────────── */

test('summarizeArgs truncates and survives unserializable args', () => {
  assert.equal(summarizeArgs({ a: 1 }), '{"a":1}')
  assert.ok(summarizeArgs({ blob: 'x'.repeat(1000) }).length <= 301)

  const circular: Record<string, unknown> = {}

  circular.self = circular
  assert.equal(summarizeArgs(circular), '(unserializable arguments)')
})

test('gateResultForModel names the tool and forbids fake completion', () => {
  const text = gateResultForModel('stripe_create_refund', { charge: 'ch_1' })
  const parsed = JSON.parse(text)

  assert.equal(parsed.status, 'confirmation_required')
  assert.equal(parsed.tool, 'stripe_create_refund')
  assert.match(parsed.instructions, /NOT executed/)
})

test('capToolOutput compacts giant tool catalogs instead of blind-truncating', () => {
  const tools = Array.from({ length: 400 }, (_, i) => ({
    name: `tool_${i}`,
    description: 'does a thing '.repeat(20)
  }))

  const giant = JSON.stringify({ result: { tools } })
  const capped = capToolOutput(giant)

  assert.ok(capped.length < giant.length)
  assert.match(capped, /400 tools:/)
  assert.match(capped, /- tool_0/)
})
