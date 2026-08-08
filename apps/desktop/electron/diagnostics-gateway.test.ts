import assert from 'node:assert/strict'

import { test } from 'vitest'

import { classifyGatewayFailure, createGatewayDiagnosticsClient, gatewayDiagnosticsPath } from './diagnostics-gateway'

// Records every (method, params) pair so a test can assert the exact wire shape
// the Python gateway has to answer.
function recordingTransport(answers: Record<string, unknown | (() => never)>) {
  const calls: { method: string; params: Record<string, unknown> }[] = []

  const request = async (method, params) => {
    calls.push({ method, params })

    const answer = answers[method]

    if (typeof answer === 'function') {
      return (answer as () => never)()
    }

    return answer
  }

  return { calls, request }
}

test('methods map onto the gateway REST paths', () => {
  assert.equal(gatewayDiagnosticsPath('diagnostics/arm'), '/api/diagnostics/arm')
  assert.equal(gatewayDiagnosticsPath('diagnostics/disarm'), '/api/diagnostics/disarm')
  assert.equal(gatewayDiagnosticsPath('diagnostics/collect'), '/api/diagnostics/collect')
})

test('arm sends snake_case params and returns the gateway monotonic anchor', async () => {
  const transport = recordingTransport({ 'diagnostics/arm': { monotonic_anchor_ms: 1234.5 } })
  const client = createGatewayDiagnosticsClient(transport.request)

  const result = await client.arm({ captureId: 'cap-1', wallClockAnchorMs: 1_700_000_000_000 })

  assert.deepEqual(result, { ok: true, monotonicAnchorMs: 1234.5 })
  assert.deepEqual(transport.calls, [
    {
      method: 'diagnostics/arm',
      params: { capture_id: 'cap-1', wall_clock_anchor_ms: 1_700_000_000_000 }
    }
  ])
})

test('a gateway that does not know the endpoint degrades to unsupported', async () => {
  const client = createGatewayDiagnosticsClient(async () => {
    throw new Error('404: Not Found')
  })

  assert.deepEqual(await client.arm({ captureId: 'cap-1', wallClockAnchorMs: 1 }), {
    ok: false,
    reason: 'unsupported'
  })
})

test('an unauthenticated pull degrades rather than failing the capture', async () => {
  const client = createGatewayDiagnosticsClient(async () => {
    throw new Error('401: {"detail":"no_cookie"}')
  })

  assert.deepEqual(await client.collect('cap-1'), { ok: false, reason: 'unauthenticated' })
})

test('a transport failure degrades to unavailable', async () => {
  const client = createGatewayDiagnosticsClient(async () => {
    throw new Error('Timed out connecting to Hermes backend after 10000ms')
  })

  assert.deepEqual(await client.arm({ captureId: 'cap-1', wallClockAnchorMs: 1 }), {
    ok: false,
    reason: 'unavailable'
  })
})

test('an answer without a monotonic anchor is unusable for alignment', async () => {
  const client = createGatewayDiagnosticsClient(async () => ({ ok: true }))

  assert.deepEqual(await client.arm({ captureId: 'cap-1', wallClockAnchorMs: 1 }), {
    ok: false,
    reason: 'unsupported'
  })
})

test('collect normalizes the ring payload', async () => {
  const transport = recordingTransport({
    'diagnostics/collect': {
      capture_id: 'cap-1',
      monotonic_anchor_ms: 42,
      events: [{ type: 'loop_stall', t: 99 }],
      dropped: 3
    }
  })

  const client = createGatewayDiagnosticsClient(transport.request)

  const result = await client.collect('cap-1')

  assert.equal(result.ok, true)
  assert.deepEqual(result.ok && result.stream, {
    captureId: 'cap-1',
    monotonicAnchorMs: 42,
    events: [{ type: 'loop_stall', t: 99 }],
    dropped: 3
  })
  assert.deepEqual(transport.calls[0].params, { capture_id: 'cap-1' })
})

test('collect tolerates a ring with no events and no dropped counter', async () => {
  const client = createGatewayDiagnosticsClient(async () => ({ capture_id: 'cap-9', monotonic_anchor_ms: 0 }))

  const result = await client.collect('cap-9')

  assert.equal(result.ok, true)
  assert.deepEqual(result.ok && result.stream.events, [])
  assert.equal(result.ok && result.stream.dropped, 0)
})

test('disarm never throws', async () => {
  const client = createGatewayDiagnosticsClient(async () => {
    throw new Error('500: boom')
  })

  await client.disarm()
})

test('failure classification reads the status prefix fetchJson rejects with', () => {
  assert.equal(classifyGatewayFailure(new Error('405: Method Not Allowed')), 'unsupported')
  assert.equal(classifyGatewayFailure(new Error('403: forbidden')), 'unauthenticated')
  assert.equal(classifyGatewayFailure(new Error('ECONNREFUSED')), 'unavailable')
  assert.equal(classifyGatewayFailure(null), 'unavailable')
})
