import { app } from 'electron'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import WebSocket from 'ws'
import { configureWindowsGatewayTicketClient, ensureLocalGateway, mintLocalGatewayTicket, nativeGatewayHttpHeaders, runGatewayEnsure } from '../../apps/desktop/electron/local-gateway'
import { mintGatewayTicketWithPython } from '../../apps/desktop/electron/local-gateway-python'

/**
 * Multiplex attach + cold-profile sidebar mutation, in real Electron main against one
 * `gateway.run` that serves default + warm + cold. Every step is the production Desktop
 * code path: `runGatewayEnsure` → `ensureLocalGateway` (the child main.ts spawns per
 * profile), `mintLocalGatewayTicket` (WS attach), `nativeGatewayHttpHeaders` (REST), and
 * the exact request shape `src/api/sessions.ts::mutateSessionHttp` sends for archive.
 */
const input = JSON.parse(fs.readFileSync(process.env.NATIVE_MUX_INPUT!, 'utf8'))
const receipt: Record<string, unknown> = { platform: process.platform, electron: process.versions.electron, processType: process.type, checks: {}, passed: false }
const checks = receipt.checks as Record<string, unknown>
const watchdog = setTimeout(() => { console.error('Native probe deadline'); app.exit(2) }, 150_000)
app.setPath('userData', input.userData)
app.disableHardwareAcceleration()

const backend = (profile: string) => ({ command: input.python, args: ['-m', 'hermes_cli.main', '--profile', profile, 'gateway', 'ensure', '--json', '--timeout', '30'], env: { PYTHONPATH: input.repo, HOME: input.home, USERPROFILE: input.home }, shell: false })

async function attach(connection: any) {
  const ticket = await mintLocalGatewayTicket(connection.gatewayEndpoint)

  return new Promise<{ ws: WebSocket; call: (method: string, params: any) => Promise<any> }>((resolve, reject) => {
    const ws = new WebSocket(connection.baseUrl.replace(/^http/, 'ws') + '/api/ws', ['hermes-gateway-v1', 'hermes-gateway-ticket.' + ticket], { handshakeTimeout: 10_000 })
    let next = 1
    const pending = new Map<number, (frame: any) => void>()
    ws.on('unexpected-response', (_r, response) => reject(new Error(`WS ${response.statusCode} for ${connection.profile}`)))
    ws.on('error', reject)
    ws.on('message', raw => { const frame = JSON.parse(raw.toString()); if (frame.id && pending.has(frame.id)) { pending.get(frame.id)!(frame); pending.delete(frame.id) } })
    ws.on('open', () => resolve({ ws, call: (method, params) => new Promise(done => { const id = next++; pending.set(id, done); ws.send(JSON.stringify({ jsonrpc: '2.0', id, method, params })) }) }))
  })
}

async function rest(connection: any, method: string, path: string, body?: unknown) {
  const url = connection.baseUrl + path
  const headers: Record<string, string> = { ...(await nativeGatewayHttpHeaders({ gatewayEndpoint: connection.gatewayEndpoint, baseUrl: connection.baseUrl }, url)) }
  if (body !== undefined) headers['Content-Type'] = 'application/json'
  const response = await fetch(url, { method, headers, body: body === undefined ? undefined : JSON.stringify(body), signal: AbortSignal.timeout(20_000) })
  const text = await response.text()
  return { status: response.status, json: text ? JSON.parse(text) : null }
}

/** Byte-for-byte `mutateSessionHttp(id, 'PATCH', payload, profile)`. */
async function sidebarPatch(connection: any, profile: string, sid: string, payload: Record<string, unknown>, requestId: string) {
  const query = `?profile=${encodeURIComponent(profile)}`
  const snapshot = await rest(connection, 'GET', `/api/sessions/${encodeURIComponent(sid)}/mutation-snapshot${query}`)
  assert.equal(snapshot.status, 200, JSON.stringify(snapshot.json))
  const identity = { request_id: requestId, expected_revision: snapshot.json.runtime_revision, expected_generation: snapshot.json.runtime_generation }
  return rest(connection, 'PATCH', `/api/sessions/${encodeURIComponent(sid)}${query}`, { ...payload, ...identity, profile })
}

async function main() {
  assert.ok(process.versions.electron, 'Must execute inside real Electron, not Node')
  assert.equal(process.type, 'browser', 'Must execute in Electron main')
  configureWindowsGatewayTicketClient((endpoint, purpose) => mintGatewayTicketWithPython({ command: input.python, env: { PYTHONPATH: input.repo } }, input.repo, endpoint, purpose))

  // 1. Desktop-side discovery for every profile: same instance, secondaries via the root's control socket.
  const connections: Record<string, any> = {}
  for (const profile of ['default', 'warm', 'cold']) {
    connections[profile] = { ...(await ensureLocalGateway(() => runGatewayEnsure(backend(profile), input.repo, input.state))), profile }
  }
  const instances = new Set(Object.values(connections).map(c => c.gatewayEndpoint.instance_id))
  assert.equal(instances.size, 1, 'every profile attaches to the one multiplexer')
  assert.equal(connections.cold.gatewayEndpoint.control_home, input.state)
  assert.equal(connections.default.gatewayEndpoint.control_home, null)
  checks.ensure = Object.fromEntries(Object.entries(connections).map(([p, c]) => [p, { profile_id: c.gatewayEndpoint.profile_id, control_home: c.gatewayEndpoint.control_home, api_origin: c.baseUrl }]))

  // 2. Cold profile: create a GUI session over an interactive attach, then detach (it goes cold).
  const coldAttach = await attach(connections.cold)
  const created = await coldAttach.call('session.create', { request_id: 'cold-create', source: 'gui' })
  assert.ok(created.result?.session_id, JSON.stringify(created))
  const coldSid: string = created.result.session_id
  coldAttach.ws.terminate()
  await new Promise(resolve => setTimeout(resolve, 300))

  // 3. Warm stays attached while default + warm are "active"; archive/unarchive cold over the Desktop REST shape.
  const warmAttach = await attach(connections.warm)
  const warmDescribe = await warmAttach.call('runtime.describe', {})
  assert.equal(warmDescribe.result.profile_id, connections.warm.gatewayEndpoint.profile_id)
  const defaultAttach = await attach(connections.default)
  try {
    const archived = await sidebarPatch(connections.cold, 'cold', coldSid, { archived: true }, 'archive')
    assert.equal(archived.status, 200, JSON.stringify(archived.json))
    assert.equal(archived.json.archived, true)
    const afterArchive = await rest(connections.cold, 'GET', `/api/sessions/${encodeURIComponent(coldSid)}?profile=cold`)
    assert.equal(afterArchive.json.archived, 1)
    assert.equal(afterArchive.json.profile, 'cold')
    const restored = await sidebarPatch(connections.cold, 'cold', coldSid, { archived: false }, 'unarchive')
    assert.equal(restored.status, 200, JSON.stringify(restored.json))
    assert.equal(restored.json.archived, false)
    assert.equal(restored.json.revision, archived.json.revision + 1)
    const afterRestore = await rest(connections.cold, 'GET', `/api/sessions/${encodeURIComponent(coldSid)}?profile=cold`)
    assert.equal(afterRestore.json.archived, 0)
    checks.coldArchive = { session_id: coldSid, archive: archived.json, unarchive: restored.json }

    // 4. Boundary: the default profile's descriptor cannot read the cold profile's rows.
    const foreign = await rest(connections.default, 'GET', `/api/sessions/${encodeURIComponent(coldSid)}/mutation-snapshot?profile=cold`)
    assert.equal(foreign.status, 403)
    checks.foreignProfileSnapshot = foreign.status
    // 5. The attached siblings are still live after the cold mutation.
    const stillWarm = await warmAttach.call('runtime.describe', {})
    assert.equal(stillWarm.result.instance_id, connections.warm.gatewayEndpoint.instance_id)
    checks.warmStillAttached = true
  } finally {
    warmAttach.ws.terminate()
    defaultAttach.ws.terminate()
  }
  receipt.passed = true
}

app.whenReady().then(main).catch(error => { receipt.error = String(error?.stack || error) }).finally(() => {
  clearTimeout(watchdog)
  fs.writeFileSync(input.receipt, JSON.stringify(receipt, null, 2) + '\n')
  console.log(JSON.stringify(receipt))
  app.exit(receipt.passed ? 0 : 1)
})
