import { app } from 'electron'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import WebSocket from 'ws'
import { configureWindowsGatewayTicketClient, mintLocalGatewayTicket, nativeGatewayHttpHeaders } from '../../apps/desktop/electron/local-gateway'
import { mintGatewayTicketWithPython } from '../../apps/desktop/electron/local-gateway-python'

const input = JSON.parse(fs.readFileSync(process.env.NATIVE_TICKET_INPUT!, 'utf8'))
const receipt: Record<string, unknown> = {
  platform: process.platform, electron: process.versions.electron, processType: process.type,
  checks: {}, passed: false,
  wrongOsOwnerSid: { status: 'not_exercised', reason: 'Requires a real second Windows account/token and permissive decoy pipe; same-user identity mutation is not SID evidence.' }
}
const checks = receipt.checks as Record<string, unknown>
const watchdog = setTimeout(() => { console.error('Native probe deadline'); app.exit(2) }, 90_000)
app.setPath('userData', input.userData)
app.disableHardwareAcceleration()

async function describe(ticket: string, replay = false) {
  return new Promise<any>((resolve, reject) => {
    const ws = new WebSocket(input.endpoint.api_origin.replace(/^http/, 'ws') + '/api/ws',
      ['hermes-gateway-v1', 'hermes-gateway-ticket.' + ticket], { handshakeTimeout: 10_000 })
    const timer = setTimeout(() => { ws.terminate(); reject(new Error('WS response deadline')) }, 15_000)
    const done = (error: Error | null, result?: unknown) => {
      clearTimeout(timer)
      ws.terminate()
      if (error) reject(error); else resolve(result)
    }
    ws.on('error', error => { if (!replay) done(error) })
    ws.on('unexpected-response', (_request, response) => {
      response.resume()
      if (replay && [401, 403].includes(response.statusCode!)) done(null, response.statusCode)
      else done(new Error(`Unexpected WS HTTP status ${response.statusCode}`))
    })
    ws.on('open', () => {
      if (replay) { done(new Error('Replayed WS ticket accepted')); return }
      assert.equal(ws.protocol, 'hermes-gateway-v1')
      ws.send(JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'runtime.describe', params: {} }))
    })
    ws.on('message', raw => {
      const frame = JSON.parse(raw.toString())
      if (frame.id === 1) {
        if (frame.error) done(new Error('runtime.describe returned an error'))
        else done(null, frame.result)
      }
    })
  })
}

async function main() {
  assert.ok(process.versions.electron, 'Must execute inside real Electron, not Node')
  assert.equal(process.type, 'browser', 'Must execute in Electron main')
  configureWindowsGatewayTicketClient((endpoint, purpose) =>
    mintGatewayTicketWithPython({ command: input.python, env: { PYTHONPATH: input.repo } }, input.repo, endpoint, purpose))
  const descriptor = { gatewayEndpoint: input.endpoint, baseUrl: input.endpoint.api_origin }
  const configUrl = descriptor.baseUrl + '/api/config'
  const headers = await nativeGatewayHttpHeaders(descriptor, configUrl)
  const response = await fetch(configUrl, { headers, signal: AbortSignal.timeout(15_000) })
  assert.equal(response.status, 200)
  const config = await response.json()
  assert.ok(config && typeof config === 'object')
  checks.httpConfig = response.status
  const replay = await fetch(configUrl, { headers, signal: AbortSignal.timeout(15_000) })
  assert.ok([401, 403].includes(replay.status), `HTTP replay status ${replay.status}`)
  await replay.arrayBuffer()
  checks.httpReplay = replay.status
  const ticket = await mintLocalGatewayTicket(input.endpoint)
  const runtime = await describe(ticket)
  assert.equal(runtime.instance_id, input.endpoint.instance_id)
  assert.equal(runtime.profile_id, input.endpoint.profile_id)
  assert.equal(runtime.authority_epoch, input.endpoint.authority_epoch)
  checks.runtimeDescribe = runtime
  checks.wsReplay = await describe(ticket, true)
  for (const purpose of ['interactive', 'native-http'] as const) {
    await assert.rejects(mintLocalGatewayTicket({ ...input.endpoint, instance_id: 'wrong-instance' }, purpose))
    // The sibling is a live owner, not a missing socket. Combining its profile
    // with the first owner's instance must not mint a usable credential.
    await assert.rejects(mintLocalGatewayTicket({ ...input.endpoint, profile_id: input.sibling.profile_id }, purpose))
  }
  checks.wrongInstance = 'denied for both purposes'
  checks.wrongProfile = 'live sibling profile / original instance denied for both purposes'
  const scopedUrl = configUrl + '?profile=sibling'
  const scoped = await fetch(scopedUrl, { headers: await nativeGatewayHttpHeaders(descriptor, scopedUrl), signal: AbortSignal.timeout(15_000) })
  assert.equal(scoped.status, 403)
  await scoped.arrayBuffer()
  checks.httpSiblingProfile = scoped.status
  // Positive control after the negatives proves the control listener survived.
  const after = await fetch(configUrl, { headers: await nativeGatewayHttpHeaders(descriptor, configUrl), signal: AbortSignal.timeout(15_000) })
  assert.equal(after.status, 200)
  await after.arrayBuffer()
  checks.httpAfterNegatives = after.status
  receipt.passed = true
}

app.whenReady().then(main).catch(error => {
  receipt.error = String(error?.stack || error)
}).finally(() => {
  clearTimeout(watchdog)
  fs.writeFileSync(input.receipt, JSON.stringify(receipt, null, 2) + '\n')
  console.log(JSON.stringify(receipt))
  app.exit(receipt.passed ? 0 : 1)
})
