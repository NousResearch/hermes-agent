// Actual package entry, preload IPC and renderer WS; no production patching.
import fs from 'node:fs'
import path from 'node:path'
import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
const input = JSON.parse(fs.readFileSync(process.env.NATIVE_TICKET_INPUT, 'utf8'))
const require = createRequire(path.join(input.repo, 'apps/desktop/package.json'))
const { _electron } = require('@playwright/test')
const receipt = { passed: false, platform: process.platform, surface: 'built package main + production preload + Chromium WebSocket' }
let app
try {
  app = await _electron.launch({
    executablePath: require('electron'), args: [path.join(input.repo, 'apps/desktop')],
    cwd: input.repo, timeout: 90000,
    env: { ...process.env,
      HERMES_DESKTOP_USER_DATA_DIR: input.userData + '-full',
      HERMES_DESKTOP_IGNORE_EXISTING: '1', HERMES_DESKTOP_HERMES_ROOT: input.repo,
      HERMES_DESKTOP_APP_NAME: 'HermesNativeTicketProbe', HERMES_DESKTOP_SKIP_QUIT_CONFIRM: '1' }
  })
  const page = await app.firstWindow({ timeout: 90000 })
  await page.waitForFunction(() => !!window.hermesDesktop, { timeout: 60000 })
  receipt.native = await app.evaluate(() => ({ electron: process.versions.electron, type: process.type }))
  receipt.checks = await page.evaluate(async expected => {
    const bridge = window.hermesDesktop
    const descriptor = await bridge.getConnection()
    if (descriptor.gatewayEndpoint?.instance_id !== expected.instance_id) throw new Error('Startup did not discover owned daemon')
    const config = await bridge.api({ path: '/api/config', method: 'GET' })
    if (!config || typeof config !== 'object') throw new Error('Native HTTP IPC returned no configuration')
    const fresh = await bridge.getGatewayWsUrl()
    if (!fresh.ok) throw new Error(fresh.error)
    const url = new URL(fresh.wsUrl)
    const ticket = url.searchParams.get('ticket')
    if (!ticket) throw new Error('No private WS grant')
    url.searchParams.delete('ticket')
    async function dial(replay) {
      return new Promise((resolve, reject) => {
        const ws = new WebSocket(url.toString(), ['hermes-gateway-v1', 'hermes-gateway-ticket.' + ticket])
        const timer = setTimeout(() => { ws.close(); reject(new Error('Renderer WS deadline')) }, 15000)
        const done = (error, value) => { clearTimeout(timer); ws.close(); error ? reject(error) : resolve(value) }
        ws.onopen = () => {
          if (replay) return done(new Error('Renderer replay accepted'))
          ws.send(JSON.stringify({jsonrpc:'2.0', id:991, method:'runtime.describe', params:{}}))
        }
        ws.onerror = () => replay ? done(null, 'rejected') : done(new Error('Renderer WS handshake failed'))
        ws.onmessage = event => {
          const frame = JSON.parse(event.data)
          if (frame.id === 991) frame.error ? done(new Error('Describe failed')) : done(null, frame.result)
        }
      })
    }
    const runtime = await dial(false)
    if (runtime.instance_id !== expected.instance_id || runtime.profile_id !== expected.profile_id || runtime.authority_epoch !== expected.authority_epoch) throw new Error('Renderer authority identity mismatch')
    const replay = await dial(true)
    const after = await bridge.api({ path: '/api/config', method: 'GET' })
    if (!after || typeof after !== 'object') throw new Error('HTTP IPC failed after replay')
    return { discoveredInstance: runtime.instance_id, profile: runtime.profile_id, epoch: runtime.authority_epoch, configIpc: true, rendererDescribe: true, rendererReplay: replay, configAfterReplay: true }
  }, input.endpoint)
  receipt.body = (await page.locator('body').innerText()).slice(0, 1000)
  assert.equal(receipt.native.type, 'browser')
  receipt.passed = true
} catch (error) {
  receipt.error = String(error?.stack || error)
} finally {
  if (app) await app.close().catch(() => {})
  fs.writeFileSync(input.fullReceipt, JSON.stringify(receipt, null, 2) + '\n')
}
console.log(JSON.stringify(receipt))
process.exit(receipt.passed ? 0 : 1)
