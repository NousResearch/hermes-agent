/**
 * A saved gateway the user removed stays removed after a relaunch, even with the
 * old userData connection.json still naming it (#142906, #150084), and removing a
 * cookie-auth gateway clears its cookie jar (#98242).
 *
 * Real Electron, real IPC: launch 1 imports the old userData files into the shared
 * gateway-connections.json, removes two gateways through `connections.remove` (the
 * call the Saved connections panel makes), and quits. The stale connection.json is
 * put back the way an older build or a Webapp removal leaves it. Launch 2 must list
 * neither gateway, keep the chosen primary, and show it on Settings → Gateways.
 */

import * as fs from 'node:fs'
import * as http from 'node:http'
import { createRequire } from 'node:module'
import type { AddressInfo } from 'node:net'
import * as path from 'node:path'

import { buildAppEnv, createSandbox, launchDesktop } from './fixtures'
import { allowErrorBanners, expect, test } from './test'

const outPath = (file: string): string =>
  process.env.HERMES_E2E_OUT ? path.join(process.env.HERMES_E2E_OUT, file) : test.info().outputPath(file)

interface FakeSocket {
  send: (data: string) => void
  on: (event: 'message', listener: (raw: unknown) => void) => void
  terminate: () => void
}

interface FakeSocketServer {
  clients: Set<FakeSocket>
  on: (event: 'connection', listener: (ws: FakeSocket) => void) => void
  close: () => void
}

// `ws` is a workspace dependency without bundled types; keep the surface we use typed here.
const { WebSocketServer } = createRequire(import.meta.url)('ws') as {
  WebSocketServer: new (options: { server: http.Server }) => FakeSocketServer
}

interface FakeGateway {
  url: string
  hits: string[]
  close: () => Promise<void>
}

async function startFakeGateway(
  name = 'x',
  health: Record<string, unknown> = { ok: true, status: 'ok', version: '0.22.0' }
): Promise<FakeGateway> {
  const hits: string[] = []

  const server = http.createServer((req, res) => {
    req.resume()
    const pathname = new URL(req.url ?? '/', 'http://x').pathname
    hits.push(`${req.method} ${req.url}`)
    res.setHeader('Content-Type', 'application/json')

    if (pathname === '/api/health') {
      return res.end(JSON.stringify(health))
    }

    if (pathname === '/api/status') {
      return res.end(JSON.stringify({ auth_required: false, version: health.version }))
    }

    if (pathname === '/api/profiles') {
      return res.end(
        JSON.stringify({ profiles: [{ name: 'default', is_default: true, active: true }], active: 'default' })
      )
    }

    if (pathname === '/api/profiles/active') {
      return res.end(JSON.stringify({ name: 'default', profile: 'default' }))
    }

    if (pathname.startsWith('/api/config')) {
      return res.end(JSON.stringify({}))
    }

    if (pathname === '/api/sessions') {
      return res.end(JSON.stringify({ sessions: [], total: 0 }))
    }

    res.statusCode = 404
    res.end(JSON.stringify({ error: 'not_found', gateway: name }))
  })

  const wss = new WebSocketServer({ server })
  wss.on('connection', ws => {
    hits.push('WS open')
    ws.send(
      JSON.stringify({
        jsonrpc: '2.0',
        method: 'event',
        params: { type: 'gateway.ready', payload: { change_events: true, heartbeat: false, replay_epoch: name } }
      })
    )
    ws.on('message', raw => {
      let msg: { id?: unknown; method?: string } = {}

      try {
        msg = JSON.parse(String(raw))
      } catch {
        return
      }

      if (msg.id !== undefined) {
        hits.push(`RPC ${msg.method}`)

        // A configured gateway, so no first-run onboarding overlay covers the rail.
        const result =
          msg.method === 'setup.status'
            ? { provider_configured: true, ready: true }
            : msg.method === 'setup.runtime_check'
              ? { ok: true, provider: 'fake', model: 'fake-model' }
              : msg.method === 'onboarding.state'
                ? { active: false, completed: true }
                : {}

        ws.send(JSON.stringify({ jsonrpc: '2.0', id: msg.id, result }))
      }
    })
  })
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))

  return {
    url: `http://127.0.0.1:${(server.address() as AddressInfo).port}`,
    hits,
    close: async () => {
      for (const client of wss.clients) {
        client.terminate()
      }

      wss.close()
      server.closeAllConnections()
      await new Promise<void>(resolve => server.close(() => resolve()))
    }
  }
}

const OAUTH_JAR = 'persist:hermes-remote-oauth-conn-signed-in'

test('a removed saved gateway stays removed after relaunch with a stale connection.json', async () => {
  test.setTimeout(300_000)
  allowErrorBanners()
  const old = await startFakeGateway()
  const signedIn = await startFakeGateway()
  const keeper = await startFakeGateway('keeper')
  const sandbox = createSandbox('gw-removal')
  const v1File = path.join(sandbox.userDataDir, 'connection.json')
  const staleV1 = JSON.stringify({
    mode: 'remote',
    remote: { url: old.url, authMode: 'token', token: { encoding: 'plain', value: 'old' } }
  })
  fs.writeFileSync(v1File, staleV1, { mode: 0o600 })
  fs.writeFileSync(
    path.join(sandbox.userDataDir, 'connections.json'),
    JSON.stringify({
      version: 2,
      primary: 'old-http',
      launchMode: 'primary',
      lastUsed: 'old-http',
      connections: [
        { id: 'local', kind: 'local', label: 'This device' },
        { id: 'old-http', kind: 'remote', label: 'Old HTTP gateway', url: old.url, authMode: 'token' },
        { id: 'signed-in', kind: 'remote', label: 'Signed-in gateway', url: signedIn.url, authMode: 'oauth' },
        {
          id: 'home-box',
          kind: 'remote',
          label: 'Home box',
          url: keeper.url,
          authMode: 'token',
          token: { encoding: 'plain', value: 'keep' }
        }
      ]
    }),
    { mode: 0o600 }
  )

  const env = buildAppEnv(sandbox, { HERMES_DESKTOP_DEV_SERVER: '' })
  const report: string[] = []
  const store = path.join(sandbox.hermesHome, 'gateway-connections.json')
  const ids = (registry: any) => registry.connections.map((c: any) => c.id).join(',')

  try {
    let { app, page } = await launchDesktop(env)
    const first = await page.evaluate(() => (window as any).hermesDesktop.connections.list())
    report.push(`launch 1 list: ${ids(first)} primary=${first.primary}`)

    await app.evaluate(
      async ({ session }, args) => {
        await session.fromPartition(args.jar).cookies.set({ url: args.url, name: 'hermes_session', value: 'stale' })
      },
      { jar: OAUTH_JAR, url: signedIn.url }
    )

    await page.evaluate(async () => {
      const c = (window as any).hermesDesktop.connections
      await c.setPrimary('home-box')
      await c.remove('old-http')
      await c.remove('signed-in')
    })

    const jar = await app.evaluate(
      async ({ session }, args) =>
        (await session.fromPartition(args.jar).cookies.get({ url: args.url })).map(c => c.name),
      { jar: OAUTH_JAR, url: signedIn.url }
    )

    report.push(`cookies left in the removed gateway's jar: [${jar.join(',')}]`)
    await app.close()

    // An older build (or a Webapp removal) leaves connection.json naming the gateway.
    fs.writeFileSync(v1File, staleV1, { mode: 0o600 })
    ;({ app, page } = await launchDesktop(env))

    try {
      const second = await page.evaluate(() => (window as any).hermesDesktop.connections.list())
      const disk = JSON.parse(fs.readFileSync(store, 'utf8'))
      report.push(`launch 2 list: ${ids(second)} primary=${second.primary}`)
      report.push(`gateway-connections.json: ${ids(disk)} primary=${disk.primary}`)
      const v1Mode = JSON.parse(fs.readFileSync(v1File, 'utf8')).mode
      report.push(`old connection.json mode after launch 2: ${v1Mode}`)

      await page.waitForTimeout(8000)
      await page.evaluate(() => {
        window.location.hash = '#/settings?tab=gateway'
      })
      await expect(page.getByText('Saved connections').first()).toBeVisible({ timeout: 60_000 })
      await page.getByText('Saved connections').first().click()
      await expect(page.getByText('Home box').first()).toBeVisible({ timeout: 30_000 })
      await page.waitForTimeout(1500)
      await page.screenshot({ path: outPath('after-relaunch.png') })

      fs.writeFileSync(outPath('report.txt'), report.join('\n') + '\n')
      expect(ids(second), 'the removed gateways came back').toBe('local,home-box')
      expect(second.primary, 'the removed gateway was made primary again').toBe('home-box')
      expect(jar, "the removed gateway's cookie jar survived").toEqual([])
      expect(v1Mode, 'connection.json still routes boot to the removed gateway').toBe('local')
    } finally {
      await app.close().catch(() => undefined)
    }
  } finally {
    fs.writeFileSync(outPath('report.txt'), report.join('\n') + '\n')
    await old.close()
    await signedIn.close()
    await keeper.close()
    sandbox.cleanup()
  }
})
