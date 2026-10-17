/**
 * The status bar's client version label follows the ACTIVE connection.
 *
 * Two loopback fake gateways registered in connections.json (v2): A is the
 * primary (reports 0.21.5), B is a second remote (reports 0.22.1, display
 * 0.22.1+72). The app boots on A, the user switches to B through the profile
 * rail, and the label must re-resolve to B's version instead of staying on A's.
 *
 * Version source: the dev build stamp has payload 'bootstrap' and no
 * displayVersion, so electron/app-version.ts lets the RUNTIME gateway version
 * (GET /api/health → displayVersion ?? version) win for appVersion.
 */

import * as fs from 'node:fs'
import * as http from 'node:http'
import { createRequire } from 'node:module'
import type { AddressInfo } from 'node:net'
import * as path from 'node:path'

import type { Page } from '@playwright/test'

import { buildAppEnv, createSandbox, launchDesktop } from './fixtures'
import { allowErrorBanners, expect, test } from './test'

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

const outPath = (file: string): string =>
  process.env.HERMES_E2E_OUT ? path.join(process.env.HERMES_E2E_OUT, file) : test.info().outputPath(file)

interface FakeGateway {
  url: string
  hits: string[]
  close: () => Promise<void>
}

async function startFakeGateway(name: string, health: Record<string, unknown>): Promise<FakeGateway> {
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

const clientLabel = (page: Page) =>
  page.evaluate(() => {
    const text = document.body.innerText
    const match = text.match(/client v(\S+)/)

    return match ? match[1] : null
  })

test('client version label follows a switch to another connection', async () => {
  test.setTimeout(240_000)
  allowErrorBanners()
  const a = await startFakeGateway('A', { ok: true, status: 'ok', version: '0.21.5' })
  const b = await startFakeGateway('B', { ok: true, status: 'ok', version: '0.22.1', displayVersion: '0.22.1+72' })
  const sandbox = createSandbox('version-label')
  fs.writeFileSync(
    path.join(sandbox.userDataDir, 'connections.json'),
    JSON.stringify({
      version: 2,
      primary: 'gw-a',
      launchMode: 'primary',
      lastUsed: 'gw-a',
      connections: [
        { id: 'local', kind: 'local', label: 'This device' },
        {
          id: 'gw-a',
          kind: 'remote',
          label: 'Gateway A',
          url: a.url,
          authMode: 'token',
          token: { encoding: 'plain', value: 'tok-a' }
        },
        {
          id: 'gw-b',
          kind: 'remote',
          label: 'Gateway B',
          url: b.url,
          authMode: 'token',
          token: { encoding: 'plain', value: 'tok-b' }
        }
      ]
    }),
    { encoding: 'utf8', mode: 0o600 }
  )

  const { app, page } = await launchDesktop(buildAppEnv(sandbox, { HERMES_DESKTOP_DEV_SERVER: '' }))

  const dump = (tag: string) =>
    fs.writeFileSync(outPath(`${tag}.log`), [`A hits:`, ...a.hits, '', 'B hits:', ...b.hits].join('\n'))

  try {
    await expect.poll(() => clientLabel(page), { timeout: 90_000 }).toBe('0.21.5')
    await page.screenshot({ path: outPath('before-switch.png') })
    const before = await page.evaluate(() => (window as any).hermesDesktop.getVersion())
    expect(before.appVersion, 'unscoped getVersion resolves the primary (A)').toBe('0.21.5')

    // Real user switch: click Gateway B's default profile square on the rail.
    const bHitsBefore = b.hits.filter(h => h.startsWith('GET /api/health')).length
    await page.locator('[data-slot="profile-rail"]').getByRole('button', { name: 'default · Gateway B' }).click()

    // The window re-homed onto B: B's default becomes the rail's active entry
    // and A drops to an at-rest "default · Gateway A" door.
    await expect(
      page.locator('[data-slot="profile-rail"]').getByRole('button', { name: 'default · Gateway A' }),
      'the window re-homed onto Gateway B'
    ).toBeVisible({ timeout: 60_000 })

    let label: null | string = null
    let scopedVersion: null | string = null

    try {
      await expect
        .poll(async () => (label = await clientLabel(page)), {
          message: 'status-bar client label follows the active connection',
          timeout: 20_000
        })
        .toBe('0.22.1+72')
    } finally {
      await page.screenshot({ path: outPath('after-switch.png') })
      // The preload half of the fix: a scoped getVersion must reach main.
      scopedVersion = (await page.evaluate(() => (window as any).hermesDesktop.getVersion({ connectionId: 'gw-b' })))
        .appVersion
      fs.writeFileSync(
        outPath('after-switch.txt'),
        `label after switch: ${label}\ngetVersion({connectionId:'gw-b'}).appVersion: ${scopedVersion}\nB /api/health hits before switch: ${bHitsBefore}, after: ${b.hits.filter(h => h.startsWith('GET /api/health')).length}\n`
      )
    }

    expect(scopedVersion, 'preload forwards the scope, so main resolves B for gw-b').toBe('0.22.1+72')
  } finally {
    dump('final')
    await app.close().catch(() => undefined)
    await a.close()
    await b.close()
    sandbox.cleanup()
  }
})
