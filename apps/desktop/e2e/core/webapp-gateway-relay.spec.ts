/**
 * The gateway relay, end to end with two real gateways and a real browser.
 *
 * Two sandboxes, each a gateway serving its own Webapp behind the sign-in gate (the bundled password
 * provider, the same account on both), with its own scripted model. Each binds loopback and declares
 * a `public_url` on its own 127.0.0.0/8 alias (the hub 127.0.0.2, Mini 127.0.0.3), which engages
 * the gate; a TCP forwarder serves that alias, so the browser and the hub reach each gateway at the
 * address it names, as on two machines. The hub pairs with Mini from Settings → Gateways: Mini's
 * consent page opens, Allow returns to the hub, and Mini joins the hub's gateway switcher. Proof:
 *  - the browser reaches Mini only through the hub: every socket and request the tab opens is on the
 *    hub's origin, under `/api/relay/<id>`;
 *  - a turn sent on Mini is answered by Mini's model, its terminal tool runs under Mini's gateway
 *    process, and the session is stored in Mini's home, not the hub's;
 *  - Mini's host terminal and files open through the relay;
 *  - revoking the grant on Mini closes the relayed socket and the hub reports a reconnect.
 *
 * Linux (the /proc ancestry walk, 127.0.0.0/8 aliases); needs the Webapp renderer
 * (`npm run build:webapp`) and a Chromium Playwright can launch (or HERMES_E2E_CHROMIUM).
 * HERMES_E2E_RELAY_VIDEO=<dir> records the tab.
 */

import { spawnSync } from 'node:child_process'
import * as fs from 'node:fs'
import { connect, createServer, type Server } from 'node:net'
import * as path from 'node:path'

import { type Browser, chromium, expect, type Page, test } from '@playwright/test'

import {
  backendProcesses,
  composer,
  coreAppEnv,
  type CoreSandbox,
  createCoreSandbox,
  DESKTOP_ROOT,
  recordWebSockets,
  REPO_ROOT,
  sandboxProcesses,
  send,
  storedSessionForMarker,
  writeProviderHome
} from './harness'
import { type ScriptedProvider, startScriptedProvider } from './provider'

const nonce = Math.random()
  .toString(36)
  .slice(2, 8)
  .replace(/[^a-z0-9]/g, 'x')
  .padEnd(4, 'q')

const U = (n: number) => `U${n}-relay${nonce}`
const A = (n: number) => `A${n}-relay${nonce}`
const AI = (n: number) => `A${n}i-relay${nonce}`

const PYTHON = process.env.HERMES_E2E_PYTHON || 'python3'
const VIDEO_DIR = process.env.HERMES_E2E_RELAY_VIDEO || ''

function freePort(host: string): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = createServer()
    server.once('error', reject)
    server.listen(0, host, () => {
      const address = server.address()
      server.close(() => resolve(typeof address === 'object' && address ? address.port : 0))
    })
  })
}

function run(sandbox: CoreSandbox, args: string[]): { code: number | null; output: string } {
  const result = spawnSync(PYTHON, args, {
    cwd: REPO_ROOT,
    env: coreAppEnv(sandbox),
    encoding: 'utf8',
    timeout: 180_000
  })

  return { code: result.status, output: `${result.stdout ?? ''}${result.stderr ?? ''}` }
}

function python(sandbox: CoreSandbox, code: string, ...args: string[]): string {
  const result = run(sandbox, ['-c', code, ...args])
  expect(result.code, result.output).toBe(0)

  return result.output.trim()
}

/** The pids from the tool's own shell up to init, as the tool wrote them. */
function toolAncestry(file: string): number[] {
  return fs.existsSync(file) ? fs.readFileSync(file, 'utf8').split(/\s+/).filter(Boolean).map(Number) : []
}

interface Gateway {
  forwarder: Server
  host: string
  model: ScriptedProvider
  origin: string
  sandbox: CoreSandbox
}

/** Serve *host*:*port* by forwarding every connection to 127.0.0.1:*port*. */
function forward(host: string, port: number): Promise<Server> {
  const server = createServer(client => {
    const upstream = connect(port, '127.0.0.1')
    client.pipe(upstream).pipe(client)
    client.on('error', () => upstream.destroy())
    upstream.on('error', () => client.destroy())
  })

  return new Promise((resolve, reject) => {
    server.once('error', reject)
    server.listen(port, host, () => resolve(server))
  })
}

/** A sandbox gateway serving its Webapp at http://*host* behind sign-in, with the relay on. */
async function startGateway(label: string, host: string, password: string): Promise<Gateway> {
  const model = await startScriptedProvider()
  const sandbox = createCoreSandbox(label)
  writeProviderHome(sandbox.hermesHome, model.url)
  const port = await freePort('127.0.0.1')
  const origin = `http://${host}:${port}`

  const hash = python(
    sandbox,
    'import sys; from plugins.dashboard_auth.basic import hash_password; print(hash_password(sys.argv[1]))',
    password
  )

  const secret = python(sandbox, 'import secrets; print(secrets.token_hex(32))')
  fs.appendFileSync(
    path.join(sandbox.hermesHome, 'config.yaml'),
    `\nwebapp:\n  enabled: true\n  port: ${port}\n  public_url: ${origin}\n  relay:\n    enabled: true\n` +
      `dashboard:\n  basic_auth:\n    username: owner\n    password_hash: '${hash}'\n    secret: '${secret}'\n`
  )
  const started = run(sandbox, ['-m', 'hermes_cli.main', 'webapp', '--no-open', '--skip-build'])
  expect(started.code, started.output).toBe(0)

  return { forwarder: await forward(host, port), host, model, origin, sandbox }
}

async function signIn(page: Page, gateway: Gateway, password: string): Promise<void> {
  const response = await page.request.post(`${gateway.origin}/auth/password-login`, {
    headers: { Origin: gateway.origin },
    data: { provider: 'basic', username: 'owner', password }
  })

  expect(response.status(), await response.text()).toBe(200)
}

/** Stand-in for a reader's pause in the recording, so each step is visible. */
const beat = (page: Page) => (VIDEO_DIR ? page.waitForTimeout(1600) : Promise.resolve())

test('a Webapp reaches another gateway through its own', async () => {
  test.skip(process.platform !== 'linux', 'walks /proc to prove which gateway ran the tool')
  test.setTimeout(420_000)
  expect(
    fs.existsSync(path.join(DESKTOP_ROOT, 'dist-webapp', 'index.html')),
    'the Webapp renderer is built (npm run build:webapp)'
  ).toBe(true)

  const password = `throwaway-${Math.random().toString(36).slice(2, 14)}`
  const gateways: Gateway[] = []
  let browser: Browser | null = null

  try {
    const hub = await startGateway('relay-hub', '127.0.0.2', password)
    gateways.push(hub)
    const mini = await startGateway('relay-mini', '127.0.0.3', password)
    gateways.push(mini)
    const ancestry = path.join(mini.sandbox.root, 'tool-ancestry.txt')

    browser = await chromium.launch({ executablePath: process.env.HERMES_E2E_CHROMIUM || undefined })

    const context = await browser.newContext({
      viewport: { width: 1280, height: 820 },
      ...(VIDEO_DIR ? { recordVideo: { dir: VIDEO_DIR, size: { width: 1280, height: 820 } } } : {})
    })

    const page = await context.newPage()
    const ws = recordWebSockets(page)
    const requests: string[] = []
    page.on('request', request => requests.push(request.url()))
    // Every socket the tab opens (the recorder keeps only gateway sockets).
    const socketUrls: string[] = []
    page.on('websocket', socket => socketUrls.push(socket.url()))

    // Signed in to both as the same account, as a user is before pairing.
    await signIn(page, hub, password)
    await signIn(page, mini, password)
    await page.goto(`${hub.origin}/`)
    await expect(composer(page)).toBeVisible({ timeout: 180_000 })
    await beat(page)

    // Pair from Settings → Gateways: Mini's consent page, Allow, back on the hub.
    await page.goto(`${hub.origin}/#/settings?tab=gateway`)
    const connect = page.locator('[data-slot="relay-connect"]')
    await expect(connect).toBeVisible({ timeout: 60_000 })
    await beat(page)
    await connect.locator('input[name="relay-url"]').fill(mini.origin)
    await connect.locator('input[name="relay-label"]').fill('Mini')
    await beat(page)
    await connect.getByRole('button', { name: 'Connect' }).click()
    await expect(page).toHaveURL(new RegExp(`^${mini.origin.replace(/\./g, '\\.')}/webapp/relay/authorize`))
    await expect(page.getByRole('heading', { name: /Allow 127\.0\.0\.2 to use this gateway\?/ })).toBeVisible()
    await beat(page)
    await page.getByRole('button', { name: 'Allow' }).click()
    await expect(page).toHaveURL(new RegExp(`^${hub.origin.replace(/\./g, '\\.')}/`), { timeout: 60_000 })
    await expect(page.getByText('Connected Mini').first()).toBeVisible({ timeout: 60_000 })
    await expect(page.getByText(mini.origin).first()).toBeVisible()
    await beat(page)

    // Mini joins the switcher; switching opens its socket through the hub.
    await page.goto(`${hub.origin}/`)
    await expect(composer(page)).toBeVisible({ timeout: 120_000 })
    const switcher = page.locator('[data-slot="connection-switcher"]')
    await expect(switcher).toBeVisible({ timeout: 60_000 })
    await switcher.getByRole('button').first().click()
    await beat(page)
    await page.getByRole('menuitemradio', { name: /Mini/ }).click()
    await expect
      .poll(
        () =>
          ws.sockets.some(
            socket => socket.url.startsWith(`ws://${hub.host}:`) && socket.url.includes('/api/relay/mini/api/ws')
          ),
        {
          timeout: 60_000,
          message: "the tab opened Mini's gateway socket through the hub"
        }
      )
      .toBe(true)
    // The switch has committed: the switcher is idle and names Mini.
    const switchStarted = Date.now()
    await expect(switcher).toHaveAttribute('aria-busy', 'false', { timeout: 60_000 })
    await expect(switcher.getByRole('button').first()).toContainText('Mini')
    console.log(`switch committed ${Date.now() - switchStarted} ms after Mini's socket opened`)
    await beat(page)

    // A turn on Mini: Mini's model answers, Mini's gateway runs the tool, Mini stores the session.
    mini.model.script(U(1), [
      {
        text: [`${AI(1)} `, 'checking'],
        toolCalls: [
          {
            name: 'terminal',
            args: {
              command:
                `sh -c 'p=$$; while [ "$p" -gt 1 ]; do echo $p; ` +
                `p=$(sed -n "s/^PPid:[[:space:]]*//p" /proc/$p/status); done' > ${ancestry}`
            }
          }
        ]
      },
      { text: [`${A(1)} `, 'ran ', 'on ', 'Mini'] }
    ])
    await send(page, `${U(1)} where do you run`, 'Enter', ws)

    try {
      await expect(page.getByText(`${A(1)} ran on Mini`).first()).toBeVisible({ timeout: 150_000 })
    } catch (error) {
      const strip = (url: string) => url.replace(/(ticket|relay_ticket)=[^&]+/g, '$1=…')
      console.log('sockets', JSON.stringify(ws.sockets.map(socket => ({ ...socket, url: strip(socket.url) }))))
      console.log('sent', JSON.stringify(ws.sent.map(frame => [frame.socket, frame.method])))
      console.log(
        'hub completions',
        JSON.stringify(hub.model.completions.map(c => [c.marker, c.step, c.sentText.slice(0, 40)]))
      )
      console.log(
        'mini completions',
        JSON.stringify(mini.model.completions.map(c => [c.marker, c.step, c.sentText.slice(0, 40)]))
      )
      throw error
    }

    await beat(page)
    expect(hub.model.completions.some(completion => completion.marker === U(1))).toBe(false)
    const [miniGateway] = backendProcesses(mini.sandbox)
    await expect.poll(() => toolAncestry(ancestry)).toContain(miniGateway.pid)
    await expect.poll(() => storedSessionForMarker(mini.sandbox, 'default', U(1))).not.toBeNull()
    expect(storedSessionForMarker(hub.sandbox, 'default', U(1))).toBeNull()

    // Mini's host terminal and files, through the relay.
    const terminal = await page.evaluate(async () => {
      const api = (window as any).hermesDesktop.terminal
      const session = await api.start({ cols: 100, restoreKey: 'relay-e2e', rows: 24 })
      let output = ''
      const off = api.onData(session.id, (chunk: string) => void (output += chunk))
      await api.write(session.id, 'echo "home=$HERMES_HOME"\r')
      const deadline = Date.now() + 20_000

      while (!/home=\//.test(output) && Date.now() < deadline) {
        await new Promise(resolve => setTimeout(resolve, 200))
      }

      off()
      await api.dispose(session.id)

      return output
    })

    expect(terminal).toContain(`home=${mini.sandbox.hermesHome}`)
    expect(
      socketUrls.some(url => url.startsWith(`ws://${hub.host}:`) && url.includes('/api/relay/mini/api/host-terminal'))
    ).toBe(true)
    const listing = await page.evaluate(home => (window as any).hermesDesktop.readDir(home), mini.sandbox.hermesHome)
    expect(JSON.stringify(listing)).toContain(`${mini.sandbox.hermesHome}/SOUL.md`)

    // Nothing in the tab went to Mini's origin after the consent page.
    const consentAt = requests.findIndex(url => url.includes('/webapp/relay/authorize'))
    const afterPairing = requests.slice(consentAt + 2)
    expect(afterPairing.filter(url => url.startsWith(mini.origin))).toEqual([])
    expect(socketUrls.filter(url => url.startsWith(`ws://${mini.host}:`))).toEqual([])

    // Revoking the grant on Mini ends the relay: the relayed socket closes, the hub says reconnect.
    const relayed = ws.sockets.filter(socket => socket.url.includes('/api/relay/mini/') && !socket.closed)
    expect(relayed.length).toBeGreaterThan(0)
    const mini2 = await context.newPage()
    await mini2.goto(`${mini.origin}/#/settings?tab=gateway`)
    await expect(mini2.getByText('Gateways that reach this one')).toBeVisible({ timeout: 120_000 })
    await mini2.getByRole('button', { name: 'Revoke' }).click()
    await mini2.getByRole('dialog').getByRole('button', { name: 'Revoke' }).click()
    await expect(mini2.getByText('No other gateway reaches this one.')).toBeVisible({ timeout: 30_000 })
    await beat(mini2)
    await mini2.close()
    await expect
      .poll(() => relayed.every(socket => socket.closed), {
        timeout: 30_000,
        message: 'revocation closed the relayed sockets'
      })
      .toBe(true)
    const after = await page.request.get(`${hub.origin}/api/relay/mini/api/profiles`)
    expect(after.status()).toBe(409)
    expect((await after.json()).error).toBe('relay_reconnect')
    await beat(page)
    await context.close()
  } finally {
    await browser?.close().catch(() => undefined)

    for (const gateway of gateways) {
      for (const proc of sandboxProcesses(gateway.sandbox)) {
        try {
          process.kill(proc.pid, 'SIGKILL')
        } catch {
          /* already gone */
        }
      }

      gateway.forwarder.close()
      await gateway.model.close()
      gateway.sandbox.cleanup()
    }
  }
})
