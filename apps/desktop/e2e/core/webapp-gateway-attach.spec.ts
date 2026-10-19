/**
 * The browser-hosted Desktop (`hermes webapp`) attaches to the profile's gateway (#134889's
 * browser-dashboard cutover), exactly like local Electron Desktop.
 *
 * Before the cutover the Webapp hosted turns itself (a standalone `serve` backend with no session
 * authority), so it and the gateway both wrote the profile's store. Now the browser's live socket is
 * relayed to the gateway under a delegated grant for the browser's own login, and nothing runs in the
 * Webapp process. Proof, end to end with a real browser:
 *  - exactly one gateway serves the sandbox, started by the Webapp's ensure;
 *  - a turn's terminal tool runs under the GATEWAY process (its ancestry), never the Webapp's;
 *  - the owner admitted the turn for the browser's login, not the host's native grant;
 *  - reload resumes the chat; a gateway killed mid-session is re-ensured and the next turn works;
 *  - stopping the Webapp leaves the gateway it started serving.
 *
 * Linux (the /proc ancestry walk); needs the Webapp renderer (`npm run build:webapp`) and a
 * Chromium Playwright can launch (`npx playwright install chromium`, or HERMES_E2E_CHROMIUM).
 */

import { type ChildProcess, spawn } from 'node:child_process'
import * as fs from 'node:fs'
import { createServer } from 'node:net'
import * as path from 'node:path'
import { DatabaseSync } from 'node:sqlite'

import { type Browser, chromium, expect, type Page, test } from '@playwright/test'

import {
  backendProcesses,
  composer,
  coreAppEnv,
  type CoreSandbox,
  createCoreSandbox,
  currentSessionId,
  DESKTOP_ROOT,
  recordWebSockets,
  REPO_ROOT,
  sandboxProcesses,
  send,
  storedSessionForMarker,
  writeProviderHome
} from './harness'
import { assertTranscriptOracle, installDuplicateSampler } from './oracle'
import { startScriptedProvider } from './provider'

const nonce = Math.random().toString(36).slice(2, 8).replace(/[^a-z0-9]/g, 'x').padEnd(4, 'q')
const U = (n: number) => `U${n}-webapp${nonce}`
const A = (n: number) => `A${n}-webapp${nonce}`
const AI = (n: number) => `A${n}i-webapp${nonce}`

const PYTHON = process.env.HERMES_E2E_PYTHON || 'python3'
/** The browser login a loopback Webapp's private session forwards (``webapp_gateway_relay``). */
const BROWSER_LOGIN = 'auth:v1:["session-token","","legacy-token-owner"]'

function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = createServer()
    server.once('error', reject)
    server.listen(0, '127.0.0.1', () => {
      const address = server.address()
      server.close(() => resolve(typeof address === 'object' && address ? address.port : 0))
    })
  })
}

function alive(pid: number): boolean {
  try {
    process.kill(pid, 0)

    return true
  } catch (error) {
    return (error as NodeJS.ErrnoException).code === 'EPERM'
  }
}

/** `hermes webapp` on loopback, as an operator starts it; resolves with its private launch link. */
async function startWebapp(sandbox: CoreSandbox): Promise<{ child: ChildProcess; link: string; log: () => string }> {
  const port = await freePort()
  let output = ''

  const child = spawn(
    PYTHON,
    ['-m', 'hermes_cli.main', 'webapp', '--host', '127.0.0.1', '--port', String(port), '--no-open', '--skip-build'],
    { cwd: REPO_ROOT, env: coreAppEnv(sandbox), stdio: ['ignore', 'pipe', 'pipe'] }
  )

  child.stdout!.on('data', chunk => (output += String(chunk)))
  child.stderr!.on('data', chunk => (output += String(chunk)))
  const log = () => output.replace(/#hermes-session=\S+/g, '#hermes-session=<redacted>').slice(-4000)

  await expect
    .poll(() => /launch link \(private[^)]*\): (\S+)/.exec(output)?.[1] ?? (child.exitCode === null ? '' : 'exited'), {
      timeout: 120_000,
      message: `the Webapp printed its private launch link\n${log()}`
    })
    .toMatch(/^http/)

  return { child, link: /launch link \(private[^)]*\): (\S+)/.exec(output)![1], log }
}

function admissionPrincipals(sandbox: CoreSandbox, sessionId: string): string[] {
  const db = new DatabaseSync(path.join(sandbox.hermesHome, 'state.db'), { readOnly: true })

  try {
    return (
      db.prepare('SELECT DISTINCT principal_id FROM session_admissions WHERE target_session_id = ?').all(sessionId) as {
        principal_id: string
      }[]
    ).map(row => row.principal_id)
  } finally {
    db.close()
  }
}

/** The pids from the tool's own shell up to init, as the tool wrote them. */
function toolAncestry(file: string): number[] {
  return fs.existsSync(file) ? fs.readFileSync(file, 'utf8').split(/\s+/).filter(Boolean).map(Number) : []
}

test('the Webapp attaches its browser to the profile gateway, which runs every turn', async () => {
  test.skip(process.platform !== 'linux', 'walks /proc to prove which process ran the tool')
  expect(
    fs.existsSync(path.join(DESKTOP_ROOT, 'dist-webapp', 'index.html')),
    'the Webapp renderer is built (npm run build:webapp)'
  ).toBe(true)

  const provider = await startScriptedProvider()
  const sandbox = createCoreSandbox('webapp')
  writeProviderHome(sandbox.hermesHome, provider.url)
  const ancestry = path.join(sandbox.root, 'tool-ancestry.txt')
  let webapp: null | Awaited<ReturnType<typeof startWebapp>> = null
  let browser: Browser | null = null

  try {
    webapp = await startWebapp(sandbox)
    browser = await chromium.launch({ executablePath: process.env.HERMES_E2E_CHROMIUM || undefined })
    const page: Page = await (await browser.newContext({ viewport: { width: 1280, height: 860 } })).newPage()
    const ws = recordWebSockets(page)
    await page.goto(webapp.link)
    await expect(composer(page)).toBeVisible({ timeout: 180_000 })
    await installDuplicateSampler(page)

    // The browser's socket is the gateway's canonical one, through the Webapp's relay.
    await expect.poll(() => ws.sockets.some(socket => socket.url.includes('native_dial=browser'))).toBe(true)
    await expect
      .poll(() => backendProcesses(sandbox).length, { message: `one gateway, started by the Webapp\n${webapp.log()}` })
      .toBe(1)
    const [gateway] = backendProcesses(sandbox)

    provider.script(U(1), [
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
      { text: [`${A(1)} `, 'ran ', 'on ', 'the ', 'gateway'] }
    ])
    await send(page, `${U(1)} where do you run`, 'Enter', ws)
    await expect
      .poll(() => provider.completions.some(c => c.marker === U(1) && c.step === 1 && c.finished), {
        timeout: 120_000,
        message: `provider finished ${U(1)}\n${webapp.log()}`
      })
      .toBe(true)
    await expect.poll(() => currentSessionId(page)).not.toBe('')
    const sessionId = await currentSessionId(page)
    await expect.poll(() => storedSessionForMarker(sandbox, 'default', U(1))).toBe(sessionId)

    // The tool ran inside the gateway, and the owner admitted it for the browser's login. The Webapp
    // started that gateway (its own session, so it outlives the Webapp), so the Webapp may appear ABOVE
    // the gateway in the chain; a turn hosted by the Webapp would reach the Webapp without it.
    await expect.poll(() => toolAncestry(ancestry)).toContain(gateway.pid)
    const chain = toolAncestry(ancestry)
    const webappAt = chain.indexOf(webapp.child.pid!)
    expect(webappAt === -1 || chain.indexOf(gateway.pid) < webappAt, `tool ancestry ${chain}`).toBe(true)
    expect(admissionPrincipals(sandbox, sessionId)).toEqual([BROWSER_LOGIN])
    await assertTranscriptOracle(page, ws, provider, { sessionId, expectUserMarkers: [U(1)] }, 'webapp turn')

    // Reload resumes the chat on the same gateway.
    await page.reload()
    await expect(composer(page)).toBeVisible({ timeout: 120_000 })
    await expect(page.getByText(A(1)).first()).toBeVisible({ timeout: 60_000 })
    expect(backendProcesses(sandbox).map(proc => proc.pid)).toEqual([gateway.pid])

    // A gateway killed mid-session: the browser re-dials, the Webapp re-ensures one, the next turn works.
    process.kill(gateway.pid, 'SIGKILL')
    await expect.poll(() => alive(gateway.pid), { timeout: 30_000 }).toBe(false)
    provider.script(U(2), [{ text: [`${A(2)} `, 'after ', 'the ', 'gateway ', 'came ', 'back'] }])
    await send(page, `${U(2)} still there`, 'Enter', ws)
    await expect
      .poll(() => provider.completions.some(c => c.marker === U(2) && c.finished), {
        timeout: 150_000,
        message: `provider finished ${U(2)} on the re-ensured gateway\n${webapp.log()}`
      })
      .toBe(true)
    const [replacement] = backendProcesses(sandbox)
    expect(backendProcesses(sandbox)).toHaveLength(1)
    expect(replacement.pid).not.toBe(gateway.pid)

    // The gateway is the profile's, not the Webapp's: stopping the Webapp leaves it serving.
    webapp.child.kill('SIGTERM')
    // SIGTERM ends the Webapp by re-raising the signal: Node reports that as signalCode, not exitCode.
    await expect
      .poll(() => webapp!.child.exitCode !== null || webapp!.child.signalCode !== null, {
        timeout: 60_000,
        message: `the Webapp exits on SIGTERM\n${webapp.log()}`
      })
      .toBe(true)
    expect(alive(replacement.pid), 'the gateway outlives the Webapp that started it').toBe(true)
  } finally {
    await browser?.close().catch(() => undefined)
    webapp?.child.kill('SIGKILL')

    for (const proc of sandboxProcesses(sandbox)) {
      try {
        process.kill(proc.pid, 'SIGKILL')
      } catch {
        /* already gone */
      }
    }

    await provider.close()
    sandbox.cleanup()
  }
})
