/**
 * The browser-hosted Desktop is served by the profile gateway's own listener, exactly like local
 * Electron Desktop talks to that gateway: one process runs every turn and owns the profile's store.
 *
 * `webapp.enabled` in the home's config.yaml turns it on; `hermes webapp` starts the gateway and prints
 * a one-use launch link it got over the gateway's control socket. Proof, end to end with a real browser:
 *  - exactly one gateway serves the sandbox, and the browser's socket is that gateway's own;
 *  - a turn's terminal tool runs under the gateway process (its ancestry);
 *  - the gateway admitted the turn for the browser's sign-in, not the host's native grant;
 *  - reload resumes the chat;
 *  - after the gateway is killed and started again, the same tab is still signed in (the browser
 *    credential is kept in the home) and the next turn works;
 *  - `hermes webapp --sign-out` cuts that tab off.
 *
 * A second test puts the Webapp and Electron Desktop on one chat: Electron answers the approval a
 * Webapp turn raised, and the Webapp stops a turn Electron started. A third puts the Webapp behind
 * the sign-in gate: `hermes webapp --sign-out` signs out a browser that signed in with a password.
 *
 * Linux (the /proc ancestry walk); needs the Webapp renderer (`npm run build:webapp`) and a
 * Chromium Playwright can launch (`npx playwright install chromium`, or HERMES_E2E_CHROMIUM).
 */

import { spawnSync } from 'node:child_process'
import * as fs from 'node:fs'
import { createServer } from 'node:net'
import * as path from 'node:path'
import { DatabaseSync } from 'node:sqlite'

import { type Browser, chromium, expect, type Page, test } from '@playwright/test'

import {
  appLogTail,
  backendProcesses,
  composer,
  coreAppEnv,
  type CoreSandbox,
  createCoreSandbox,
  currentSessionId,
  DESKTOP_ROOT,
  launchCoreApp,
  recordWebSockets,
  REPO_ROOT,
  sandboxProcesses,
  send,
  storedSessionForMarker,
  waitForInteractive,
  writeProviderHome
} from './harness'
import { assertTranscriptOracle, installDuplicateSampler } from './oracle'
import { shellQuote } from './process-census'
import { gate, startScriptedProvider } from './provider'

const nonce = Math.random().toString(36).slice(2, 8).replace(/[^a-z0-9]/g, 'x').padEnd(4, 'q')
const U = (n: number) => `U${n}-webapp${nonce}`
const A = (n: number) => `A${n}-webapp${nonce}`
const AI = (n: number) => `A${n}i-webapp${nonce}`

const PYTHON = process.env.HERMES_E2E_PYTHON || 'python3'
/** The browser's sign-in on a loopback Webapp: the gateway's session credential. */
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

/** `hermes <args>` in the sandbox, as an operator runs it. */
function hermesCli(sandbox: CoreSandbox, args: string[]): { code: number | null; output: string } {
  const result = spawnSync(PYTHON, ['-m', 'hermes_cli.main', ...args], {
    cwd: REPO_ROOT,
    env: coreAppEnv(sandbox),
    encoding: 'utf8',
    timeout: 180_000
  })

  return { code: result.status, output: `${result.stdout ?? ''}${result.stderr ?? ''}` }
}

/** `hermes webapp <args>` in the sandbox. */
function hermesWebapp(sandbox: CoreSandbox, ...args: string[]): { code: number | null; output: string } {
  return hermesCli(sandbox, ['webapp', ...args])
}

/** Starts the gateway if needed and returns a one-use launch link for it (`hermes -p <profile> webapp`
 * when a profile is named). */
function launchLink(sandbox: CoreSandbox, profile?: string): string {
  const profileArgs = profile ? ['-p', profile] : []
  const { code, output } = hermesCli(sandbox, [...profileArgs, 'webapp', '--no-open', '--skip-build'])
  const link = /^\s+(http:\/\/127\.0\.0\.1:\d+\/webapp\/launch#\S+)$/m.exec(output)?.[1]
  expect(code, `hermes webapp --no-open\n${output.replace(/ticket=\S+/g, 'ticket=<redacted>')}`).toBe(0)
  expect(link, 'hermes webapp printed a one-use launch link').toBeTruthy()

  return link!
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

test('the gateway serves the Webapp and runs every turn its browser sends', async () => {
  test.skip(process.platform !== 'linux', 'walks /proc to prove which process ran the tool')
  expect(
    fs.existsSync(path.join(DESKTOP_ROOT, 'dist-webapp', 'index.html')),
    'the Webapp renderer is built (npm run build:webapp)'
  ).toBe(true)

  const provider = await startScriptedProvider()
  const sandbox = createCoreSandbox('webapp')
  writeProviderHome(sandbox.hermesHome, provider.url)
  const port = await freePort()
  fs.appendFileSync(path.join(sandbox.hermesHome, 'config.yaml'), `\nwebapp:\n  enabled: true\n  port: ${port}\n`)
  const ancestry = path.join(sandbox.root, 'tool-ancestry.txt')
  let browser: Browser | null = null

  try {
    const link = launchLink(sandbox)
    expect(new URL(link).port).toBe(String(port))
    browser = await chromium.launch({ executablePath: process.env.HERMES_E2E_CHROMIUM || undefined })
    const page: Page = await (await browser.newContext({ viewport: { width: 1280, height: 860 } })).newPage()
    const ws = recordWebSockets(page)
    await page.goto(link)
    await expect(composer(page)).toBeVisible({ timeout: 180_000 })
    expect(page.url()).not.toContain('ticket=')
    expect(page.url()).not.toContain('hermes-session=')
    await installDuplicateSampler(page)

    // The browser's socket is the gateway's own, on the Webapp's origin.
    await expect
      .poll(() => ws.sockets.some(socket => socket.url.startsWith(`ws://127.0.0.1:${port}/api/ws?native_dial=browser`)))
      .toBe(true)
    await expect.poll(() => backendProcesses(sandbox).length, { message: 'one gateway serves the sandbox' }).toBe(1)
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
        message: `provider finished ${U(1)}`
      })
      .toBe(true)
    await expect.poll(() => currentSessionId(page)).not.toBe('')
    const sessionId = await currentSessionId(page)
    await expect.poll(() => storedSessionForMarker(sandbox, 'default', U(1))).toBe(sessionId)

    // The tool ran inside the gateway, admitted for the browser's sign-in.
    await expect.poll(() => toolAncestry(ancestry)).toContain(gateway.pid)
    expect(admissionPrincipals(sandbox, sessionId)).toEqual([BROWSER_LOGIN])
    await assertTranscriptOracle(page, ws, provider, { sessionId, expectUserMarkers: [U(1)] }, 'webapp turn')

    // Reload resumes the chat on the same gateway.
    await page.reload()
    await expect(composer(page)).toBeVisible({ timeout: 120_000 })
    await expect(page.getByText(A(1)).first()).toBeVisible({ timeout: 60_000 })
    expect(backendProcesses(sandbox).map(proc => proc.pid)).toEqual([gateway.pid])

    // A gateway killed and started again (here by `hermes webapp`, whose new link this tab never
    // uses): the same tab is still signed in and the next turn runs on the new gateway.
    process.kill(gateway.pid, 'SIGKILL')
    await expect.poll(() => alive(gateway.pid), { timeout: 30_000 }).toBe(false)
    launchLink(sandbox)
    await page.reload()
    await expect(composer(page)).toBeVisible({ timeout: 120_000 })
    provider.script(U(2), [{ text: [`${A(2)} `, 'after ', 'the ', 'gateway ', 'came ', 'back'] }])
    await send(page, `${U(2)} still there`, 'Enter', ws)
    await expect
      .poll(() => provider.completions.some(c => c.marker === U(2) && c.finished), {
        timeout: 150_000,
        message: `provider finished ${U(2)} on the restarted gateway`
      })
      .toBe(true)
    const [replacement] = backendProcesses(sandbox)
    expect(backendProcesses(sandbox)).toHaveLength(1)
    expect(replacement.pid).not.toBe(gateway.pid)

    // Signing out replaces the browser credential: the tab loses access.
    const signedOut = hermesWebapp(sandbox, '--sign-out')
    expect(signedOut.code, signedOut.output).toBe(0)
    await page.reload()
    await expect(composer(page)).toBeHidden({ timeout: 60_000 })
  } finally {
    await browser?.close().catch(() => undefined)

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

/**
 * `hermes -p <profile> webapp` opens the default gateway's Webapp on that profile. The tab's own socket
 * answers for the gateway's profile, so each request for the tab's profile names it: a turn whose tool
 * stays quiet past the 45 s silence probe stays live, and a reload reads the transcript back.
 */
test('a Webapp tab opened on a named profile runs and reads that profile', async () => {
  test.skip(process.platform !== 'linux', 'the quiet tool is a POSIX sleep')
  test.setTimeout(420_000)
  expect(
    fs.existsSync(path.join(DESKTOP_ROOT, 'dist-webapp', 'index.html')),
    'the Webapp renderer is built (npm run build:webapp)'
  ).toBe(true)

  const provider = await startScriptedProvider()
  const sandbox = createCoreSandbox('webapp-profile')
  writeProviderHome(sandbox.hermesHome, provider.url)
  writeProviderHome(path.join(sandbox.hermesHome, 'profiles', 'p2'), provider.url)
  const port = await freePort()
  fs.appendFileSync(path.join(sandbox.hermesHome, 'config.yaml'), `\nwebapp:\n  enabled: true\n  port: ${port}\n`)
  let browser: Browser | null = null

  try {
    launchLink(sandbox)
    const link = launchLink(sandbox, 'p2')
    browser = await chromium.launch({ executablePath: process.env.HERMES_E2E_CHROMIUM || undefined })
    const page: Page = await (await browser.newContext({ viewport: { width: 1280, height: 860 } })).newPage()
    const ws = recordWebSockets(page)
    await page.goto(link)
    await expect(composer(page)).toBeVisible({ timeout: 180_000 })
    await expect.poll(() => new URL(page.url()).searchParams.get('profile')).toBe('p2')

    // No text before the tool: a turn that has replied already never shows the ended card.
    provider.script(U(1), [
      { toolCalls: [{ name: 'terminal', args: { command: 'sleep 55' } }] },
      { text: [`${A(1)} `, 'after ', 'the ', 'quiet ', 'tool'] }
    ])
    await send(page, `${U(1)} wait quietly`, 'Enter', ws)
    await expect.poll(() => currentSessionId(page)).not.toBe('')
    const sessionId = await currentSessionId(page)
    await expect.poll(() => storedSessionForMarker(sandbox, 'p2', U(1)), { timeout: 60_000 }).toBe(sessionId)
    expect(storedSessionForMarker(sandbox, 'default', U(1))).toBeNull()

    const ended = page.getByText('Hermes ended this turn without a reply')
    await page.waitForTimeout(50_000)
    await expect(ended).toHaveCount(0)
    await expect(page.getByText(A(1)).first()).toBeVisible({ timeout: 120_000 })
    await expect(ended).toHaveCount(0)

    await page.reload()
    await expect(composer(page)).toBeVisible({ timeout: 120_000 })
    expect(new URL(page.url()).searchParams.get('profile')).toBe('p2')
    await expect(page.getByText(A(1)).first()).toBeVisible({ timeout: 60_000 })
  } finally {
    await browser?.close().catch(() => undefined)

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

function viewport(page: Page) {
  return page.locator('[data-slot="aui_thread-viewport"]').filter({ visible: true }).first()
}

function approvalRun(page: Page) {
  return page.locator('[data-approval-run]').filter({ visible: true })
}

function stopButton(page: Page) {
  return page.locator('[data-slot="composer-root"] button[aria-label="Stop"]').filter({ visible: true })
}

/**
 * One chat, two surfaces on one gateway: the Webapp in a browser and Electron Desktop attached to the
 * same gateway. A prompt raised in a turn one surface started is answered from the other, resolves
 * once, and leaves no stale card on the first; Stop on one surface ends the turn the other started.
 */
test("the Webapp and Electron Desktop answer and stop each other's turns on one gateway", async () => {
  test.skip(process.platform !== 'linux', 'the gated command and the process census are Linux-only here')
  test.setTimeout(480_000)
  expect(
    fs.existsSync(path.join(DESKTOP_ROOT, 'dist-webapp', 'index.html')),
    'the Webapp renderer is built (npm run build:webapp)'
  ).toBe(true)

  const provider = await startScriptedProvider()
  const sandbox = createCoreSandbox('webapp-cross')
  writeProviderHome(sandbox.hermesHome, provider.url, '', 'manual')
  const port = await freePort()
  fs.appendFileSync(path.join(sandbox.hermesHome, 'config.yaml'), `\nwebapp:\n  enabled: true\n  port: ${port}\n`)
  const hold = gate()
  let browser: Browser | null = null
  let desktop: Awaited<ReturnType<typeof launchCoreApp>> | null = null

  try {
    browser = await chromium.launch({ executablePath: process.env.HERMES_E2E_CHROMIUM || undefined })
    const page: Page = await (await browser.newContext({ viewport: { width: 1280, height: 860 } })).newPage()
    const ws = recordWebSockets(page)
    await page.goto(launchLink(sandbox))
    await expect(composer(page)).toBeVisible({ timeout: 180_000 })
    await installDuplicateSampler(page)
    await expect.poll(() => backendProcesses(sandbox).length, { message: 'one gateway serves the sandbox' }).toBe(1)
    const [gateway] = backendProcesses(sandbox)

    // The Webapp's turn stops at an approval.
    const victim = path.join(sandbox.root, `victim-cross-${nonce}`)
    fs.mkdirSync(victim)
    provider.script(U(1), [
      {
        text: [`${AI(1)} `, 'needs ', 'approval'],
        toolCalls: [{ name: 'terminal', args: { command: `rm -rf ${shellQuote(victim)}` } }]
      },
      { text: [`${A(1)} `, 'approved ', 'elsewhere'] }
    ])
    await send(page, `${U(1)} delete the dir`, 'Enter', ws)
    await expect(approvalRun(page)).toHaveCount(1, { timeout: 120_000 })
    await expect.poll(() => currentSessionId(page)).not.toBe('')
    const sessionId = await currentSessionId(page)

    // Electron Desktop attaches to the same gateway and opens the same chat.
    desktop = await launchCoreApp(coreAppEnv(sandbox))
    const desktopWs = recordWebSockets(desktop.page)
    await waitForInteractive(desktop.app, desktop.page)
    expect(backendProcesses(sandbox).map(proc => proc.pid), 'Electron attached to the serving gateway').toEqual([
      gateway.pid
    ])
    await desktop.page.evaluate(id => {
      window.location.hash = `#/${encodeURIComponent(id)}`
    }, sessionId)
    await expect.poll(() => currentSessionId(desktop!.page)).toBe(sessionId)

    // Electron answers the Webapp's approval: it resolves once and the Webapp's card clears.
    await expect(approvalRun(desktop.page), `Electron shows the pending approval\n${appLogTail(desktop.app)}`).toHaveCount(
      1,
      { timeout: 60_000 }
    )
    expect(fs.existsSync(victim), 'the gated command waits for the answer').toBe(true)
    await approvalRun(desktop.page).click()
    await expect
      .poll(() => provider.completions.some(c => c.marker === U(1) && c.step === 1 && c.finished), {
        timeout: 120_000,
        message: `provider finished ${U(1)} after the approval`
      })
      .toBe(true)
    expect(fs.existsSync(victim), 'the approved command ran').toBe(false)
    await expect(approvalRun(page), "the Webapp's approval card clears").toHaveCount(0, { timeout: 30_000 })
    await expect(approvalRun(desktop.page)).toHaveCount(0)
    expect(desktopWs.sent.filter(frame => frame.method === 'approval.respond')).toHaveLength(1)
    expect(ws.sent.filter(frame => frame.method === 'approval.respond')).toHaveLength(0)
    expect(provider.completions.filter(c => c.marker === U(1) && c.step === 1)).toHaveLength(1)
    await assertTranscriptOracle(page, ws, provider, { sessionId, expectUserMarkers: [U(1)] }, 'approval from Electron')
    await expect(viewport(desktop.page)).toContainText(A(1), { timeout: 60_000 })

    // Electron starts a turn; the Webapp shows it and stops it.
    provider.script(U(2), [{ text: [`${A(2)} `, 'held ', 'until ', 'stopped'], holdAfterFirstChunk: hold }])
    await send(desktop.page, `${U(2)} keep going`, 'Enter', desktopWs)
    await provider.streamStarted(U(2))
    await expect(viewport(page), "the Webapp shows Electron's turn").toContainText(A(2), { timeout: 60_000 })
    await expect(stopButton(page), "the Webapp offers Stop for Electron's turn").toHaveCount(1, { timeout: 60_000 })
    await stopButton(page).click()

    const interrupted = () =>
      desktopWs.events.find(
        event =>
          event.type === 'message.complete' &&
          String(event.payload?.text ?? '').includes(A(2)) &&
          event.payload?.status === 'interrupted'
      )

    await expect
      .poll(() => Boolean(interrupted()), { timeout: 60_000, message: `Electron sees ${U(2)} end interrupted` })
      .toBe(true)
    hold.open()
    await expect(stopButton(desktop.page)).toHaveCount(0, { timeout: 30_000 })
    await expect(stopButton(page)).toHaveCount(0, { timeout: 30_000 })
    expect(provider.completions.find(c => c.marker === U(2))?.finished, 'the held reply never streamed to its end').toBe(
      false
    )
    expect(ws.sent.some(frame => frame.method === 'session.interrupt'), 'the Webapp sent the interrupt').toBe(true)
    expect(desktopWs.sent.some(frame => frame.method === 'session.interrupt')).toBe(false)
    expect(backendProcesses(sandbox).map(proc => proc.pid)).toEqual([gateway.pid])
  } finally {
    hold.open()
    await desktop?.app.close().catch(() => undefined)
    await browser?.close().catch(() => undefined)

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

/**
 * The config.yaml lines that serve the Webapp on a loopback port behind the sign-in gate (the URL
 * browsers reach it at) with the bundled password provider. The one place this spec names those settings.
 */
function signInGateYaml(port: number, publicUrl: string, user: string, passwordHash: string, secret: string): string {
  return (
    `\nwebapp:\n  enabled: true\n  port: ${port}\n  public_url: ${publicUrl}\n` +
    `dashboard:\n  basic_auth:\n    username: ${user}\n    password_hash: '${passwordHash}'\n    secret: '${secret}'\n`
  )
}

/** A Python one-liner in the sandbox, as the gateway's own interpreter runs it. */
function python(sandbox: CoreSandbox, code: string, ...args: string[]): string {
  const result = spawnSync(PYTHON, ['-c', code, ...args], {
    cwd: REPO_ROOT,
    env: coreAppEnv(sandbox),
    encoding: 'utf8'
  })

  expect(result.status, result.stderr).toBe(0)

  return result.stdout.trim()
}

/**
 * Behind the sign-in gate the browser signs in with the dashboard's password, not the browser
 * credential `hermes webapp` hands out; `hermes webapp --sign-out` must still sign it out: the open
 * tab is cut off, a reload lands on the sign-in page, and signing in again works.
 */
test('signing out of a Webapp behind sign-in signs the browser out', async () => {
  test.skip(process.platform !== 'linux', 'runs with the other Webapp journeys (Linux CI)')
  expect(
    fs.existsSync(path.join(DESKTOP_ROOT, 'dist-webapp', 'index.html')),
    'the Webapp renderer is built (npm run build:webapp)'
  ).toBe(true)

  const provider = await startScriptedProvider()
  const sandbox = createCoreSandbox('webapp-gated')
  writeProviderHome(sandbox.hermesHome, provider.url)
  const port = await freePort()
  const origin = `http://127.0.0.1:${port}`
  const password = python(sandbox, 'import secrets; print("throwaway-" + secrets.token_urlsafe(12))')

  const hash = python(
    sandbox,
    'import sys; from plugins.dashboard_auth.basic import hash_password; print(hash_password(sys.argv[1]))',
    password
  )

  const secret = python(sandbox, 'import secrets; print(secrets.token_hex(32))')
  fs.appendFileSync(
    path.join(sandbox.hermesHome, 'config.yaml'),
    signInGateYaml(port, `http://hermes-webapp.test:${port}`, 'owner', hash, secret)
  )
  let browser: Browser | null = null

  try {
    const started = hermesWebapp(sandbox, '--no-open', '--skip-build')
    expect(started.code, started.output).toBe(0)
    expect(started.output).toContain('sign in with your dashboard account')
    browser = await chromium.launch({ executablePath: process.env.HERMES_E2E_CHROMIUM || undefined })
    const page: Page = await (await browser.newContext({ viewport: { width: 1280, height: 860 } })).newPage()
    const ws = recordWebSockets(page)

    const signIn = () =>
      page.request.post(`${origin}/auth/password-login`, {
        headers: { Origin: origin },
        data: { provider: 'basic', username: 'owner', password }
      })

    expect((await signIn()).status()).toBe(200)
    await page.goto(`${origin}/`)
    await expect(composer(page)).toBeVisible({ timeout: 180_000 })
    await expect
      .poll(() => ws.sockets.some(socket => socket.url.startsWith(`ws://127.0.0.1:${port}/api/ws`)))
      .toBe(true)

    const open = ws.sockets.filter(socket => !socket.closed)
    const signedOut = hermesWebapp(sandbox, '--sign-out')
    expect(signedOut.code, signedOut.output).toBe(0)
    await expect
      .poll(() => open.every(socket => socket.closed), { timeout: 30_000, message: "sign-out closed the tab's socket" })
      .toBe(true)
    expect(signedOut.output).toMatch(/closed [1-9]\d* browser connections?\./)
    expect(signedOut.output).toContain('Every browser login behind sign-in has ended')
    expect((await page.request.get(`${origin}/api/sessions`)).status()).toBe(401)
    await page.reload()
    await expect(page).toHaveURL(/\/login/, { timeout: 60_000 })
    await expect(composer(page)).toBeHidden()

    expect((await signIn()).status()).toBe(200)
    await page.goto(`${origin}/`)
    await expect(composer(page)).toBeVisible({ timeout: 120_000 })
  } finally {
    await browser?.close().catch(() => undefined)

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
