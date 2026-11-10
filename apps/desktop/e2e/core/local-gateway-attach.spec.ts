/**
 * Desktop attaching to a local gateway that was already running (#134889 W3).
 *
 * The shape every Windows login item, `hermes gateway run` console and CLI
 * session leaves behind: the profile's gateway is up BEFORE the Desktop opens.
 * Desktop must attach to that process through the real main-process path
 * (electron/local-gateway.ts `hermes gateway ensure`, then the ticket mint:
 * the SID-checked named pipe through the installed runtime's helper on
 * Windows, the private control socket elsewhere), never start a second one,
 * and serve a turn on it. Quitting leaves the gateway it did not start
 * running; a relaunch attaches to the same process again.
 *
 * Runs on every OS. The Windows Desktop lane (.github/workflows/
 * windows-desktop-attach-e2e.yml) runs it on a real Windows runner against the
 * dev tree and, with HERMES_E2E_PACKAGED_EXE, against the electron-builder
 * `--dir` build of the same commit.
 */

import { spawn } from 'node:child_process'
import * as fs from 'node:fs'
import * as path from 'node:path'

import { _electron, type ElectronApplication, expect, type Page, test } from '@playwright/test'

import {
  appLogTail,
  backendProcesses,
  coreAppEnv,
  type CoreSandbox,
  createCoreSandbox,
  currentSessionId,
  launchCoreApp,
  recordWebSockets,
  REPO_ROOT,
  sandboxProcesses,
  send,
  stopSandboxGateway,
  storedSessionForMarker,
  waitForInteractive,
  writeProviderHome
} from './harness'
import { assertTranscriptOracle, installDuplicateSampler } from './oracle'
import { startScriptedProvider } from './provider'

const nonce = Math.random().toString(36).slice(2, 8).replace(/[^a-z0-9]/g, 'x').padEnd(4, 'q')
const U = (n: number) => `U${n}-attach${nonce}`
const A = (n: number) => `A${n}-attach${nonce}`

const PYTHON = process.env.HERMES_E2E_PYTHON || (process.platform === 'win32' ? 'python' : 'python3')
const PACKAGED_EXE = process.env.HERMES_E2E_PACKAGED_EXE || ''

function gatewayState(sandbox: CoreSandbox): Record<string, any> {
  try {
    return JSON.parse(fs.readFileSync(path.join(sandbox.hermesHome, 'gateway_state.json'), 'utf8').replace(/^\uFEFF/, ''))
  } catch {
    return {}
  }
}

function alive(pid: number): boolean {
  try {
    process.kill(pid, 0)

    return true
  } catch (error) {
    return (error as NodeJS.ErrnoException).code === 'EPERM'
  }
}

/** `hermes gateway run`, detached and windowless, as a login item / console starts it. */
async function startGatewayFirst(sandbox: CoreSandbox): Promise<{ pid: number; log: string }> {
  const log = path.join(sandbox.root, 'gateway-run.log')
  const out = fs.openSync(log, 'a')

  const child = spawn(PYTHON, ['-m', 'hermes_cli.main', 'gateway', 'run'], {
    cwd: REPO_ROOT,
    env: coreAppEnv(sandbox),
    detached: true,
    windowsHide: true,
    stdio: ['ignore', out, out]
  })

  child.unref()
  fs.closeSync(out)

  await expect
    .poll(
      () => {
        const state = gatewayState(sandbox)

        return state.gateway_state === 'running' && Number(state.pid) > 0 && alive(Number(state.pid))
      },
      {
        timeout: 180_000,
        intervals: [500],
        message: `the pre-started gateway reports running\n${fs.existsSync(log) ? fs.readFileSync(log, 'utf8').slice(-4000) : ''}`
      }
    )
    .toBe(true)

  return { pid: Number(gatewayState(sandbox).pid), log }
}

async function launchApp(sandbox: CoreSandbox): Promise<{ app: ElectronApplication; page: Page }> {
  if (!PACKAGED_EXE) {
    return launchCoreApp(coreAppEnv(sandbox))
  }

  const app = await _electron.launch({
    executablePath: PACKAGED_EXE,
    args: ['--disable-gpu', '--no-sandbox'],
    env: coreAppEnv(sandbox),
    cwd: sandbox.root
  })

  return { app, page: await app.firstWindow() }
}

test(`Desktop${PACKAGED_EXE ? ' (packaged)' : ''} attaches to the gateway already running and serves a turn`, async () => {
  const provider = await startScriptedProvider()
  const sandbox = createCoreSandbox('attach')
  writeProviderHome(sandbox.hermesHome, provider.url)
  let live: null | { app: ElectronApplication; page: Page } = null

  try {
    const gateway = await startGatewayFirst(sandbox)
    expect(backendProcesses(sandbox).map(p => p.pid), 'only the pre-started gateway runs before Desktop opens').toEqual([
      gateway.pid
    ])

    let sessionId = ''

    for (let launch = 1; launch <= 2; launch++) {
      await test.step(`launch ${launch}: attach to pid ${gateway.pid}`, async () => {
        live = await launchApp(sandbox)
        const { app, page } = live

        if (PACKAGED_EXE) {
          const facts = await app.evaluate(({ app: a }) => ({ isPackaged: a.isPackaged }))
          expect(facts.isPackaged, 'launched the packaged build').toBe(true)
        }

        const ws = recordWebSockets(page)
        await waitForInteractive(app, page)
        await installDuplicateSampler(page)

        // Main's descriptor names the very gateway that was running first.
        const descriptor = await page.evaluate(() => (window as any).hermesDesktop.getConnection())
        const state = gatewayState(sandbox)
        expect(descriptor?.mode, `connection descriptor: ${JSON.stringify(descriptor)}`).toBe('local')
        expect(String(descriptor?.gatewayEndpoint?.api_origin ?? descriptor?.baseUrl ?? '')).toMatch(
          /^http:\/\/(127\.0\.0\.1|\[::1\]):\d+\/?$/
        )
        expect(Number(state.pid), 'gateway_state.json still names the pre-started gateway').toBe(gateway.pid)
        expect(backendProcesses(sandbox).map(p => p.pid), `launch ${launch} attached, never spawned a second gateway`).toEqual([
          gateway.pid
        ])

        if (launch === 1) {
          provider.script(U(1), [{ text: [`${A(1)} `, 'served ', 'by ', 'the ', 'running ', 'gateway'] }])
          await send(page, `${U(1)} hello`, 'Enter', ws)
          await expect
            .poll(() => provider.completions.some(c => c.marker === U(1) && c.finished), {
              timeout: 120_000,
              message: `provider finished ${U(1)}\n${appLogTail(app)}`
            })
            .toBe(true)
          await expect.poll(() => currentSessionId(page)).not.toBe('')
          sessionId = await currentSessionId(page)
          await expect
            .poll(() => storedSessionForMarker(sandbox, 'default', U(1)), { message: 'turn persisted by the gateway' })
            .toBe(sessionId)
        } else {
          await page.evaluate(id => {
            window.location.hash = `#/${encodeURIComponent(id)}`
          }, sessionId)
          provider.script(U(2), [{ text: [`${A(2)} `, 'after ', 'relaunch'] }])
          await send(page, `${U(2)} again`, 'Enter', ws)
          await expect
            .poll(() => provider.completions.some(c => c.marker === U(2) && c.finished), {
              timeout: 120_000,
              message: `provider finished ${U(2)}\n${appLogTail(app)}`
            })
            .toBe(true)
        }

        await assertTranscriptOracle(
          page,
          ws,
          provider,
          { sessionId, expectUserMarkers: launch === 1 ? [U(1)] : [U(1), U(2)] },
          `attach launch ${launch}`
        )

        const electronPid = app.process().pid!
        await app.close()
        live = null
        await expect
          .poll(() => alive(electronPid), { timeout: 60_000, message: `Electron main (pid ${electronPid}) exits on quit #${launch}` })
          .toBe(false)
        expect(alive(gateway.pid), `the gateway Desktop attached to outlives quit #${launch}`).toBe(true)
        expect(backendProcesses(sandbox).map(p => p.pid)).toEqual([gateway.pid])
      })
    }

    const stopped = stopSandboxGateway(sandbox)
    expect(stopped.code, stopped.output).toBe(0)
    await expect
      .poll(() => alive(gateway.pid), { timeout: 90_000, message: 'hermes gateway stop ends the gateway' })
      .toBe(false)
  } finally {
    if (live) {
      await (live as { app: ElectronApplication }).app.close().catch(() => undefined)
    }

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
