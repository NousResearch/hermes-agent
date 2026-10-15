/**
 * Failure class: FIRST RUN on a healthy install.
 *
 * A user who installed Hermes with scripts/install.sh and built the Desktop
 * app with `hermes desktop` opens the app. It must find that install and go
 * straight to chat — on the first launch and on every later one — and never
 * show the first-run setup chooser or start the bootstrap installer
 * (#123888, #123800: the chooser / installer came back on every launch while
 * the local install was healthy). The second launch drops the Desktop
 * bootstrap marker first: a usable install is recognised off the filesystem.
 *
 * Real entry point: the packaged app the install built, launched with the
 * install's environment and no HERMES_DESKTOP_HERMES_ROOT override, so the app
 * resolves the backend the way it does for a user ($HERMES_HOME/hermes-agent).
 */

import * as fs from 'node:fs'
import * as path from 'node:path'

import { expect, test } from '@playwright/test'

import { recordWebSockets, send, waitForInteractive } from '../core/harness'

import {
  backendServeProcesses,
  closeQuietly,
  diagnostics,
  firstRunScreensSeen,
  type InstallFacts,
  installFirstRunSampler,
  installProcesses,
  launchInstalledApp,
  startInstallSession
} from './harness'

const RUN = Date.now().toString(36)
const U = (n: number) => `U${n}-${RUN}`
const A = (n: number) => `A${n}-${RUN}`

/** Every process of the install that is a bootstrap/installer run, sampled for the whole boot. */
function startInstallerSampler(facts: InstallFacts): { stop: () => string[] } {
  const seen = new Set<string>()

  const scan = () => {
    for (const proc of installProcesses(facts)) {
      if (/install\.sh|install\.ps1|bootstrap-runner|\bpm (bootstrap|install)\b/.test(proc.cmdline)) {
        seen.add(proc.cmdline.slice(0, 240))
      }
    }
  }

  const timer = setInterval(scan, 100)

  return {
    stop: () => {
      clearInterval(timer)
      scan()

      return [...seen]
    }
  }
}

test('a healthy local install opens straight to chat on every launch: no setup chooser, no bootstrap installer', async () => {
  const session = await startInstallSession()
  const { facts, provider, env } = session

  try {
    for (const launch of [1, 2]) {
      await test.step(launch === 1 ? 'launch 1' : 'launch 2, bootstrap marker absent', async () => {
        if (launch === 2) {
          // A usable install the Desktop bootstrap never stamped (a CLI install from before the
          // marker, or one whose marker was lost): usability, not the marker, decides.
          fs.rmSync(path.join(facts.checkout, '.hermes-bootstrap-complete'), { force: true })
        }

        const installer = startInstallerSampler(facts)
        const { app, page, logTail } = await launchInstalledApp(facts, env)

        try {
          await installFirstRunSampler(page)
          const ws = recordWebSockets(page)
          await waitForInteractive(app, page, 180_000).catch(error => {
            throw new Error(`${(error as Error).message}\n${diagnostics(facts, logTail())}`)
          })

          expect(
            await firstRunScreensSeen(page),
            `launch ${launch} of a healthy install showed the first-run setup chooser\n${diagnostics(facts, logTail())}`
          ).toEqual([])

          // The backend the app started is the user's install, not something it provisioned.
          const serve = backendServeProcesses(facts)
          expect(serve.length, `one backend for the install\n${diagnostics(facts)}`).toBe(1)
          const cwd = fs.realpathSync(`/proc/${serve[0].pid}/cwd`)
          expect(
            [serve[0].cmdline, cwd].some(s => s.includes(facts.hermesHome)),
            `backend runs from the install under ${facts.hermesHome}: ${serve[0].cmdline} (cwd ${cwd})`
          ).toBe(true)

          provider.script(U(launch), [{ text: [`${A(launch)} `, 'healthy ', 'install'] }])
          await send(page, `${U(launch)} hello`, 'Enter', ws)
          await expect(page.getByText(`${A(launch)} healthy install`)).toBeVisible({ timeout: 120_000 })

          expect(
            installer.stop(),
            `launch ${launch} of a healthy install started the bootstrap installer\n${diagnostics(facts, logTail())}`
          ).toEqual([])
        } finally {
          installer.stop()
          await closeQuietly(app)
        }

        await expect
          .poll(() => installProcesses(facts).map(p => `${p.pid} ${p.cmdline.slice(0, 160)}`), {
            timeout: 30_000,
            message: 'quitting the app leaves nothing of the install running'
          })
          .toEqual([])
      })
    }

    expect(
      fs.existsSync(path.join(facts.hermesHome, 'hermes-agent', '.git')),
      'the install is still the git checkout the installer made'
    ).toBe(true)
  } finally {
    await session.close()
  }
})
