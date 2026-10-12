/**
 * Failure class: ABOUT PANEL / UPDATE UX — one update at a time, and the
 * panel tells the truth afterwards.
 *
 * #123117: the About panel's update buttons let a second update start while
 * one was already running, and the panel's state stopped matching the
 * install. Here the user starts `hermes update` in a terminal, then clicks
 * "Update now" in the running app before the terminal update is done:
 *   - the app refuses and says an update is already running (no second
 *     updater process is started, the app stays up);
 *   - the terminal update finishes successfully;
 *   - afterwards the same running app reports the install as current on the
 *     new commit (no stale "Update now") and a chat turn completes.
 */

import { spawn } from 'node:child_process'
import * as fs from 'node:fs'
import * as path from 'node:path'

import { expect, test } from '@playwright/test'

import { recordWebSockets, send, waitForInteractive } from '../core/harness'

import {
  backendServeProcesses,
  closeQuietly,
  currentAs,
  desktopLog,
  diagnostics,
  git,
  installProcesses,
  isAlive,
  launchInstalledApp,
  openAbout,
  publishUpstream,
  readText,
  startInstallSession,
  updateLogLines,
  waitFor,
  waitForUpdateOffer
} from './harness'

const RUN = Date.now().toString(36)
const U = (n: number) => `U${n}-${RUN}`
const A = (n: number) => `A${n}-${RUN}`

test('Update now while a terminal update runs is refused, and the panel is truthful once that update lands', async () => {
  test.setTimeout(20 * 60_000)
  const session = await startInstallSession()
  const { facts, provider, env } = session
  const target = publishUpstream('release: next upstream main', 'UPSTREAM_RELEASE.md', `next release ${RUN}\n`)
  const marker = path.join(facts.hermesHome, '.hermes-update-in-progress')
  const cliLog = path.join(facts.sandboxRoot, 'terminal-update.log')
  let cliExit: null | number = null

  try {
    const launched = await launchInstalledApp(facts, env)
    const { app, page, logTail } = launched
    // Track the app's exit ourselves: once the terminal update stops it, Playwright's handle is dead.
    let appExited = false
    app.process().once('exit', () => {
      appExited = true
    })

    const explain = (extra = '') =>
      diagnostics(
        facts,
        `${extra}\n── terminal hermes update (tail) ──\n${readText(cliLog).split('\n').slice(-40).join('\n')}\n${logTail()}`
      )

    try {
      const ws = recordWebSockets(page)
      await waitForInteractive(app, page).catch(error => {
        throw new Error(`${(error as Error).message}\n${explain()}`)
      })
      const oldBackends = backendServeProcesses(facts).map(proc => proc.pid)
      await openAbout(page)
      await waitForUpdateOffer(page, target)

      await test.step('a terminal `hermes update` is running', async () => {
        const out = fs.openSync(cliLog, 'w')

        const cli = spawn(facts.hermes, ['update', '--yes'], {
          cwd: facts.home,
          env: { ...env, TERM: 'dumb' },
          stdio: ['ignore', out, out]
        })

        cli.once('exit', code => {
          cliExit = code ?? -1
        })
        await waitFor('the terminal update to own the update marker', () => fs.existsSync(marker) || cliExit !== null, {
          timeout: 120_000,
          interval: 100,
          explain
        })
        expect(cliExit, `the terminal update is still running when the user clicks\n${explain()}`).toBeNull()
      })

      await test.step('clicking Update now is refused while it runs', async () => {
        const offset = desktopLog(facts).length
        await page
          .getByRole('button', { name: /^update now$/i })
          .first()
          .click()
        await expect(page.getByText(/An update is already running/).first()).toBeVisible({ timeout: 30_000 })
        expect(appExited, 'the app stays up').toBe(false)
        // desktop.log is flushed on a timer, so the refusal line can trail the dialog.
        await expect
          .poll(() => updateLogLines(facts, offset), { timeout: 20_000, message: 'the app refused the hand-off' })
          .toMatch(/refusing (posix )?hand-off: An update is already running/)
        expect(
          installProcesses(facts)
            .filter(proc => /desktop-update\/posix\.sh|hermes-desktop-update/.test(proc.cmdline))
            .map(p => p.cmdline),
          'no second updater was started'
        ).toEqual([])
      })

      await test.step('the terminal update lands', async () => {
        await waitFor('the terminal update to finish', () => cliExit !== null, {
          timeout: 8 * 60_000,
          interval: 1_000,
          explain
        })
        expect(cliExit, `terminal hermes update exits 0\n${explain()}`).toBe(0)
        expect(git(facts.checkout, 'rev-parse', 'HEAD'), 'the checkout is on the new commit').toBe(target)
      })

      await test.step('reopened after the terminal update, the app is truthful: current, no stale Update now, chat works', async () => {
        // The terminal update's stage-and-swap stops a Desktop running from the release tree it
        // replaces (#109643); if it did not, the user quits it. Either way nothing from before
        // the update may keep running.
        if (!appExited) {
          await closeQuietly(app)
        }

        await waitFor(
          'the pre-update backend to be gone once its app is',
          () => oldBackends.every(pid => !isAlive(pid)),
          { timeout: 60_000, explain }
        )
        const reopened = await launchInstalledApp(facts, env)

        try {
          const ws2 = recordWebSockets(reopened.page)
          await waitForInteractive(reopened.app, reopened.page).catch(error => {
            throw new Error(`${(error as Error).message}\n${explain(reopened.logTail())}`)
          })
          await expect
            .poll(() => currentAs(reopened.page), {
              timeout: 120_000,
              intervals: [2_000],
              message: 'the app reports the install current on the new commit'
            })
            .toEqual({ currentSha: target, updateAvailable: false })
          // Chat first: the About panel replaces the chat view.
          provider.script(U(1), [{ text: [`${A(1)} `, 'after ', 'terminal ', 'update'] }])
          await send(reopened.page, `${U(1)} still here?`, 'Enter', ws2)
          await expect(reopened.page.getByText(`${A(1)} after terminal update`)).toBeVisible({ timeout: 120_000 })
          await openAbout(reopened.page)
          await expect(reopened.page.getByRole('button', { name: /^update now$/i })).toHaveCount(0, { timeout: 60_000 })
        } finally {
          await closeQuietly(reopened.app)
        }
      })
    } finally {
      if (!appExited) {
        await closeQuietly(app)
      }
    }
  } finally {
    await session.close()
  }
})
