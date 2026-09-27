/**
 * Failure class: APP-DRIVEN UPDATE completes.
 *
 * A user on a healthy local install (scripts/install.sh + `hermes desktop`)
 * sees "Update now" in Settings → About when upstream main moves, clicks it,
 * and the app hands off to the updater, quits, updates, and comes back:
 *   - the backend checkout is on the new upstream commit, clean;
 *   - the app relaunched by itself (a new Desktop process from the rebuilt
 *     package, with its own backend) and logged the update as finished OK;
 *   - the updated app boots to chat and a chat turn completes;
 *   - the updated app reports itself current (no update offered again).
 *
 * Real entry points only: the About panel's button and the app's own
 * hand-off (scripts/desktop-update/posix.sh → `hermes update`). Fakes: the git
 * server (local bare origin) and the LLM provider.
 */

import * as fs from 'node:fs'

import { expect, test } from '@playwright/test'

import { recordWebSockets, send, waitForInteractive } from '../core/harness'

import {
  backendServeProcesses,
  clickUpdateNowAndExpectHandoff,
  closeQuietly,
  currentAs,
  desktopLog,
  desktopMainProcesses,
  diagnostics,
  git,
  handoffLog,
  installProcesses,
  launchInstalledApp,
  openAbout,
  publishUpstream,
  startInstallSession,
  waitFor,
  waitForUpdateOffer
} from './harness'

const RUN = Date.now().toString(36)
const U = (n: number) => `U${n}-${RUN}`
const A = (n: number) => `A${n}-${RUN}`

test('clicking Update now moves the backend to the new commit, relaunches the app, and chat works after', async () => {
  test.setTimeout(20 * 60_000)
  const session = await startInstallSession()
  const { facts, provider, env } = session
  const before = git(facts.checkout, 'rev-parse', 'HEAD')
  const target = publishUpstream('release: next upstream main', 'UPSTREAM_RELEASE.md', `next release ${RUN}\n`)
  const explain = (extra = '') => diagnostics(facts, extra)

  try {
    let firstMainPid = 0

    await test.step('the running app offers the new upstream commit and hands off on click', async () => {
      const launched = await launchInstalledApp(facts, env)
      firstMainPid = launched.app.process().pid ?? 0
      await waitForInteractive(launched.app, launched.page).catch(error => {
        throw new Error(`${(error as Error).message}\n${explain(launched.logTail())}`)
      })
      await openAbout(launched.page)
      await waitForUpdateOffer(launched.page, target)
      await clickUpdateNowAndExpectHandoff(launched, facts)
    })

    await test.step('the update lands and the app relaunches itself', async () => {
      await waitFor(
        'the updater to finish `hermes update`',
        () => /hermes update exit code: \d+/.test(handoffLog(facts)),
        {
          timeout: 8 * 60_000,
          interval: 2_000,
          explain: () => explain(`handoff log:\n${handoffLog(facts)}`)
        }
      )
      expect(handoffLog(facts), `the updater's hermes update succeeded\n${explain()}`).toMatch(
        /hermes update exit code: 0\s*$|retry exit code: 0/m
      )
      expect(git(facts.checkout, 'rev-parse', 'HEAD'), 'the backend checkout is on the new upstream commit').toBe(
        target
      )

      // The updater's progress window can be the same Electron binary (--app=http://127.0.0.1:...): not a relaunch.
      const relaunched = await waitFor(
        'the app to relaunch itself',
        () =>
          desktopMainProcesses(facts).find(
            proc => proc.pid !== firstMainPid && !/--app=|https?:\/\/127\.0\.0\.1/.test(proc.cmdline)
          ),
        { timeout: 3 * 60_000, interval: 1_000, explain: () => explain(`handoff log:\n${handoffLog(facts)}`) }
      )

      await waitFor(
        'the relaunched app to log the update result',
        () => /\[updates\] detached update (finished|FAILED)/.test(desktopLog(facts)),
        {
          timeout: 3 * 60_000,
          explain
        }
      )
      expect(desktopLog(facts), `the relaunched app reports the update as finished OK\n${explain()}`).toMatch(
        /\[updates\] detached update finished OK/
      )
      await waitFor('the relaunched app to start its backend', () => backendServeProcesses(facts).length > 0, {
        timeout: 3 * 60_000,
        explain
      })
      expect(
        git(facts.checkout, 'status', '--porcelain', '--untracked-files=no'),
        'the updated checkout is clean'
      ).toBe('')

      // Hand the machine back to Playwright: quit the relaunched app the way the OS asks it to.
      process.kill(relaunched.pid, 'SIGTERM')
      await expect
        .poll(() => installProcesses(facts).map(p => `${p.pid} ${p.cmdline.slice(0, 160)}`), {
          timeout: 60_000,
          message: 'the relaunched app quits cleanly on SIGTERM'
        })
        .toEqual([])
    })

    await test.step('the updated app boots to chat, a turn completes, and it reports itself current', async () => {
      const { app, page, logTail } = await launchInstalledApp(facts, env)

      try {
        const ws = recordWebSockets(page)
        await waitForInteractive(app, page).catch(error => {
          throw new Error(`${(error as Error).message}\n${explain(logTail())}`)
        })
        provider.script(U(1), [{ text: [`${A(1)} `, 'after ', 'update'] }])
        await send(page, `${U(1)} hello after the update`, 'Enter', ws)
        await expect(page.getByText(`${A(1)} after update`)).toBeVisible({ timeout: 120_000 })
        expect(await currentAs(page), `the updated app is current (was ${before})`).toEqual({
          currentSha: target,
          updateAvailable: false
        })
      } finally {
        await closeQuietly(app)
      }
    })

    expect(fs.existsSync(facts.checkout)).toBe(true)
  } finally {
    await session.close()
  }
})
