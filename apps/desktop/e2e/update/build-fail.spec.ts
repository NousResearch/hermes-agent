/**
 * Failure class: APP-DRIVEN UPDATE whose Desktop build fails.
 *
 * #124040: the Desktop build failed inside the update takeover's completion
 * stage and the failure was swallowed / kept looping instead of reaching the
 * user. Here upstream main carries a Desktop source file that does not
 * compile; the user clicks "Update now".
 *
 * The build runs after the code committed, so under contract C3 the update
 * succeeds with an owed follow-up: Hermes IS on the new code and only this
 * app was not rebuilt. The relaunched (old-build) app must say exactly that
 * and what to do — the action-required dialog, logged as `[updates] detached
 * update finished with manual action` — never "finished OK" (a silent stale
 * app) and never "FAILED / still on the previous version" (false: the code
 * moved). The updater must not be left running.
 */

import * as path from 'node:path'

import { expect, test } from '@playwright/test'

import { waitForInteractive } from '../core/harness'

import {
  clickUpdateNowAndExpectHandoff,
  desktopLog,
  diagnostics,
  git,
  handoffLog,
  installProcesses,
  launchInstalledApp,
  openAbout,
  publishUpstream,
  readText,
  startInstallSession,
  waitFor,
  waitForUpdateOffer
} from './harness'

test('a Desktop build failure during Update now tells the user the app was not rebuilt and how to fix it', async () => {
  test.setTimeout(20 * 60_000)
  const session = await startInstallSession()
  const { facts, env } = session
  const rel = 'apps/desktop/src/main.tsx'
  const broken = `${readText(path.join(facts.checkout, rel))}\nexport const brokenRelease = (\n`
  const target = publishUpstream('release: next upstream main (desktop does not compile)', rel, broken)
  const explain = (extra = '') => diagnostics(facts, `${extra}\n── handoff log ──\n${handoffLog(facts)}`)

  try {
    await test.step('the user clicks Update now', async () => {
      const launched = await launchInstalledApp(facts, env)
      await waitForInteractive(launched.app, launched.page).catch(error => {
        throw new Error(`${(error as Error).message}\n${explain(launched.logTail())}`)
      })
      await openAbout(launched.page)
      await waitForUpdateOffer(launched.page, target)
      await clickUpdateNowAndExpectHandoff(launched, facts)
    })

    await test.step('the failed build reaches the user as an app that still needs rebuilding', async () => {
      await waitFor(
        'the relaunched app to report the update outcome',
        () => /\[updates\] detached update (finished|FAILED)/.test(desktopLog(facts)),
        { timeout: 12 * 60_000, interval: 2_000, explain }
      )

      const outcome = desktopLog(facts)
        .split('\n')
        .filter(line => /\[updates\] detached update/.test(line))
        .join('\n')

      expect(outcome, `the owed rebuild reaches the user as an action, not a silent OK\n${explain()}`).toMatch(
        /detached update finished with manual action/
      )
      expect(outcome, 'the user is told the Desktop app could not be rebuilt and how to rebuild it').toMatch(
        /Desktop app could not be rebuilt.*hermes desktop --force-build/
      )
      expect(outcome).not.toMatch(/detached update finished OK/)
      // The code committed: "FAILED / still on the previous version" would be false.
      expect(outcome).not.toMatch(/detached update FAILED|previous version/)
      // The relaunched app reports the result while posix.sh is still inside launch_app's 1.5 s
      // acceptance window, so the updater gets a bounded moment to exit instead of none.
      await expect
        .poll(
          () =>
            installProcesses(facts)
              .filter(proc => /desktop-update\/posix\.sh| update --yes/.test(proc.cmdline))
              .map(p => p.cmdline),
          { timeout: 30_000, message: 'no updater is left running after the failure' }
        )
        .toEqual([])
      // The checkout moved (that is why it is not "previous version"); the app bundle did not.
      expect(git(facts.checkout, 'rev-parse', 'HEAD'), 'the code update committed').toBe(target)
    })
  } finally {
    await session.close()
  }
})
