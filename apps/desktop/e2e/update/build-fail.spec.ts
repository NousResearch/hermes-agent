/**
 * Failure class: APP-DRIVEN UPDATE whose Desktop build fails.
 *
 * #124040: the Desktop build failed inside the update takeover's completion
 * stage and the failure was swallowed / kept looping instead of reaching the
 * user. Here upstream main carries a Desktop source file that does not
 * compile; the user clicks "Update now". The update must end with the app
 * telling the user the update FAILED (the relaunched app's result dialog is
 * logged as `[updates] detached update FAILED`), never "finished OK", and the
 * updater must not be left running.
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

test('a Desktop build failure during Update now is reported to the user as a failed update', async () => {
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

    await test.step('the failed build reaches the user as a failed update', async () => {
      await waitFor(
        'the relaunched app to report the update outcome',
        () => /\[updates\] detached update (finished|FAILED)/.test(desktopLog(facts)),
        { timeout: 12 * 60_000, interval: 2_000, explain }
      )

      const outcome = desktopLog(facts)
        .split('\n')
        .filter(line => /\[updates\] detached update/.test(line))
        .join('\n')

      expect(outcome, `the update is reported as FAILED, not finished\n${explain()}`).toMatch(/detached update FAILED/)
      expect(outcome).not.toMatch(/detached update finished OK/)
      expect(
        installProcesses(facts)
          .filter(proc => /desktop-update\/posix\.sh| update --yes/.test(proc.cmdline))
          .map(p => p.cmdline),
        'no updater is left running after the failure'
      ).toEqual([])
      // What the user is left on: record it for triage (the checkout moved; the app bundle did not).
      test
        .info()
        .annotations.push({ type: 'checkout-after-failure', description: git(facts.checkout, 'rev-parse', 'HEAD') })
    })
  } finally {
    await session.close()
  }
})
