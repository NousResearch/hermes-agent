/**
 * E2E first boot without a provider.
 *
 * A fresh desktop install claims the Nous welcome trial and opens the first-run setup chat on it
 * (`tui_gateway/methods_onboarding.py` `onboarding.start`). The claim creates a guest account at
 * the Nous portal, so a test never makes it: the sandbox switches guests off (`nous.guest: false`,
 * `no-provider-sandbox.ts`). With no trial and no provider the first boot must land on provider
 * setup instead of a setup chat that has no model, and nothing may leave the machine.
 *
 * Prerequisite: `pnpm run build` must have been run so dist/ exists.
 */

import * as fs from 'node:fs'
import * as path from 'node:path'

import { type NoProviderFixture, providerSetup, setupNoProvider } from './fixtures'
import { nousAccountAttempts } from './no-provider-sandbox'
import { expect, test } from './test'
import { expectVisualSnapshot } from './visual-snapshot'

let fixture: NoProviderFixture | null = null

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

/** The sandbox profile holds a Nous identity (a minted guest is saved before anything else). */
function hasNousIdentity(hermesHome: string): boolean {
  const authPath = path.join(hermesHome, 'auth.json')

  if (!fs.existsSync(authPath)) {
    return false
  }

  const store = JSON.parse(fs.readFileSync(authPath, 'utf8')) as { providers?: Record<string, unknown> }

  return Boolean(store.providers?.nous)
}

/**
 * Nothing reached for a Nous account: no sandbox process, the app included, even tried a Nous
 * account or inference host. Everything else the sandbox asked for off the machine (the catalog
 * and update fetches a backend makes at boot, the theme's web font) reached only the fixture's
 * recorder, which refused it; it is listed on the test, not failed on.
 */
function expectStayedLocal({ egress }: NoProviderFixture): void {
  const attempts = egress()

  test.info().annotations.push({ type: 'refused egress', description: attempts.join(', ') || 'none' })
  expect(nousAccountAttempts(attempts)).toEqual([])
}

test.describe('first boot with no provider configured', () => {
  test('lands on provider setup without reaching the network', async () => {
    fixture = await setupNoProvider()
    const { page, sandbox } = fixture

    // hermes serve boots without a provider; the renderer finds nothing to chat on.
    await expect(providerSetup(page)).toHaveCount(1, { timeout: 90_000 })

    // No setup chat was opened on a trial this sandbox cannot have.
    await expect(page.getByRole('button', { name: 'Skip setup' })).toHaveCount(0)
    await expect(page.getByText('What should I call you?')).toHaveCount(0)

    expect(hasNousIdentity(sandbox.hermesHome)).toBe(false)
    expectStayedLocal(fixture)
  })

  test('screenshot of provider setup', async () => {
    if (!fixture) {
      test.skip(true, 'Previous test failed — no app running')

      return
    }

    // Photograph the surface the first test proved, never whatever a failed boot left on screen.
    await expect(providerSetup(fixture.page)).toHaveCount(1)
    await expectVisualSnapshot(fixture.page, { name: 'onboarding-overlay', app: fixture.app })
    expectStayedLocal(fixture)
  })
})
