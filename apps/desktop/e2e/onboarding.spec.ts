/**
 * E2E onboarding tests. The redesign's deterministic guide renders without an inference provider,
 * provider setup remaining an explicit choice from the model step. The classic shell (the redesign
 * switched off) still shows the provider picker when no inference provider is configured.
 *
 * Prerequisite: `pnpm run build` must have been run so dist/ exists.
 */

import { type NoProviderFixture, setupNoProvider, waitForOnboarding } from './fixtures'
import { expect, test } from './test'
import { CLASSIC_SHELL } from './ui-preferences-seed'
import { expectVisualSnapshot } from './visual-snapshot'

let fixture: NoProviderFixture | null = null

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

test.describe('onboarding with no provider configured', () => {
  test('onboarding overlay appears on first boot', async () => {
    fixture = await setupNoProvider()

    const guide = fixture.page.getByRole('dialog')
    await expect(guide.getByRole('heading', { name: 'Set up Hermes' })).toBeVisible({ timeout: 90_000 })
    await guide.getByRole('button', { name: 'Continue', exact: true }).click()
    await expect(guide.getByRole('heading', { name: 'Choose a model', exact: true })).toBeVisible()
    await expect(guide.getByRole('button', { name: 'Finish setup', exact: true })).toBeDisabled()
    await expect(guide.getByRole('button', { name: 'Connect a provider', exact: true })).toBeEnabled()
  })

  test('screenshot of onboarding overlay', async () => {
    if (!fixture) {
      test.skip(true, 'Previous test failed — no app running')

      return
    }

    await expectVisualSnapshot(fixture.page, { name: 'onboarding-overlay', app: fixture.app })
  })
})

test.describe('classic shell onboarding with no provider configured', () => {
  let classic: NoProviderFixture | null = null

  test.afterAll(async () => {
    await classic?.cleanup()
    classic = null
  })

  test('onboarding overlay appears on first boot', async () => {
    classic = await setupNoProvider(CLASSIC_SHELL)

    // The app should boot (hermes serve starts fine even without a provider),
    // but the renderer should show the onboarding overlay because no
    // provider is configured.
    await waitForOnboarding(classic.page, 90_000)
    await expect(classic.page.locator('html')).not.toHaveAttribute('data-shell', /.*/)
  })
})
