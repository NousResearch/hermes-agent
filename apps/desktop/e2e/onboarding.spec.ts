/**
 * E2E onboarding tests — verify the provider picker appears when no
 * inference provider is configured.
 *
 * Launches the app with an empty config.yaml (no providers). The renderer
 * should detect the unconfigured state and show the DesktopOnboardingOverlay
 * with provider options / API key form.
 *
 * Prerequisite: `npm run build` must have been run so dist/ exists.
 */

import { test } from './test'

import {
  type NoProviderFixture,
  setupNoProvider,
  waitForOnboarding,
} from './fixtures'
import { expectVisualSnapshot } from './visual-snapshot'

let fixture: NoProviderFixture | null = null

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

test.describe('onboarding with no provider configured', () => {
  test('onboarding overlay appears on first boot', async () => {
    fixture = await setupNoProvider()

    // The app should boot (hermes serve starts fine even without a provider),
    // but the renderer should show the onboarding overlay because no
    // provider is configured.
    await waitForOnboarding(fixture.page, 90_000)
  })

  test('screenshot of onboarding overlay', async () => {
    if (!fixture) {
      test.skip(true, 'Previous test failed — no app running')

      return
    }

    await expectVisualSnapshot(fixture.page, { name: 'onboarding-overlay', app: fixture.app })
  })
})
