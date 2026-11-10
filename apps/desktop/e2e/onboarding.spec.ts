/**
 * E2E onboarding tests — the deterministic guide renders without an inference provider.
 * Provider setup remains an explicit choice from the model step.
 *
 * Prerequisite: `pnpm run build` must have been run so dist/ exists.
 */

import { type NoProviderFixture, setupNoProvider } from './fixtures'
import { expect, test } from './test'
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
