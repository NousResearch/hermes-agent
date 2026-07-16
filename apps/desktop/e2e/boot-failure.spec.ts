/**
 * E2E boot-failure tests — verify the app shows an error overlay when the
 * backend can't reach the inference provider.
 *
 * Launches the app with a provider pointing at a dead endpoint (port 1).
 * The `hermes serve` backend starts, but when the renderer tries to connect
 * or when a runtime check fails, the app should show a boot failure or
 * onboarding error overlay.
 *
 * Prerequisite: `npm run build` must have been run so dist/ exists.
 */

import { test } from '@playwright/test'

import {
  type DeadBackendFixture,
  setupDeadBackend,
  waitForBootFailure,
} from './fixtures'
import { expectVisualSnapshot } from './visual-snapshot'

let fixture: DeadBackendFixture | null = null

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

test.describe('boot failure with dead provider endpoint', () => {
  test('app shows error state or onboarding', async () => {
    fixture = await setupDeadBackend()

    // With a dead provider endpoint, the app should eventually show either:
    //  1. A boot failure overlay (if the backend fails to start), or
    //  2. An onboarding overlay with an error (if the runtime check fails)
    // Both outcomes prove the app is handling provider failures gracefully.
    //
    // We give it a generous timeout — the backend needs to start, the
    // renderer needs to boot, and then the runtime check needs to fail.
    await waitForBootFailure(fixture.page, 90_000)
  })

  test('screenshot of error state', async () => {
    if (!fixture) {
      test.skip(true, 'Previous test failed — no app running')

      return
    }

    await expectVisualSnapshot(fixture!.page, { name: 'boot-failure-error-state', app: fixture.app })
  })
})
