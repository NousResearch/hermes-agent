/**
 * E2E boot-failure tests — verify the app shows an error overlay when the
 * backend can't start.
 *
 * Injects a fake boot error (HERMES_DESKTOP_BOOT_FAKE_ERROR) so the backend
 * resolution fails with a controlled error message. The app should show the
 * BootFailureOverlay with retry/repair actions.
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

test.describe('boot failure with dead backend', () => {
  test('app shows error state', async () => {
    // Inject a fake boot error so the backend resolution "fails" with a
    // controlled error message. This is the only reliable way to trigger
    // BootFailureOverlay in dev mode.
    fixture = await setupDeadBackend({ fakeError: true })

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
