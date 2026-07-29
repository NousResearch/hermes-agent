/**
 * E2E smoke tests for the dev-mode desktop app.
 *
 * These tests launch the Electron app from the built dist/ (not the
 * packaged binary) with a real `hermes serve` backend pointed at a mock
 * inference server. The full chain is exercised:
 *
 *   electron → hermes serve (python) → mock provider → renderer
 *
 * Prerequisite: `npm run build` must have been run so dist/ exists.
 * Run from the nix devshell:
 *   npm exec playwright test e2e/boot.spec.ts --reporter=list
 */
import { expect, test } from './test'

import {
  type MockBackendFixture,
  setupMockBackend,
  waitForAppReady,
} from './fixtures'
import { expectVisualSnapshot } from './visual-snapshot'

let fixture: MockBackendFixture | null = null

test.beforeAll(async () => {
  fixture = await setupMockBackend()
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

test.describe('dev-mode boot with mock backend', () => {
  test('window opens with Hermes title', async () => {
    const title = await fixture!.page.title()
    expect(title).toContain('Hermes')
  })

  test('renderer mounts and shows DOM content', async () => {
    const page = fixture!.page
    // Wait for the React root to mount. The app renders into #root
    // (see src/main.tsx), but content may arrive through portals — so
    // check the body for any interactive content instead.
    await page.waitForSelector('body', { state: 'attached' })
    // Wait for the main app shell — the composer is always present.
    await page.waitForSelector('textarea, [contenteditable="true"]', {
      state: 'attached',
      timeout: 30_000,
    })
  })

  test('backend boots and app becomes ready', async () => {
    // This is the big one — wait for the full boot chain to complete:
    // electron starts → hermes serve is spawned → WS connects → config
    // loaded → sessions loaded → boot overlay dismissed → composer visible.
    await waitForAppReady(fixture!, 120_000)
  })

  test('screenshot after boot', async () => {
    await expectVisualSnapshot(fixture!.page, { name: 'boot-ready', app: fixture!.app })
  })
})
