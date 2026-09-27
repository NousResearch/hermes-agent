/**
 * Desktop RTL visual regression — the Electron renderer under Persian.
 *
 * Boots the app with `display.language: fa` in config.yaml (the persisted
 * locale the I18nProvider applies at startup) and pins the renderer's
 * RTL behavior:
 *
 *  - `applyDocumentLocale('fa')` must set `html[dir="rtl"][lang="fa"]`
 *    (checked with `expectRtlBoot` below so a direction flip or a locale
 *    fallback to English fails the test even though visual diffs are
 *    soft-failed elsewhere).
 *  - `html[dir='rtl'] body` swaps in the Vazirmatn webfont stack and
 *    mirrors the glass sidebar — the font half is asserted via CSS.
 *  - Key surfaces (session view, settings) are snapshotted against
 *    committed baselines with the repo's soft-diff policy
 *    (`expectVisualSnapshot`): diffs never fail the suite, they surface
 *    in CI artifacts for human review.
 *
 * Hard assertions (locale + direction + font + real fa anchors) gate;
 * pixels only report. Baselines live in `rtl-snapshots.spec.ts-snapshots/`
 * next to this file and are generated on `main` via `--update-snapshots`
 * (same policy as boot.spec.ts and e2e-desktop.yml).
 *
 * NOTE on readiness: `waitForAppReady` is called exactly once (first test).
 * After the app is ready it must NOT be re-polled — the translucent glass
 * tint installs a full-viewport `position: fixed` layer once connected,
 * which the overlay-center heuristic reads as "still booting". Later tests
 * wait on concrete anchors instead (html attrs, composer, pane buttons).
 *
 * Run: `npx playwright test e2e/rtl-snapshots.spec.ts` from apps/desktop.
 */
import type { MockBackendFixture } from './fixtures'
import {
  setupMockBackend,
  waitForAppReady,
} from './fixtures'
import { expect, test } from './test'
import { expectVisualSnapshot } from './visual-snapshot'

let fixture: MockBackendFixture | null = null

test.beforeAll(async () => {
  // display.language: fa — the exact config the I18nProvider reads at
  // startup (same path a real Persian user's config.yaml takes).
  fixture = await setupMockBackend({ extraDisplayConfig: '  language: fa' })
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

/** Hard gate: renderer actually booted Persian RTL with the fa webfont. */
async function expectRtlBoot(page: MockBackendFixture['page']): Promise<void> {
  await expect(page.locator('html')).toHaveAttribute('dir', 'rtl')
  await expect(page.locator('html')).toHaveAttribute('lang', 'fa')

  // Vazirmatn rides the font stack only under html[dir='rtl'] (styles.css).
  const font = await page.evaluate(() => getComputedStyle(document.body).fontFamily)
  expect(font).toContain('Vazirmatn')
}

test.describe('desktop Persian RTL', () => {
  test('boots with html[dir=rtl][lang=fa] and Vazirmatn', async () => {
    const page = fixture!.page
    await waitForAppReady(fixture!)
    await expectRtlBoot(page)
  })

  test('main session view is mirrored (baseline)', async () => {
    const page = fixture!.page
    // App is already ready (test 1) — anchor on the always-present composer.
    await page.waitForSelector('textarea, [contenteditable="true"]', {
      state: 'attached',
      timeout: 30_000,
    })
    await expectRtlBoot(page)
    await page.waitForLoadState('networkidle')
    await expectVisualSnapshot(page, {
      app: fixture!.app,
      name: 'fa-session-view',
    })
  })

  test('settings pane renders Persian (baseline)', async () => {
    const page = fixture!.page

    // Settings opens via the nav rail; label comes from the fa catalog.
    const settings = page
      .getByRole('button', { name: /تنظیمات/ })
      .or(page.getByRole('tab', { name: /تنظیمات/ }))
      .first()

    await settings.click()
    // The open pane exposes its close control with the fa aria-label —
    // concrete proof the pane actually rendered in Persian.
    await expect(
      page.getByRole('button', { name: 'بستن تنظیمات', exact: true }),
    ).toBeVisible({ timeout: 15_000 })
    await page.waitForLoadState('networkidle')
    await expectVisualSnapshot(page, {
      app: fixture!.app,
      name: 'fa-settings',
    })
  })
})
