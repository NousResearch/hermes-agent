/**
 * E2E: Theme Scope (Settings → Appearance) on a device with several profiles.
 *
 * Skin and light/dark mode are saved per profile, so switching profiles (or
 * gateways) switches the look. "Shared" routes every profile at the one
 * global slot instead. What would bite a user if it regressed:
 *
 *  - flipping to Shared repaints the window to some OTHER profile's theme
 *    (the live look must be promoted, not replaced);
 *  - under Shared a profile switch still repaints per profile;
 *  - flipping back to Per profile loses the assignments profiles had before
 *    (they must be untouched — Shared only ever writes the global slot);
 *  - the choice does not survive a relaunch.
 *
 * Profile switches are driven through the real statusbar rail
 * (`[data-slot="profile-rail"]` squares / home pill), which re-homes the live
 * gateway on the picked profile's backend. The painted theme is read straight
 * off `document.documentElement` (`data-hermes-theme`, `data-hermes-mode`, the
 * `.dark` class — see applyTheme in src/themes/context.tsx) and cross-checked
 * against the localStorage slots the context persists to.
 *
 * Prerequisite: `npm run build` (dist/) and the repo venv (.venv).
 */

import * as fs from 'node:fs'
import * as path from 'node:path'

import {
  buildAppEnv,
  createSandbox,
  launchDesktop,
  type MockBackendFixture,
  type Sandbox,
  waitForAppReady,
  writeEnvFile,
  writeMockProviderConfig,
} from './fixtures'
import { startMockServer } from './mock-server'
import { type ElectronApplication, expect, type Page, test } from './test'

// localStorage slots owned by src/themes/context.tsx.
const SKIN_KEY = 'hermes-desktop-theme-v2'
const MODE_KEY = 'hermes-desktop-mode-v1'
const PROFILE_SKINS_KEY = 'hermes-desktop-profile-themes-v1'
const PROFILE_MODES_KEY = 'hermes-desktop-profile-modes-v1'
const SCOPE_KEY = 'hermes-desktop-theme-scope-v1'
// Written by ThemeProvider whenever the profile it paints for changes — the
// most direct "the theme layer has seen the switch" signal there is.
const LAST_PROFILE_KEY = 'hermes-desktop-active-profile-v1'

const APPEARANCE_ROUTE = '/settings?tab=config%3Aappearance'

/**
 * Seed `<home>/profiles/<name>/` so the backend's /api/profiles lists it. Each
 * profile gets the same mock provider as the primary home, so re-homing on it
 * brings up a gateway that reports "ready" rather than "inference unavailable".
 */
function seedProfiles(home: string, mockUrl: string, names: string[]): void {
  for (const name of names) {
    const dir = path.join(home, 'profiles', name)
    fs.mkdirSync(dir, { recursive: true })
    writeMockProviderConfig(dir, mockUrl)
    writeEnvFile(dir)
  }
}

const storage = (page: Page, key: string) => page.evaluate(k => window.localStorage.getItem(k), key)

const storageRecord = async (page: Page, key: string): Promise<Record<string, string>> => {
  const raw = await storage(page, key)

  return raw ? (JSON.parse(raw) as Record<string, string>) : {}
}

const html = (page: Page) => page.locator('html')
const rail = (page: Page) => page.locator('[data-slot="profile-rail"]')
const scopeRow = (page: Page) => page.locator('[id="setting-field-appearance.theme-scope"]')
const themeRow = (page: Page) => page.locator('[id="setting-field-appearance.theme"]')
const scopeButton = (page: Page, label: 'Per profile' | 'Shared') => scopeRow(page).getByRole('button', { name: label, exact: true })
const modeButton = (page: Page, label: 'Light' | 'Dark' | 'System') => themeRow(page).getByRole('button', { name: label, exact: true })

// A card's text is label + description, so match the label node exactly.
const themeCard = (page: Page, label: string) =>
  themeRow(page).getByRole('button').filter({ has: page.getByText(label, { exact: true }) })

async function waitReady(page: Page): Promise<void> {
  await expect(page.locator('[data-slot="statusbar"]').getByText('ready', { exact: true })).toBeVisible({ timeout: 120_000 })
}

async function gotoAppearance(page: Page): Promise<void> {
  await page.evaluate(route => {
    window.location.hash = route
  }, APPEARANCE_ROUTE)
  await expect(scopeRow(page)).toBeVisible({ timeout: 60_000 })
}

async function leaveSettings(page: Page): Promise<void> {
  await page.evaluate(() => {
    window.location.hash = '/'
  })
  await expect(page.locator('[data-overlay-surface]')).toHaveCount(0, { timeout: 30_000 })
}

/**
 * Re-home the live gateway on `name` through the rail, then wait until both
 * the rail and the theme layer agree the switch landed. A switch starts a
 * fresh session (leaves Settings), so callers re-open Appearance after.
 */
async function switchProfile(page: Page, name: 'default' | 'work' | 'research'): Promise<void> {
  test.setTimeout(180_000)

  // Settings is a full-window overlay that sits over the statusbar rail —
  // leave it first, the way a user would, so the click reaches a square.
  await leaveSettings(page)

  if (name === 'default') {
    // Off `default` the pinned pill is the way home; on it, it's the all-profiles toggle.
    await rail(page).getByRole('button', { name: 'Switch to default', exact: true }).click()
  } else {
    await rail(page).getByRole('button', { name, exact: true }).click()
  }

  if (name !== 'default') {
    await expect(rail(page).getByRole('button', { name, exact: true })).toHaveAttribute('aria-pressed', 'true', { timeout: 120_000 })
  }

  await expect.poll(() => storage(page, LAST_PROFILE_KEY), { timeout: 120_000 }).toBe(name)
  await waitReady(page)
}

const expectSkin = (page: Page, skin: string) => expect(html(page)).toHaveAttribute('data-hermes-theme', skin, { timeout: 30_000 })

test.describe('theme scope — per-profile vs shared appearance', () => {
  test.describe.configure({ mode: 'serial' })

  let mock: Awaited<ReturnType<typeof startMockServer>>
  let sandbox: Sandbox
  let app: ElectronApplication
  let page: Page

  async function launch(): Promise<void> {
    ;({ app, page } = await launchDesktop(buildAppEnv(sandbox)))
    await waitForAppReady({ app, page } as MockBackendFixture, 120_000)
    await waitReady(page)
    // Let the boot-time profile restore settle before any click lands.
    await page.waitForTimeout(1_500)
  }

  test.beforeAll(async () => {
    test.setTimeout(240_000)
    mock = await startMockServer()
    sandbox = createSandbox('theme-scope')
    writeMockProviderConfig(sandbox.hermesHome, mock.url)
    writeEnvFile(sandbox.hermesHome)
    seedProfiles(sandbox.hermesHome, mock.url, ['work', 'research'])
    await launch()
  })

  test.afterAll(async () => {
    await app?.close().catch(() => undefined)
    await mock?.close()
    sandbox?.cleanup()
  })

  test('shows the scope row, Per profile by default, captioned with the live profile', async () => {
    await gotoAppearance(page)

    await expect(scopeButton(page, 'Per profile')).toHaveAttribute('aria-pressed', 'true')
    await expect(scopeButton(page, 'Shared')).toHaveAttribute('aria-pressed', 'false')
    await expect(scopeRow(page)).toContainText('Each profile and gateway remembers its own theme')
    await expect(themeRow(page)).toContainText('Saved for the default profile')

    // Nothing persisted yet: a fresh install is per-profile without a key.
    expect(await storage(page, SCOPE_KEY)).toBeNull()
  })

  test('per profile: each profile keeps its own skin across switches', async () => {
    test.setTimeout(300_000)

    // `default` IS the global slot — its pick lands on the legacy key.
    await themeCard(page, 'Catppuccin').click()
    await expectSkin(page, 'catppuccin')
    // Pin a light base so the Dark pick later is a real change, not a coin flip on the OS.
    await modeButton(page, 'Light').click()
    await expect(html(page)).toHaveAttribute('data-hermes-mode', 'light')
    expect(await storage(page, SKIN_KEY)).toBe('catppuccin')
    expect(await storage(page, MODE_KEY)).toBe('light')

    // `work` has no assignment yet, so it inherits the global.
    await switchProfile(page, 'work')
    await expectSkin(page, 'catppuccin')
    await gotoAppearance(page)
    await expect(themeRow(page)).toContainText('Saved for the work profile')

    await themeCard(page, 'Everforest').click()
    await expectSkin(page, 'everforest')
    expect(await storageRecord(page, PROFILE_SKINS_KEY)).toEqual({ work: 'everforest' })
    // The global slot is untouched by a named profile's pick.
    expect(await storage(page, SKIN_KEY)).toBe('catppuccin')

    await switchProfile(page, 'default')
    await expectSkin(page, 'catppuccin')

    await switchProfile(page, 'work')
    await expectSkin(page, 'everforest')
  })

  test('flipping to Shared promotes the live look without repainting', async () => {
    await gotoAppearance(page)
    await expectSkin(page, 'everforest')

    await scopeButton(page, 'Shared').click()
    await expect(scopeButton(page, 'Shared')).toHaveAttribute('aria-pressed', 'true')

    // The window must not change under the user: still `work`'s Everforest,
    // and it stays that way once every effect has settled.
    await expectSkin(page, 'everforest')
    await page.waitForTimeout(500)
    await expectSkin(page, 'everforest')

    await expect(scopeRow(page)).toContainText('One theme and light/dark mode for the whole app')
    await expect(themeRow(page)).toContainText('Shared across every profile and gateway')

    expect(await storage(page, SCOPE_KEY)).toBe('shared')
    // Promotion: the live profile's skin + mode are now the global slot…
    expect(await storage(page, SKIN_KEY)).toBe('everforest')
    expect(await storage(page, MODE_KEY)).toBe('light')
    // …and its own per-profile entry is left as it was.
    expect(await storageRecord(page, PROFILE_SKINS_KEY)).toEqual({ work: 'everforest' })
  })

  test('under Shared every profile paints the one global skin', async () => {
    test.setTimeout(300_000)

    await switchProfile(page, 'default')
    await expectSkin(page, 'everforest')

    await gotoAppearance(page)
    await themeCard(page, 'Solarized').click()
    await expectSkin(page, 'solarized')
    expect(await storage(page, SKIN_KEY)).toBe('solarized')

    await switchProfile(page, 'work')
    await expectSkin(page, 'solarized')

    await switchProfile(page, 'research')
    await expectSkin(page, 'solarized')

    // Shared only ever writes the global slot: no per-profile entry moved.
    expect(await storageRecord(page, PROFILE_SKINS_KEY)).toEqual({ work: 'everforest' })
  })

  test('flipping back to Per profile restores each profile\'s own assignment', async () => {
    test.setTimeout(300_000)

    await gotoAppearance(page)
    await scopeButton(page, 'Per profile').click()
    await expect(scopeButton(page, 'Per profile')).toHaveAttribute('aria-pressed', 'true')
    expect(await storage(page, SCOPE_KEY)).toBe('per-profile')
    await expect(themeRow(page)).toContainText('Saved for the research profile')

    // `research` was never assigned, so it inherits the global — now Solarized.
    await expectSkin(page, 'solarized')
    await page.waitForTimeout(500)
    await expectSkin(page, 'solarized')

    // `work` gets its earlier Everforest back.
    await switchProfile(page, 'work')
    await expectSkin(page, 'everforest')

    // `default` is the global slot itself, so it keeps what Shared last wrote there.
    await switchProfile(page, 'default')
    await expectSkin(page, 'solarized')
    expect(await storage(page, SKIN_KEY)).toBe('solarized')
    expect(await storageRecord(page, PROFILE_SKINS_KEY)).toEqual({ work: 'everforest' })
  })

  test('light/dark mode follows the same scope', async () => {
    test.setTimeout(300_000)

    await gotoAppearance(page)
    await expect(html(page)).toHaveAttribute('data-hermes-mode', 'light')

    // Back to Shared from `default` (Solarized / light): a no-op promotion.
    await scopeButton(page, 'Shared').click()
    await expect(scopeButton(page, 'Shared')).toHaveAttribute('aria-pressed', 'true')
    await expectSkin(page, 'solarized')

    await modeButton(page, 'Dark').click()
    await expect(html(page)).toHaveAttribute('data-hermes-mode', 'dark')
    await expect(html(page)).toHaveClass(/\bdark\b/)
    expect(await storage(page, MODE_KEY)).toBe('dark')
    expect(await storageRecord(page, PROFILE_MODES_KEY)).toEqual({})

    await switchProfile(page, 'work')
    await expect(html(page)).toHaveAttribute('data-hermes-mode', 'dark', { timeout: 30_000 })
    await expect(html(page)).toHaveClass(/\bdark\b/)
    await expectSkin(page, 'solarized')

    await switchProfile(page, 'research')
    await expect(html(page)).toHaveAttribute('data-hermes-mode', 'dark', { timeout: 30_000 })
    await expect(html(page)).toHaveClass(/\bdark\b/)
    await expectSkin(page, 'solarized')
  })

  test('the shared look survives a relaunch', async () => {
    test.setTimeout(300_000)

    await app.close()
    await launch()

    expect(await storage(page, SCOPE_KEY)).toBe('shared')
    await expectSkin(page, 'solarized')
    await expect(html(page)).toHaveAttribute('data-hermes-mode', 'dark')
    await expect(html(page)).toHaveClass(/\bdark\b/)

    await gotoAppearance(page)
    await expect(scopeButton(page, 'Shared')).toHaveAttribute('aria-pressed', 'true')
    await expect(themeRow(page)).toContainText('Shared across every profile and gateway')
  })
})
