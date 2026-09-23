/**
 * E2E: Interface mode round trip against a real backend.
 *
 * Simple is a resolver input, not a preference writer. The scenario walks the
 * Advanced → Simple → Advanced loop from the layout editor and asserts what an
 * install would notice: the instrumentation leaves (titlebar tools beyond
 * Settings + Layout editor, the statusbar, the machinery rows in the sidebar),
 * the one-key reveal still opens the terminal for the session, and on the way
 * back every stored preference is byte-identical — nothing under
 * `hermes.desktop.*` changed except the mode key itself, which is gone again.
 *
 * Prerequisite: `npm run build` must have been run so dist/ exists.
 */
import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import { expect, test } from './test'

const MODE_KEY = 'hermes.desktop.interfaceMode.v1'

let fixture: MockBackendFixture | null = null

test.beforeAll(async () => {
  fixture = await setupMockBackend()
  await waitForAppReady(fixture)
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

/** Every `hermes.desktop.*` record except the mode key — the user's actual preferences. */
async function preferenceSnapshot(page: MockBackendFixture['page']): Promise<Record<string, string>> {
  return page.evaluate((modeKey: string) => {
    const out: Record<string, string> = {}
    for (let i = 0; i < window.localStorage.length; i += 1) {
      const key = window.localStorage.key(i)
      if (key && key.startsWith('hermes.desktop.') && key !== modeKey) {
        out[key] = window.localStorage.getItem(key) ?? ''
      }
    }
    return out
  }, MODE_KEY)
}

async function pickMode(page: MockBackendFixture['page'], mode: 'Simple' | 'Advanced'): Promise<void> {
  await page.getByRole('button', { name: 'Layout editor', exact: true }).click()
  const card = page.getByRole('button', { name: new RegExp(`^${mode}\\b`) })
  await expect(card).toBeVisible()
  await card.click()
  await expect(card).toHaveAttribute('aria-pressed', 'true')
  await page.getByRole('button', { name: 'Done', exact: true }).click()
}

test('Simple hides the machinery, keeps the doors, and leaves preferences untouched on the way back', async () => {
  const page = fixture!.page
  const appControls = page.getByRole('toolbar', { name: 'App controls' }).or(page.locator('[aria-label="App controls"]')).first()

  // Advanced (default): the usual chrome, and no mode record on disk.
  await expect(appControls.getByRole('button', { name: 'HUD mode', exact: true })).toBeVisible()
  await expect(page.locator('[data-slot="statusbar"]')).toBeVisible()
  expect(await page.evaluate((key: string) => window.localStorage.getItem(key), MODE_KEY)).toBeNull()

  const before = await preferenceSnapshot(page)

  await pickMode(page, 'Simple')

  // Titlebar: only Settings and the layout editor remain among the app actions.
  await expect(appControls.getByRole('button', { name: 'HUD mode', exact: true })).toHaveCount(0)
  await expect(appControls.getByRole('button', { name: 'Swap sidebar sides', exact: true })).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Open settings', exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Layout editor', exact: true })).toBeVisible()

  // Statusbar and the sidebar's machinery rows are gone; the conversation isn't.
  await expect(page.locator('[data-slot="statusbar"]')).toHaveCount(0)
  await expect(page.getByRole('button', { name: /^Scheduled jobs/ })).toHaveCount(0)
  await expect(page.getByRole('button', { name: /^Capabilities/ })).toHaveCount(0)
  await expect(page.locator('textarea, [contenteditable="true"]').first()).toBeVisible()

  // Simple is a default, not a lock: the terminal reveal still answers for the session.
  await expect(page.locator('[data-terminal]').first()).toBeHidden()
  await page.keyboard.press('Control+`')
  await expect(page.locator('[data-terminal]').first()).toBeVisible()

  // Simple wrote exactly one thing: the mode key.
  expect(await page.evaluate((key: string) => window.localStorage.getItem(key), MODE_KEY)).not.toBeNull()
  expect(await preferenceSnapshot(page)).toEqual(before)

  await pickMode(page, 'Advanced')

  // Everything is back, the session reveal is cleared, and the disk looks like we were never here.
  await expect(appControls.getByRole('button', { name: 'HUD mode', exact: true })).toBeVisible()
  await expect(page.locator('[data-slot="statusbar"]')).toBeVisible()
  await expect(page.locator('[data-terminal]').first()).toBeHidden()
  expect(await page.evaluate((key: string) => window.localStorage.getItem(key), MODE_KEY)).toBeNull()
  expect(await preferenceSnapshot(page)).toEqual(before)
})
