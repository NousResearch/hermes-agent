/**
 * E2E: the All-profiles view names each chat's owner.
 *
 * One gateway — the Electron-managed local backend with mock inference —
 * carrying three named profiles beside default, each holding real sessions
 * the real agent wrote. Switching the sidebar to "Show all profiles" merges
 * them into one list: every one-line row from a named profile leads with that
 * profile's name, default-profile rows carry no mark, and the mark takes the
 * profile's colour while the row is hovered or selected.
 *
 * Prerequisite: `npm run build` must have been run so dist/ exists, and the
 * repo's Python venv (`.venv`) must exist (and be active for `uv run`) for
 * the backend and the session builder.
 *
 * OWNER_LEAD_SCREENSHOT_DIR=<dir> saves sidebar captures at the key states
 * (single scope, merged at rest, hovered, selected) for design review; never
 * part of the assertions. OWNER_LEAD_SCREENSHOT_ONLY=1 skips the lead
 * assertions so the same spec can capture a build that predates the feature.
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
import { RealSessionBuilder } from './real-session-builder'
import { type ElectronApplication, expect, type Page, test } from './test'

// Fictional profiles and chats — nothing here comes from a real install.
const DEFAULT_SESSIONS = ['Plan the week', 'Tidy the downloads folder']
const PROFILE_SESSIONS: Record<string, string[]> = {
  inbox: ['Draft the release notes', 'Reply to the vendor thread'],
  ops: ['Check the deploy logs', 'Rotate the staging certificate'],
  research: ['Compare the two vector stores'],
}

const screenshotOnly = process.env.OWNER_LEAD_SCREENSHOT_ONLY === '1'

/** Seed `<home>/profiles/<name>/` so the backend's /api/profiles lists it. */
function seedProfiles(home: string, names: string[]): void {
  for (const name of names) {
    const dir = path.join(home, 'profiles', name)
    fs.mkdirSync(dir, { recursive: true })
    fs.writeFileSync(path.join(dir, 'config.yaml'), '', 'utf8')
  }
}

/** Real sessions in one profile home, written by the real agent against the
 *  mock provider. The first turn doubles as the title the sidebar shows. */
async function seedSessions(home: string, mockUrl: string, titles: string[]): Promise<void> {
  writeMockProviderConfig(home, mockUrl)
  writeEnvFile(home)
  const builder = await RealSessionBuilder.start(home)

  try {
    for (const title of titles) {
      await builder.createSession({ title, turns: [title] })
    }
  } finally {
    await builder.close()
  }
}

test.describe('All-profiles view — owner lead on every row', () => {
  test.describe.configure({ mode: 'serial' })

  let mock: Awaited<ReturnType<typeof startMockServer>>
  let sandbox: Sandbox
  let app: ElectronApplication
  let page: Page

  const sidebar = () => page.locator('[data-slot="sidebar"]')
  const row = (title: string) => sidebar().locator('button').filter({ hasText: title }).first()
  const leads = () => sidebar().locator('[data-profile-lead]')

  // Off the list, over the empty chat pane — Electron reports no viewport
  // size, so the window's known 1220×800 bounds stand in for it.
  const parkPointer = () => page.mouse.move(900, 400)

  async function capture(name: string): Promise<void> {
    const dir = process.env.OWNER_LEAD_SCREENSHOT_DIR

    if (!dir) {
      return
    }

    fs.mkdirSync(dir, { recursive: true })
    await sidebar().screenshot({ path: path.join(dir, `${name}.png`) })
  }

  test.beforeAll(async () => {
    test.setTimeout(300_000)
    mock = await startMockServer()
    sandbox = createSandbox('owner-lead')
    seedProfiles(sandbox.hermesHome, Object.keys(PROFILE_SESSIONS))
    await seedSessions(sandbox.hermesHome, mock.url, DEFAULT_SESSIONS)

    for (const [name, titles] of Object.entries(PROFILE_SESSIONS)) {
      await seedSessions(path.join(sandbox.hermesHome, 'profiles', name), mock.url, titles)
    }

    ;({ app, page } = await launchDesktop(buildAppEnv(sandbox)))
    await waitForAppReady({ app, page } as MockBackendFixture, 120_000)
    await expect(page.locator('[data-slot="statusbar"]').getByText('ready', { exact: true })).toBeVisible({ timeout: 120_000 })
    // Playwright-driven Electron reports no hover-capable pointer on some
    // Linux desktops, which switches off every `@media (hover: hover)` rule —
    // the sidebar's whole hover vocabulary, this feature's included. Declare
    // one so the hover state is real and testable everywhere.
    const cdp = await page.context().newCDPSession(page)
    await cdp.send('Emulation.setEmulatedMedia', {
      features: [
        { name: 'hover', value: 'hover' },
        { name: 'pointer', value: 'fine' },
      ],
    })
  })

  test.afterAll(async () => {
    await app?.close().catch(() => undefined)
    await mock?.close()
    sandbox?.cleanup()
  })

  test('a single-profile scope marks nothing', async () => {
    await expect(row('Plan the week')).toBeVisible({ timeout: 60_000 })
    await expect(row('Draft the release notes')).toHaveCount(0)
    await expect(leads()).toHaveCount(0)
    await capture('scope-default')
  })

  test('the merged list leads every named-profile row with its owner', async () => {
    await page.getByRole('button', { name: 'Show all profiles' }).click()
    await expect(row('Draft the release notes')).toBeVisible({ timeout: 60_000 })
    await expect(row('Compare the two vector stores')).toBeVisible()
    await expect(row('Check the deploy logs')).toBeVisible()
    await expect(row('Plan the week')).toBeVisible()
    // Park the pointer off the list so the capture shows the resting state.
    await parkPointer()
    await capture('all-profiles-rest')

    if (screenshotOnly) {
      return
    }

    await expect(row('Draft the release notes').locator('[data-profile-lead="inbox"]')).toBeVisible()
    await expect(row('Compare the two vector stores').locator('[data-profile-lead="research"]')).toBeVisible()
    await expect(row('Check the deploy logs').locator('[data-profile-lead="ops"]')).toBeVisible()
    // The lead is the first thing on the title line.
    await expect(row('Draft the release notes')).toContainText(/inbox\s*›\s*Draft the release notes/)
    // The default profile is never marked.
    await expect(row('Plan the week').locator('[data-profile-lead]')).toHaveCount(0)
    await expect(row('Tidy the downloads folder').locator('[data-profile-lead]')).toHaveCount(0)
  })

  test('hover and selection paint the lead in the profile colour', async () => {
    const target = row('Check the deploy logs')
    const lead = target.locator('[data-profile-lead="ops"]')
    const color = () => lead.evaluate(el => getComputedStyle(el).color)

    const rest = screenshotOnly ? '' : await color()
    await target.hover()
    await capture('all-profiles-hover')

    if (!screenshotOnly) {
      await expect.poll(color).not.toBe(rest)
    }

    await target.click()

    if (screenshotOnly) {
      await page.waitForTimeout(5_000)
    } else {
      await expect(lead).toHaveClass(/font-medium/, { timeout: 60_000 })
    }

    await parkPointer()
    await capture('all-profiles-selected')

    if (!screenshotOnly) {
      // Selected keeps the colour after the pointer leaves — it never falls
      // back to the resting grey.
      await expect.poll(color).not.toBe(rest)
    }
  })
})
