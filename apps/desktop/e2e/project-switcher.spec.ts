/**
 * E2E coverage for the backend-authoritative project switcher.
 *
 * Drives the real Electron app through ⌘K → Switch project → projects.* RPCs
 * → a project-scoped session. The native folder dialog is the only stub,
 * because Playwright cannot click an OS-owned window.
 */

import * as fs from 'node:fs'
import * as os from 'node:os'
import * as path from 'node:path'

import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import { type ElectronApplication, expect, type Page, test } from './test'

const SCREENSHOT_DIR = path.join(os.tmpdir(), 'hermes-project-switcher-e2e-shots')

function makeProjectDir(name: string): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), `hermes-e2e-project-${name}-`))

  fs.writeFileSync(path.join(dir, 'README.md'), `# ${name}\n`, 'utf8')

  return fs.realpathSync(dir)
}

function removeTempDir(dir: string): void {
  const tmpRoot = fs.realpathSync(os.tmpdir())

  if (dir && dir.startsWith(tmpRoot) && path.basename(dir).startsWith('hermes-e2e-project-')) {
    fs.rmSync(dir, { force: true, recursive: true })
  }
}

async function stubFolderDialog(app: ElectronApplication, dir: string): Promise<void> {
  await app.evaluate(async ({ dialog }, targetDir) => {
    dialog.showOpenDialog = (async () => ({ canceled: false, filePaths: [targetDir] })) as typeof dialog.showOpenDialog
  }, dir)
}

async function statusbarLabels(page: Page): Promise<string[]> {
  return page.evaluate(() => {
    const statusbar = document.querySelector('[data-slot="statusbar"]')

    return statusbar ? [...statusbar.querySelectorAll('button')].map(button => (button.textContent ?? '').trim()) : []
  })
}

const canonical = (value: string): string => value.replace(/^\/private\//, '/')

async function hoveredWorkspacePath(page: Page, projectDir: string): Promise<string> {
  await page
    .locator('[data-slot="statusbar"] button')
    .filter({ hasText: path.basename(projectDir) })
    .hover()

  const tooltip = page
    .getByRole('tooltip')
    .filter({ hasText: path.basename(projectDir) })
    .first()
  await expect(tooltip).toBeVisible({ timeout: 10_000 })

  return canonical((await tooltip.textContent())?.trim() ?? '')
}

async function openCommandPalette(page: Page): Promise<void> {
  await page.keyboard.press(process.platform === 'darwin' ? 'Meta+k' : 'Control+k')
  await expect(page.locator('[data-slot="command-input"]')).toBeVisible({ timeout: 10_000 })
}

function switcherInput(page: Page) {
  return page.locator('[data-slot="command-input"][placeholder="Search projects…"]')
}

async function openSwitcherViaPalette(page: Page): Promise<void> {
  await openCommandPalette(page)

  const paletteInput = page.locator('[data-slot="command-input"]').first()
  await paletteInput.fill('switch project')

  const entry = page.getByRole('option', { name: /Switch project/ }).first()
  await expect(entry).toBeVisible({ timeout: 10_000 })
  await paletteInput.press('ArrowDown')
  await expect(entry).toHaveAttribute('aria-selected', 'true')
  await paletteInput.press('Enter')
  await expect(switcherInput(page)).toBeVisible({ timeout: 10_000 })
}

async function closeSwitcher(page: Page): Promise<void> {
  await page.keyboard.press('Escape')
  await expect(switcherInput(page)).toHaveCount(0, { timeout: 10_000 })
}

function switcherRow(page: Page, projectDir: string) {
  return page.getByRole('option').filter({ hasText: path.basename(projectDir) })
}

async function chooseOpenFolder(page: Page): Promise<void> {
  const input = switcherInput(page)
  const row = page.getByRole('option', { name: /Open folder as project/ })

  await input.fill('open folder')
  await expect(row).toBeVisible({ timeout: 10_000 })
  if ((await row.getAttribute('aria-selected')) !== 'true') {
    await input.press('ArrowDown')
  }
  await expect(row).toHaveAttribute('aria-selected', 'true')
  await input.press('Enter')
}

async function chooseProject(page: Page, projectDir: string): Promise<void> {
  const input = switcherInput(page)

  await input.fill(path.basename(projectDir))

  const row = switcherRow(page, projectDir)

  await expect(row).toBeVisible({ timeout: 10_000 })
  if ((await row.getAttribute('aria-selected')) !== 'true') {
    await input.press('ArrowDown')
  }
  await expect(row).toHaveAttribute('aria-selected', 'true')
  await input.press('Enter')
}

test('switches between backend-authoritative projects without a window-global MRU', async () => {
  test.setTimeout(240_000)
  fs.mkdirSync(SCREENSHOT_DIR, { recursive: true })

  const projectAlpha = makeProjectDir('alpha')
  const projectBeta = makeProjectDir('beta')
  const fixture: MockBackendFixture = await setupMockBackend()
  const { page } = fixture

  try {
    await waitForAppReady(fixture, 120_000)

    await test.step('the picker is discoverable and uses the shared open-folder path', async () => {
      await openSwitcherViaPalette(page)
      await expect(page.getByRole('option', { name: /Open folder as project/ })).toBeVisible()
      await page.screenshot({ path: path.join(SCREENSHOT_DIR, 'switcher-open.png') })
      await closeSwitcher(page)
    })

    await test.step('a picked folder is upserted through Projects with no renderer MRU', async () => {
      await stubFolderDialog(fixture.app, projectAlpha)
      await openSwitcherViaPalette(page)
      await chooseOpenFolder(page)

      await expect.poll(() => statusbarLabels(page), { timeout: 30_000 }).toContain(path.basename(projectAlpha))
      expect(await hoveredWorkspacePath(page, projectAlpha)).toBe(canonical(projectAlpha))
      expect(await page.evaluate(() => window.localStorage.getItem('hermes.desktop.recentProjects'))).toBeNull()

      await openSwitcherViaPalette(page)
      await expect(switcherRow(page, projectAlpha)).toBeVisible({ timeout: 20_000 })
      await page.screenshot({ path: path.join(SCREENSHOT_DIR, 'switcher-projects-tree.png') })
      await closeSwitcher(page)
    })

    await test.step('selecting a project follows the existing project-session path', async () => {
      await stubFolderDialog(fixture.app, projectBeta)
      await openSwitcherViaPalette(page)
      await chooseOpenFolder(page)

      await expect.poll(() => statusbarLabels(page), { timeout: 30_000 }).toContain(path.basename(projectBeta))
      expect(await hoveredWorkspacePath(page, projectBeta)).toBe(canonical(projectBeta))

      await openSwitcherViaPalette(page)
      await expect(switcherRow(page, projectAlpha)).toBeVisible({ timeout: 20_000 })
      await expect(switcherRow(page, projectBeta)).toBeVisible({ timeout: 20_000 })
      await chooseProject(page, projectAlpha)

      await expect.poll(() => statusbarLabels(page), { timeout: 30_000 }).toContain(path.basename(projectAlpha))
      expect(await hoveredWorkspacePath(page, projectAlpha)).toBe(canonical(projectAlpha))
    })
  } finally {
    await fixture.cleanup()
    removeTempDir(projectAlpha)
    removeTempDir(projectBeta)
  }
})
