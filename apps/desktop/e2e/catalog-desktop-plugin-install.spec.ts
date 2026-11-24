/**
 * A desktop-only catalog plugin installed from its catalog card (Accent Picker, reported on X on
 * 2026-11-07): the install worked, but the card kept its "+", the plugin sat switched off (it ships
 * `defaultEnabled: false`), and a second Install answered "already exists" with Force out of sight.
 *
 * Real Electron and real `hermes serve`. The catalog feed is answered with one entry that points at a
 * local Git repository laid out like hermes-official-plugins (plugin.yaml + desktop/plugin.js, no agent
 * half), so the install clones, checks out the pinned commit and publishes for real, with no network.
 *
 * The second half is the copy an older build left behind: the folder with no recorded catalog entry.
 * Installing from the card again adopts it in place instead of refusing.
 */
import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import { expect, test } from './test'

const CATALOG_FEED = 'https://nousresearch.github.io/hermes-agent/docs/api/plugins.json'
const NAME = 'accent-picker'
const INSTALL_ORIGIN = '.hermes-origin.json'

const PLUGIN_JS = `export default {
  id: 'accent',
  name: 'Accent Picker',
  description: 'E2E stand-in for the catalog accent picker.',
  defaultEnabled: false,
  register() {}
}
`

let fixture: MockBackendFixture | null = null
let repo = ''
let sha = ''

function git(cwd: string, ...args: string[]): string {
  return execFileSync('git', args, { cwd, encoding: 'utf8' }).trim()
}

/** A repository shaped like hermes-official-plugins: the entry is a subdirectory, desktop half only. */
function officialPluginsRepo(root: string): void {
  const plugin = path.join(root, NAME)
  fs.mkdirSync(path.join(plugin, 'desktop'), { recursive: true })
  fs.writeFileSync(path.join(plugin, 'plugin.yaml'), `name: ${NAME}\nlabel: Accent Picker\nversion: 1.0.0\n`)
  fs.writeFileSync(path.join(plugin, 'desktop', 'plugin.js'), PLUGIN_JS)
  git(root, 'init', '-q', '-b', 'main')
  git(root, 'add', '-A')
  git(root, '-c', 'user.email=e2e@example.invalid', '-c', 'user.name=e2e', 'commit', '-q', '-m', 'accent-picker')
}

test.beforeAll(async () => {
  fixture = await setupMockBackend()
  repo = path.join(fixture.sandbox.root, 'hermes-official-plugins')
  fs.mkdirSync(repo, { recursive: true })
  officialPluginsRepo(repo)
  sha = git(repo, 'rev-parse', 'HEAD')

  const entry = {
    category: 'desktop',
    description: 'OKLCH accent colour picker for the status bar. Desktop-only.',
    maintainer: 'NousResearch',
    name: NAME,
    repo: pathToFileURL(repo).href,
    sha,
    subdir: NAME,
    tier: 'official',
    version: '1.0.0'
  }

  await fixture.app
    .context()
    .route(CATALOG_FEED, route =>
      route.fulfill({ body: JSON.stringify([entry]), contentType: 'application/json', status: 200 })
    )
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

test('a desktop-only catalog plugin shows installed, turns on, and an older copy is adopted', async () => {
  test.setTimeout(300_000)
  const { page, sandbox } = fixture!
  await waitForAppReady(fixture!, 120_000)
  const folder = path.join(sandbox.hermesHome, 'desktop-plugins', NAME)
  const addButton = page.getByRole('button', { name: `Add ${NAME}` })
  const pluginSwitch = page.getByRole('switch', { name: 'Accent Picker' })

  await page.evaluate(() => {
    window.location.hash = '#/capabilities?tab=plugins'
  })
  const search = page.getByPlaceholder(/search plugins/i).first()
  await search.waitFor({ timeout: 60_000 })
  await search.fill('accent')

  const installFromCard = async () => {
    await addButton.first().click({ timeout: 60_000 })
    const dialog = page.getByRole('dialog', { name: NAME })
    await dialog.getByRole('button', { name: /^install$/i }).click({ timeout: 120_000 })
    await dialog.waitFor({ state: 'detached', timeout: 120_000 })
  }

  // First install: the card records its entry, stops offering Install, and the plugin is on.
  await installFromCard()
  expect(JSON.parse(fs.readFileSync(path.join(folder, INSTALL_ORIGIN), 'utf8'))).toEqual({
    catalogName: NAME,
    repo: pathToFileURL(repo).href,
    sha
  })
  expect(fs.existsSync(path.join(folder, '.hermes-package.json'))).toBe(false)
  await expect(addButton).toHaveCount(0, { timeout: 30_000 })
  await expect(pluginSwitch).toHaveAttribute('aria-checked', 'true', { timeout: 30_000 })

  // What an older build left: the same folder with no recorded entry. Its card offers Install again.
  fs.rmSync(path.join(folder, INSTALL_ORIGIN))
  fs.writeFileSync(path.join(folder, 'state.json'), 'kept\n')
  await page.getByRole('button', { name: 'Rescan' }).click()
  await expect(addButton).toHaveCount(1, { timeout: 30_000 })

  // Installing again adopts that copy in place: no "already exists", nothing in the folder replaced.
  await installFromCard()
  expect(JSON.parse(fs.readFileSync(path.join(folder, INSTALL_ORIGIN), 'utf8')).catalogName).toBe(NAME)
  expect(fs.readFileSync(path.join(folder, 'state.json'), 'utf8')).toBe('kept\n')
  await expect(addButton).toHaveCount(0, { timeout: 30_000 })
  await expect(pluginSwitch).toHaveAttribute('aria-checked', 'true')
})
