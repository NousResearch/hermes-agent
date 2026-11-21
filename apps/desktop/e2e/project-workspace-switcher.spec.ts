import { execFile } from 'node:child_process'
import * as fs from 'node:fs'
import * as path from 'node:path'
import { promisify } from 'node:util'

import { expect, type Locator, type Page, test } from '@playwright/test'

import { MOCK_REPLY } from '../../../tests-js/scripts/mock-server'

import { setupMockBackend, waitForAppReady } from './fixtures'

const execFileAsync = promisify(execFile)

const DESKTOP_ROOT = path.resolve(import.meta.dirname, '..')
const REPO_ROOT = path.resolve(DESKTOP_ROOT, '..', '..')
const SCREENSHOT_DIR = process.env.WORKSPACE_SCREENSHOT_DIR ? path.resolve(process.env.WORKSPACE_SCREENSHOT_DIR) : null
// A cold sandbox must not wait on the remote model catalog before the composer
// becomes ready; these scenarios exercise Projects RPCs, not provider discovery.
const OFFLINE_PROJECT_E2E_CONFIG = 'model_catalog:\n  enabled: false'

/**
 * Project/workspace switcher E2E: real Electron app, real `hermes serve`
 * backend (isolated sandbox HERMES_HOME), real projects.* + session.workspace.move
 * RPCs against the mock provider. Two projects are seeded through the real
 * `hermes_cli.projects_db` module (the same code the projects.* RPC handlers
 * call) into the sandbox home — never the user's projects.
 *
 * Wave 6: "Switch current chat" and "New chat" are separate nested submenus
 * (a multi-root project asks for an explicit folder in each).
 * Wave 9: each saved project's submenu ends with a "Default for new chats"
 * star menuitem (after a separator) that drives the projects.set_default RPC.
 * Home/auto rows have no star. Parent project rows are submenu triggers only.
 */

interface SeedProject {
  folders: string[]
  name: string
}

/** Seed projects into the sandbox's `$HERMES_HOME/projects.db` via the real backend module. */
async function seedProjects(hermesHome: string, projects: SeedProject[]): Promise<void> {
  const python = path.join(REPO_ROOT, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python')
  const scriptPath = path.join(hermesHome, 'seed-projects-e2e.py')

  fs.writeFileSync(
    scriptPath,
    [
      'import json',
      'from hermes_cli import projects_db as pdb',
      `spec = json.loads(${JSON.stringify(JSON.stringify(projects))})`,
      'with pdb.connect_closing() as conn:',
      '    for item in spec:',
      '        pdb.create_project(conn, name=item["name"], folders=item["folders"])',
      'print("seeded")',
      ''
    ].join('\n'),
    'utf8'
  )

  await execFileAsync(python, [scriptPath], {
    cwd: REPO_ROOT,
    env: { ...process.env, HERMES_HOME: hermesHome, PYTHONPATH: REPO_ROOT },
    timeout: 60_000
  })
  fs.rmSync(scriptPath, { force: true })
}

/** Read the persisted default project id straight from the sandbox database. */
async function readPersistedDefaultId(hermesHome: string): Promise<null | string> {
  const python = path.join(REPO_ROOT, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python')
  const scriptPath = path.join(hermesHome, 'read-default-e2e.py')

  fs.writeFileSync(
    scriptPath,
    [
      'import json',
      'from hermes_cli import projects_db as pdb',
      'with pdb.connect_closing() as conn:',
      '    row = pdb.get_default_id(conn)',
      'print(json.dumps({"default_project_id": row}))',
      ''
    ].join('\n'),
    'utf8'
  )

  try {
    const { stdout } = await execFileAsync(python, [scriptPath], {
      cwd: REPO_ROOT,
      env: { ...process.env, HERMES_HOME: hermesHome, PYTHONPATH: REPO_ROOT },
      timeout: 60_000
    })

    const parsed = JSON.parse(stdout.trim()) as { default_project_id: null | string }

    return parsed.default_project_id
  } finally {
    fs.rmSync(scriptPath, { force: true })
  }
}

/**
 * The workspace-cwd statusbar item carries no stable DOM hook; it is the only
 * statusbar menu whose popover lists projects + "Manage projects". Probe the
 * menu-variant triggers until that menu opens.
 */
async function openWorkspaceSwitcher(page: Page): Promise<void> {
  const triggers = page.locator('footer[data-slot="statusbar"] button[aria-haspopup="menu"]')
  const count = await triggers.count()

  for (let i = 0; i < count; i++) {
    await triggers.nth(i).click()
    const switcher = page.locator('[role="menu"]').filter({ hasText: 'Manage projects' })

    if (await switcher.first().isVisible().catch(() => false)) {
      return
    }

    await page.keyboard.press('Escape')
  }

  throw new Error('workspace switcher statusbar item not found')
}

/** The open switcher popover root. */
function switcherMenu(page: Page) {
  return page.locator('[role="menu"]').filter({ hasText: 'Manage projects' }).first()
}

/** A saved-project parent row in the switcher (a nested-submenu trigger only). */
function projectRow(page: Page, name: string) {
  return page.locator('[data-project-row]').filter({ hasText: name }).first()
}

/**
 * Open a project's submenu and return the bottom "Default for new chats" star
 * menuitem (Wave 9: the star lives at the submenu bottom, after a separator).
 */
async function defaultStar(page: Page, name: string): Promise<Locator> {
  const submenu = await openProjectSubmenu(page, name)

  return submenu.getByRole('menuitem', { name: 'Default for new chats' })
}

/** Open a project's submenu and wait for its own content to render. */
async function openProjectSubmenu(page: Page, name: string): Promise<Locator> {
  const root = switcherMenu(page)

  // A sibling project's submenu may still be open from the previous assertion
  // (e.g. reading its default star). Radix then holds a pointer-grace polygon
  // toward that submenu for 300 ms and swallows the next trigger hover
  // (onItemEnter defaultPrevented), so the requested submenu never opens.
  // Park the pointer above the menu long enough for the grace window to
  // expire, then hover the trigger.
  const rootBox = await root.boundingBox()

  if (rootBox) {
    await page.mouse.move(rootBox.x + rootBox.width / 2, Math.max(1, rootBox.y - 40), { steps: 6 })
    await page.waitForTimeout(400)
  }

  const trigger = root.getByRole('menuitem', { name: new RegExp(name) }).first()
  await trigger.hover()
  await expect(trigger).toHaveAttribute('aria-controls', /.+/)
  const contentId = await trigger.getAttribute('aria-controls')

  if (!contentId) {
    throw new Error(`Project submenu trigger has no aria-controls: ${name}`)
  }

  const submenu = page.locator(`[id="${contentId}"]`)
  await expect(submenu.getByRole('menuitem', { name: 'Open project' })).toBeVisible({ timeout: 10_000 })

  return submenu
}

/** Open the nested "Switch current chat" folder list of a (multi-root) project. */
async function openMoveFolderSubmenu(page: Page, name: string): Promise<void> {
  const submenu = await openProjectSubmenu(page, name)
  await submenu.getByRole('menuitem', { name: 'Switch current chat' }).hover()
}

/** Save a screenshot into both the test-output dir and the shared artifact dir. */
async function captureArtifact(page: Page, testInfo: { outputPath: (name: string) => string }, name: string): Promise<string> {
  const outputPath = testInfo.outputPath(`${name}.png`)
  await page.screenshot({ path: outputPath, fullPage: false })

  if (SCREENSHOT_DIR) {
    try {
      fs.mkdirSync(SCREENSHOT_DIR, { recursive: true })
      fs.copyFileSync(outputPath, path.join(SCREENSHOT_DIR, `${name}.png`))
    } catch {
      /* the artifact copy is best-effort; the outputPath original stays authoritative */
    }
  }

  return outputPath
}

/** The statusbar button that opens the switcher (identified by its current label). */
function workspaceButton(page: Page, label: string) {
  return page.locator('footer[data-slot="statusbar"] button', { hasText: label })
}

test('statusbar workspace switcher moves an idle chat between projects, guards a busy one, and opens the workspace manager', async () => {
  test.setTimeout(300_000)

  const busyPrompt = 'Hold this stream while the switcher is probed'

  const fixture = await setupMockBackend({
    mockServer: { holdFirstStreamForPrompt: busyPrompt },
    // Pin the UI language: the switcher assertions read English copy, while the
    // host OS locale would otherwise leak in.
    extraDisplayConfig: '  language: en',
    extraConfig: OFFLINE_PROJECT_E2E_CONFIG
  })

  const { page, mock, sandbox } = fixture

  try {
    const events: string[] = []
    page.on('websocket', socket => {
      socket.on('framereceived', frame => {
        try {
          const message = JSON.parse(String(frame.payload))

          if (message.method === 'event') {events.push(message.params?.type)}
        } catch {
          /* non-JSON transport frame */
        }
      })
    })

    // Real directories the moves retarget onto; isolated under the sandbox root.
    const alphaDir = path.join(sandbox.root, 'workspaces', 'alpha-app')
    const multiDirA = path.join(sandbox.root, 'workspaces', 'multi-a')
    const multiDirB = path.join(sandbox.root, 'workspaces', 'multi-b')
    fs.mkdirSync(alphaDir, { recursive: true })
    fs.mkdirSync(multiDirA, { recursive: true })
    fs.mkdirSync(multiDirB, { recursive: true })

    await seedProjects(sandbox.hermesHome, [
      { folders: [alphaDir], name: 'Alpha' },
      { folders: [multiDirA, multiDirB], name: 'MultiRoot' }
    ])

    await waitForAppReady(fixture, 120_000)

    const composer = page
      .locator('[data-slot="composer-root"] [contenteditable="true"]')
      .filter({ visible: true })
      .first()

    // One real turn: the draft becomes a durable stored session (the row a move writes to).
    await composer.fill('First request: establish an idle stored chat')
    await composer.press('Enter')
    const answers = page.locator('[data-slot="aui_assistant-message-content"]').filter({ hasText: MOCK_REPLY })
    await expect(answers).toHaveCount(1, { timeout: 60_000 })

    // The workspace-cwd item only renders once the focused chat has a cwd; it is
    // the third menu-variant trigger (gateway, approval, workspace).
    await expect
      .poll(async () => page.locator('footer[data-slot="statusbar"] button[aria-haspopup="menu"]').count(), {
        timeout: 30_000
      })
      .toBeGreaterThanOrEqual(3)

    // The switcher lists both seeded projects; the multi-root project nests its
    // folder choice under the labelled "Switch current chat" / "New chat" submenus.
    await openWorkspaceSwitcher(page)
    await expect(page.getByRole('menuitem', { name: /Alpha/ }).first()).toBeVisible()
    await openProjectSubmenu(page, 'MultiRoot')

    // Wave 6: no flat "Choose a folder" hint — two explicit nested submenus instead.
    await expect(page.getByText('Choose a folder')).toHaveCount(0)
    const switchCurrent = page.getByRole('menuitem', { name: 'Switch current chat' }).first()
    const newChat = page.getByRole('menuitem', { name: 'New chat' }).first()
    await expect(switchCurrent).toBeVisible()
    await expect(newChat).toBeVisible()

    // One nested folder list at a time: the "Switch current chat" folder list.
    await switchCurrent.hover()
    await expect(page.getByRole('menuitem', { name: /multi-a/ })).toBeVisible()
    await expect(page.getByRole('menuitem', { name: /multi-b/ })).toBeVisible()
    await page.keyboard.press('Escape')
    await page.keyboard.press('Escape')
    await page.keyboard.press('Escape')

    // Move the idle chat into the single-root project: a real session.workspace.move RPC.
    await openWorkspaceSwitcher(page)
    await openProjectSubmenu(page, 'Alpha')
    await page.getByRole('menuitem', { name: 'Switch current chat' }).click()
    await expect
      .poll(async () => (await workspaceButton(page, 'Alpha').textContent()) ?? '', { timeout: 15_000 })
      .toBe('Alpha')

    // Multi-root move requires picking a folder; pick the non-primary one from
    // the nested "Switch current chat" folder list.
    await openWorkspaceSwitcher(page)
    await openMoveFolderSubmenu(page, 'MultiRoot')
    await page.getByRole('menuitem', { name: /multi-b/ }).first().click()
    await expect
      .poll(async () => (await workspaceButton(page, 'MultiRoot').textContent()) ?? '', { timeout: 15_000 })
      .toBe('MultiRoot')

    // Busy guard: a held turn disables the move action and surfaces the guidance copy.
    await composer.fill(busyPrompt)
    await composer.press('Enter')
    await Promise.race([
      mock.waitForHeldStream(),
      page.waitForTimeout(60_000).then(() => {
        throw new Error('held stream never started')
      })
    ])

    await openWorkspaceSwitcher(page)
    await expect(page.getByText('Stop the running turn, then move the chat')).toBeVisible()
    await openProjectSubmenu(page, 'Alpha')
    await expect(page.getByRole('menuitem', { name: 'Switch current chat' })).toBeDisabled()
    await page.keyboard.press('Escape')
    await page.keyboard.press('Escape')

    // Let the held turn finish so teardown never waits on a running stream.
    mock.releaseHeldStream()
    await expect
      .poll(() => events.filter(type => type === 'message.complete').length, { timeout: 60_000 })
      .toBeGreaterThanOrEqual(2)

    // Manage projects navigates to the workspace manager showing both projects.
    await openWorkspaceSwitcher(page)
    await page.getByRole('menuitem', { name: /Manage projects/ }).click()
    await expect
      .poll(async () => page.evaluate(() => window.location.hash))
      .toContain('page=projects')
    await expect(page.getByRole('button', { name: 'New project' })).toBeVisible()
    await expect(page.getByRole('listitem', { name: 'Alpha' })).toBeVisible()
    await expect(page.getByRole('listitem', { name: 'MultiRoot' })).toBeVisible()
  } finally {
    await fixture.cleanup()
  }
})

// No Playwright browser fixtures: this scenario launches isolated Electron itself.
// eslint-disable-next-line no-empty-pattern
test('project default star toggles via its submenu, persists across reload, and never switches the active project or chat', async ({}, testInfo) => {
  test.setTimeout(300_000)

  const fixture = await setupMockBackend({
    extraDisplayConfig: '  language: en',
    extraConfig: OFFLINE_PROJECT_E2E_CONFIG
  })

  const { page, sandbox } = fixture

  try {
    const alphaDir = path.join(sandbox.root, 'workspaces', 'alpha-app')
    const multiDirA = path.join(sandbox.root, 'workspaces', 'multi-a')
    const multiDirB = path.join(sandbox.root, 'workspaces', 'multi-b')
    fs.mkdirSync(alphaDir, { recursive: true })
    fs.mkdirSync(multiDirA, { recursive: true })
    fs.mkdirSync(multiDirB, { recursive: true })

    await seedProjects(sandbox.hermesHome, [
      { folders: [alphaDir], name: 'Alpha' },
      { folders: [multiDirA, multiDirB], name: 'MultiRoot' }
    ])

    await waitForAppReady(fixture, 120_000)

    const composer = page
      .locator('[data-slot="composer-root"] [contenteditable="true"]')
      .filter({ visible: true })
      .first()

    // A durable stored session is required for moves; the default pointer works regardless.
    await composer.fill('Establish a stored chat before exercising the default star')
    await composer.press('Enter')
    const answers = page.locator('[data-slot="aui_assistant-message-content"]').filter({ hasText: MOCK_REPLY })
    await expect(answers).toHaveCount(1, { timeout: 60_000 })

    await expect
      .poll(async () => page.locator('footer[data-slot="statusbar"] button[aria-haspopup="menu"]').count(), {
        timeout: 30_000
      })
      .toBeGreaterThanOrEqual(3)

    // (a) Each saved project's submenu ends with an unpressed "Default for
    // new chats" star; the parent row is only a submenu trigger.
    await openWorkspaceSwitcher(page)
    await captureArtifact(page, testInfo, '00-switcher-unified-action-icons')
    await expect(await defaultStar(page, 'Alpha')).toHaveAttribute('aria-pressed', 'false')
    await expect(await defaultStar(page, 'MultiRoot')).toHaveAttribute('aria-pressed', 'false')

    await captureArtifact(page, testInfo, '01-switcher-project-submenus-default-stars')

    // Set Alpha as the default from the bottom of its submenu. Selecting the
    // menuitem dismisses the popover, so the pressed state is verified on a
    // freshly reopened switcher.
    await (await defaultStar(page, 'Alpha')).click()
    await openWorkspaceSwitcher(page)
    await expect(await defaultStar(page, 'Alpha')).toHaveAttribute('aria-pressed', 'true')
    await expect(await defaultStar(page, 'MultiRoot')).toHaveAttribute('aria-pressed', 'false')
    expect(await readPersistedDefaultId(sandbox.hermesHome)).not.toBeNull()
    await page.keyboard.press('Escape')

    // Move the chat into MultiRoot: the default pointer must not follow.
    await openWorkspaceSwitcher(page)
    await openMoveFolderSubmenu(page, 'MultiRoot')
    await page.getByRole('menuitem', { name: /multi-b/ }).first().click()
    await expect
      .poll(async () => (await workspaceButton(page, 'MultiRoot').textContent()) ?? '', { timeout: 15_000 })
      .toBe('MultiRoot')

    await openWorkspaceSwitcher(page)
    await expect(await defaultStar(page, 'Alpha')).toHaveAttribute('aria-pressed', 'true')
    await expect(await defaultStar(page, 'MultiRoot')).toHaveAttribute('aria-pressed', 'false')
    // The active-project marker stays on MultiRoot, not on the default project.
    await expect(projectRow(page, 'MultiRoot').getByRole('menuitem', { name: /MultiRoot/ })).toHaveAttribute(
      'aria-current',
      'true'
    )

    // Repoint the default to MultiRoot from its submenu: the active
    // project/session must not move. Selecting the star dismisses the popover
    // — reopen to verify.
    await (await defaultStar(page, 'MultiRoot')).click()
    await openWorkspaceSwitcher(page)
    await expect(await defaultStar(page, 'MultiRoot')).toHaveAttribute('aria-pressed', 'true')
    await expect(await defaultStar(page, 'Alpha')).toHaveAttribute('aria-pressed', 'false')
    await page.keyboard.press('Escape')
    await expect
      .poll(async () => (await workspaceButton(page, 'MultiRoot').textContent()) ?? '', { timeout: 15_000 })
      .toBe('MultiRoot')
    // The same stored chat (its completed assistant turn) is still focused.
    await expect(page.locator('[data-slot="aui_assistant-message-content"]').filter({ hasText: MOCK_REPLY })).toHaveCount(1)

    // (b) One multi-root nested folder list at a time.
    await openWorkspaceSwitcher(page)
    await openProjectSubmenu(page, 'MultiRoot')
    await page.getByRole('menuitem', { name: 'Switch current chat' }).first().hover()
    await expect(page.getByRole('menuitem', { name: /multi-a/ })).toBeVisible()
    await captureArtifact(page, testInfo, '02-multiroot-nested-switch-current-chat')
    await page.keyboard.press('Escape')
    await page.keyboard.press('Escape')
    await page.keyboard.press('Escape')

    // Settings → Workspace Projects: the accepted icon-only star stays beside
    // each project name and reflects the same default pointer.
    await openWorkspaceSwitcher(page)
    await page.getByRole('menuitem', { name: /Manage projects/ }).click()
    await expect
      .poll(async () => page.evaluate(() => window.location.hash))
      .toContain('page=projects')

    const multiRootDefault = page.getByRole('listitem', { name: 'MultiRoot' }).getByRole('button', { name: /Default for new chats/ })
    const alphaDefault = page.getByRole('listitem', { name: 'Alpha' }).getByRole('button', { name: /Default for new chats/ })
    await expect(multiRootDefault).toBeVisible()
    await expect(multiRootDefault).toHaveAttribute('aria-pressed', 'true')
    await expect(alphaDefault).toHaveAttribute('aria-pressed', 'false')

    // (c) Settings project cards retain the accepted icon-only default star.
    await captureArtifact(page, testInfo, '03-settings-project-cards-default-star')

    // Toggle off, then back on, straight from the settings card.
    await multiRootDefault.click()
    await expect(multiRootDefault).toHaveAttribute('aria-pressed', 'false')
    await expect(alphaDefault).toHaveAttribute('aria-pressed', 'false')
    expect(await readPersistedDefaultId(sandbox.hermesHome)).toBeNull()

    await multiRootDefault.click()
    await expect(multiRootDefault).toHaveAttribute('aria-pressed', 'true')
    await expect(alphaDefault).toHaveAttribute('aria-pressed', 'false')
    expect(await readPersistedDefaultId(sandbox.hermesHome)).not.toBeNull()

    // Close the settings overlay before reloading so the app restores the
    // chat route (and its composer) deterministically.
    const closeSettings = page.getByRole('button', { name: 'Close settings' })

    if (await closeSettings.isVisible().catch(() => false)) {
      await closeSettings.click()
    }

    await waitForAppReady(fixture)

    // Reload the renderer: the default star must come back pressed from the
    // backend-backed store.
    await page.reload()
    await waitForAppReady(fixture)

    await openWorkspaceSwitcher(page)
    await expect(await defaultStar(page, 'MultiRoot')).toHaveAttribute('aria-pressed', 'true')
    await expect(await defaultStar(page, 'Alpha')).toHaveAttribute('aria-pressed', 'false')

    // The same persistence must be visible through a direct Settings open;
    // the page hydrates its detailed list without a grouped-sidebar detour.
    await page.getByRole('menuitem', { name: /Manage projects/ }).click()
    await expect
      .poll(async () => page.evaluate(() => window.location.hash))
      .toContain('page=projects')

    const multiRootDefaultAfterReload = page
      .getByRole('listitem', { name: 'MultiRoot' })
      .getByRole('button', { name: /Default for new chats/ })

    await expect(multiRootDefaultAfterReload).toHaveAttribute('aria-pressed', 'true', { timeout: 30_000 })

    await captureArtifact(page, testInfo, '04-settings-default-star-persisted-after-reload')
  } finally {
    await fixture.cleanup()
  }
})
