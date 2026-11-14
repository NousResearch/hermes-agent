import { execFile } from 'node:child_process'
import * as fs from 'node:fs'
import * as path from 'node:path'
import { promisify } from 'node:util'

import { expect, type Page, test } from '@playwright/test'

import { MOCK_REPLY } from '../../../tests-js/scripts/mock-server'

import { setupMockBackend, waitForAppReady } from './fixtures'

const execFileAsync = promisify(execFile)

const DESKTOP_ROOT = path.resolve(import.meta.dirname, '..')
const REPO_ROOT = path.resolve(DESKTOP_ROOT, '..', '..')

/**
 * Project/workspace switcher E2E: real Electron app, real `hermes serve`
 * backend (isolated sandbox HERMES_HOME), real projects.* + session.workspace.move
 * RPCs against the mock provider. Two projects are seeded through the real
 * `hermes_cli.projects_db` module (the same code the projects.* RPC handlers
 * call) into the sandbox home — never the user's projects.
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

/** Open a project's submenu and return nothing — the submenu stays open for the caller. */
async function openProjectSubmenu(page: Page, name: string): Promise<void> {
  const root = page.locator('[role="menu"]').filter({ hasText: 'Manage projects' }).first()
  await root.getByRole('menuitem', { name: new RegExp(name) }).first().hover()
  await expect(
    page.getByRole('menuitem', { name: 'Open project' }).first()
  ).toBeVisible({ timeout: 10_000 })
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
    extraDisplayConfig: '  language: en'
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

    await waitForAppReady(fixture)

    const composer = page
      .locator('[data-slot="composer-root"] [contenteditable="true"]')
      .filter({ visible: true })
      .first()

    // One real turn: the draft becomes a durable stored session (the row a move writes to).
    await composer.fill('First request: establish an idle stored chat')
    await composer.press('Enter')
    const answers = page.locator('[data-slot="aui_assistant-message-content"]').filter({ hasText: MOCK_REPLY })
    await expect(answers).toHaveCount(1, { timeout: 60_000 })

    // The completed turn has bound the composer to a durable stored session;
    // the switcher must expose the real move action for that idle chat.

    // The workspace-cwd item only renders once the focused chat has a cwd; it is
    // the third menu-variant trigger (gateway, approval, workspace).
    await expect
      .poll(async () => page.locator('footer[data-slot="statusbar"] button[aria-haspopup="menu"]').count(), {
        timeout: 30_000
      })
      .toBeGreaterThanOrEqual(3)

    // The switcher lists both seeded projects; the multi-root project asks for an explicit folder.
    await openWorkspaceSwitcher(page)
    await expect(page.getByRole('menuitem', { name: /Alpha/ }).first()).toBeVisible()
    await openProjectSubmenu(page, 'MultiRoot')
    await expect(page.getByText('Choose a folder')).toBeVisible()
    const multiRootMenu = page.getByRole('menu', { name: /MultiRoot/ })
    await expect(multiRootMenu.getByRole('menuitem', { name: /multi-a/ }).first()).toBeVisible()
    await expect(multiRootMenu.getByRole('menuitem', { name: /multi-b/ }).first()).toBeVisible()
    await page.keyboard.press('Escape')
    await page.keyboard.press('Escape')

    // Move the idle chat into the single-root project: a real session.workspace.move RPC.
    await openWorkspaceSwitcher(page)
    await openProjectSubmenu(page, 'Alpha')
    await page.getByRole('menuitem', { name: 'Switch current chat' }).click()
    await expect
      .poll(async () => (await workspaceButton(page, 'Alpha').textContent()) ?? '', { timeout: 15_000 })
      .toBe('Alpha')

    // Multi-root move requires picking a folder; pick the non-primary one. The
    // move rows precede the same-named "New chat" rows, so .first() is the move.
    await openWorkspaceSwitcher(page)
    await openProjectSubmenu(page, 'MultiRoot')
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
