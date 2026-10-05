/**
 * Project picker (`composer.actions`, bundled plugin `project-picker`) — live proof.
 *
 * Boots the real Electron app against the mock backend, seeds genuine project
 * rows through `hermes_cli.projects_db` (the same per-profile store the sidebar
 * and the project tree read), and then asserts, from the real DOM:
 *
 *   1. the picker lives INSIDE `[data-slot="composer-root"]` and precedes the
 *      model selector *in the same child row* (`grid-area: controls`);
 *   2. it lists the profile's project folders;
 *   3. picking one starts a fresh chat anchored at that folder — a session row
 *      appears in `state.db` with `cwd` = the picked folder.
 *
 * Prerequisites: `npm run build` (the specs launch the built app), plus a
 * Python runtime for the helper (`HERMES_E2E_PYTHON`, default the installed
 * venv) because the sandbox home has none of its own.
 */
import { execFileSync } from 'node:child_process'
import * as fs from 'node:fs'
import * as path from 'node:path'

import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import { expect, test } from './test'

const HERMES_PYTHON = process.env.HERMES_E2E_PYTHON ?? '/usr/local/lib/hermes-agent/.venv/bin/python'
const HERMES_ROOT = process.env.HERMES_E2E_ROOT ?? '/usr/local/lib/hermes-agent'
const HELPER = path.join(process.cwd(), 'e2e', 'project-picker-helper.py')
const ARTIFACTS = process.env.HERMES_E2E_ARTIFACTS ?? path.join(process.cwd(), 'e2e-artifacts')
// The pill's accessible name depends on its branch (`model-pill.tsx`: the live
// menu trigger is labelled `Model · provider: model`, only the no-menu fallback
// reads "Open model picker"), so neither name is a stable hook. `data-tour` is
// written on both branches, primary chat only — that IS the model selector.
const MODEL_PILL = 'button[data-tour="model-pill"]'
const PICKER = 'button[aria-label="Project"]'

interface SeededProject {
  id: string
  name: string
}

function runHelper<T>(hermesHome: string, args: string[]): T {
  const stdout = execFileSync(HERMES_PYTHON, [HELPER, ...args], {
    cwd: HERMES_ROOT,
    encoding: 'utf8',
    env: { ...process.env, HERMES_HOME: hermesHome, PYTHONPATH: HERMES_ROOT },
  })

  return JSON.parse(stdout.trim().split('\n').pop() as string) as T
}

let fixture: MockBackendFixture | null = null
let seeded: SeededProject[] = []
let websiteFolder = ''

test.describe('project picker (composer.actions)', () => {
  test.describe.configure({ timeout: 240_000 })

  test.beforeAll(async () => {
    test.setTimeout(300_000)

    fixture = await setupMockBackend()
    await waitForAppReady(fixture, 120_000)

    websiteFolder = path.join(fixture.sandbox.root, 'atlas-website')
    const apiFolder = path.join(fixture.sandbox.root, 'atlas-api')

    for (const folder of [websiteFolder, apiFolder]) {
      fs.mkdirSync(folder, { recursive: true })
    }

    seeded = runHelper<SeededProject[]>(fixture.sandbox.hermesHome, [
      'seed',
      fixture.sandbox.hermesHome,
      JSON.stringify([
        { name: 'Atlas Website', folder: websiteFolder },
        { name: 'Atlas API', folder: apiFolder },
      ]),
    ])

    console.log('[project-picker] seeded projects:', JSON.stringify(seeded))
  })

  test.afterAll(async () => {
    await fixture?.cleanup()
    fixture = null
  })

  test('sits in the composer controls row, immediately before the model selector', async () => {
    test.setTimeout(300_000)

    const page = fixture!.page

    // The project tree is fetched on boot: reload so the picker sees the seeds.
    await page.reload()
    await waitForAppReady(fixture!, 120_000)

    const picker = page.locator(PICKER)
    await expect(picker).toBeVisible({ timeout: 60_000 })

    const placement = await page.evaluate(
      ({ modelPill }) => {
        const trigger = document.querySelector('button[aria-label="Project"]')

        if (!trigger) {
          return null
        }

        const row = trigger.parentElement as HTMLElement
        const pill = document.querySelector(modelPill)

        return {
          insideComposerRoot: !!trigger.closest('[data-slot="composer-root"]'),
          pillInSameRow: !!pill && row.contains(pill),
          precedesPill:
            !!pill &&
            trigger.compareDocumentPosition(pill) === Node.DOCUMENT_POSITION_FOLLOWING,
          pickerIndexInRow: Array.from(row.children).indexOf(trigger),
          rowChildren: Array.from(row.children).map(
            el => `${el.tagName.toLowerCase()}${el.getAttribute('aria-label') ? `[${el.getAttribute('aria-label')}]` : ''}`,
          ),
          model: pill?.textContent?.trim().slice(0, 40) ?? null,
        }
      },
      { modelPill: MODEL_PILL },
    )

    console.log('[project-picker] placement:', JSON.stringify(placement))

    expect(placement).not.toBeNull()
    expect(placement!.insideComposerRoot).toBe(true)
    expect(placement!.pillInSameRow).toBe(true)
    expect(placement!.precedesPill).toBe(true)
    expect(placement!.pickerIndexInRow).toBe(0)
  })

  test('lists the profile project folders', async () => {
    test.setTimeout(120_000)

    const page = fixture!.page
    const picker = page.locator(PICKER)

    await expect(picker).toBeVisible({ timeout: 30_000 })
    await picker.click()

    const menu = page.getByRole('menu')
    await expect(menu).toBeVisible({ timeout: 30_000 })

    const options = (await page.getByRole('menuitem').allTextContents()).map(t => t.trim())
    console.log('[project-picker] options:', JSON.stringify(options))

    for (const project of seeded) {
      expect(options).toContain(project.name)
    }

    await page.keyboard.press('Escape')

    fs.mkdirSync(ARTIFACTS, { recursive: true })
    const rowShot = path.join(ARTIFACTS, 'composer-controls-row.png')
    await page.locator('[data-slot="composer-root"]').first().screenshot({ path: rowShot })
    await page.screenshot({ path: path.join(ARTIFACTS, 'app-window.png') })
    console.log('[project-picker] screenshots:', rowShot, path.join(ARTIFACTS, 'app-window.png'))
  })

  test('picking a project opens a chat anchored at that folder', async () => {
    test.setTimeout(240_000)

    const page = fixture!.page
    const home = fixture!.sandbox.hermesHome

    await page.locator(PICKER).click()
    await page.getByRole('menuitem', { name: 'Atlas Website' }).click()

    // One turn makes the session durable, so the anchoring is observable in state.db.
    const input = page.locator('[contenteditable="true"]').first()
    await input.click()
    await input.type('hello from the project picker e2e')
    await page.keyboard.press('Enter')

    await expect
      .poll(
        () =>
          runHelper<{ rows: Array<{ id: string; cwd: string | null }> }>(home, ['sessions', home]).rows.some(
            row => row.cwd === websiteFolder,
          ),
        { timeout: 90_000, message: `expected a session with cwd=${websiteFolder}` },
      )
      .toBe(true)

    const rows = runHelper<{ rows: Array<{ id: string; cwd: string | null }> }>(home, ['sessions', home]).rows
    console.log('[project-picker] sessions:', JSON.stringify(rows))
  })
})
