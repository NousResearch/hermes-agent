/**
 * E2E: sidebar full-text search is keyed to the sidebar's profile.
 *
 * Two real profile homes on This device — `default` and `ssprobe-b` — each
 * hold one durable session. Both transcripts contain the shared word "zebra"
 * only in the assistant reply, so a loaded row never matches client-side and
 * every hit comes from the backend's `/api/sessions/search`. Each reply also
 * carries a profile-only marker. With the sidebar on a profile, searching the
 * shared word must surface that profile's marker and never the other one's,
 * across default → ssprobe-b → default.
 *
 * Prerequisite: `npm run build` (dist/) and the repo's `.venv`.
 */

import { spawnSync } from 'node:child_process'
import * as fs from 'node:fs'
import * as path from 'node:path'

import { writeEnvFile, writeMockProviderConfig } from '../../../tests-js/scripts/mock-provider-config'
import { startMockServer } from '../../../tests-js/scripts/mock-server'

import { buildAppEnv, createSandbox, launchDesktop, type MockBackendFixture, type Sandbox, waitForAppReady } from './fixtures'
import { type ElectronApplication, expect, type Page, test } from './test'

const REPO_ROOT = path.resolve(import.meta.dirname, '..', '..', '..')
const PROFILE_B = 'ssprobe-b'
const MARKER = { default: 'alphaonlymarker', [PROFILE_B]: 'bravoonlymarker' } as const

/** Write one session into ``home``'s state.db through the real SessionDB. */
function seedSession(sandbox: Sandbox, home: string, sessionId: string, marker: string): void {
  const script = [
    'import sys',
    'from pathlib import Path',
    'from hermes_state import SessionDB',
    'home, sid, marker = sys.argv[1:4]',
    "db = SessionDB(db_path=Path(home) / 'state.db')",
    "db.create_session(sid, source='cli')",
    "db.append_message(sid, role='user', content=f'hello {marker}')",
    "db.append_message(sid, role='assistant', content=f'zebra {marker}')",
    'db.close()'
  ].join('\n')

  const python = path.join(REPO_ROOT, '.venv', 'bin', 'python')

  const result = spawnSync(python, ['-c', script, home, sessionId, marker], {
    cwd: REPO_ROOT,
    encoding: 'utf8',
    env: { ...process.env, HERMES_HOME: sandbox.hermesHome, HOME: sandbox.root, PYTHONPATH: REPO_ROOT }
  })

  if (result.status !== 0) {
    throw new Error(`seeding ${home} failed:\n${result.stderr}`)
  }
}

const sidebar = (page: Page) => page.locator('[data-slot="sidebar"]').first()
const rail = (page: Page) => page.getByRole('group', { name: 'Profiles' })
const railButton = (page: Page, name: string) => rail(page).getByRole('button', { name, exact: true })

// The launch profile is the rail's home pill: "Switch to default" while another
// profile is selected, relabelled "Show all profiles" (pressed) once on default.
async function selectProfile(page: Page, name: string): Promise<void> {
  if (name === 'default') {
    const showAll = railButton(page, 'Show all profiles')

    if (!(await showAll.count()) || (await showAll.getAttribute('aria-pressed')) !== 'true') {
      await railButton(page, 'Switch to default').click()
    }

    await expect(showAll).toHaveAttribute('aria-pressed', 'true', { timeout: 60_000 })

    return
  }

  await railButton(page, name).click()
  await expect(railButton(page, name)).toHaveAttribute('aria-pressed', 'true', { timeout: 60_000 })
}

async function expectSearchScopedTo(page: Page, profile: keyof typeof MARKER): Promise<void> {
  const other = profile === 'default' ? PROFILE_B : 'default'
  const search = page.getByRole('textbox', { name: 'Search sessions' })

  await search.fill('')
  await search.fill('zebra')
  // Only the backend can match "zebra" here, so this waits for the server hits.
  await expect(sidebar(page).getByText(MARKER[profile])).toBeVisible({ timeout: 30_000 })
  await expect(sidebar(page).getByText(MARKER[other])).toHaveCount(0)
  await search.fill('')
}

test.describe('sidebar session search — profile scope', () => {
  test.describe.configure({ mode: 'serial' })

  let mock: Awaited<ReturnType<typeof startMockServer>>
  let sandbox: Sandbox
  let app: ElectronApplication
  let page: Page

  test.beforeAll(async () => {
    test.setTimeout(240_000)
    mock = await startMockServer()
    sandbox = createSandbox('search-scope')
    writeMockProviderConfig(sandbox.hermesHome, mock.url)
    writeEnvFile(sandbox.hermesHome)

    const homeB = path.join(sandbox.hermesHome, 'profiles', PROFILE_B)
    fs.mkdirSync(homeB, { recursive: true })
    writeMockProviderConfig(homeB, mock.url)
    writeEnvFile(homeB)
    seedSession(sandbox, sandbox.hermesHome, 'search-scope-default', MARKER.default)
    seedSession(sandbox, homeB, 'search-scope-b', MARKER[PROFILE_B])

    ;({ app, page } = await launchDesktop(buildAppEnv(sandbox)))
    await waitForAppReady({ app, page } as MockBackendFixture, 120_000)
    await expect(page.locator('[data-slot="statusbar"]').getByText('ready', { exact: true })).toBeVisible({
      timeout: 120_000
    })
  })

  test.afterAll(async () => {
    await app?.close().catch(() => undefined)
    await mock?.close()
    sandbox?.cleanup()
  })

  test('search returns only the selected profile across A → B → A', async () => {
    test.setTimeout(240_000)
    // Start on whichever profile the app booted into, then switch away and back.
    await expect(rail(page)).toBeVisible({ timeout: 60_000 })
    const bootedOnB = (await railButton(page, PROFILE_B).getAttribute('aria-pressed')) === 'true'
    const first: keyof typeof MARKER = bootedOnB ? PROFILE_B : 'default'
    const second: keyof typeof MARKER = first === 'default' ? PROFILE_B : 'default'

    await selectProfile(page, first)
    await expectSearchScopedTo(page, first)
    await selectProfile(page, second)
    await expectSearchScopedTo(page, second)
    await selectProfile(page, first)
    await expectSearchScopedTo(page, first)
  })
})
