/**
 * E2E regression: an archived session stays reachable from Settings → Sessions
 * even when it is also `hidden` (Bot Mode marks its sessions hidden).
 *
 * The archived view is the only recovery surface for rows that dropped out of
 * the default Sessions list; a row that is both archived AND hidden used to be
 * excluded from it too, so the transcript was unrecoverable from any UI.
 */

import { execFileSync } from 'node:child_process'
import * as fs from 'node:fs'
import * as path from 'node:path'

import { startMockServer } from '../../../tests-js/scripts/mock-server'

import { writeEnvFile, writeMockProviderConfig } from '../../../tests-js/scripts/mock-provider-config'
import {
  buildAppEnv,
  createSandbox,
  launchDesktop,
  type MockBackendFixture,
  waitForAppReady
} from './fixtures'
import { expect, test } from './test'

const HIDDEN_ARCHIVED_TEXT = 'E2E_ARCHIVED_HIDDEN_ROW'
const CONTROL_TEXT = 'E2E_ARCHIVED_VISIBLE_ROW'

const DESKTOP_ROOT = path.resolve(import.meta.dirname, '..')
const REPO_ROOT = path.resolve(DESKTOP_ROOT, '..', '..')
const RUNTIME_PYTHON = path.join(REPO_ROOT, 'venv', 'bin', 'python')

/**
 * Seed two durable rows through the real SessionDB (the same code the desktop
 * backend reads): one archived+hidden (the recovery case) and one plain
 * archived control. Direct SessionDB writes keep the seed deterministic under
 * host load; the desktop still reads them through the real `hermes serve`.
 */
function seedArchivedRows(hermesHome: string, hiddenId: string, controlId: string): void {
  execFileSync(
    RUNTIME_PYTHON,
    [
      '-c',
      [
        'import sys',
        'from hermes_state import SessionDB',
        'hidden_id, control_id, hidden_text, control_text = sys.argv[1:5]',
        'db = SessionDB()',
        'for sid, text in ((hidden_id, hidden_text), (control_id, control_text)):',
        '    db.create_session(sid, source="desktop")',
        '    db.append_message(sid, "user", text)',
        '    db.append_message(sid, "assistant", "ok " + text)',
        '    assert db.set_session_archived(sid, True) is True',
        'assert db.set_session_hidden(hidden_id, True) is True',
        'rows = {r["id"]: (r["archived"], r["hidden"]) for r in db.list_sessions_rich(include_archived=True, include_hidden=True)}',
        'db.close()',
        'assert rows[hidden_id] == (1, 1), rows',
        'assert rows[control_id] == (1, 0), rows',
      ].join('\n'),
      hiddenId,
      controlId,
      HIDDEN_ARCHIVED_TEXT,
      CONTROL_TEXT,
    ],
    { cwd: REPO_ROOT, env: { ...process.env, HERMES_HOME: hermesHome }, stdio: 'pipe' },
  )
  expect(fs.existsSync(path.join(hermesHome, 'state.db'))).toBe(true)
}

async function setupSeededMockBackend(): Promise<MockBackendFixture & { controlId: string; hiddenId: string }> {
  const mock = await startMockServer()
  const sandbox = createSandbox('archived-hidden')
  writeMockProviderConfig(sandbox.hermesHome, mock.url)
  writeEnvFile(sandbox.hermesHome)

  const hiddenId = '20260916_000001_e2ehid'
  const controlId = '20260916_000002_e2ectl'
  seedArchivedRows(sandbox.hermesHome, hiddenId, controlId)

  const { app, page } = await launchDesktop(buildAppEnv(sandbox))

  return {
    app,
    page,
    mock,
    mockUrl: mock.url,
    sandbox,
    hiddenId,
    controlId,
    cleanup: async () => {
      await app.close().catch(() => undefined)
      await mock.close()
      sandbox.cleanup()
    },
  }
}

test.describe('archived + hidden session stays recoverable', () => {
  let fixture: Awaited<ReturnType<typeof setupSeededMockBackend>>

  test.beforeAll(async () => {
    fixture = await setupSeededMockBackend()
    await waitForAppReady(fixture)
  })

  test.afterAll(async () => {
    await fixture?.cleanup()
  })

  test('Settings → Sessions lists the archived+hidden row next to the plain archived row', async () => {
    const { page, hiddenId, controlId } = fixture

    // Hidden rows never show in the default Sessions list (unchanged contract).
    await expect(page.getByText(HIDDEN_ARCHIVED_TEXT, { exact: false })).toHaveCount(0)

    await page.evaluate(() => {
      window.location.hash = '#/settings?tab=sessions'
    })

    const controlRow = page.locator(`#archived-session-${controlId}`)
    await controlRow.waitFor({ state: 'visible', timeout: 60_000 })

    const hiddenRow = page.locator(`#archived-session-${hiddenId}`)
    await expect(hiddenRow, 'archived+hidden session must be listed in the archived view').toBeVisible({
      timeout: 15_000,
    })
    await expect(hiddenRow.getByRole('button', { name: /unarchive/i })).toBeVisible()

    await page.screenshot({ path: test.info().outputPath('archived-hidden-row.png'), fullPage: false })
  })
})
