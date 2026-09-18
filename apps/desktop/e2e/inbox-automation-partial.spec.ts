/**
 * E2E evidence capture for real backend-seeded automation and partial fixture.
 *
 * Two distinct test scenarios:
 *  1. REAL BACKEND-SEEDED: Seeds a session + goal into state.db via Python
 *     before the desktop app starts. The inbox.list RPC reads from the real DB.
 *     This is genuine persisted automation, not a WebSocket stub.
 *
 *  2. PARTIAL FIXTURE: Intercepts inbox.list to return partial=true with errors.
 *     Verifies the renderer's partial-read warning and retained rows.
 *     This is a response-stub interception, not backend proof.
 *
 * Output: .inbox-work/screenshots/real-automation-*.png, fixture-partial-*.png
 *         .inbox-work/screenshots/receipt.json
 *
 * Prerequisite: `npm run build` in apps/desktop.
 */

import * as childProcess from 'node:child_process'
import * as fs from 'node:fs'
import * as path from 'node:path'

import { expect, test, type Page } from '@playwright/test'

import {
  type MockBackendFixture,
  buildAppEnv,
  createSandbox,
  findElectron,
  launchDesktop,
  writeEnvFile,
  writeMockProviderConfig,
} from './fixtures'
import { startMockServer } from '../../../tests-js/scripts/mock-server'
import { installErrorBannerGuard } from './test'

const DESKTOP_ROOT = path.resolve(import.meta.dirname, '..')
const REPO_ROOT = path.resolve(DESKTOP_ROOT, '..', '..')
const SCREENSHOTS_DIR = path.resolve(REPO_ROOT, '.inbox-work', 'screenshots')
const PYTHON = process.env.HERMES_DESKTOP_PYTHON || 'python'
const SEED_SCRIPT = path.resolve(import.meta.dirname, 'fixtures', 'seed_inbox_automation.py')

// ── Helpers ──────────────────────────────────────────────────────────────────

function assertDistBuilt(): void {
  const electronMain = path.join(DESKTOP_ROOT, 'dist', 'electron-main.mjs')
  if (!fs.existsSync(electronMain)) {
    throw new Error(`Desktop dist not built. Run 'cd apps/desktop && npm run build' first.`)
  }
}

async function forceSize(app: MockBackendFixture['app'], width: number, height: number): Promise<void> {
  await app.evaluate(({ BrowserWindow }, { width, height }) => {
    const win = BrowserWindow.getAllWindows()[0]
    if (win) {
      win.unmaximize()
      win.setMinimumSize(width, height)
      win.setSize(width, height, false)
      win.setBounds({ x: 0, y: 0, width, height })
    }
  }, { width, height })
}

function inboxChip(page: Page) {
  return page.getByRole('button', { name: /inbox/i }).first()
}

async function waitForInboxReady(page: Page, timeoutMs = 30_000): Promise<void> {
  await page.waitForFunction(
    () => {
      const text = document.body.textContent ?? ''
      if (text.includes('All clear')) return true
      if (text.includes('Inbox not supported')) return true
      if (text.includes('Partial read')) return true
      if (text.includes('Incomplete data')) return true
      if (text.includes('Loading inbox')) return false
      if (document.querySelector('[data-panel-row]')) return true
      return false
    },
    undefined,
    { timeout: timeoutMs },
  )
}

async function closePanelWithEscape(page: Page): Promise<void> {
  await page.keyboard.press('Escape')
  await page.waitForTimeout(500)
}

// ── Manifest ─────────────────────────────────────────────────────────────────

interface ManifestEntry {
  dimensions: string
  fixture: string
  path: string
  state: string
}

const manifest: ManifestEntry[] = []

function record(state: string, fixtureType: string, dims: string, filePath: string): void {
  manifest.push({ dimensions: dims, fixture: fixtureType, path: filePath, state })
}

async function capture(
  testInfo: { outputPath: (n: string) => string },
  name: string,
  state: string,
  fixtureType: string,
  page: Page,
  dims = '1220x800',
): Promise<void> {
  const shotPath = testInfo.outputPath(name)
  await page.screenshot({ path: shotPath })
  record(state, fixtureType, dims, shotPath)

  const dest = path.join(SCREENSHOTS_DIR, name)
  fs.mkdirSync(path.dirname(dest), { recursive: true })
  fs.copyFileSync(shotPath, dest)
  record(state, fixtureType, dims, dest)
}

// ── Test 1: Real backend-seeded automation ────────────────────────────────────

const SESSION_KEY = 'e2e-seeded-deploy-staging'
const SESSION_TITLE = 'Deploy to staging'
const GOAL_TEXT = 'Ship the inbox feature end-to-end'

test.describe('real backend-seeded automation', () => {
  let fixture: MockBackendFixture | null = null

  test.beforeAll(async () => {
    assertDistBuilt()

    // 1. Start mock server
    const mock = await startMockServer()

    // 2. Create sandbox + write config (same as setupMockBackend)
    const sandbox = createSandbox('real-auto')
    writeMockProviderConfig(sandbox.hermesHome, mock.url)
    writeEnvFile(sandbox.hermesHome)

    // 3. Seed real session + goal into state.db BEFORE app launch
    console.log(`[real-auto] Seeding session into ${sandbox.hermesHome}`)
    const seedResult = childProcess.spawnSync(
      PYTHON,
      [SEED_SCRIPT, sandbox.hermesHome, SESSION_KEY, SESSION_TITLE, GOAL_TEXT],
      { timeout: 15_000, encoding: 'utf8', env: { ...process.env, HERMES_HOME: sandbox.hermesHome } },
    )
    console.log(`[real-auto] Seed stdout: ${seedResult.stdout}`)
    if (seedResult.stderr) {
      console.log(`[real-auto] Seed stderr: ${seedResult.stderr}`)
    }
    console.log(`[real-auto] Seed status: ${seedResult.status}`)
    if (seedResult.status !== 0) {
      throw new Error(`Seed script failed (exit ${seedResult.status}): ${seedResult.stderr || seedResult.stdout}`)
    }
    if (!seedResult.stdout.includes('SEED_OK')) {
      throw new Error(`Seed script did not output SEED_OK. Output: ${seedResult.stdout}`)
    }

    // Verify DB file exists and has the session
    const dbPath = path.join(sandbox.hermesHome, 'state.db')
    if (!fs.existsSync(dbPath)) {
      throw new Error(`state.db not found at ${dbPath} after seeding`)
    }
    console.log(`[real-auto] state.db exists: ${dbPath} (${fs.statSync(dbPath).size} bytes)`)

    // 4. Build env + launch desktop
    const env = buildAppEnv(sandbox)
    const { app, page } = await launchDesktop(env)

    fixture = { app, page, mock, mockUrl: mock.url, sandbox, cleanup: async () => { /* cleanup below */ } }

    // Install error banner guard
    installErrorBannerGuard(page)

    // Wait for app to be ready
    const { waitForAppReady } = await import('./fixtures')
    await waitForAppReady(fixture, 120_000)
  })

  test.afterAll(async () => {
    if (fixture) {
      await fixture.app.close().catch(() => undefined)
      await fixture.mock.close()
      fixture.sandbox.cleanup()
    }
  })

  test('open inbox and verify automation row appears', async ({}, testInfo) => {
    const page = fixture!.page
    await forceSize(fixture!.app, 1220, 800)
    await page.waitForTimeout(500)

    const chip = inboxChip(page)
    await chip.click()

    // Wait for inbox content to load
    await waitForInboxReady(page, 15_000)

    expect(await page.locator('h2:has-text("Agent Inbox")').isVisible()).toBe(true)

    // Switch to Automation tab
    const autoTab = page.getByRole('button', { name: 'Automation' })
    await expect(autoTab).toBeVisible()
    await autoTab.click()
    await page.waitForTimeout(500)

    // Verify our seeded row exists
    const seededRow = page.locator(`[data-panel-row="${SESSION_KEY}"]`)
    await expect(seededRow).toBeVisible({ timeout: 10_000 })
    console.log(`[real-auto] Seeded automation row visible: ${SESSION_KEY}`)

    // Capture the automation tab with our row
    await capture(testInfo, 'real-automation-tab.png', 'real-automation-tab', 'real-backend-seeded', page)
  })

  test('select row and verify detail shows title', async ({}, testInfo) => {
    const page = fixture!.page

    // Click the seeded row
    const seededRow = page.locator(`[data-panel-row="${SESSION_KEY}"]`)
    await expect(seededRow).toBeVisible()
    await seededRow.click()
    await page.waitForTimeout(500)

    // Verify detail pane shows the session title
    const titleText = page.getByRole('definition').filter({ hasText: SESSION_TITLE })
    await expect(titleText).toBeVisible({ timeout: 5_000 })
    console.log(`[real-auto] Detail pane shows title: ${SESSION_TITLE}`)

    // Verify session key is shown in detail
    const sessionKeyText = page.getByText(SESSION_KEY)
    await expect(sessionKeyText).toBeVisible()
    console.log(`[real-auto] Detail pane shows session key: ${SESSION_KEY}`)

    // Verify goal section is present
    const goalSection = page.locator('text=Goal').first()
    await expect(goalSection).toBeVisible()
    console.log(`[real-auto] Goal section visible in detail`)

    // Verify the goal title text
    const goalTitle = page.getByText(GOAL_TEXT)
    await expect(goalTitle).toBeVisible()
    console.log(`[real-auto] Goal title visible: ${GOAL_TEXT}`)

    // Capture the detail view
    await capture(testInfo, 'real-automation-detail.png', 'real-automation-detail', 'real-backend-seeded', page)
  })

  test('open session navigates to real seeded durable session without creating new one', async ({}, testInfo) => {
    const page = fixture!.page

    // Compare durable rows before and after navigation using read-only SQLite.
    const countSessions = () => {
      const result = childProcess.spawnSync(PYTHON, ['-c',
        "import sqlite3,sys; from pathlib import Path; c=sqlite3.connect(Path(sys.argv[1]).as_uri()+'?mode=ro',uri=True); print(c.execute('SELECT COUNT(*) FROM sessions').fetchone()[0]); c.close()",
        path.join(fixture!.sandbox.hermesHome, 'state.db')], { encoding: 'utf8' })
      expect(result.status).toBe(0)
      return Number(result.stdout.trim())
    }
    const sessionsBefore = countSessions()
    // Record URL before navigation
    const urlBefore = page.url()
    console.log(`[real-auto] URL before Open session: ${urlBefore}`)

    // Click the seeded row to select it
    const seededRow = page.locator(`[data-panel-row="${SESSION_KEY}"]`)
    await expect(seededRow).toBeVisible()
    await seededRow.click()
    await page.waitForTimeout(300)

    // Click Open session button in the detail pane
    const openBtn = page.getByRole('button', { name: 'Open session' })
    await expect(openBtn).toBeVisible({ timeout: 5_000 })
    await openBtn.click()

    // Wait for navigation to complete
    await page.waitForTimeout(1000)

    // Verify URL now contains the real seeded session key
    const urlAfter = page.url()
    console.log(`[real-auto] URL after Open session: ${urlAfter}`)
    expect(urlAfter).toContain(encodeURIComponent(SESSION_KEY))

    // The inbox panel should no longer be visible (navigated away)
    const panelHeading = page.locator('h2:has-text("Agent Inbox")')
    await expect(panelHeading).not.toBeVisible()
    expect(countSessions()).toBe(sessionsBefore)

    console.log(`[real-auto] Open session navigated to real seeded session: ${SESSION_KEY}`)
  })

  test('escape closes inbox and returns focus to chip', async ({}, testInfo) => {
    const page = fixture!.page

    // Re-open inbox by clicking the chip
    const chip = inboxChip(page)
    await chip.click()
    await page.waitForTimeout(500)

    // Verify inbox is open
    const heading = page.locator('h2:has-text("Agent Inbox")')
    await expect(heading).toBeVisible({ timeout: 5_000 })

    // Press Escape to close
    await closePanelWithEscape(page)

    // Verify the inbox chip has focus after Escape
    await expect(chip).toBeFocused()
  })
})

// ── Test 2: Partial fixture ──────────────────────────────────────────────────

const PARTIAL_ERRORS = ['Fixture source unavailable']

const PARTIAL_FULL = {
  inbox: {
    coverage: {
      approval_scope: 'live gateway approval queue',
      clarify_scope: 'live open sessions only',
      connection_scope: 'active connection and profile only',
      errors: PARTIAL_ERRORS,
      partial: true,
      profile: 'default',
      scanned_sessions: 3,
    },
    items: [
      {
        session_key: 'sess-partial-running',
        title: 'Partial session alpha',
        source: 'cli',
        cwd: '/work/hermes-agent',
        lanes: ['running'],
        goal: { status: 'active', title: 'Partial goal alpha', turns_used: 1, max_turns: 5 },
        loop: null,
        heartbeat: null,
        pending_approval: null,
        pending_clarify: null,
      },
      {
        session_key: 'sess-partial-waiting',
        title: 'Partial session beta',
        source: 'cli',
        cwd: '/work/hermes-agent',
        lanes: ['waiting'],
        goal: { status: 'paused', title: 'Partial goal beta', wait_barrier: 'waiting for input' },
        loop: null,
        heartbeat: null,
        pending_approval: null,
        pending_clarify: null,
      },
    ],
    counts: { needs_you: 0, running: 1, waiting: 1, scheduled: 0, total: 2 },
    badge: 'none',
  },
}

test.describe('partial fixture inbox', () => {
  let fixture: MockBackendFixture | null = null

  test.beforeAll(async () => {
    assertDistBuilt()

    const mock = await startMockServer()
    const sandbox = createSandbox('partial')
    writeMockProviderConfig(sandbox.hermesHome, mock.url)
    writeEnvFile(sandbox.hermesHome)

    const env = buildAppEnv(sandbox)
    const { app, page } = await launchDesktop(env)
    installErrorBannerGuard(page)

    fixture = { app, page, mock, mockUrl: mock.url, sandbox, cleanup: async () => {} }

    // Patch WebSocket to intercept inbox.list and return partial data
    await page.addInitScript((fixtureJson: string) => {
      const fixtureData = JSON.parse(fixtureJson)
      const origSend = WebSocket.prototype.send
      WebSocket.prototype.send = function patchedSend(data: string | ArrayBuffer | Blob) {
        try {
          const raw = typeof data === 'string' ? data : ''
          if (raw) {
            const frame = JSON.parse(raw)
            if (frame.method === 'inbox.list' && frame.id !== undefined) {
              const responseId = frame.id
              const response = { jsonrpc: '2.0', id: responseId, result: fixtureData }
              const ws = this
              setTimeout(() => {
                try {
                  ws.dispatchEvent(new MessageEvent('message', {
                    data: JSON.stringify(response),
                    origin: ws.url,
                  }))
                } catch { /* WebSocket may be closed */ }
              }, 30)
              // This request is fully stubbed; never race a real response.
              return
            }
          }
        } catch { /* Not JSON or not our concern */ }
        return origSend.call(this, data)
      }
    }, JSON.stringify(PARTIAL_FULL))

    // Reload so addInitScript takes effect before gateway connects
    await page.reload({ waitUntil: 'domcontentloaded' })

    const { waitForAppReady } = await import('./fixtures')
    await waitForAppReady(fixture, 120_000)
  })

  test.afterAll(async () => {
    if (fixture) {
      await fixture.app.close().catch(() => undefined)
      await fixture.mock.close()
      fixture.sandbox.cleanup()
    }
  })

  test('partial read warning and retained rows visible', async ({}, testInfo) => {
    const page = fixture!.page
    await forceSize(fixture!.app, 1220, 800)
    await page.waitForTimeout(500)

    const chip = inboxChip(page)
    await chip.click()

    await waitForInboxReady(page, 15_000)

    expect(await page.locator('h2:has-text("Agent Inbox")').isVisible()).toBe(true)

    // Switch to Automation tab to see our partial rows
    const autoTab = page.getByRole('button', { name: 'Automation' })
    await expect(autoTab).toBeVisible()
    await autoTab.click()
    await page.waitForTimeout(500)

    // Verify rows are visible
    const rows = page.locator('[data-panel-row]')
    const rowCount = await rows.count()
    console.log(`[partial] Automation section row count: ${rowCount}`)
    expect(rowCount).toBeGreaterThanOrEqual(1)

    // Verify partial read warning is shown (hasErrors path)
    const partialWarning = page.getByText('Partial read')
    await expect(partialWarning).toBeVisible({ timeout: 5_000 })
    console.log('[partial] Partial read warning visible: true')

    // Capture the partial state
    await capture(testInfo, 'fixture-partial-warning.png', 'fixture-partial-warning', 'response-stub-interception', page)

    // Select a row to verify it's retained
    const firstRow = rows.first()
    await firstRow.click()
    await page.waitForTimeout(300)

    // Verify detail shows session key
    const sessionKey = page.getByText('sess-partial-running')
    await expect(sessionKey).toBeVisible({ timeout: 5_000 })
    console.log('[partial] Session key visible in detail: true')

    await capture(testInfo, 'fixture-partial-detail.png', 'fixture-partial-detail', 'response-stub-interception', page)
  })

  test('refresh button visible, triggers additional inbox.list, retains rows', async ({}, testInfo) => {
    const page = fixture!.page

    // Install a WebSocket message counter BEFORE clicking Refresh so we can
    // detect the additional inbox.list request the button triggers.
    await page.evaluate(() => {
      (window as any).__inboxListRequestCount = 0
      const origSend = WebSocket.prototype.send
      WebSocket.prototype.send = function countedSend(data: string | ArrayBuffer | Blob) {
        try {
          const raw = typeof data === 'string' ? data : ''
          if (raw) {
            const frame = JSON.parse(raw)
            if (frame.method === 'inbox.list') {
              (window as any).__inboxListRequestCount++
            }
          }
        } catch { /* not JSON */ }
        return origSend.call(this, data)
      }
    })

    // The Refresh button MUST be visible — no conditional skip
    const refreshButton = page.getByRole('button', { name: /refresh/i })
    await expect(refreshButton).toBeVisible({ timeout: 5_000 })
    console.log('[partial] Refresh button visible: true')

    // Record request count before click
    const beforeCount: number = await page.evaluate(() => (window as any).__inboxListRequestCount ?? 0)
    console.log(`[partial] inbox.list requests before Refresh: ${beforeCount}`)

    // Click Refresh
    await refreshButton.click()
    await page.waitForTimeout(1500)

    // Verify an additional inbox.list request was fired
    const afterCount: number = await page.evaluate(() => (window as any).__inboxListRequestCount ?? 0)
    console.log(`[partial] inbox.list requests after Refresh: ${afterCount}`)
    expect(afterCount).toBeGreaterThan(beforeCount)

    // Rows should still be visible after refresh (retained from fixture)
    const rows = page.locator('[data-panel-row]')
    const rowCount = await rows.count()
    console.log(`[partial] Row count after Refresh: ${rowCount}`)
    expect(rowCount).toBeGreaterThanOrEqual(1)

    // Verify partial/error state persists (rows retained, warning still present)
    const partialWarning = page.getByText('Partial read')
    await expect(partialWarning).toBeVisible()

    await capture(testInfo, 'fixture-partial-after-retry.png', 'fixture-partial-after-retry', 'response-stub-interception', page)
  })
})

// ── Write receipt ────────────────────────────────────────────────────────────

test.afterAll(async () => {
  fs.mkdirSync(SCREENSHOTS_DIR, { recursive: true })

  const receipt = {
    timestamp: new Date().toISOString(),
    test_file: 'inbox-automation-partial.spec.ts',
    manifest,
    assertions: manifest.map((e) => ({
      state: e.state,
      fixture_kind: e.fixture,
      screenshot: e.path,
    })),
  }

  const receiptPath = path.join(SCREENSHOTS_DIR, 'receipt.json')
  fs.writeFileSync(receiptPath, JSON.stringify(receipt, null, 2), 'utf8')
  console.log(`[inbox-automation-partial] receipt written: ${receiptPath}`)
})
