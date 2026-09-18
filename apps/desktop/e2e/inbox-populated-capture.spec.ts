/**
 * E2E visual capture for POPULATED inbox states.
 *
 * Uses the existing setupMockBackend fixture and patches the gateway's
 * WebSocket transport via page.addInitScript before the page loads.
 * All `inbox.list` calls return populated fixture data.
 *
 * Fixture classification: response-stub interception (not real backend-seeded).
 * Labels this clearly in the manifest.
 *
 * Output: .inbox-work/screenshots/populated-*.png
 *
 * Prerequisite: `npm run build` must have been run so dist/ exists.
 */

import * as fs from 'node:fs'
import * as path from 'node:path'

import { expect, test, type Page } from '@playwright/test'

import {
  type MockBackendFixture,
  setupMockBackend,
  waitForAppReady,
} from './fixtures'

const DESKTOP_ROOT = path.resolve(import.meta.dirname, '..')
const REPO_ROOT = path.resolve(DESKTOP_ROOT, '..', '..')
const SCREENSHOTS_DIR = path.resolve(REPO_ROOT, '.inbox-work', 'screenshots')

// ── populated inbox.list fixtures ─────────────────────────────────────────────

const POPULATED_FULL = {
  inbox: {
    coverage: {
      approval_scope: 'live gateway approval queue',
      clarify_scope: 'live open sessions only',
      connection_scope: 'active connection and profile only',
      errors: [],
      partial: false,
      profile: 'default',
      scanned_sessions: 5,
    },
    items: [
      {
        session_key: 'sess-deploy-approval',
        title: 'Deploy to staging',
        source: 'cli',
        cwd: '/work/hermes-agent',
        lanes: ['needs_you', 'running'],
        goal: { status: 'active', title: 'Ship inbox feature', turns_used: 3, max_turns: 6 },
        loop: null,
        heartbeat: null,
        pending_approval: { count: 1, description: 'pending approval', command_redacted: true },
        pending_clarify: null,
      },
      {
        session_key: 'sess-clarify-question',
        title: 'Code review agent',
        source: 'cli',
        cwd: '/work/hermes-agent',
        lanes: ['needs_you'],
        goal: null,
        loop: null,
        heartbeat: null,
        pending_approval: null,
        pending_clarify: { count: 2 },
      },
      {
        session_key: 'sess-running-loop',
        title: 'Monitor deployment health',
        source: 'cli',
        cwd: '/work/hermes-agent',
        lanes: ['running'],
        goal: null,
        loop: { status: 'active', awaiting_response: true, prompt: 'Check server status' },
        heartbeat: null,
        pending_approval: null,
        pending_clarify: null,
      },
      {
        session_key: 'sess-waiting-goal',
        title: 'Database migration',
        source: 'cli',
        cwd: '/work/hermes-agent',
        lanes: ['waiting'],
        goal: { status: 'paused', title: 'Run DB migration', wait_barrier: 'waiting for approval' },
        loop: null,
        heartbeat: null,
        pending_approval: null,
        pending_clarify: null,
      },
      {
        session_key: 'sess-scheduled-heartbeat',
        title: 'Nightly backup check',
        source: 'cli',
        cwd: '/work/hermes-agent',
        lanes: ['scheduled'],
        goal: null,
        loop: null,
        heartbeat: { status: 'active', prompt: 'Run backup verification', fire_count: 12 },
        pending_approval: null,
        pending_clarify: null,
      },
    ],
    counts: { needs_you: 2, running: 1, waiting: 1, scheduled: 1, total: 5 },
    badge: 'amber',
  },
}

// ── helpers ──────────────────────────────────────────────────────────────────

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

async function waitForInboxConnected(page: Page, timeoutMs = 30_000): Promise<void> {
  await page.waitForFunction(
    () => {
      const text = document.body.textContent ?? ''
      if (text.includes('All clear')) return true
      if (text.includes('Inbox not supported')) return true
      if (text.includes('Partial read')) return true
      if (text.includes('Incomplete data')) return true
      if (document.querySelector('[data-panel-row]')) return true
      return false
    },
    undefined,
    { timeout: timeoutMs },
  )
}

function isPanelVisible(page: Page): Promise<boolean> {
  return page.locator('h2:has-text("Agent Inbox")').isVisible()
}

async function closePanelWithEscape(page: Page): Promise<void> {
  await page.keyboard.press('Escape')
  await page.waitForTimeout(500)
}

// ── manifest ─────────────────────────────────────────────────────────────────

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
  page: Page,
  dims = '1220x800',
): Promise<void> {
  const shotPath = testInfo.outputPath(name)
  await page.screenshot({ path: shotPath })
  record(state, 'response-stub-interception', dims, shotPath)

  const dest = path.join(SCREENSHOTS_DIR, name)
  fs.mkdirSync(path.dirname(dest), { recursive: true })
  fs.copyFileSync(shotPath, dest)
  record(state, 'response-stub-interception', dims, dest)
}

// ── fixture ──────────────────────────────────────────────────────────────────

let fixture: MockBackendFixture | null = null

test.beforeAll(async () => {
  fixture = await setupMockBackend()

  // Patch WebSocket.prototype.send via addInitScript BEFORE the page reloads.
  // This intercepts inbox.list JSON-RPC requests and returns our fixture data.
  await fixture.page.addInitScript((fixtureJson: string) => {
    const fixtureData = JSON.parse(fixtureJson)

    const origSend = WebSocket.prototype.send
    WebSocket.prototype.send = function patchedSend(data: string | ArrayBuffer | Blob) {
      try {
        const raw = typeof data === 'string' ? data : ''
        if (raw) {
          const frame = JSON.parse(raw)
          if (frame.method === 'inbox.list' && frame.id !== undefined) {
            const responseId = frame.id
            const response = {
              jsonrpc: '2.0',
              id: responseId,
              result: fixtureData,
            }
            const ws = this
            setTimeout(() => {
              try {
                ws.dispatchEvent(new MessageEvent('message', {
                  data: JSON.stringify(response),
                  origin: ws.url,
                }))
              } catch { /* WebSocket may be closed */ }
            }, 30)
            return // Fully intercepted: do not race the real backend response.
          }
        }
      } catch {
        // Not JSON or not our concern
      }
      return origSend.call(this, data)
    }
  }, JSON.stringify(POPULATED_FULL))

  // Reload so the addInitScript takes effect before the gateway connects
  await fixture.page.reload({ waitUntil: 'domcontentloaded' })

  await waitForAppReady(fixture, 120_000)
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null

  fs.mkdirSync(SCREENSHOTS_DIR, { recursive: true })
  fs.writeFileSync(
    path.join(SCREENSHOTS_DIR, 'manifest-populated.json'),
    JSON.stringify(manifest, null, 2),
    'utf8',
  )
  console.log(`[inbox-populated-capture] manifest written`)
})

// ── Visual captures ──────────────────────────────────────────────────────────

test.describe('populated inbox visual capture', () => {
  test('populated Needs you — question + approval visible', async ({}, testInfo) => {
    const page = fixture!.page
    await forceSize(fixture!.app, 1220, 800)
    await page.waitForTimeout(500)

    const chip = inboxChip(page)
    await chip.click()

    await page.waitForFunction(
      () => document.querySelector('[data-panel-row]') !== null,
      undefined,
      { timeout: 15_000 },
    )

    expect(await isPanelVisible(page)).toBe(true)

    const rows = page.locator('[data-panel-row]')
    const rowCount = await rows.count()
    console.log(`[populated] Needs you section row count: ${rowCount}`)

    await capture(testInfo, 'inbox-populated-needs.png', 'populated-needs', page)
  })

  test('populated Automation section', async ({}, testInfo) => {
    const page = fixture!.page

    const autoTab = page.getByRole('button', { name: 'Automation' })
    await expect(autoTab).toBeVisible()
    await autoTab.click()
    await page.waitForTimeout(300)

    const rows = page.locator('[data-panel-row]')
    const rowCount = await rows.count()
    console.log(`[populated] Automation section row count: ${rowCount}`)

    await capture(testInfo, 'inbox-populated-automation.png', 'populated-automation', page)

    await page.getByRole('button', { name: 'Needs you' }).click()
    await page.waitForTimeout(300)
  })

  test('populated rows with reusable row navigation', async ({}, testInfo) => {
    const page = fixture!.page

    // Select a row to verify navigation target is a real session key
    const firstRow = page.locator('[data-panel-row="sess-deploy-approval"]')
    await firstRow.click()
    await page.waitForTimeout(300)

    // Detail pane should show session metadata
    const sessionKey = page.getByText('sess-deploy-approval')
    await expect(sessionKey).toBeVisible()

    await capture(testInfo, 'inbox-populated-detail.png', 'populated-detail', page)
  })

  test('narrow populated window', async ({}, testInfo) => {
    const page = fixture!.page

    await closePanelWithEscape(page)
    await page.waitForTimeout(500)

    await forceSize(fixture!.app, 600, 700)
    await page.waitForTimeout(800)

    const chip = inboxChip(page)
    await chip.click()

    await page.waitForFunction(
      () => document.querySelector('[data-panel-row]') !== null,
      undefined,
      { timeout: 15_000 },
    )

    await capture(testInfo, 'inbox-populated-narrow.png', 'populated-narrow', page, '600x700')

    await closePanelWithEscape(page)
  })

  test('escape closes populated panel — focus restored', async ({}, testInfo) => {
    const page = fixture!.page

    await forceSize(fixture!.app, 1220, 800)
    await page.waitForTimeout(500)

    const chip = inboxChip(page)
    await chip.click()

    await page.waitForFunction(
      () => document.querySelector('[data-panel-row]') !== null,
      undefined,
      { timeout: 15_000 },
    )

    expect(await isPanelVisible(page)).toBe(true)

    await closePanelWithEscape(page)

    expect(await isPanelVisible(page)).toBe(false)
    await expect(chip).toBeFocused()

    await capture(testInfo, 'inbox-populated-focus-restored.png', 'populated-focus-restored', page)
  })
})
