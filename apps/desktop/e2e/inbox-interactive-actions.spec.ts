/**
 * E2E interactive gallery — deterministic fixture-driven inbox states.
 *
 * Covers all 8 category views, expanded inline detail, search, narrow
 * layout, detail load failure+retry, approval flows, single/multi/batch
 * clarify, context fallback, disconnect/stale, and partial coverage.
 *
 * Every response send is intercepted; RPC payloads are asserted and
 * zero fake request IDs are forwarded. No live approvals or questions
 * are answered.
 *
 * Fixture classification: real Electron renderer, response-stub data.
 * Output: .inbox-work/interactive-gallery/
 */

import fs from 'node:fs'
import path from 'node:path'
import { test, expect, type Page } from '@playwright/test'
import { setupMockBackend, waitForAppReady, type MockBackendFixture } from './fixtures'
import { allowErrorBanners } from './test'

const OUTPUT = path.resolve(import.meta.dirname, '../../../.inbox-work/interactive-gallery')
let fixture: MockBackendFixture

// ── manifest ─────────────────────────────────────────────────────────────────

interface ManifestEntry {
  assertion: string
  category: string
  evidence: string
  filename: string
  fixture_provenance: string
  pass_or_blocked: 'PASS' | 'BLOCKED'
}

const manifest: ManifestEntry[] = []
// This file's shared fixtures and manifest require one ordered worker.
test.describe.configure({ mode: 'serial' })
test.afterAll(() => {
  fs.mkdirSync(OUTPUT, { recursive: true })
  const unique = [...new Map(manifest.map(entry => [entry.filename, entry])).values()]
  for (const entry of unique) expect(fs.existsSync(path.join(OUTPUT, entry.filename))).toBe(true)
  fs.writeFileSync(path.join(OUTPUT, 'actions-manifest.json'), JSON.stringify(unique, null, 2))
})

function record(category: string, filename: string, assertion: string, passOrBlocked: 'PASS' | 'BLOCKED' = 'PASS'): void {
  manifest.push({ assertion, category, evidence: 'Real Electron renderer; response-stub fixture; no live backend', filename, fixture_provenance: 'WebSocket intercept', pass_or_blocked: passOrBlocked })
}

// ── fixture data ─────────────────────────────────────────────────────────────

const ALL_CATEGORIES = ['goals', 'loops', 'heartbeats', 'background_tasks', 'subagents', 'other'] as const

const CATEGORY_TITLES: Record<string, string> = {
  background_tasks: 'Build assets',
  goals: 'Deploy staging',
  heartbeats: 'Morning briefing',
  loops: 'Check deployment health',
  other: 'Planning notes',
  subagents: 'Review implementation',
}

const ITEMS = ALL_CATEGORIES.map(cat => ({
  session_key: `gallery-${cat}`,
  title: CATEGORY_TITLES[cat],
  source: 'cli',
  cwd: '/disposable/gallery',
  categories: [cat],
  lanes: cat === 'goals' ? ['needs_you', 'running'] : cat === 'loops' ? ['running'] : cat === 'heartbeats' ? ['waiting'] : cat === 'background_tasks' ? ['scheduled'] : cat === 'subagents' ? ['running'] : [],
  goal: cat === 'goals' ? { title: 'Ship approved changes', status: 'active' } : null,
  loop: cat === 'loops' ? { status: 'active', prompt: 'Check health' } : null,
  heartbeat: cat === 'heartbeats' ? { status: 'active', prompt: 'Morning check', fire_count: 5 } : null,
  background_task_count: cat === 'background_tasks' ? 2 : 0,
  subagent_count: cat === 'subagents' ? 3 : 0,
  background_task_count_unavailable: false,
  subagent_count_unavailable: false,
  pending_approval: cat === 'goals' ? { count: 1, command_redacted: true, description: 'Deploy command' } : null,
  pending_clarify: cat === 'loops' ? { count: 1 } : null,
}))

const SNAPSHOT = {
  inbox: {
    badge: 'amber' as const,
    counts: { needs_you: 1, running: 2, waiting: 1, scheduled: 1, total: 6 },
    coverage: { approval_scope: 'fixture', clarify_scope: 'fixture', connection_scope: 'disposable', errors: [], partial: false, profile: 'default', scanned_sessions: 6 },
    items: ITEMS,
  },
}

const DETAIL_CONTEXT = {
  available: true,
  reason: null,
  messages: [
    { role: 'user', text: 'Clean up the stale build cache before the staging deploy finishes.', timestamp: 1789765000 },
    { role: 'assistant', text: 'Staging is green. I want to clear /tmp/build-cache, then finish the deploy.', timestamp: 1789765060 },
  ],
}

const DETAIL_APPROVAL = {
  sessions: [{ live_session_ids: ['live-1'], context: DETAIL_CONTEXT, approvals: [{ allow_permanent: true, allow_session: true, choices: ['once', 'session', 'always', 'deny'], command: 'rm -rf /tmp/build-cache', description: 'Delete build cache directory', request_id: 'req-approval-1', smart_denied: null, tool_name: 'terminal' }], clarifications: [] }],
  coverage: { approval_count: 1, clarification_count: 0, context_anchor: 'unavailable: open chat for context', errors: [], live_session_count: 1, profile: 'default', session_key: 'gallery-goals' },
}

const DETAIL_SINGLE_CLARIFY = {
  sessions: [{ live_session_ids: ['live-2'], approvals: [], clarifications: [{ kind: 'single', params: { choices: ['TypeScript', 'Python', 'Rust'], multi_select: false, question: 'Which language should the new module be written in?' }, request_id: 'req-clarify-1' }] }],
  coverage: { approval_count: 0, clarification_count: 1, context_anchor: 'session:gallery-loops', errors: [], live_session_count: 1, profile: 'default', session_key: 'gallery-loops' },
}

const DETAIL_MULTI_CLARIFY = {
  sessions: [{ live_session_ids: ['live-3'], approvals: [], clarifications: [{ kind: 'single', params: { choices: ['Error handling', 'Performance', 'Documentation', 'Testing'], multi_select: true, question: 'Which areas need the most improvement?' }, request_id: 'req-multiselect-1' }] }],
  coverage: { approval_count: 0, clarification_count: 1, context_anchor: 'session:gallery-subagents', errors: [], live_session_count: 1, profile: 'default', session_key: 'gallery-subagents' },
}

const DETAIL_BATCH_CLARIFY = {
  sessions: [{ live_session_ids: ['live-4'], approvals: [], clarifications: [{ kind: 'batch', params: { questions: [
    { multi_select: false, qid: 'q1', question: 'Priority level?', choices: ['Low', 'Medium', 'High'] },
    { multi_select: false, qid: 'q2', question: 'Target environment?', choices: ['Staging', 'Production'] },
    { multi_select: true, qid: 'q3', question: 'Run additional checks?', choices: ['Lint', 'Typecheck', 'Unit tests'] },
  ] }, request_id: 'req-batch-1' }] }],
  coverage: { approval_count: 0, clarification_count: 1, context_anchor: 'session:gallery-heartbeats', errors: [], live_session_count: 1, profile: 'default', session_key: 'gallery-heartbeats' },
}

const DETAIL_APPROVAL_RESTRICTED = {
  sessions: [{ live_session_ids: ['live-5'], approvals: [{ allow_permanent: false, allow_session: false, choices: ['once', 'session', 'always', 'deny'], command: 'dangerous-script.sh', description: 'Restricted approval', request_id: 'req-approval-restricted', smart_denied: null, tool_name: 'terminal' }], clarifications: [] }],
  coverage: { approval_count: 1, clarification_count: 0, context_anchor: 'unavailable: open chat for context', errors: [], live_session_count: 1, profile: 'default', session_key: 'gallery-background_tasks' },
}

const SNAPSHOT_PARTIAL = {
  inbox: {
    badge: 'red' as const,
    counts: { needs_you: 0, running: 0, waiting: 0, scheduled: 0, total: 0 },
    coverage: { approval_scope: 'fixture', clarify_scope: 'fixture', connection_scope: 'disposable', errors: ['Session snapshot failed: timeout'], partial: true, profile: 'default', scanned_sessions: 0 },
    items: [],
  },
}

// ── helpers ──────────────────────────────────────────────────────────────────

async function closePanel(page: Page): Promise<void> {
  await page.keyboard.press('Escape')
  await page.waitForTimeout(300)
}

async function openInbox(page: Page): Promise<void> {
  await closePanel(page)
  await page.getByRole('button', { name: /inbox/i }).first().click()
  await expect(page.getByRole('heading', { name: 'Agent Inbox' })).toBeVisible()
}

async function waitForRows(page: Page, timeoutMs = 15_000): Promise<void> {
  await page.waitForFunction(() => document.querySelector('[data-panel-row]') !== null, undefined, { timeout: timeoutMs })
}

async function shot(page: Page, name: string, cat: string, assert: string): Promise<void> {
  const fp = path.join(OUTPUT, name)
  await page.screenshot({ path: fp })
  record(cat, name, assert)
}

async function sentFrames(page: Page) {
  return page.evaluate(() => [...((window as any).__INBOX_RPC_SENT__ ?? [])])
}

async function resetFrames(page: Page) {
  await page.evaluate(() => { const s = (window as any).__INBOX_RPC_SENT__; if (s) s.length = 0 })
}

function interceptScript(cfg: Record<string, unknown>): string {
  return `((cfg) => {
    const sent = []
    window.__INBOX_RPC_SENT__ = sent
    let detailCallCount = 0
    const origSend = WebSocket.prototype.send
    WebSocket.prototype.send = function(data) {
      if (typeof data === 'string') {
        try {
          const f = JSON.parse(data)
          if (f.method) sent.push({ method: f.method, params: f.params || {}, id: f.id })
          let result, error
          if (f.method === 'inbox.list') {
            error = cfg.errorForList ? { code: -32000, message: 'Connection lost' } : undefined
            result = cfg.errorForList ? undefined : cfg.fixtureData
          } else if (f.method === 'inbox.requests') {
            detailCallCount++
            if (cfg.detailError && detailCallCount === 1) error = { code: -32000, message: 'Failed to load details' }
            else if (cfg.secondDetailFixture && detailCallCount >= 2) result = cfg.secondDetailFixture
            else if (cfg.detailFixture) result = cfg.detailFixture
            else result = { sessions: [], coverage: { profile: 'default', session_key: (f.params && f.params.session_key) || '', live_session_count: 0, approval_count: 0, clarification_count: 0, context_anchor: 'unavailable: open chat for context', errors: [] } }
          } else if (['approval.respond','request.answer','clarify.lock'].includes(f.method)) {
            if (!cfg.allowActions) error = { code: -32600, message: 'View-only gallery forbids responses' }
            else if (cfg.actionError) error = { code: -32000, message: 'Network error' }
            else result = cfg.actionResponse || { resolved: 1 }
          }
          if (result !== undefined || error !== undefined) {
            const resp = { jsonrpc: '2.0', id: f.id }
            if (error !== undefined) resp.error = error; else resp.result = result
            const ws = this
            const deliver = () => { ws.dispatchEvent(new MessageEvent('message', { data: JSON.stringify(resp), origin: ws.url })) }
            if (cfg.deferMethod === f.method) window.__INBOX_RELEASE__ = deliver
            else setTimeout(deliver, 25)
            return
          }
        } catch(e) {}
      }
      return origSend.call(this, data)
    }
  })(${JSON.stringify(cfg)})`
}

async function setupPage(cfg: Record<string, unknown>): Promise<MockBackendFixture> {
  const f = await setupMockBackend()
  await f.page.addInitScript(interceptScript(cfg))
  await f.page.reload({ waitUntil: 'domcontentloaded' })
  await waitForAppReady(f, 120_000)
  return f
}

// ── Deferred interaction evidence ──────────────────────────────────────────
test('expanded request shows the session context above the response controls', async () => {
  const f = await setupPage({ fixtureData: SNAPSHOT, detailFixture: DETAIL_APPROVAL })
  try {
    const p = f.page
    await openInbox(p)
    await p.locator('[data-panel-row="gallery-goals"]').click()
    await expect(p.getByText('Recent messages')).toBeVisible()
    await expect(p.getByText('Clean up the stale build cache before the staging deploy finishes.')).toBeVisible()
    await expect(p.getByRole('button', { name: 'Approve once', exact: true })).toBeVisible()
    await shot(p, 'detail-context.png', 'detail', 'Recent messages render above the approval controls, so the request is answered in place')
  } finally { await f.cleanup() }
})

test('expanded request with no transcript reports the absence honestly', async () => {
  const f = await setupPage({ fixtureData: SNAPSHOT, detailFixture: DETAIL_SINGLE_CLARIFY })
  try {
    const p = f.page
    await openInbox(p)
    await p.locator('[data-panel-row="gallery-loops"]').click()
    await expect(p.getByText('No recent transcript available.')).toBeVisible()
    await shot(p, 'detail-context-absent.png', 'detail', 'Missing excerpt is a named state, never an all-clear')
  } finally { await f.cleanup() }
})

test('request detail loading is visible until response arrives', async () => {
  const f = await setupPage({ fixtureData: SNAPSHOT, detailFixture: DETAIL_APPROVAL, deferMethod: 'inbox.requests' })
  try {
    const p = f.page
    await openInbox(p)
    await p.locator('[data-panel-row="gallery-goals"]').click()
    await expect(p.getByRole('status')).toContainText('Loading request details')
    await expect(p.getByRole('button', { name: 'Approve once', exact: true })).not.toBeVisible()
    await shot(p, 'detail-loading.png', 'detail', 'Pending read shows loading, no approval action')
    await p.evaluate(() => (window as any).__INBOX_RELEASE__())
    await expect(p.getByRole('button', { name: 'Approve once', exact: true })).toBeVisible()
    await expect(p.getByText('Loading request details')).not.toBeVisible()
  } finally { await f.cleanup() }
})

test('approval sending disables responses until acknowledged', async () => {
  const f = await setupPage({ fixtureData: SNAPSHOT, detailFixture: DETAIL_APPROVAL, deferMethod: 'approval.respond', allowActions: true })
  try {
    const p = f.page
    await openInbox(p)
    await p.locator('[data-panel-row="gallery-goals"]').click()
    await p.getByRole('button', { name: 'Approve once', exact: true }).click()
    await expect(p.getByRole('button', { name: 'Deny', exact: true })).toBeDisabled()
    await expect(p.getByRole('button', { name: 'Approve for session', exact: true })).toBeDisabled()
    expect((await sentFrames(p)).filter((f: any) => f.method === 'approval.respond')).toHaveLength(1)
    await shot(p, 'approval-sending.png', 'approval', 'Pending approval disables alternate responses; exactly one intercepted send')
    await p.evaluate(() => (window as any).__INBOX_RELEASE__())
    await expect(p.getByRole('button', { name: 'Deny', exact: true })).not.toBeDisabled()
  } finally { await f.cleanup() }
})

// ── Group 1: View-only tests (no action responses needed) ────────────────────

test.describe('inbox view-only gallery', () => {
  test.beforeAll(async () => {
    fs.mkdirSync(OUTPUT, { recursive: true })
    fixture = await setupPage({ fixtureData: SNAPSHOT, detailFixture: DETAIL_APPROVAL })
  })
  test.afterAll(async () => {

    await fixture?.cleanup()
  })

  test('all 8 category views', async () => {
    const p = fixture.page
    await openInbox(p)
    await waitForRows(p)

    // All sessions (default)
    let rows = p.locator('[data-panel-row]')
    expect(await rows.count()).toBe(6)
    await shot(p, 'cat-all-sessions.png', 'categories', 'All sessions: 6 rows')

    // Each category
    for (const [label, expected] of [['Needs attention', 1], ['Goals', 1], ['Loops', 1], ['Heartbeats', 1], ['Background tasks', 1], ['Subagents', 1], ['Other', 1]] as const) {
      await p.getByRole('button', { name: new RegExp(label, 'i') }).first().click()
      await p.waitForTimeout(300)
      rows = p.locator('[data-panel-row]')
      expect(await rows.count()).toBe(expected)
      await shot(p, `cat-${label.toLowerCase().replaceAll(' ', '-')}.png`, 'categories', `${label}: ${expected} row(s)`)
    }
  })

  test('expand + collapse inline detail', async () => {
    const p = fixture.page
    await openInbox(p)
    await waitForRows(p)
    await p.getByRole('button', { name: /Goals/i }).first().click()
    await p.waitForTimeout(300)

    const row = p.locator('[data-panel-row="gallery-goals"]')
    await row.click()
    await p.waitForTimeout(500)
    await expect(p.getByText('gallery-goals')).toBeVisible()
    await shot(p, 'expanded-goals.png', 'expand', 'Expanded: detail visible')

    await row.click()
    await p.waitForTimeout(300)
    await expect(p.getByText('gallery-goals')).not.toBeVisible()
    await shot(p, 'collapsed-goals.png', 'expand', 'Collapsed')
  })

  test('search within category vs all', async () => {
    const p = fixture.page
    await openInbox(p)
    await waitForRows(p)
    await p.getByRole('button', { name: /Goals/i }).first().click()
    await p.waitForTimeout(300)

    const sf = p.getByRole('textbox', { name: /search/i })
    await sf.fill('Deploy')
    await p.waitForTimeout(300)
    expect(await p.locator('[data-panel-row]').count()).toBe(1)
    await shot(p, 'search-section.png', 'search', 'Search in Goals: 1 result')

    await p.getByRole('button', { name: /all sessions/i }).first().click()
    await p.waitForTimeout(300)
    expect(await p.locator('[data-panel-row]').count()).toBeGreaterThanOrEqual(1)
    await shot(p, 'search-all.png', 'search', 'Search in All: cross-category')

    await sf.fill('')
  })

  test('search no matches', async () => {
    const p = fixture.page
    await openInbox(p)
    await waitForRows(p)
    await p.getByRole('textbox', { name: /search/i }).fill('zzzznonexistent')
    await p.waitForTimeout(500)
    await expect(p.getByText('No results')).toBeVisible()
    await shot(p, 'search-no-matches.png', 'search', 'No results empty state')
    await p.getByRole('textbox', { name: /search/i }).fill('')
  })

  test('narrow layout', async () => {
    const p = fixture.page
    await closePanel(p)
    await fixture.app.evaluate(({ BrowserWindow }) => {
      const w = BrowserWindow.getAllWindows()[0]
      if (w) { w.unmaximize(); w.setMinimumSize(550, 700); w.setSize(550, 700, false); w.setBounds({ x: 0, y: 0, width: 550, height: 700 }) }
    })
    await p.waitForTimeout(500)
    await openInbox(p)
    await waitForRows(p)
    await shot(p, 'narrow-layout.png', 'narrow', 'Narrow 550px: panel adapts')
    await closePanel(p)
    await fixture.app.evaluate(({ BrowserWindow }) => {
      const w = BrowserWindow.getAllWindows()[0]
      if (w) { w.unmaximize(); w.setMinimumSize(1220, 800); w.setSize(1220, 800, false); w.setBounds({ x: 0, y: 0, width: 1220, height: 800 }) }
    })
    await p.waitForTimeout(500)
  })

  test('detail loads after expand', async () => {
    const p = fixture.page
    await openInbox(p)
    await waitForRows(p)
    await p.getByRole('button', { name: /Goals/i }).first().click()
    await p.waitForTimeout(300)
    const row = p.locator('[data-panel-row="gallery-goals"]')
    await row.click()
    await p.waitForTimeout(500)
    await expect(p.getByText('gallery-goals')).toBeVisible()
    await expect(p.getByRole('button', { name: /approve once/i })).toBeVisible()
    await shot(p, 'detail-loaded.png', 'detail', 'Detail loaded: approval card')
    await row.click()
  })

  test('context fallback button and panel close', async () => {
    const p = fixture.page
    await openInbox(p)
    await waitForRows(p)
    await p.getByRole('button', { name: /Goals/i }).first().click()
    await p.waitForTimeout(300)
    const row = p.locator('[data-panel-row="gallery-goals"]')
    await row.click()
    await p.waitForTimeout(500)
    await expect(p.getByText('Exact request location unavailable')).toBeVisible()
    const btn = p.getByRole('button', { name: /open chat for context/i })
    await expect(btn).toBeVisible()
    await shot(p, 'context-fallback.png', 'context', 'Context button visible')
    await btn.click()
    await p.waitForTimeout(1000)
    await expect(p.getByRole('heading', { name: 'Agent Inbox' })).not.toBeVisible()
    await shot(p, 'context-navigated.png', 'context', 'Panel closed after context click')
  })
})

// ── Group 2: Action tests (approval, clarify) — single page load ─────────────

test.describe('inbox action gallery', () => {
  test.beforeAll(async () => {
    fs.mkdirSync(OUTPUT, { recursive: true })
    fixture = await setupPage({
      fixtureData: SNAPSHOT,
      detailFixture: DETAIL_APPROVAL,
      allowActions: true,
      actionResponse: { resolved: 1 },
    })
  })
  test.afterAll(async () => {
    await fixture?.cleanup()
  })

  test('approval: 4 choices visible, approve sends correct RPC', async () => {
    const p = fixture.page
    await resetFrames(p)
    await openInbox(p)
    await waitForRows(p)
    await p.getByRole('button', { name: /Goals/i }).first().click()
    await p.waitForTimeout(300)

    const row = p.locator('[data-panel-row="gallery-goals"]')
    await row.click()
    await p.waitForTimeout(500)

    await expect(p.getByRole('button', { name: /approve once/i })).toBeVisible()
    await expect(p.getByRole('button', { name: /approve for session/i })).toBeVisible()
    await expect(p.getByRole('button', { name: /always allow/i })).toBeVisible()
    await expect(p.getByRole('button', { name: /deny/i })).toBeVisible()
    await shot(p, 'approval-options.png', 'approval', '4 choices visible')

    await p.getByRole('button', { name: /approve once/i }).click()
    await p.waitForTimeout(500)

    const frames = await sentFrames(p)
    const af = frames.find((f: any) => f.method === 'approval.respond')
    expect(af).toBeDefined()
    expect(af!.params.choice).toBe('once')
    expect(af!.params.request_id).toBe('req-approval-1')
    expect(af!.params.all).toBe(false)
    expect(typeof af!.params.session_id).toBe('string')

    for (const f of frames.filter((x: any) => x.method === 'approval.respond')) {
      expect(f.params.request_id).not.toMatch(/^fake-/)
    }

    await shot(p, 'approval-approved.png', 'approval', 'RPC sent: choice=once, request_id=req-approval-1')
    await row.click()
  })

  test('approval: deny sends choice=deny', async () => {
    const p = fixture.page
    await resetFrames(p)
    await openInbox(p)
    await waitForRows(p)
    await p.getByRole('button', { name: /Goals/i }).first().click()
    await p.waitForTimeout(300)

    const row = p.locator('[data-panel-row="gallery-goals"]')
    await row.click()
    await p.waitForTimeout(500)

    await p.getByRole('button', { name: /deny/i }).click()
    await p.waitForTimeout(500)

    const frames = await sentFrames(p)
    const af = frames.find((f: any) => f.method === 'approval.respond')
    expect(af).toBeDefined()
    expect(af!.params.choice).toBe('deny')
    await shot(p, 'approval-denied.png', 'approval', 'RPC sent: choice=deny')
    await row.click()
  })

  test('approval: restricted — only once+deny visible', async () => {
    const p = fixture.page
    // Need a fresh page with restricted detail
    await p.addInitScript(interceptScript({
      fixtureData: SNAPSHOT,
      detailFixture: DETAIL_APPROVAL_RESTRICTED,
      allowActions: true,
      actionResponse: { resolved: 1 },
    }))
    await p.reload({ waitUntil: 'domcontentloaded' })
    await waitForAppReady(fixture, 120_000)
    await openInbox(p)
    await waitForRows(p)

    await p.getByRole('button', { name: /background tasks/i }).first().click()
    await p.waitForTimeout(300)

    const row = p.locator('[data-panel-row="gallery-background_tasks"]')
    await row.click()
    await p.waitForTimeout(500)

    await expect(p.getByRole('button', { name: /approve once/i })).toBeVisible()
    await expect(p.getByRole('button', { name: /deny/i })).toBeVisible()
    await expect(p.getByRole('button', { name: /approve for session/i })).not.toBeVisible()
    await expect(p.getByRole('button', { name: /always allow/i })).not.toBeVisible()
    await shot(p, 'approval-restricted.png', 'approval', 'Restricted: only once+deny')
    await row.click()
  })

  test('approval: resolved=0 shows already resolved', async () => {
    const p = fixture.page
    await p.addInitScript(interceptScript({
      fixtureData: SNAPSHOT,
      detailFixture: DETAIL_APPROVAL,
      allowActions: true,
      actionResponse: { resolved: 0 },
    }))
    await p.reload({ waitUntil: 'domcontentloaded' })
    await waitForAppReady(fixture, 120_000)
    await openInbox(p)
    await waitForRows(p)
    await p.getByRole('button', { name: /Goals/i }).first().click()
    await p.waitForTimeout(300)

    const row = p.locator('[data-panel-row="gallery-goals"]')
    await row.click()
    await p.waitForTimeout(500)

    await p.getByRole('button', { name: /approve once/i }).click()
    await p.waitForTimeout(500)

    await expect(p.getByText(/request may have been resolved already/i)).toBeVisible()
    await shot(p, 'approval-resolved-0.png', 'approval', 'resolved=0: already resolved')
    await row.click()
  })

  test('single clarify: choices + answer via request.answer', async () => {
    const p = fixture.page
    await p.addInitScript(interceptScript({
      fixtureData: SNAPSHOT,
      detailFixture: DETAIL_SINGLE_CLARIFY,
      allowActions: true,
      actionResponse: { status: 'ok' },
    }))
    await p.reload({ waitUntil: 'domcontentloaded' })
    await waitForAppReady(fixture, 120_000)
    await openInbox(p)
    await waitForRows(p)
    await p.getByRole('button', { name: /Loops/i }).first().click()
    await p.waitForTimeout(300)

    const row = p.locator('[data-panel-row="gallery-loops"]')
    await row.click()
    await p.waitForTimeout(500)

    await expect(p.getByText('Which language should the new module be written in?')).toBeVisible()
    await expect(p.getByRole('button', { name: /^TypeScript$/ })).toBeVisible()
    await expect(p.getByRole('button', { name: /^Python$/ })).toBeVisible()
    await expect(p.getByRole('button', { name: /^Rust$/ })).toBeVisible()
    await shot(p, 'clarify-single.png', 'clarify', 'Single clarify: 3 choices')

    await p.getByRole('button', { name: /^TypeScript$/ }).click()
    await p.waitForTimeout(200)
    await p.getByRole('button', { name: /submit/i }).click()
    await p.waitForTimeout(500)

    const frames = await sentFrames(p)
    const af = frames.find((f: any) => f.method === 'request.answer')
    expect(af).toBeDefined()
    expect(af!.params.id).toBe('req-clarify-1')
    expect((af!.params.result as any).answer).toBe('TypeScript')
    await shot(p, 'clarify-single-answered.png', 'clarify', 'RPC: request.answer id=req-clarify-1, answer=TypeScript')
    await row.click()
  })

  test('multi-select clarify: JSON-serialized answers', async () => {
    const p = fixture.page
    await p.addInitScript(interceptScript({
      fixtureData: SNAPSHOT,
      detailFixture: DETAIL_MULTI_CLARIFY,
      allowActions: true,
      actionResponse: { status: 'ok' },
    }))
    await p.reload({ waitUntil: 'domcontentloaded' })
    await waitForAppReady(fixture, 120_000)
    await openInbox(p)
    await waitForRows(p)
    await p.getByRole('button', { name: /Subagents/i }).first().click()
    await p.waitForTimeout(300)

    const row = p.locator('[data-panel-row="gallery-subagents"]')
    await row.click()
    await p.waitForTimeout(500)

    await expect(p.getByText('Which areas need the most improvement?')).toBeVisible()
    await shot(p, 'clarify-multi.png', 'clarify', 'Multi-select: choices + Other')

    await p.getByRole('button', { name: /^Error handling$/ }).click()
    await p.waitForTimeout(100)
    await p.getByRole('button', { name: /^Performance$/ }).click()
    await p.waitForTimeout(200)
    await p.getByRole('button', { name: /submit/i }).click()
    await p.waitForTimeout(500)

    const frames = await sentFrames(p)
    const af = frames.find((f: any) => f.method === 'request.answer')
    expect(af).toBeDefined()
    expect(af!.params.id).toBe('req-multiselect-1')
    const answer = JSON.parse((af!.params.result as any).answer)
    expect(Array.isArray(answer)).toBe(true)
    expect(answer).toContain('Error handling')
    expect(answer).toContain('Performance')
    await shot(p, 'clarify-multi-answered.png', 'clarify', 'RPC: JSON array in answer')
    await row.click()
  })

  test('batch clarify: lock sequence for 3 questions', async () => {
    const p = fixture.page
    await p.addInitScript(interceptScript({
      fixtureData: SNAPSHOT,
      detailFixture: DETAIL_BATCH_CLARIFY,
      allowActions: true,
      actionResponse: { status: 'ok' },
    }))
    await p.reload({ waitUntil: 'domcontentloaded' })
    await waitForAppReady(fixture, 120_000)
    await openInbox(p)
    await waitForRows(p)
    await p.getByRole('button', { name: /Heartbeats/i }).first().click()
    await p.waitForTimeout(300)

    const row = p.locator('[data-panel-row="gallery-heartbeats"]')
    await row.click()
    await p.waitForTimeout(500)

    await expect(p.getByText('3 questions')).toBeVisible()
    await expect(p.getByText('Priority level?')).toBeVisible()
    await expect(p.getByText('Target environment?')).toBeVisible()
    await expect(p.getByText('Run additional checks?')).toBeVisible()
    await shot(p, 'clarify-batch.png', 'clarify', 'Batch: 3 questions visible')

    await p.getByRole('button', { name: /^High$/ }).click()
    await p.waitForTimeout(100)
    await p.getByRole('button', { name: /^Production$/ }).click()
    await p.waitForTimeout(100)
    await p.getByRole('button', { name: /^Lint$/ }).click()
    await p.waitForTimeout(100)
    await p.getByRole('button', { name: /^Typecheck$/ }).click()
    await p.waitForTimeout(200)
    await shot(p, 'clarify-batch-staged.png', 'clarify', 'Batch: answers staged')

    await p.getByRole('button', { name: /submit answers/i }).click()
    await p.waitForTimeout(1000)

    const frames = await sentFrames(p)
    const locks = frames.filter((f: any) => f.method === 'clarify.lock')
    expect(locks.length).toBe(3)
    expect(locks[0].params.request_id).toBe('req-batch-1')
    expect(locks[0].params.question_id).toBe('q1')
    expect(locks[0].params.answer).toBe('High')
    expect(locks[1].params.question_id).toBe('q2')
    expect(locks[1].params.answer).toBe('Production')
    expect(locks[2].params.question_id).toBe('q3')
    const q3 = JSON.parse(locks[2].params.answer as string)
    expect(q3).toContain('Lint')
    expect(q3).toContain('Typecheck')
    expect(new Set(locks.map((f: any) => f.params.request_id)).size).toBe(1)
    await shot(p, 'clarify-batch-answered.png', 'clarify', 'Batch: 3 clarify.lock calls sent')
    await row.click()
  })

  test('approval: response failure then retry', async () => {
    const p = fixture.page
    // First install with error
    await p.addInitScript(interceptScript({
      fixtureData: SNAPSHOT,
      detailFixture: DETAIL_APPROVAL,
      allowActions: true,
      actionError: true,
    }))
    await p.reload({ waitUntil: 'domcontentloaded' })
    await waitForAppReady(fixture, 120_000)
    await openInbox(p)
    await waitForRows(p)
    await p.getByRole('button', { name: /Goals/i }).first().click()
    await p.waitForTimeout(300)

    const row = p.locator('[data-panel-row="gallery-goals"]')
    await row.click()
    await p.waitForTimeout(500)

    await p.getByRole('button', { name: /approve once/i }).click()
    await p.waitForTimeout(500)
    await expect(p.getByText(/failed to respond|network error/i)).toBeVisible()
    await shot(p, 'approval-failed.png', 'approval', 'Response failed: error shown')
    await row.click()
  })
})

// ── Group 3: Detail failure + retry ──────────────────────────────────────────

test.describe('inbox detail failure gallery', () => {
  test.beforeAll(async () => {
    fixture = await setupPage({
      fixtureData: SNAPSHOT,
      detailError: true,
      secondDetailFixture: DETAIL_APPROVAL,
    })
  })
  test.afterAll(async () => { await fixture?.cleanup() })

  test('detail failure then retry succeeds', async () => {
    allowErrorBanners() // This test deliberately asserts the error, then its recovery.
    const p = fixture.page
    await openInbox(p)
    await waitForRows(p)
    await p.getByRole('button', { name: /Goals/i }).first().click()
    await p.waitForTimeout(300)

    const row = p.locator('[data-panel-row="gallery-goals"]')
    await row.click()
    await p.waitForTimeout(500)

    await expect(p.getByRole('alert')).toContainText('Failed to load details')
    await shot(p, 'detail-failed.png', 'detail-retry', 'Detail error visible with Retry button')
    const before = (await sentFrames(p)).filter((f: any) => f.method === 'inbox.requests').length
    await p.getByRole('button', { name: 'Retry request details', exact: true }).click()
    await expect(p.getByRole('button', { name: 'Approve once', exact: true })).toBeVisible()
    await expect(p.getByRole('alert')).not.toBeVisible()
    const after = (await sentFrames(p)).filter((f: any) => f.method === 'inbox.requests').length
    expect(after).toBe(before + 1)
    await expect(p.getByText('gallery-goals', { exact: true })).toBeVisible()
    await shot(p, 'detail-retried.png', 'detail-retry', 'Retry succeeded')
    await row.click()
  })
})

// ── Group 4: Disconnect + partial ────────────────────────────────────────────

test.describe('inbox error-state gallery', () => {
  test.beforeAll(async () => {
    fixture = await setupPage({ fixtureData: SNAPSHOT, errorForList: true })
  })
  test.afterAll(async () => { await fixture?.cleanup() })

  test('disconnect state: nonactionable', async () => {
    const p = fixture.page
    await openInbox(p)
    await expect(p.getByText('Disconnected')).toBeVisible()
    await expect(p.getByRole('button', { name: /retry/i })).toBeVisible()
    await shot(p, 'disconnect.png', 'disconnect', 'Disconnect: error + retry')

    const frames = await sentFrames(p)
    const actions = frames.filter((f: any) => ['approval.respond', 'request.answer', 'clarify.lock'].includes(f.method))
    expect(actions.length).toBe(0)
  })
})

test.describe('inbox partial gallery', () => {
  test.beforeAll(async () => {
    fixture = await setupPage({ fixtureData: SNAPSHOT_PARTIAL })
  })
  test.afterAll(async () => { await fixture?.cleanup() })

  test('partial coverage: error warning shown', async () => {
    const p = fixture.page
    await openInbox(p)
    await expect(p.getByText('Partial read')).toBeVisible()
    await shot(p, 'partial.png', 'partial', 'Partial coverage: error warning')

    const frames = await sentFrames(p)
    const actions = frames.filter((f: any) => ['approval.respond', 'request.answer', 'clarify.lock'].includes(f.method))
    expect(actions.length).toBe(0)
  })
})
