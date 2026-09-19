import fs from 'node:fs'
import path from 'node:path'
import { test, expect } from '@playwright/test'
import { setupMockBackend, waitForAppReady, type MockBackendFixture } from './fixtures'

const output = path.resolve(import.meta.dirname, '../../../.inbox-work/interactive-gallery')
let fixture: MockBackendFixture
const manifest: { view: string; file: string; evidence: string }[] = []
const categories = ['goals', 'loops', 'heartbeats', 'background_tasks', 'subagents', 'other']
const items = categories.map((cat, i) => ({
  session_key: `gallery-${cat}`, title: ['Deploy staging', 'Check deployment health', 'Morning briefing', 'Build assets', 'Review implementation', 'Planning notes'][i],
  source: 'cli', cwd: '/disposable/gallery', categories: [cat], lanes: i === 0 ? ['needs_you', 'running'] : [],
  goal: cat === 'goals' ? { title: 'Ship approved changes', status: 'active' } : null,
  loop: cat === 'loops' ? { status: 'paused', prompt: 'Check health' } : null,
  heartbeat: cat === 'heartbeats' ? { status: 'active' } : null,
  background_task_count: cat === 'background_tasks' ? 2 : 0, subagent_count: cat === 'subagents' ? 3 : 0,
  background_task_count_unavailable: false, subagent_count_unavailable: false,
  pending_approval: i === 0 ? { count: 1, command_redacted: true, description: 'pending approval' } : null,
  pending_clarify: null
}))

test.beforeAll(async () => {
  fs.mkdirSync(output, { recursive: true })
  fixture = await setupMockBackend()
  await fixture.page.addInitScript(({ items }) => {
    const send = WebSocket.prototype.send
    WebSocket.prototype.send = function(data) {
      if (typeof data === 'string') {
        const f = JSON.parse(data)
        let result: unknown
        if (f.method === 'inbox.list') result = { inbox: { items, badge: 'amber', counts: { total: 6, needs_you: 1, running: 1, waiting: 0, scheduled: 0 }, coverage: { profile: 'default', connection_scope: 'this disposable connection only', approval_scope: 'fixture', clarify_scope: 'fixture', scanned_sessions: 6, partial: false, errors: [] } } }
        if (f.method === 'inbox.requests') result = { sessions: [{ live_session_ids: [], approvals: [], clarifications: [] }], coverage: { profile: 'default', session_key: f.params.session_key, live_session_count: 0, approval_count: 0, clarification_count: 0, context_anchor: 'unavailable: open chat for context', errors: [] } }
        // No action RPC is permitted through this gallery fixture.
        if (['approval.respond', 'request.answer', 'clarify.lock'].includes(f.method)) throw new Error('View-only gallery forbids responses')
        if (result !== undefined) {
          setTimeout(() => this.dispatchEvent(new MessageEvent('message', { data: JSON.stringify({ jsonrpc: '2.0', id: f.id, result }) })), 25)
          return
        }
      }
      return send.call(this, data)
    }
  }, { items })
  await fixture.page.reload({ waitUntil: 'domcontentloaded' })
  await waitForAppReady(fixture, 120000)
})

test.afterAll(async () => {
  fs.writeFileSync(path.join(output, 'views-manifest.json'), JSON.stringify(manifest, null, 2))
  await fixture?.cleanup()
})

test('capture category views from the real renderer', async () => {
  const page = fixture.page
  await page.getByRole('button', { name: /Action Center/ }).first().click()
  await expect(page.getByRole('heading', { name: 'Action Center' })).toBeVisible()
  for (const label of ['All sessions', 'Needs attention', 'Goals', 'Loops', 'Heartbeats', 'Background tasks', 'Subagents', 'Other']) {
    await page.getByRole('button', { name: new RegExp(label, 'i') }).first().click()
    await expect(page.locator('[data-panel-row]').first()).toBeVisible()
    const file = path.join(output, label.toLowerCase().replaceAll(' ', '-') + '.png')
    await page.screenshot({ path: file })
    manifest.push({ view: label, file, evidence: 'Real Electron renderer; response-stub data; view-only, no actions submitted' })
  }
})
