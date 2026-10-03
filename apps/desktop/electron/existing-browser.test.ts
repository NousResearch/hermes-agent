import assert from 'node:assert/strict'
import test from 'node:test'
import { ExistingBrowsers } from './existing-browser'

test('existing-window grant uses its own transport and exact window binding', async () => {
  const calls: { name: string; args: Record<string, unknown> }[] = []
  let created = 0, closed = 0
  const driver = { close: () => { closed++ }, call: async (name: string, args: Record<string, unknown>) => {
    calls.push({ name, args })
    return name === 'get_browser_state' ? { structuredContent: { target_id: 'bt-owned-fixture', tab_id: 'tab-fixture' }, content: [] } : { content: [] }
  } }
  const existing = new ExistingBrowsers('verified-driver', 'window:gateway:chat', (command, args) => {
    created++
    assert.equal(command, 'verified-driver')
    assert.deepEqual(args, ['mcp', '--grant', 'existing-profile'])
    return driver
  })
  assert.equal(created, 0)
  await existing.prepare({ pid: 10, window_id: 20 })
  assert.equal(created, 1)
  const prepare = calls.find(call => call.name === 'browser_prepare')!
  assert.deepEqual(prepare.args.strategy, { kind: 'existing_profile' })
  assert.equal(prepare.args.pid, 10)
  assert.equal(prepare.args.window_id, 20)
  assert.equal(existing.handles({ pid: 10, window_id: 21 }), false)
  await existing.call('get_browser_state', { pid: 10, window_id: 20 })
  assert.equal(existing.scope({ target_id: 'bt-owned-fixture' }), 'existing_profile:10:20')
  assert.equal(existing.handles({ target_id: 'bt-foreign' }), false)
  await existing.call('browser_click', { target_id: 'bt-owned-fixture', tab_id: 'tab-fixture', ref: 'p1:2' })
  assert.equal(calls.at(-1)?.args.session, prepare.args.session)
  await assert.rejects(existing.call('launch_app', { target_id: 'bt-owned-fixture' }), /only browser/)
  existing.close()
  assert.equal(closed, 1)
  assert.equal(existing.handles({ target_id: 'bt-owned-fixture' }), false)
})
