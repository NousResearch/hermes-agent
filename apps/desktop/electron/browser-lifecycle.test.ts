import assert from 'node:assert/strict'
import test from 'node:test'
import { refreshLiveLifecycle, refreshConversationLifecycle, startBrowserLifecycle } from './browser-lifecycle'

test('adapter-only conversation keeps native PC inspection alive beyond five minutes', async () => {
  let now = 0, lastActivity = 0
  const calls: string[] = []
  const driver = { call: async (name: string, args: Record<string, unknown>) => {
    assert.deepEqual(args, {})
    calls.push(name)
    if (name === 'get_session') return { structuredContent: { state: 'active', expires_in_seconds: 300 - (now - lastActivity) } }
    lastActivity = now
    return { structuredContent: { revived: false } }
  } }
  for (now = 60; now <= 600; now += 60) assert.equal(await refreshConversationLifecycle(driver), true)
  assert.equal(lastActivity, 600)
  assert.equal(calls.length, 20)
})

test('conversation maintenance never revives an ended PC transport or its browser', async () => {
  const calls: unknown[] = []
  const driver = { call: async (name: string, args: Record<string, unknown>) => {
    calls.push([name, args]); return { structuredContent: { state: 'ended', expires_in_seconds: 0 } }
  } }
  assert.equal(await refreshConversationLifecycle(driver, 'owned-browser'), false)
  assert.deepEqual(calls, [['get_session', {}]])
})

test('persistent conversation refreshes only a confirmed live lifecycle', async () => {
  const calls: string[] = []
  const driver = { call: async (name: string) => {
    calls.push(name)
    return name === 'get_session' ? { structuredContent: { state: 'active', expires_in_seconds: 240 } } : { structuredContent: { revived: false } }
  } }
  assert.equal(await refreshLiveLifecycle(driver, { session: 'owned-chat' }), true)
  assert.deepEqual(calls, ['get_session', 'start_session'])
})
test('expired or nearly expired lifecycle is not revived', async () => {
  for (const status of [{ isError: true }, { structuredContent: { state: 'ending', expires_in_seconds: 200 } }, { structuredContent: { state: 'active', expires_in_seconds: 10 } }]) {
    const calls: string[] = []
    assert.equal(await refreshLiveLifecycle({ call: async name => { calls.push(name); return status } }, {}), false)
    assert.deepEqual(calls, ['get_session'])
  }
})
test('explicit browser preparation starts both transport and named lifecycles', async () => {
  const calls: unknown[] = []
  await startBrowserLifecycle({ call: async (name, args) => { calls.push([name, args]); return {} } }, 'owned-chat')
  assert.deepEqual(calls, [['start_session', {}], ['start_session', { session: 'owned-chat' }]])
})

test('live renewal spans more than the five-minute idle window', async () => {
  let now = 0
  let lastActivity = 0
  const driver = { call: async (name: string) => {
    if (name === 'get_session') return { structuredContent: { state: 'active', expires_in_seconds: 300 - (now - lastActivity) } }
    lastActivity = now
    return { structuredContent: { revived: false } }
  } }
  for (now = 60; now <= 420; now += 60) assert.equal(await refreshLiveLifecycle(driver, { session: 'owned-chat' }), true)
  assert.equal(lastActivity, 420)
})
