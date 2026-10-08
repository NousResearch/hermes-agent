import assert from 'node:assert/strict'
import test from 'node:test'
import { browserArguments } from './browser-session'

test('browser actions share an explicit host-owned label within one conversation', () => {
  const first = browserArguments('browser_prepare', {}, 'chat-one')
  const second = browserArguments('get_browser_state', {}, 'chat-one')
  assert.equal(first.session, second.session)
  assert.notEqual(first.session, browserArguments('browser_click', {}, 'chat-two').session)
})
test('caller cannot select browser lifecycle and native PC arguments remain unchanged', () => {
  assert.throws(() => browserArguments('browser_click', { session: 'another-chat' }, 'chat-one'), /Desktop owns/)
  assert.deepEqual(browserArguments('list_windows', {}, 'chat-one'), {})
})
