import assert from 'node:assert/strict'
import test from 'node:test'
import { retainBrowserAfterError } from './browser-error-policy'
test('scoped browser errors permit inspection without replay or broader access', () => {
  const args = { target_id: 'obt-exact', tab_id: 'tab-exact' }
  assert.equal(retainBrowserAfterError('browser_click', args, 'Protocol timeout'), true)
  assert.equal(retainBrowserAfterError('get_browser_state', args, 'Stale reference'), true)
  assert.equal(retainBrowserAfterError('browser_prepare', args, 'Timeout'), false)
  assert.equal(retainBrowserAfterError('browser_click', args, 'Private Edge profile enabled sync. Preparation refused.'), false)
  assert.equal(retainBrowserAfterError('click', args, 'Timeout'), false)
  assert.equal(retainBrowserAfterError('browser_click', { target_id: 'bt-native' }, 'Timeout'), false)
})
