import assert from 'node:assert/strict'
import test from 'node:test'
import { actionConsentScope, consentCovers, type ActionConsentScope } from './action-consent'

test('conversation background permission never authorizes foreground input', () => {
  const grants = new Set<ActionConsentScope>(['background'])
  assert(consentCovers(grants, 'type_text', { delivery_mode: 'background' }))
  assert(consentCovers(grants, 'browser_click', { target_id: 'tab' }))
  assert(!consentCovers(grants, 'type_text', { delivery_mode: 'foreground' }))
  assert(!consentCovers(grants, 'bring_to_front', {}))
  assert(!consentCovers(grants, 'click', { target: { kind: 'desktop' } }))
  assert(!consentCovers(grants, 'browser_prepare', { profile: { mode: 'existing_profile' } }))
})
test('new conversations have independent permissions and revoke removes them', () => {
  const first = new Set<ActionConsentScope>(['background'])
  const second = new Set<ActionConsentScope>()
  assert(!consentCovers(second, 'click', {}))
  first.clear()
  assert(!consentCovers(first, 'click', {}))
  assert.equal(actionConsentScope('click', { bring_to_front: true }), 'foreground')
})
