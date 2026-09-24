import { expect, test } from 'vitest'

import { normalizeProfileKey, profileLabel } from './profile'

// The UI label contract: every switcher surface (rail, dropdown, Manage
// Profiles) renders `profileLabel(profile)`, so the display_name from
// profile.yaml must keep showing there — including when it differs from the
// profile's directory name, which is the ONLY thing that ever reaches the
// backend spawn (see electron/profile-id.ts).
test('profileLabel shows the display_name label, not the profile id', () => {
  expect(profileLabel({ display_name: 'SmartHome', name: 'smarthome' })).toBe('SmartHome')
  expect(profileLabel({ display_name: 'E-Mail', name: 'e-mail' })).toBe('E-Mail')
})

test('profileLabel falls back to the canonical id when no display_name is set', () => {
  expect(profileLabel({ display_name: '', name: 'worker' })).toBe('worker')
  expect(profileLabel({ display_name: '   ', name: 'worker' })).toBe('worker')
  expect(profileLabel({ name: 'worker' })).toBe('worker')
})

test('a label identical to the id renders unchanged', () => {
  expect(profileLabel({ display_name: 'virtualisierung', name: 'virtualisierung' })).toBe('virtualisierung')
})

// The label is presentation-only: the identity surfaces keep reading `name`.
test('normalizeProfileKey keeps reading the id, never the label', () => {
  const profile = { display_name: 'SmartHome', name: 'smarthome' }

  expect(normalizeProfileKey(profile.name)).toBe('smarthome')
  expect(normalizeProfileKey(profile.name)).not.toBe(profileLabel(profile))
})
