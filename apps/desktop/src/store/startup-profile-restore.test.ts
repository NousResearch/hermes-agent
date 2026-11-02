import { expect, it } from 'vitest'

import { servedProfileSet, startupProfileForConnection } from './startup-profile-restore'

const installed = [{ name: 'default' }, { name: 'lab' }, { name: 'work' }]

it.each([
  // `/api/status` gateway_shared_with is the served set; installed is not served.
  { name: 'remembered profile parked', status: { gateway_shared_with: ['default', 'work'] }, active: 'gone', remembered: 'lab', want: 'default' },
  { name: 'active profile parked', status: { gateway_shared_with: ['default', 'lab'] }, active: 'work', remembered: 'lab', want: 'lab' },
  { name: 'single-profile gateway serves default only', status: { gateway_shared_with: [] }, active: 'work', want: 'default' },
  { name: 'active served', status: { gateway_shared_with: ['default', 'work'] }, active: 'work', want: null },
  // Unknown served set (no gateway, older backend, failed read): installed counts as served.
  { name: 'served set unknown', status: { gateway_shared_with: null }, active: 'work', want: null },
  { name: 'status read failed', status: null, active: 'gone', remembered: 'lab', want: 'lab' }
])('$name', ({ active, remembered, status, want }) => {
  expect(startupProfileForConnection(active, remembered, installed, servedProfileSet(status))).toBe(want)
})
