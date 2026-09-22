import { describe, expect, it } from 'vitest'
import { targetIdentity } from './target-row'

describe('managed rollout target identity', () => {
  it('keeps aliases display-only and keys selection by machine plus install identity', () => {
    expect(targetIdentity({ installId: 'install-a', machineId: 'machine-a', label: 'Alias', alias: 'old-name', supported: true, sharedMachine: false })).toBe(
      JSON.stringify(['machine-a', 'install-a'])
    )
    expect(targetIdentity({ installId: 'install-a', machineId: 'machine-a', label: 'Alias', alias: 'new-name', supported: true, sharedMachine: false })).toBe(
      JSON.stringify(['machine-a', 'install-a'])
    )
  })
})
