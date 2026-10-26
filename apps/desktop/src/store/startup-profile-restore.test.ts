import { describe, expect, it } from 'vitest'

import { startupProfileForConnection } from './startup-profile-restore'

describe('startup profile reconciliation', () => {
  it('replaces a Windows-only legacy profile with the remote default', () => {
    expect(
      startupProfileForConnection('office-evals-windows', 'office-evals-windows', [
        { name: 'default' },
        { name: 'marina' }
      ])
    ).toBe('default')
  })

  it('preserves a profile that actually exists on the chosen source', () => {
    expect(
      startupProfileForConnection('office-evals-windows', 'marina', [
        { name: 'default' },
        { name: 'office-evals-windows' }
      ])
    ).toBeNull()
  })

  it('uses a valid connection-specific preference before the default', () => {
    expect(
      startupProfileForConnection('office-evals-windows', 'marina', [{ name: 'default' }, { name: 'marina' }])
    ).toBe('marina')
  })

  it('chooses the only available profile when no default exists and does not invent one for an empty roster', () => {
    expect(startupProfileForConnection('office-evals-windows', undefined, [{ name: 'special' }])).toBe('special')
    expect(startupProfileForConnection('office-evals-windows', undefined, [])).toBeNull()
  })
})
