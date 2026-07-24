import { describe, expect, it } from 'vitest'

import { shouldRestoreRememberedRoute } from './use-desktop-integrations'

describe('shouldRestoreRememberedRoute', () => {
  it('restores the remembered chat on a normal cold app launch', () => {
    expect(shouldRestoreRememberedRoute({ freshInstanceWindow: false, locationPathname: '/', restored: false })).toBe(
      true
    )
  })

  it("never restores another window's remembered chat into a new peer instance", () => {
    expect(shouldRestoreRememberedRoute({ freshInstanceWindow: true, locationPathname: '/', restored: false })).toBe(
      false
    )
  })

  it('does not restore after the cold-start decision has already run', () => {
    expect(shouldRestoreRememberedRoute({ freshInstanceWindow: false, locationPathname: '/', restored: true })).toBe(
      false
    )
  })
})
