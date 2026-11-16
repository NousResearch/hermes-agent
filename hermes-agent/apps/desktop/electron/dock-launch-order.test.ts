import { describe, expect, it } from 'vitest'

import { deepLinkRegistersBeforeSingleInstanceLock, preReadyDockLaunchSteps } from './dock-launch-order'

describe('macOS Dock launch order', () => {
  it('registers the deep link before the single-instance lock', () => {
    expect(preReadyDockLaunchSteps('darwin')).toEqual(['register-deep-link', 'single-instance-lock'])
    expect(deepLinkRegistersBeforeSingleInstanceLock('darwin')).toBe(true)
  })

  it('keeps non-macOS platforms on the lock at module load, with no Dock race', () => {
    for (const platform of ['win32', 'linux', 'freebsd']) {
      expect(preReadyDockLaunchSteps(platform)).toEqual(['single-instance-lock'])
      expect(deepLinkRegistersBeforeSingleInstanceLock(platform)).toBe(false)
    }
  })
})
