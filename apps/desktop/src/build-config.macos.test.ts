import desktopPackage from '../package.json'
import { describe, expect, it } from 'vitest'

describe('macOS Desktop bundle metadata', () => {
  it('keeps the real Desktop app visible in Dock and Cmd-Tab', () => {
    const macExtendInfo = desktopPackage.build.mac.extendInfo

    expect(desktopPackage.build.appId).toBe('com.nousresearch.hermes')
    expect(macExtendInfo).not.toHaveProperty('LSUIElement')
  })
})
