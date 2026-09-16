import { describe, expect, it } from 'vitest'

import { isValidSidebarPortrait } from './sidebar-portrait'

const valid = {
  version: 1,
  width: 2,
  height: 2,
  luminance: [0, 128, 200, 255]
}

describe('isValidSidebarPortrait', () => {
  it('accepts a well-formed payload', () => {
    expect(isValidSidebarPortrait(valid)).toBe(true)
  })

  it('rejects non-objects and arrays', () => {
    expect(isValidSidebarPortrait(null)).toBe(false)
    expect(isValidSidebarPortrait(undefined)).toBe(false)
    expect(isValidSidebarPortrait('portrait')).toBe(false)
    expect(isValidSidebarPortrait([1, 2, 3])).toBe(false)
  })

  it('rejects a wrong or missing version', () => {
    expect(isValidSidebarPortrait({ ...valid, version: 2 })).toBe(false)
    expect(isValidSidebarPortrait({ width: 2, height: 2, luminance: valid.luminance })).toBe(false)
  })

  it('rejects non-integer or non-positive dimensions', () => {
    expect(isValidSidebarPortrait({ ...valid, width: 1.5 })).toBe(false)
    expect(isValidSidebarPortrait({ ...valid, width: 0 })).toBe(false)
    expect(isValidSidebarPortrait({ ...valid, height: -2 })).toBe(false)
  })

  it('rejects a pixel count over the cap', () => {
    expect(isValidSidebarPortrait({ ...valid, width: 1001, height: 1000 })).toBe(false)
  })

  it('rejects a luminance array whose length does not match the dimensions', () => {
    expect(isValidSidebarPortrait({ ...valid, luminance: [0, 128] })).toBe(false)
    expect(isValidSidebarPortrait({ ...valid, luminance: undefined })).toBe(false)
  })

  it('rejects out-of-range or non-integer luminance entries', () => {
    expect(isValidSidebarPortrait({ ...valid, luminance: [0, 128, 200, 256] })).toBe(false)
    expect(isValidSidebarPortrait({ ...valid, luminance: [0, 128, 200, -1] })).toBe(false)
    expect(isValidSidebarPortrait({ ...valid, luminance: [0, 128, 200, 12.5] })).toBe(false)
  })
})
