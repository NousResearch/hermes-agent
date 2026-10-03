import { afterEach, describe, expect, it, vi } from 'vitest'

import { preservesNativeTouchContextMenu } from './user-message'

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('user message touch context menus', () => {
  it('preserves the OS long-press menu on coarse pointers', () => {
    vi.stubGlobal('matchMedia', vi.fn().mockReturnValue({ matches: true, media: '(hover: none), (pointer: coarse)' }))

    expect(preservesNativeTouchContextMenu()).toBe(true)
  })

  it('keeps the desktop reaction gesture when hover is available', () => {
    vi.stubGlobal('matchMedia', vi.fn().mockReturnValue({ matches: false }))

    expect(preservesNativeTouchContextMenu()).toBe(false)
  })
})
