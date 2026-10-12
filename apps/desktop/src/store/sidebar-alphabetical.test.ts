import { afterEach, describe, expect, it } from 'vitest'

import { $sidebarAlphabetical, setSidebarAlphabetical } from './sidebar-alphabetical'

afterEach(() => {
  setSidebarAlphabetical(false)
  window.localStorage.clear()
})

describe('sidebar alphabetical preference', () => {
  it('is off by default so stock ordering applies', () => {
    expect($sidebarAlphabetical.get()).toBe(false)
  })

  it('persists under its own key', () => {
    setSidebarAlphabetical(true)

    expect($sidebarAlphabetical.get()).toBe(true)
    expect(window.localStorage.getItem('hermes.desktop.sidebarAlphabetical')).toBe('true')
  })
})
