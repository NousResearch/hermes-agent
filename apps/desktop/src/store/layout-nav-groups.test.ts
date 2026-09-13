import { beforeEach, describe, expect, it } from 'vitest'

import { $sidebarNavGroupOpen, setSidebarNavGroupOpen } from './layout'

const KEY = 'hermes.desktop.sidebarNavGroupOpen'

beforeEach(() => {
  window.localStorage.removeItem(KEY)
  $sidebarNavGroupOpen.set({})
})

describe('contributed sidebar nav groups', () => {
  it('remembers which plugin groups the user opened', () => {
    expect($sidebarNavGroupOpen.get()).toEqual({})
    setSidebarNavGroupOpen('workflow', false)
    expect($sidebarNavGroupOpen.get()).toEqual({ workflow: false })
    expect(window.localStorage.getItem(KEY)).toContain('workflow')
    setSidebarNavGroupOpen('workflow', true)
    expect($sidebarNavGroupOpen.get()).toEqual({ workflow: true })
  })
})
