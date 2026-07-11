import { beforeEach, describe, expect, it } from 'vitest'

import { $fileBrowserOpen, $rightRailActiveTabId, setFileBrowserOpen } from '@/store/layout'
import { $utilityPreviewTabs, clearSessionPreviewRegistry, openUtilityPreviewTab } from '@/store/preview'

import { $terminalTakeover, setTerminalTakeover } from './store'

describe('terminal utility tab adapter', () => {
  beforeEach(() => {
    window.localStorage.clear()
    clearSessionPreviewRegistry()
    setFileBrowserOpen(false)
  })

  it('opens and closes Terminal as a peer tab without changing File Browser', () => {
    setFileBrowserOpen(true)

    setTerminalTakeover(true)

    expect($terminalTakeover.get()).toBe(true)
    expect($utilityPreviewTabs.get()).toEqual([{ id: 'utility:terminal', kind: 'terminal' }])
    expect($rightRailActiveTabId.get()).toBe('utility:terminal')
    expect($fileBrowserOpen.get()).toBe(true)

    openUtilityPreviewTab('host-vnc')
    expect($rightRailActiveTabId.get()).toBe('utility:host-vnc')

    setTerminalTakeover(true)
    expect($rightRailActiveTabId.get()).toBe('utility:terminal')
    expect($utilityPreviewTabs.get()).toEqual([
      { id: 'utility:terminal', kind: 'terminal' },
      { id: 'utility:host-vnc', kind: 'host-vnc' }
    ])
    expect($fileBrowserOpen.get()).toBe(true)

    setTerminalTakeover(false)

    expect($terminalTakeover.get()).toBe(false)
    expect($utilityPreviewTabs.get()).toEqual([{ id: 'utility:host-vnc', kind: 'host-vnc' }])
    expect($fileBrowserOpen.get()).toBe(true)
  })
})
