import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { $hiddenTreePanes } from '@/components/pane-shell/tree/store'
import { registry } from '@/contrib/registry'
import { $paneStates, ensurePaneRegistered } from '@/store/panes'

import { host, TITLEBAR_AREAS } from './index'

const STORAGE_KEY = 'hermes.desktop.paneStates.v1'

describe('plugin SDK pane controls', () => {
  beforeEach(() => {
    $paneStates.set({})
    $hiddenTreePanes.set(new Set())
    window.localStorage.clear()
  })

  afterEach(() => {
    $paneStates.set({})
    $hiddenTreePanes.set(new Set())
    window.localStorage.clear()
  })

  it('exposes a reactive open atom and bounded set/toggle actions', () => {
    ensurePaneRegistered('plugin:browser', { open: true })
    const open = host.panes.open('plugin:browser')

    expect(open.get()).toBe(true)

    host.panes.setOpen('plugin:browser', false)
    expect(open.get()).toBe(false)
    expect($hiddenTreePanes.get().has('plugin:browser')).toBe(true)

    host.panes.toggle('plugin:browser')
    expect(open.get()).toBe(true)
    expect($hiddenTreePanes.get().has('plugin:browser')).toBe(false)
    expect(JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? '{}')).toEqual({
      'plugin:browser': { open: true }
    })
  })

  it('starts an unregistered pane closed and opens it on toggle', () => {
    const open = host.panes.open('plugin:new-pane')

    expect(open.get()).toBe(false)

    host.panes.toggle('plugin:new-pane')
    expect(open.get()).toBe(true)
  })

  it('refuses core-owned panes but drives registered plugin-sourced panes', () => {
    const disposeCore = registry.register({ area: 'panes', id: 'sessions' })
    const disposePlugin = registry.register({ area: 'panes', id: 'wb:browser', source: 'plugin:wb' })

    try {
      // A core pane's visibility belongs to its app store; the SDK must not
      // write it behind the store's back (titlebar would desync from the tree).
      host.panes.setOpen('sessions', false)
      expect($paneStates.get()['sessions']).toBeUndefined()
      expect($hiddenTreePanes.get().has('sessions')).toBe(false)

      host.panes.toggle('sessions')
      expect($paneStates.get()['sessions']).toBeUndefined()

      // A plugin-sourced registered pane stays fully drivable.
      host.panes.setOpen('wb:browser', false)
      expect(host.panes.open('wb:browser').get()).toBe(false)
      expect($hiddenTreePanes.get().has('wb:browser')).toBe(true)
    } finally {
      disposeCore()
      disposePlugin()
    }
  })

  it('exposes an app-controls slot adjacent to the fixed titlebar controls', () => {
    expect(TITLEBAR_AREAS.appControls).toBe('titleBar.appControls')
  })
})
