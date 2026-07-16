import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { $paneStates, ensurePaneRegistered } from '@/store/panes'

import { host, TITLEBAR_AREAS } from './index'

const STORAGE_KEY = 'hermes.desktop.paneStates.v1'

describe('plugin SDK pane controls', () => {
  beforeEach(() => {
    $paneStates.set({})
    window.localStorage.clear()
  })

  afterEach(() => {
    $paneStates.set({})
    window.localStorage.clear()
  })

  it('exposes a reactive open atom and bounded set/toggle actions', () => {
    ensurePaneRegistered('plugin:browser', { open: true })
    const open = host.panes.open('plugin:browser')

    expect(open.get()).toBe(true)

    host.panes.setOpen('plugin:browser', false)
    expect(open.get()).toBe(false)

    host.panes.toggle('plugin:browser')
    expect(open.get()).toBe(true)
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

  it('exposes an app-controls slot adjacent to the fixed titlebar controls', () => {
    expect(TITLEBAR_AREAS.appControls).toBe('titleBar.appControls')
  })
})
