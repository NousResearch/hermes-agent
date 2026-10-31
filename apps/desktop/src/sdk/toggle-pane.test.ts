import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { allPaneIds, findGroupOfPane, group, split } from '@/components/pane-shell/tree/model'
import { $dismissedPanes, $hiddenTreePanes, $layoutTree, isPaneVisible } from '@/components/pane-shell/tree/store'
import { setPluginEnabled } from '@/contrib/plugins-store'
import { registry } from '@/contrib/registry'
import { host } from '@/sdk'

// A plugin's shortcut must be able to put its own pane away and bring it back,
// the way ⌘G does for Review. The SDK could only reveal, and Close is not a
// hide for a closer-less pane: for a single-pane plugin it is the Capabilities
// switch, so "hide" would mean "disabled until re-enabled".

vi.mock('@/contrib/plugins-store', () => ({ setPluginEnabled: vi.fn() }))
vi.mock('@/store/notifications', async importOriginal => ({
  ...((await importOriginal()) as Record<string, unknown>),
  notify: vi.fn()
}))

const disposers: (() => void)[] = []

function registerPane(id: string, data: Record<string, unknown>, source?: string) {
  disposers.push(registry.register({ area: 'panes', data, id, render: () => null, source, title: id }))
}

const zoneOf = (paneId: string) => findGroupOfPane($layoutTree.get()!, paneId)!

beforeEach(() => {
  window.localStorage.clear()
  $dismissedPanes.set(new Set())
  $hiddenTreePanes.set(new Set())
  vi.mocked(setPluginEnabled).mockReset()
  registerPane('workspace', { placement: 'main', uncloseable: true })
  registerPane('files', { placement: 'right' })
  registerPane('gr:pane', { placement: 'right' }, 'plugin:gr')
})

afterEach(() => {
  disposers.splice(0).forEach(dispose => dispose())
})

describe('host.togglePane on a single-pane plugin in its own zone', () => {
  beforeEach(() => {
    $layoutTree.set(
      split('row', [
        group(['workspace'], { active: 'workspace', id: 'g-main' }),
        group(['gr:pane'], { active: 'gr:pane', id: 'g-gr' })
      ])
    )
  })

  it('minimizes the zone instead of closing the pane or disabling the plugin', () => {
    host.togglePane('gr:pane')

    expect(isPaneVisible('gr:pane')).toBe(false)
    expect(zoneOf('gr:pane').minimized).toBe(true)
    expect(allPaneIds($layoutTree.get()!)).toContain('gr:pane')
    expect($dismissedPanes.get().has('gr:pane')).toBe(false)
    expect(setPluginEnabled).not.toHaveBeenCalled()
  })

  it('brings the pane back on the second press', () => {
    host.togglePane('gr:pane')
    host.togglePane('gr:pane')

    expect(isPaneVisible('gr:pane')).toBe(true)
    expect(zoneOf('gr:pane').minimized).toBeFalsy()
  })
})

describe('host.togglePane on a plugin pane stacked with the conversation', () => {
  it('hands the zone back to the conversation, never minimizing it', () => {
    $layoutTree.set(group(['workspace', 'gr:pane'], { active: 'gr:pane', id: 'g-main' }))

    host.togglePane('gr:pane')

    expect(zoneOf('gr:pane').active).toBe('workspace')
    expect(zoneOf('gr:pane').minimized).toBeFalsy()
    expect(isPaneVisible('workspace')).toBe(true)

    host.togglePane('gr:pane')

    expect(isPaneVisible('gr:pane')).toBe(true)
  })
})

describe('host.togglePane on a plugin pane sharing a zone with an app pane', () => {
  it('folds the zone like its Minimize and brings the plugin pane back to the front', () => {
    $layoutTree.set(
      split('row', [
        group(['workspace'], { active: 'workspace', id: 'g-main' }),
        group(['files', 'gr:pane'], { active: 'gr:pane', id: 'g-right' })
      ])
    )

    host.togglePane('gr:pane')

    expect(zoneOf('gr:pane').minimized).toBe(true)
    expect(allPaneIds($layoutTree.get()!)).toEqual(expect.arrayContaining(['files', 'gr:pane']))

    host.togglePane('gr:pane')

    expect(isPaneVisible('gr:pane')).toBe(true)
    expect(zoneOf('gr:pane').active).toBe('gr:pane')
  })
})

describe('host.togglePane outside a plugin pane', () => {
  it('leaves panes the app owns alone', () => {
    $layoutTree.set(
      split('row', [
        group(['workspace'], { active: 'workspace', id: 'g-main' }),
        group(['files'], { active: 'files', id: 'g-files' })
      ])
    )
    const before = $layoutTree.get()

    host.togglePane('files')
    host.togglePane('workspace')
    host.togglePane('  ')

    expect($layoutTree.get()).toBe(before)
    expect(isPaneVisible('files')).toBe(true)
    expect(isPaneVisible('workspace')).toBe(true)
  })
})
