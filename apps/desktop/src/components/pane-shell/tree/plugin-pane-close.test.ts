import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { setPluginEnabled } from '@/contrib/plugins-store'
import { registry } from '@/contrib/registry'

import { allPaneIds, group, split } from './model'
import { $dismissedPanes, $layoutTree, closeTreePane } from './store'

vi.mock('@/contrib/plugins-store', () => ({ setPluginEnabled: vi.fn() }))
vi.mock('@/store/notifications', () => ({ notify: vi.fn() }))

const disposers: (() => void)[] = []

function registerPluginPane(pluginId: string, paneId: string) {
  disposers.push(
    registry.register({
      area: 'panes',
      data: { placement: 'main' },
      id: paneId,
      render: () => null,
      source: `plugin:${pluginId}`,
      title: paneId
    })
  )
}

beforeEach(() => {
  window.localStorage.clear()
  $dismissedPanes.set(new Set())
  vi.mocked(setPluginEnabled).mockReset()
})

afterEach(() => {
  disposers.splice(0).forEach(dispose => dispose())
})

describe('closing plugin panes', () => {
  it('dismisses one pane without disabling a plugin that contributes multiple panes', () => {
    registerPluginPane('bots', 'bots:pane')
    registerPluginPane('bots', 'bots:routines')
    $layoutTree.set(
      split('row', [
        group(['workspace'], { active: 'workspace', id: 'g-main' }),
        group(['bots:pane'], { active: 'bots:pane', id: 'g-bots' }),
        group(['bots:routines'], { active: 'bots:routines', id: 'g-routines' })
      ])
    )

    closeTreePane('bots:routines')

    expect(allPaneIds($layoutTree.get()!)).not.toContain('bots:routines')
    expect(allPaneIds($layoutTree.get()!)).toContain('bots:pane')
    expect($dismissedPanes.get()).toContain('bots:routines')
    expect(setPluginEnabled).not.toHaveBeenCalled()

    // Any later registry mutation runs pane adoption. The dismissal must
    // survive that cycle instead of immediately resurrecting Cronjobs.
    registerPluginPane('other', 'other:pane')
    expect(allPaneIds($layoutTree.get()!)).not.toContain('bots:routines')
  })

  it('keeps disabling a plugin whose only pane is closed', () => {
    registerPluginPane('single', 'single:pane')
    $layoutTree.set(group(['single:pane'], { active: 'single:pane', id: 'g-single' }))

    closeTreePane('single:pane')

    expect(setPluginEnabled).toHaveBeenCalledWith('single', false)
    expect(allPaneIds($layoutTree.get()!)).toContain('single:pane')
  })
})
