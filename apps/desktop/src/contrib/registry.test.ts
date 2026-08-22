import { atom } from 'nanostores'
import { describe, expect, it, vi } from 'vitest'

import { ContributionRegistry } from './registry'

describe('ContributionRegistry visibility authority', () => {
  it('invalidates the owning area when a live condition changes', () => {
    const registry = new ContributionRegistry()
    const visible = atom(true)
    const onAreaChange = vi.fn()

    registry.register({ area: 'statusBar.right', id: 'live', when: visible })
    registry.subscribeArea('statusBar.right', onAreaChange)

    const first = registry.getArea('statusBar.right')
    expect(first.map(item => item.id)).toEqual(['live'])
    expect(registry.getArea('statusBar.right')).toBe(first)

    visible.set(false)

    expect(onAreaChange).toHaveBeenCalledTimes(1)
    expect(registry.getArea('statusBar.right')).toEqual([])

    visible.set(true)

    expect(onAreaChange).toHaveBeenCalledTimes(2)
    expect(registry.getArea('statusBar.right').map(item => item.id)).toEqual(['live'])
  })

  it('detaches a replaced contribution from its previous authority', () => {
    const registry = new ContributionRegistry()
    const stale = atom(true)
    const current = atom(true)
    const onAreaChange = vi.fn()

    registry.register({ area: 'panes', id: 'same', when: stale })
    registry.register({ area: 'panes', id: 'same', when: current })
    registry.subscribeArea('panes', onAreaChange)

    stale.set(false)

    expect(onAreaChange).not.toHaveBeenCalled()
    expect(registry.getArea('panes').map(item => item.id)).toEqual(['same'])

    current.set(false)

    expect(onAreaChange).toHaveBeenCalledTimes(1)
    expect(registry.getArea('panes')).toEqual([])
  })

  it('detaches visibility authority when a contribution is removed', () => {
    const registry = new ContributionRegistry()
    const visible = atom(true)
    const onAreaChange = vi.fn()

    const dispose = registry.register({ area: 'titleBar.right', id: 'temporary', when: visible })
    registry.subscribeArea('titleBar.right', onAreaChange)

    dispose()
    onAreaChange.mockClear()
    visible.set(false)

    expect(onAreaChange).not.toHaveBeenCalled()
    expect(registry.getArea('titleBar.right')).toEqual([])
  })
})
