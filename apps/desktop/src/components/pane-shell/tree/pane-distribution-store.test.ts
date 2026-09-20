import { afterEach, beforeEach, describe, expect, test } from 'vitest'

import { registry } from '@/contrib/registry'
import { $paneDistributionMode } from '@/store/pane-distribution'

import { group, split } from './model'
import {
  $collapsedTreeSides,
  $dismissedPanes,
  $hiddenTreePanes,
  $layoutTree,
  $userPlacedPanes,
  dockPaneBeside,
  equalizeCurrentPaneTree,
  moveTreePane
} from './store'

const disposers: (() => void)[] = []

function registerPane(id: string, data: Record<string, unknown>) {
  disposers.push(registry.register({ area: 'panes', data, id, render: () => null, title: id }))
}

beforeEach(() => {
  window.localStorage.clear()
  $paneDistributionMode.set('equal-flex')
  $collapsedTreeSides.set(new Set())
  $dismissedPanes.set(new Set())
  $hiddenTreePanes.set(new Set())
  $userPlacedPanes.set(new Set())

  registerPane('fixed', { placement: 'right', width: '240px' })
  registerPane('main', { placement: 'main', uncloseable: true })
  registerPane('left', { placement: 'main' })
  registerPane('right', { placement: 'main' })
  registerPane('new', { placement: 'right' })
  registerPane('collapsed-a', { placement: 'right' })
  registerPane('collapsed-b', { placement: 'right' })
})

afterEach(() => {
  disposers.splice(0).forEach(dispose => dispose())
  $layoutTree.set(null)
  $collapsedTreeSides.set(new Set())
})

describe('structural pane distribution', () => {
  test('equal-flex keeps a fixed sidebar and equalizes moved flex panes', () => {
    $layoutTree.set(
      split(
        'row',
        [
          group(['fixed'], { id: 'fixed-zone' }),
          group(['left'], { id: 'left-zone' }),
          group(['right'], { id: 'right-zone' })
        ],
        [1, 1, 2],
        'root'
      )
    )

    moveTreePane('left', { groupId: 'right-zone', pos: 'right' })

    const tree = $layoutTree.get()
    expect(tree?.type).toBe('split')

    if (tree?.type !== 'split') {
      return
    }

    expect(tree.children[0].id).toBe('fixed-zone')
    expect(tree.weights[0]).toBe(1)
    expect(tree.weights[1]).toBeCloseTo(tree.weights[2])
  })

  test('equal-flex can redistribute the existing tree in one operation', () => {
    $layoutTree.set(
      split(
        'row',
        [
          group(['fixed'], { id: 'fixed-zone' }),
          group(['left'], { id: 'left-zone' }),
          group(['right'], { id: 'right-zone' })
        ],
        [1, 1, 2],
        'root'
      )
    )

    equalizeCurrentPaneTree()

    const tree = $layoutTree.get()
    expect(tree?.type).toBe('split')

    if (tree?.type !== 'split') {
      return
    }

    expect(tree.weights).toEqual([1, 1.5, 1.5])
  })

  test('equal-all includes a fixed pane when a new pane is added beside it', () => {
    $paneDistributionMode.set('equal-all')
    $layoutTree.set(
      split('row', [group(['fixed'], { id: 'fixed-zone' }), group(['main'], { id: 'main-zone' })], [1, 3], 'root')
    )

    dockPaneBeside('new', 'main')

    const tree = $layoutTree.get()
    expect(tree?.type).toBe('split')

    if (tree?.type !== 'split') {
      return
    }

    expect(tree.children).toHaveLength(3)
    expect(tree.weights[0]).toBeCloseTo(tree.weights[1])
    expect(tree.weights[1]).toBeCloseTo(tree.weights[2])
  })

  test('does not redistribute tracks on a collapsed root side', () => {
    for (const mode of ['equal-flex', 'equal-all'] as const) {
      $paneDistributionMode.set(mode)
      $collapsedTreeSides.set(new Set(['right']))

      const tree = split(
        'row',
        [
          group(['main'], { id: 'main-zone' }),
          group(['collapsed-a'], { id: 'collapsed-a-zone' }),
          group(['collapsed-b'], { id: 'collapsed-b-zone' })
        ],
        [1, 4, 7],
        `collapsed-${mode}`
      )

      $layoutTree.set(tree)
      equalizeCurrentPaneTree()

      expect($layoutTree.get()).toBe(tree)
      expect(tree.weights).toEqual([1, 4, 7])
    }
  })
})
