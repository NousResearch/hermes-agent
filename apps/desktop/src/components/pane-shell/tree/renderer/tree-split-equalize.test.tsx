import { useStore } from '@nanostores/react'
import { cleanup, fireEvent, render } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'
import { readJson } from '@/lib/storage'
import { $paneStates } from '@/store/panes'
import { stubResizeObserver } from '@/test/jsdom'

import { group, type LayoutNode, split } from '../model'
import { $hiddenTreePanes, $layoutTree, persistTree } from '../store'

import { TreeSplit } from './tree-split'

const disposers: (() => void)[] = []

beforeEach(() => {
  window.localStorage.clear()
  stubResizeObserver()
  vi.stubGlobal('CSS', { ...globalThis.CSS, escape: (value: string) => value })

  for (const [id, data] of [
    ['sidebar', { placement: 'left', width: '200px', height: '120px' }],
    ['workspace', { placement: 'main', minWidth: '22vw', minHeight: '100px' }],
    ['files', { placement: 'right', width: '200px', height: '120px' }],
    ['session', { placement: 'main', minWidth: '20rem', minHeight: '80px' }],
    ['tools', { placement: 'right', width: '160px', height: '100px' }],
    ['hidden', { placement: 'right', width: '160px', height: '100px' }],
    ['minimized', { placement: 'right', width: '160px', height: '100px' }],
    ['nested-minimized', { placement: 'right', width: '160px', height: '100px' }],
    ['nested-a', { placement: 'main' }],
    ['nested-b', { placement: 'main' }]
  ] as const) {
    disposers.push(registry.register({ area: 'panes', data, id, render: () => null, title: id }))
  }

  $hiddenTreePanes.set(new Set(['hidden']))
  $paneStates.set(
    Object.fromEntries(
      ['sidebar', 'files', 'tools', 'hidden', 'minimized', 'nested-minimized'].map(id => [
        id,
        { open: true, widthOverride: 350, heightOverride: 250 }
      ])
    )
  )
})

afterEach(() => {
  cleanup()
  $layoutTree.set(null)
  $hiddenTreePanes.set(new Set())
  $paneStates.set({})
  disposers.splice(0).forEach(dispose => dispose())
  vi.unstubAllGlobals()
})

function LiveSplit() {
  const tree = useStore($layoutTree)

  return tree?.type === 'split' ? <TreeSplit node={tree} root rootRow /> : null
}

function wrapper(groupId: string): HTMLElement {
  const element = window.document.querySelector<HTMLElement>(`[data-tree-group="${groupId}"]`)?.parentElement

  if (!element) {
    throw new Error(`Missing group ${groupId}`)
  }

  return element
}

it.each(['row', 'column'] as const)(
  'double click equalizes the %s flex siblings and persists their sizes without changing other axes or splits',
  orientation => {
    const nested = split(
      orientation === 'row' ? 'column' : 'row',
      [
        split(orientation, [group(['nested-a']), group(['nested-minimized'], { minimized: true })], [2, 3]),
        group(['nested-b'])
      ],
      [2, 5],
      'nested'
    )

    const tree = split(
      orientation,
      [
        group(['sidebar'], { id: 'sidebar-zone' }),
        group(['workspace', 'files'], { id: 'workspace-zone' }),
        group(['session'], { id: 'session-zone' }),
        group(['tools'], { id: 'tools-zone' }),
        group(['hidden'], { id: 'hidden-zone' }),
        group(['minimized'], { id: 'minimized-zone', minimized: true }),
        nested
      ],
      [9, 8, 4, 7, 2, 6, 3],
      'root'
    )

    if (orientation === 'row') {
      disposers.push(registry.register({ area: 'layouts', data: tree, id: 'uneven-preset', title: 'Uneven' }))
    }

    $layoutTree.set(tree)
    persistTree()
    const { getAllByRole } = render(<LiveSplit />)

    fireEvent.doubleClick(getAllByRole('separator')[1])

    const updated = $layoutTree.get()

    if (updated?.type !== 'split') {
      throw new Error('Expected the root split')
    }

    const flexWeights = [1, 2, 6].map(index => updated.weights[index])
    expect(flexWeights.every(weight => weight === flexWeights[0])).toBe(true)
    expect(wrapper('workspace-zone').style.flexGrow).toBe(wrapper('session-zone').style.flexGrow)

    const axisOverride = orientation === 'row' ? 'widthOverride' : 'heightOverride'
    const otherOverride = orientation === 'row' ? 'heightOverride' : 'widthOverride'
    const otherSize = orientation === 'row' ? 250 : 350

    for (const id of ['sidebar', 'files', 'tools']) {
      expect($paneStates.get()[id]?.[axisOverride]).toBeUndefined()
      expect($paneStates.get()[id]?.[otherOverride]).toBe(otherSize)
    }

    expect(wrapper('sidebar-zone').style.flexBasis).toBe(orientation === 'row' ? '200px' : '120px')
    expect(wrapper('tools-zone').style.flexBasis).toBe(orientation === 'row' ? '160px' : '100px')

    for (const index of [0, 3, 4, 5]) {
      expect(updated.weights[index]).toBe(tree.weights[index])
    }

    expect(wrapper('hidden-zone').style.display).toBe('none')
    expect(updated.children[5]).toEqual(tree.children[5])
    expect(updated.children[6]).toEqual(nested)
    expect($paneStates.get().hidden?.widthOverride).toBe(350)
    expect($paneStates.get().minimized?.heightOverride).toBe(250)
    expect($paneStates.get()['nested-minimized']).toEqual({ open: true, widthOverride: 350, heightOverride: 250 })
    expect(readJson<LayoutNode>('hermes.desktop.layoutTree.v2')).toEqual(updated)
  }
)
