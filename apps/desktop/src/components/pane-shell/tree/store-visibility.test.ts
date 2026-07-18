import { atom } from 'nanostores'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'

import { group, split } from './model'
import {
  $collapsedTreeSides,
  $hiddenTreePanes,
  bindTreeSideVisibility,
  declareDefaultTree,
  setTreePaneHidden
} from './store'

describe('tree pane visibility intent', () => {
  const disposers: Array<() => void> = []

  afterEach(() => {
    disposers.splice(0).forEach(dispose => dispose())
  })

  it('does not reopen a collapsed side for a structural unhide', () => {
    disposers.push(
      registry.register({ id: 'files', area: 'panes', title: 'files', data: { placement: 'right' } }),
      registry.register({ id: 'workspace', area: 'panes', title: 'workspace', data: { placement: 'main' } })
    )
    declareDefaultTree(split('row', [group(['workspace']), group(['files'])], [3, 1]))

    const sideOpen = atom(false)
    const setSideOpen = vi.fn()
    bindTreeSideVisibility('right', sideOpen, setSideOpen)
    expect($collapsedTreeSides.get().has('right')).toBe(true)

    setTreePaneHidden('files', true)
    setTreePaneHidden('files', false, { revealOnShow: false })

    expect($hiddenTreePanes.get().has('files')).toBe(false)
    expect($collapsedTreeSides.get().has('right')).toBe(true)
    expect(setSideOpen).not.toHaveBeenCalled()

    setTreePaneHidden('files', true)
    setTreePaneHidden('files', false)
    expect(setSideOpen).toHaveBeenCalledWith(true)
  })
})
