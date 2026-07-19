import { beforeEach, describe, expect, it } from 'vitest'

import { group, split } from './model'
import {
  $collapsedTreeSides,
  $hiddenTreePanes,
  declareDefaultTree,
  setTreePaneHidden,
  setTreeSideCollapsed
} from './store'

describe('setTreePaneHidden reveal option', () => {
  beforeEach(() => {
    declareDefaultTree(
      split('row', [group(['sessions']), group(['workspace']), group(['files'])], [1, 3, 1])
    )
    // Start from a known closed right rail + hidden files pane.
    setTreeSideCollapsed('right', true)
    setTreePaneHidden('files', true, { reveal: false })
  })

  it('unhides without opening the side when reveal is false', () => {
    expect($hiddenTreePanes.get().has('files')).toBe(true)
    expect($collapsedTreeSides.get().has('right')).toBe(true)

    setTreePaneHidden('files', false, { reveal: false })

    expect($hiddenTreePanes.get().has('files')).toBe(false)
    // Right rail stays collapsed: cwd adoption can fill the tree without
    // force-opening the files panel.
    expect($collapsedTreeSides.get().has('right')).toBe(true)
  })

  it('unhides and opens the side when reveal is true (default)', () => {
    setTreePaneHidden('files', false)

    expect($hiddenTreePanes.get().has('files')).toBe(false)
    expect($collapsedTreeSides.get().has('right')).toBe(false)
  })
})
