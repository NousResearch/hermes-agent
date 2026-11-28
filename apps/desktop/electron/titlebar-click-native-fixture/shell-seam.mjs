import { atom } from 'nanostores'
import { createElement as h, useRef } from 'react'

import { usePanelTitlebar } from '../../src/components/pane-shell/tree/renderer/panel-titlebar'

// Only unrelated shell services and heavy panel content are replaced. The
// bundler loads the real LayoutTreeRoot, titlebar hook and resize observer.
// JS keeps this renderer-only fixture out of Electron's composite TS project.
export const $layoutTree = atom(null)
export const $connection = atom(null)
export const trackActiveTreeGroup = () => {}
export const useLayoutEditHotkey = () => {}
export const useTabKeyHints = () => {}
export const publishWorkspaceGeometry = () => {}
export const ZoneEditor = () => null
export const TreeEditBar = () => null
export const FloatingPanes = () => null
export const NarrowOverlays = () => null
export const KeepAlivePanes = ({ children }) => children

export function TreeNode({ node, topEdge }) {
  return node.type === 'split'
    ? h(
        'div',
        { className: 'fixture-split' },
        ...node.children.map(child => h(TreeNode, { key: child.id, node: child, topEdge }))
      )
    : h(Panel, { enabled: topEdge, id: node.id })
}

function Panel({ enabled, id }) {
  const ref = useRef(null)
  const below = usePanelTitlebar(ref, enabled, false)

  return h(
    'div',
    { className: 'panel', 'data-below': below, 'data-tree-group': id, ref },
    h(
      'div',
      { className: 'band' },
      h('div', { className: 'leftpad' }),
      h('div', { className: 'drag', 'data-window-drag-handle': '' }),
      h('div', { className: 'rightpad' })
    ),
    id
  )
}
