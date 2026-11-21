import { createElement as h, Fragment, useState } from 'react'
import { createRoot } from 'react-dom/client'

import { mirrorTreeHorizontal } from '../../src/components/pane-shell/tree/model'
import { LayoutTreeRoot } from '../../src/components/pane-shell/tree/renderer'

import { $layoutTree } from './shell-seam.mjs'

const document = globalThis.document
const initialTree = {
  type: 'split',
  id: 'root',
  orientation: 'row',
  weights: [1, 3],
  children: ['sidebar', 'main'].map(id => ({ type: 'group', id, panes: [id], active: id }))
}
const counts = {}
const originals = new Map()
$layoutTree.set(initialTree)

function record(id) {
  counts[id] = (counts[id] ?? 0) + 1

  if (id === 'swap') {
    $layoutTree.set(mirrorTreeHorizontal($layoutTree.get()))
  }
}

function snapshot() {
  const panels = [...document.querySelectorAll('[data-tree-group]')].map(element => {
    const id = element.dataset.treeGroup

    if (!originals.has(id)) {
      originals.set(id, element)
    }

    const bounds = element.getBoundingClientRect()

    return {
      id,
      x: bounds.x,
      width: bounds.width,
      retained: originals.get(id) === element,
      left: element.style.getPropertyValue('--panel-titlebar-left'),
      right: element.style.getPropertyValue('--panel-titlebar-right')
    }
  })
  const drags = [...document.querySelectorAll('.drag')].map(element => {
    const { x, y, width, height } = element.getBoundingClientRect()

    return { x, y, width, height }
  })
  const buttons = [...document.querySelectorAll('button')].map(element => {
    const bounds = element.getBoundingClientRect()
    const x = bounds.x + bounds.width / 2
    const y = bounds.y + bounds.height / 2

    return { id: element.id, x, y, domTarget: document.elementFromPoint(x, y)?.id }
  })

  return { counts, panels, drags, buttons }
}

function App() {
  const [negativeControl, setNegativeControl] = useState(false)
  Object.assign(globalThis.window, { titlebarFixture: { snapshot, setNegativeControl } })
  const button = id => h('button', { id, key: id, onClick: () => record(id), type: 'button' }, id)

  return h(
    Fragment,
    null,
    h('div', { className: 'cluster', 'data-titlebar-cluster': 'left', id: 'left' }, button('hide')),
    h(
      'div',
      { className: 'cluster', 'data-titlebar-cluster': 'right', id: 'right' },
      ...['settings', 'layoutTool', 'hud', 'swap', 'rhide'].map(button)
    ),
    h('div', { id: 'layout' }, h(LayoutTreeRoot, { titlebar: true })),
    button('body'),
    negativeControl && h('div', { id: 'native-negative-control' })
  )
}

createRoot(document.getElementById('root')).render(h(App))
