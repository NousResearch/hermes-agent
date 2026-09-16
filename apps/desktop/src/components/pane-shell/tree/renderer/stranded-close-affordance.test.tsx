import { useStore } from '@nanostores/react'
import { cleanup, render } from '@testing-library/react'
import { act } from 'react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'
import { setTabStripDefault } from '@/store/tabstrip-prefs'

import { group, split } from '../model'
import {
  $dismissedPanes,
  $hiddenTreePanes,
  $layoutTree,
  closeTreePane,
  registerPaneCloser,
  setTreePaneHidden
} from '../store'

import { TreeGroup } from './tree-group'

class TestResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

beforeAll(() => {
  vi.stubGlobal('ResizeObserver', TestResizeObserver)
  vi.stubGlobal('CSS', { ...globalThis.CSS, escape: (value: string) => value })
  Element.prototype.hasPointerCapture ??= () => false
  Element.prototype.setPointerCapture ??= () => undefined
  Element.prototype.releasePointerCapture ??= () => undefined
  HTMLElement.prototype.scrollIntoView ??= () => undefined
})

const disposers: (() => void)[] = []

function registerPane(id: string, data: Record<string, unknown>, source?: string) {
  disposers.push(
    registry.register({
      area: 'panes',
      data,
      id,
      render: () => <div>{id}</div>,
      source,
      title: id
    })
  )
}

function LiveTreeGroups() {
  const tree = useStore($layoutTree)

  if (!tree) {
    return null
  }

  const groups = tree.type === 'group' ? [tree] : tree.children.filter(child => child.type === 'group')
  const parentAxis = tree.type === 'split' ? tree.orientation : 'row'

  return groups.map(node => <TreeGroup key={node.id} node={node} parentAxis={parentAxis} />)
}

const tab = (paneId: string) => globalThis.document.querySelector(`[data-tree-tab="${paneId}"]`)
const closeButton = (paneId: string) => tab(paneId)?.querySelector('button[aria-label]')

beforeEach(() => {
  window.localStorage.clear()
  setTabStripDefault('auto')
  $dismissedPanes.set(new Set())
  $hiddenTreePanes.set(new Set())
  $layoutTree.set(null)
})

afterEach(() => {
  cleanup()
  registerPaneCloser('workspace')
  registerPaneCloser('core-side-pane')
  disposers.splice(0).forEach(dispose => dispose())
})

describe('a closeable pane never loses its only visible Close affordance', () => {
  it('keeps a private runtime-plugin tab after its core sibling closes', () => {
    registerPane('core-side-pane', { placement: 'right' })
    registerPane('custom-plugin:pane', { placement: 'right', width: '320px' }, 'plugin:custom-plugin')
    registerPaneCloser('core-side-pane', () => setTreePaneHidden('core-side-pane', true))
    $layoutTree.set(
      group(['core-side-pane', 'custom-plugin:pane'], {
        active: 'core-side-pane',
        id: 'grp-right'
      })
    )

    render(<LiveTreeGroups />)
    expect(tab('custom-plugin:pane')).not.toBeNull()

    act(() => closeTreePane('core-side-pane'))

    expect(tab('custom-plugin:pane')).not.toBeNull()
  })

  it('keeps the closable workspace tab after its session sibling closes', () => {
    registerPane('workspace', { placement: 'main', uncloseable: true })
    registerPane('session-tile:other', { placement: 'main' })
    registerPaneCloser('workspace', vi.fn())
    $layoutTree.set(
      group(['workspace', 'session-tile:other'], {
        active: 'session-tile:other',
        id: 'grp-main'
      })
    )

    render(<LiveTreeGroups />)
    expect(tab('workspace')).not.toBeNull()

    act(() => closeTreePane('session-tile:other'))

    expect(tab('workspace')).not.toBeNull()
  })

  it('keeps title, drag and Close tabs on both vertically split chat zones', () => {
    registerPane('workspace', { placement: 'main', uncloseable: true })
    registerPane('session-tile:lower', { placement: 'main' })
    $layoutTree.set(
      split('column', [
        group(['workspace'], { active: 'workspace', id: 'grp-chat-top' }),
        group(['session-tile:lower'], { active: 'session-tile:lower', id: 'grp-chat-bottom' })
      ])
    )

    render(<LiveTreeGroups />)
    act(() => registerPaneCloser('workspace', vi.fn()))

    expect(tab('workspace')).not.toBeNull()
    expect(closeButton('workspace')).not.toBeNull()
    expect(tab('session-tile:lower')).not.toBeNull()
    expect(closeButton('session-tile:lower')).not.toBeNull()
  })
})
