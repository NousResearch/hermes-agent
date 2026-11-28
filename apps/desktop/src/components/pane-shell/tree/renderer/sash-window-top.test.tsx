import { cleanup, render } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'
import { $shellChoice } from '@/store/shell-variant'
import { stubResizeObserver } from '@/test/jsdom'

import { group, split } from '../model'
import { $hiddenTreePanes, $layoutTree } from '../store'

import { TreeSplit } from './tree-split'

// The redesign keeps a seam off the titlebar band when the pane after it reaches the window top. The
// seam carries that fact itself: a stylesheet :has() looking into the trailing pane re-searched the
// whole transcript on every DOM change inside it (portal-shell.css).

beforeAll(() => {
  stubResizeObserver()
  vi.stubGlobal('CSS', { ...globalThis.CSS, escape: (value: string) => value })
})

const disposers: (() => void)[] = []

beforeEach(() => {
  window.localStorage.clear()
  $hiddenTreePanes.set(new Set())

  for (const [id, data] of [
    ['sessions', { placement: 'left', width: '237px' }],
    ['workspace', { placement: 'main', uncloseable: true }],
    ['terminal', { placement: 'bottom', height: '38vh' }]
  ] as const) {
    disposers.push(registry.register({ area: 'panes', data, id, render: () => null, title: id }))
  }
})

afterEach(() => {
  cleanup()
  $shellChoice.set(null)
  $layoutTree.set(null)
  disposers.splice(0).forEach(dispose => dispose())
})

/** Sidebar | (workspace over terminal): one seam beside the column, one inside it. */
function renderTree() {
  const tree = split('row', [
    group(['sessions'], { id: 'grp-sessions' }),
    split('column', [group(['workspace'], { id: 'grp-main' }), group(['terminal'], { id: 'grp-terminal' })])
  ])

  $layoutTree.set(tree)
  render(
    <MemoryRouter>
      <TreeSplit leftEdge node={tree} rightEdge root rootRow topEdge />
    </MemoryRouter>
  )

  const sashes = [...document.querySelectorAll<HTMLElement>('[data-sash-overlap="trailing"]')]
  const beside = sashes.find(sash => sash.getAttribute('aria-orientation') === 'vertical')
  const under = sashes.find(sash => sash.getAttribute('aria-orientation') === 'horizontal')

  return { beside, sashes, under }
}

describe('a seam beside a pane that reaches the window top', () => {
  it('is named on the seam in the redesign, and the seam under the top zone is not', () => {
    $shellChoice.set(true)
    const { beside, under } = renderTree()

    expect(beside?.hasAttribute('data-trailing-window-top')).toBe(true)
    expect(under).toBeDefined()
    expect(under?.hasAttribute('data-trailing-window-top')).toBe(false)
  })

  it('is not named in classic', () => {
    $shellChoice.set(false)
    const { sashes } = renderTree()

    expect(sashes.length).toBe(2)
    expect(sashes.some(sash => sash.hasAttribute('data-trailing-window-top'))).toBe(false)
  })
})
