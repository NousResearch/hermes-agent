import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'
import { stubMenuDomApis, stubResizeObserver } from '@/test/jsdom'

import { group, split } from '../model'
import { $layoutTree, declareDefaultTree, markCollapsePane } from '../store'

import { TreeGroup } from './tree-group'

const disposers: (() => void)[] = []

beforeAll(() => {
  stubResizeObserver()
  stubMenuDomApis()
  vi.stubGlobal('CSS', { ...globalThis.CSS, escape: (value: string) => value })
})

beforeEach(() => {
  window.localStorage.clear()

  for (const [id, data] of [
    ['sessions', { hideOnly: true, placement: 'left' }],
    ['bots', { hideOnly: true, placement: 'left' }],
    ['workspace', { uncloseable: true, placement: 'main' }],
    ['terminal', { placement: 'bottom' }]
  ] as const) {
    disposers.push(registry.register({ area: 'panes', data, id, render: () => <div>{id} content</div>, title: id }))
  }
})

afterEach(() => {
  cleanup()
  disposers.splice(0).forEach(dispose => dispose())
})

it('keeps standing sidebar tabs usable without a duplicate collapse button', () => {
  const sidebar = group(['sessions', 'bots'], { active: 'sessions', id: 'sidebar' })
  declareDefaultTree(split('row', [sidebar, group(['workspace'], { id: 'main' })]))
  render(<TreeGroup node={sidebar} parentAxis="row" />)

  expect(screen.queryByRole('button', { name: /minimize|hide/i })).toBeNull()
  expect(screen.getAllByRole('tab').map(tab => tab.textContent)).toEqual(['sessions', 'bots'])

  fireEvent.pointerDown(screen.getByRole('tab', { name: 'bots' }), { button: 0, pointerType: 'mouse' })
  fireEvent.pointerUp(screen.getByRole('tab', { name: 'bots' }), { button: 0, pointerType: 'mouse' })

  const tree = $layoutTree.get()!
  expect(tree.type === 'split' && tree.children[0].type === 'group' && tree.children[0].active).toBe('bots')
})

it('retains the collapse button for tool panels', () => {
  markCollapsePane('terminal')
  const terminal = group(['terminal'], { active: 'terminal', id: 'tools' })
  declareDefaultTree(split('column', [group(['workspace'], { id: 'main' }), terminal]))
  render(<TreeGroup node={terminal} parentAxis="column" />)

  expect(screen.getByRole('button', { name: /minimize|hide/i })).toBeTruthy()
})
