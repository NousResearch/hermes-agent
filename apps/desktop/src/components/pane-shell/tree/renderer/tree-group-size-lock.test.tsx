import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'
import { $paneStates } from '@/store/panes'
import { stubMenuDomApis, stubResizeObserver } from '@/test/jsdom'

import { $layoutEditMode } from '../../edit-mode'
import { group, split } from '../model'
import { $hiddenTreePanes, $layoutTree } from '../store'

import { TreeGroup } from './tree-group'

const disposers: (() => void)[] = []

beforeAll(() => {
  stubResizeObserver()
  stubMenuDomApis()
  vi.stubGlobal('CSS', { ...globalThis.CSS, escape: (value: string) => value })
})

beforeEach(() => {
  window.localStorage.clear()
  $hiddenTreePanes.set(new Set())
  $layoutEditMode.set(false)
  $paneStates.set({})
  disposers.push(
    registry.register({ area: 'panes', data: { placement: 'main' }, id: 'chat', render: () => null, title: 'Chat' }),
    registry.register({ area: 'panes', data: { placement: 'main' }, id: 'browser', render: () => null, title: 'Browser' })
  )
})

afterEach(() => {
  cleanup()
  $layoutTree.set(null)
  $layoutEditMode.set(false)
  $paneStates.set({})
  disposers.splice(0).forEach(dispose => dispose())
})

function openContextMenu(target: HTMLElement) {
  fireEvent.pointerDown(target, { button: 2, pointerType: 'mouse' })
  fireEvent.contextMenu(target, { button: 2 })
}

describe('zone size locks', () => {
  it('keeps size locks out of the right-click menu', async () => {
    const tree = split('row', [group(['chat'], { id: 'chat-zone' }), group(['browser'], { id: 'browser-zone' })])
    $layoutTree.set(tree)

    render(<TreeGroup node={tree.children[1] as ReturnType<typeof group>} parentAxis="row" />)
    openContextMenu(globalThis.document.querySelector<HTMLElement>('[data-tree-tab="browser"]')!)

    await screen.findByRole('menu')
    expect(screen.queryByRole('menuitem', { name: /lock (column )?(width|height)/i })).toBeNull()
    expect(screen.queryByRole('menuitem', { name: /unlock (column )?(width|height)/i })).toBeNull()
  })

  it('uses one current-state pane lock for every applicable axis in layout editor', async () => {
    const rightColumn = split(
      'column',
      [group(['browser'], { id: 'browser-zone' }), group(['files'], { id: 'files-zone' })],
      [1, 1],
      'right-column'
    )

    const tree = split('row', [group(['chat'], { id: 'chat-zone' }), rightColumn], [1, 1], 'root-row')
    $layoutTree.set(tree)
    $layoutEditMode.set(true)
    disposers.push(registry.register({ area: 'panes', data: { placement: 'right' }, id: 'files', render: () => null, title: 'Files' }))

    render(<TreeGroup node={rightColumn.children[0] as ReturnType<typeof group>} parentAxis="column" />)

    Object.defineProperty(globalThis.document.querySelector('[data-tree-group="browser-zone"]'), 'getBoundingClientRect', {
      configurable: true,
      value: () => ({ height: 260, width: 420 })
    })

    const resizable = await screen.findAllByRole('button', { name: /^pane is resizable/i })
    expect(resizable).toHaveLength(1)
    expect(resizable[0]!.getAttribute('aria-pressed')).toBe('false')
    expect(resizable[0]!.querySelector('i')!.className).toContain('codicon-unlock')
    fireEvent.click(resizable[0]!)

    expect($paneStates.get().browser).toMatchObject({ heightLocked: true, heightOverride: 260, widthLocked: true, widthOverride: 420 })
    expect($paneStates.get().files).toMatchObject({ widthLocked: true, widthOverride: 420 })

    const locked = await screen.findAllByRole('button', { name: /^pane is locked/i })
    expect(locked).toHaveLength(1)
    expect(locked[0]!.getAttribute('aria-pressed')).toBe('true')
    expect(locked[0]!.querySelector('i')!.className).toContain('codicon-lock')
    fireEvent.click(locked[0]!)

    expect($paneStates.get().browser?.heightLocked).toBeUndefined()
    expect($paneStates.get().browser?.widthLocked).toBeUndefined()
    expect($paneStates.get().files?.widthLocked).toBeUndefined()
  })
})
