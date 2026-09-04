import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'
import { $paneStates } from '@/store/panes'
import { stubMenuDomApis, stubResizeObserver } from '@/test/jsdom'

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
  $paneStates.set({})
  disposers.push(
    registry.register({ area: 'panes', data: { placement: 'main' }, id: 'chat', render: () => null, title: 'Chat' }),
    registry.register({ area: 'panes', data: { placement: 'main' }, id: 'browser', render: () => null, title: 'Browser' })
  )
})

afterEach(() => {
  cleanup()
  $layoutTree.set(null)
  $paneStates.set({})
  disposers.splice(0).forEach(dispose => dispose())
})

function openContextMenu(target: HTMLElement) {
  fireEvent.pointerDown(target, { button: 2, pointerType: 'mouse' })
  fireEvent.contextMenu(target, { button: 2 })
}

describe('zone size locks', () => {
  it('offers Lock width for a panel that has a horizontal neighbor and persists the lock', async () => {
    const tree = split('row', [group(['chat'], { id: 'chat-zone' }), group(['browser'], { id: 'browser-zone' })])
    $layoutTree.set(tree)
    $paneStates.set({ browser: { open: true, widthOverride: 420 } })

    render(<TreeGroup node={tree.children[1] as ReturnType<typeof group>} parentAxis="row" />)

    Object.defineProperty(document.querySelector('[data-tree-group="browser-zone"]'), 'getBoundingClientRect', {
      configurable: true,
      value: () => ({ height: 600, width: 420 })
    })

    openContextMenu(document.querySelector<HTMLElement>('[data-tree-tab="browser"]')!)
    const lock = await screen.findByRole('menuitem', { name: /^lock width$/i })
    expect(screen.queryByRole('menuitem', { name: /^lock height$/i })).toBeNull()
    fireEvent.click(lock)

    expect($paneStates.get().browser).toMatchObject({ open: true, widthLocked: true, widthOverride: 420 })
    expect(JSON.parse(window.localStorage.getItem('hermes.desktop.paneStates.v1') ?? '{}')).toMatchObject({
      browser: { open: true, widthLocked: true, widthOverride: 420 }
    })

    openContextMenu(document.querySelector<HTMLElement>('[data-tree-tab="browser"]')!)
    fireEvent.click(await screen.findByRole('menuitem', { name: /^unlock width$/i }))
    expect($paneStates.get().browser).toMatchObject({ open: true, widthOverride: 420 })
    expect($paneStates.get().browser?.widthLocked).toBeUndefined()

    openContextMenu(document.querySelector<HTMLElement>('[data-tree-tab="browser"]')!)
    fireEvent.click(await screen.findByRole('menuitem', { name: /^lock width$/i }))
    expect($paneStates.get().browser).toMatchObject({ open: true, widthLocked: true, widthOverride: 420 })
  })

  it('offers Lock height for a panel that has a vertical neighbor and omits width', async () => {
    const tree = split('column', [group(['chat'], { id: 'chat-zone' }), group(['browser'], { id: 'browser-zone' })])
    $layoutTree.set(tree)

    render(<TreeGroup node={tree.children[1] as ReturnType<typeof group>} parentAxis="column" />)

    Object.defineProperty(document.querySelector('[data-tree-group="browser-zone"]'), 'getBoundingClientRect', {
      configurable: true,
      value: () => ({ height: 260, width: 900 })
    })

    openContextMenu(document.querySelector<HTMLElement>('[data-tree-tab="browser"]')!)
    const lock = await screen.findByRole('menuitem', { name: /^lock height$/i })
    expect(screen.queryByRole('menuitem', { name: /^lock width$/i })).toBeNull()
    fireEvent.click(lock)

    expect($paneStates.get().browser).toMatchObject({ heightLocked: true, heightOverride: 260, open: false })
  })

  it('offers Lock height from a hidden-tab-strip zone body', async () => {
    const tree = split(
      'column',
      [group(['chat'], { id: 'chat-zone' }), group(['files'], { id: 'files-zone', tabStrip: 'never' })],
      [1, 1]
    )

    $layoutTree.set(tree)
    disposers.push(registry.register({ area: 'panes', data: { placement: 'right' }, id: 'files', render: () => null, title: 'Files' }))

    render(<TreeGroup node={tree.children[1] as ReturnType<typeof group>} parentAxis="column" />)

    Object.defineProperty(document.querySelector('[data-tree-group="files-zone"]'), 'getBoundingClientRect', {
      configurable: true,
      value: () => ({ height: 260, width: 420 })
    })

    openContextMenu(document.querySelector<HTMLElement>('[data-tree-group="files-zone"]')!)
    expect(await screen.findByRole('menuitem', { name: /^lock height$/i })).toBeTruthy()
  })

  it('offers Lock column width and locks every zone in a vertically stacked column', async () => {
    const stack = split(
      'column',
      [group(['chat'], { id: 'files-zone' }), group(['browser'], { id: 'browser-zone' })],
      [1, 1],
      'right-column'
    )

    const tree = split('row', [group(['left'], { id: 'left-zone' }), stack], [1, 1], 'root-row')
    $layoutTree.set(tree)
    disposers.push(registry.register({ area: 'panes', data: { placement: 'main' }, id: 'left', render: () => null, title: 'Left' }))

    render(<TreeGroup node={stack.children[1] as ReturnType<typeof group>} parentAxis="column" />)

    Object.defineProperty(document.querySelector('[data-tree-group="browser-zone"]'), 'getBoundingClientRect', {
      configurable: true,
      value: () => ({ height: 260, width: 420 })
    })

    openContextMenu(document.querySelector<HTMLElement>('[data-tree-tab="browser"]')!)
    expect(await screen.findByRole('menuitem', { name: /^lock height$/i })).toBeTruthy()
    const lockColumn = await screen.findByRole('menuitem', { name: /^lock column width$/i })
    fireEvent.click(lockColumn)

    expect($paneStates.get().chat).toMatchObject({ widthLocked: true, widthOverride: 420 })
    expect($paneStates.get().browser).toMatchObject({ widthLocked: true, widthOverride: 420 })
  })
})
