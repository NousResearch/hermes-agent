import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'
import { $paneStates } from '@/store/panes'
import { stubMenuDomApis, stubResizeObserver } from '@/test/jsdom'

import { $layoutEditMode } from '../../edit-mode'
import { group, split } from '../model'
import { $hiddenTreePanes, $layoutTree } from '../store'

import { edgeFixedZone, fixedTrackSize } from './track-model'
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

  it('locks only Terminal at its measured size in the Default layout', async () => {
    const upperRail = split(
      'row',
      [group(['review'], { id: 'review-zone' }), group(['files'], { id: 'files-zone' })],
      [1, 1.2],
      'rail-row'
    )

    const rightColumn = split(
      'column',
      [upperRail, group(['terminal'], { id: 'terminal-zone' })],
      [1.6, 1],
      'right-column'
    )

    const tree = split('row', [group(['chat'], { id: 'chat-zone' }), rightColumn], [1, 1], 'root-row')
    $layoutTree.set(tree)
    $layoutEditMode.set(true)
    disposers.push(
      registry.register({ area: 'panes', data: { placement: 'right' }, id: 'review', render: () => null, title: 'Review' }),
      registry.register({ area: 'panes', data: { placement: 'right' }, id: 'files', render: () => null, title: 'Files' }),
      registry.register({ area: 'panes', data: { placement: 'bottom' }, id: 'terminal', render: () => null, title: 'Terminal' })
    )

    render(<TreeGroup node={rightColumn.children[1] as ReturnType<typeof group>} parentAxis="column" />)

    Object.defineProperty(globalThis.document.querySelector('[data-tree-group="terminal-zone"]'), 'getBoundingClientRect', {
      configurable: true,
      value: () => ({ height: 260, width: 420 })
    })

    const resizable = await screen.findAllByRole('button', { name: /^pane is resizable/i })
    expect(resizable).toHaveLength(1)
    fireEvent.click(resizable[0]!)

    expect($paneStates.get().terminal).toMatchObject({ heightLocked: true, heightOverride: 260, widthLocked: true, widthOverride: 420 })
    expect($paneStates.get().review?.widthLocked).toBeUndefined()
    expect($paneStates.get().review?.widthOverride).toBeUndefined()
    expect($paneStates.get().files?.widthLocked).toBeUndefined()
    expect($paneStates.get().files?.widthOverride).toBeUndefined()
  })

  it('keeps a shared column within the width locked by one child', () => {
    const upperRail = split(
      'row',
      [group(['review'], { id: 'review-zone' }), group(['files'], { id: 'files-zone' })],
      [1, 1.2],
      'rail-row'
    )

    const rightColumn = split(
      'column',
      [upperRail, group(['terminal'], { id: 'terminal-zone' })],
      [1.6, 1],
      'right-column'
    )

    const paneFor = (id: string) =>
      ({
        data: id === 'review' || id === 'files' ? { width: '320px' } : {},
        id
      }) as never

    expect(
      fixedTrackSize(rightColumn, 'row', {
        overrides: { terminal: { widthLocked: true, widthOverride: 420 } },
        paneFor,
        paneGone: () => false
      })
    ).toBe('420px')
  })

  it('uses the largest cross-axis lock as the shared sash owner', () => {
    const sharedColumn = split(
      'column',
      [group(['terminal'], { id: 'terminal-zone' }), group(['logs'], { id: 'logs-zone' })],
      [1, 1],
      'right-column'
    )

    const ctx = {
      overrides: {
        logs: { widthLocked: true, widthOverride: 480 },
        terminal: { widthLocked: true, widthOverride: 420 }
      },
      paneFor: (id: string) => ({ data: {}, id }) as never,
      paneGone: () => false
    }

    expect(fixedTrackSize(sharedColumn, 'row', ctx)).toBe('480px')
    expect(edgeFixedZone(sharedColumn, 'start', 'row', ctx)?.id).toBe('logs-zone')
    expect(edgeFixedZone(sharedColumn, 'end', 'row', ctx)?.id).toBe('logs-zone')
  })

  it('routes an outer seam past a locally locked edge pane to an unlocked sibling', () => {
    const topRail = split(
      'row',
      [group(['review'], { id: 'review-zone' }), group(['files'], { id: 'files-zone' })],
      [1, 1],
      'top-rail'
    )

    const ctx = {
      overrides: {
        files: { widthOverride: 200 },
        review: { widthLocked: true, widthOverride: 300 }
      },
      paneFor: (id: string) => ({ data: { width: id === 'review' ? '300px' : '200px' }, id }) as never,
      paneGone: () => false
    }

    expect(edgeFixedZone(topRail, 'start', 'row', ctx)?.id).toBe('files-zone')
  })

  it('composes adjacent locks before comparing a shared cross-axis boundary', () => {
    const topRail = split(
      'row',
      [group(['review'], { id: 'review-zone' }), group(['files'], { id: 'files-zone' })],
      [1, 1],
      'top-rail'
    )
    const rightColumn = split('column', [topRail, group(['terminal'], { id: 'terminal-zone' })], [1, 1], 'right-column')

    const ctx = {
      overrides: {
        files: { widthLocked: true, widthOverride: 200 },
        review: { widthLocked: true, widthOverride: 300 },
        terminal: { widthLocked: true, widthOverride: 400 }
      },
      paneFor: (id: string) => ({ data: {}, id }) as never,
      paneGone: () => false
    }

    expect(fixedTrackSize(rightColumn, 'row', ctx)).toBe('500px')
  })

  it('keeps a shared row within the height locked by one child', () => {
    const sharedRow = split(
      'row',
      [group(['review'], { id: 'review-zone' }), group(['files'], { id: 'files-zone' })],
      [1, 1.2],
      'shared-row'
    )

    const paneFor = (id: string) =>
      ({
        data: id === 'files' ? { height: '320px' } : {},
        id
      }) as never

    expect(
      fixedTrackSize(sharedRow, 'column', {
        overrides: { review: { heightLocked: true, heightOverride: 260 } },
        paneFor,
        paneGone: () => false
      })
    ).toBe('260px')
  })
})
