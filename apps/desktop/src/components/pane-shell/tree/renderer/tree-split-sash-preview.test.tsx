import { cleanup, fireEvent, render } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'
import { $paneStates } from '@/store/panes'

import { group, split } from '../model'
import { $hiddenTreePanes, $layoutTree } from '../store'

import { TreeSplit } from './tree-split'

class TestResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

const disposers: (() => void)[] = []

beforeAll(() => {
  vi.stubGlobal('ResizeObserver', TestResizeObserver)
  vi.stubGlobal('CSS', { ...globalThis.CSS, escape: (value: string) => value })
  // Flush the drag preview on the same turn so mid-drag styles are observable.
  vi.stubGlobal('requestAnimationFrame', (cb: FrameRequestCallback) => {
    cb(0)

    return 1
  })
  vi.stubGlobal('cancelAnimationFrame', () => undefined)
  Element.prototype.hasPointerCapture ??= () => false
  Element.prototype.setPointerCapture ??= () => undefined
  Element.prototype.releasePointerCapture ??= () => undefined
})

beforeEach(() => {
  window.localStorage.clear()
  $hiddenTreePanes.set(new Set())
  $paneStates.set({})

  disposers.push(
    registry.register({
      area: 'panes',
      data: { placement: 'main' },
      id: 'workspace',
      render: () => null,
      title: 'Chat'
    }),
    registry.register({
      area: 'panes',
      data: { maxWidth: '320px', minWidth: '160px', placement: 'right', width: '237px' },
      id: 'review',
      render: () => null,
      title: 'Review'
    }),
    registry.register({
      area: 'panes',
      data: { maxWidth: '320px', minWidth: '160px', placement: 'right', width: '237px' },
      id: 'files',
      render: () => null,
      title: 'Files'
    })
  )
})

afterEach(() => {
  cleanup()
  $layoutTree.set(null)
  $paneStates.set({})
  disposers.splice(0).forEach(dispose => dispose())
})

function rect(width: number): DOMRect {
  return {
    bottom: 600,
    height: 600,
    left: 0,
    right: width,
    toJSON: () => ({}),
    top: 0,
    width,
    x: 0,
    y: 0
  } as DOMRect
}

function setWidth(element: HTMLElement, width: number) {
  Object.defineProperty(element, 'getBoundingClientRect', { configurable: true, value: () => rect(width) })
}

/** Default-tree shape: workspace | column[row[review|files]]. The seam's resize
 *  target is the inner review zone, not the section wrapper. */
function nestedRail() {
  return split(
    'row',
    [
      group(['workspace'], { id: 'grp-main' }),
      split(
        'column',
        [
          split(
            'row',
            [group(['review'], { id: 'grp-review' }), group(['files'], { id: 'grp-files' })],
            [1, 1.2],
            'spl-rail'
          )
        ],
        [1],
        'spl-right'
      )
    ],
    [3.4, 1.25],
    'spl-root'
  )
}

describe('TreeSplit nested section sash preview', () => {
  it('sizes the section wrapper from its pointerdown width and previews the inner zone', () => {
    const tree = nestedRail()

    $layoutTree.set(tree)
    render(<TreeSplit node={tree} root rootRow />)

    const container = document.querySelector<HTMLElement>('[data-tree-split="spl-root"]')!
    const [workspace, section] = [...container.children] as HTMLElement[]
    const review = document.querySelector<HTMLElement>('[data-tree-group="grp-review"]')!
    const reviewItem = review.parentElement as HTMLElement

    setWidth(container, 1000)
    setWidth(workspace, 526)
    setWidth(section, 474)
    setWidth(review, 237)
    setWidth(document.querySelector<HTMLElement>('[data-tree-group="grp-files"]')!, 237)

    const sash = section.querySelector('[role="separator"]')!

    fireEvent.pointerDown(sash, { button: 0, clientX: 700, pointerId: 1, pointerType: 'mouse' })
    fireEvent.pointerMove(window, { clientX: 620, pointerId: 1, pointerType: 'mouse' })

    // 474px section + 80px drag, not the 237px review zone wearing the section's basis.
    expect(section.style.flexBasis).toBe('554px')
    expect(reviewItem).not.toBe(section)
    expect(reviewItem.style.flexBasis).toBe('317px')

    fireEvent.pointerUp(window, { clientX: 620, pointerId: 1, pointerType: 'mouse' })

    expect($paneStates.get().review?.widthOverride).toBe(317)
    expect($paneStates.get().files?.widthOverride).toBeUndefined()
    // React commits the section as a calc() of the zone override plus its
    // siblings. A stuck preview pin would still be the raw `554px` basis.
    expect(section.style.flexBasis).not.toBe('554px')
  })

  it('previews a nested section reached by a cascade past the seam partner', () => {
    disposers.push(
      registry.register({
        area: 'panes',
        data: { placement: 'main', width: '100px' },
        id: 'cron',
        render: () => null,
        title: 'Cron'
      })
    )

    const tree = split(
      'row',
      [
        group(['workspace'], { id: 'grp-main' }),
        group(['cron'], { id: 'grp-cron' }),
        split(
          'column',
          [
            split(
              'row',
              [group(['review'], { id: 'grp-review' }), group(['files'], { id: 'grp-files' })],
              [1, 1.2],
              'spl-rail'
            )
          ],
          [1],
          'spl-right'
        )
      ],
      [3, 1, 1.25],
      'spl-root'
    )

    $layoutTree.set(tree)
    render(<TreeSplit node={tree} root rootRow />)

    const container = document.querySelector<HTMLElement>('[data-tree-split="spl-root"]')!
    const [workspace, cron, section] = [...container.children] as HTMLElement[]
    const review = document.querySelector<HTMLElement>('[data-tree-group="grp-review"]')!
    const reviewItem = review.parentElement as HTMLElement

    setWidth(container, 1000)
    setWidth(workspace, 426)
    setWidth(cron, 100)
    setWidth(document.querySelector<HTMLElement>('[data-tree-group="grp-cron"]')!, 100)
    setWidth(section, 474)
    setWidth(review, 237)
    setWidth(document.querySelector<HTMLElement>('[data-tree-group="grp-files"]')!, 237)

    const sash = cron.querySelector('[role="separator"]')!

    fireEvent.pointerDown(sash, { button: 0, clientX: 426, pointerId: 1, pointerType: 'mouse' })
    fireEvent.pointerMove(window, { clientX: 486, pointerId: 1, pointerType: 'mouse' })

    // Cron gives 20px down to its 80px floor, then the section's review zone
    // gives the other 40px: the wrapper shrinks from 474px, the zone from 237px.
    expect(cron.style.flexBasis).toBe('80px')
    expect(section.style.flexBasis).toBe('434px')
    expect(reviewItem.style.flexBasis).toBe('197px')

    fireEvent.pointerUp(window, { clientX: 486, pointerId: 1, pointerType: 'mouse' })

    expect($paneStates.get().cron?.widthOverride).toBe(80)
    expect($paneStates.get().review?.widthOverride).toBe(197)
    expect($paneStates.get().files?.widthOverride).toBeUndefined()
    expect(section.style.flexBasis).not.toBe('434px')
  })
})
