import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { beginSashDrag, endSashDrag } from '@/components/pane-shell/geometry'
import { registry } from '@/contrib/registry'
import type * as PanesStore from '@/store/panes'
import { $paneStates, setPaneWidthOverride } from '@/store/panes'

import type { LayoutNode } from '../model'
import { group, split } from '../model'

import { TreeSplit } from './tree-split'

const panesMock = vi.hoisted(() => ({
  setPaneWidthOverride: vi.fn<(id: string, width: number | undefined) => void>(),
  throwOnOverride: { current: false }
}))

vi.mock('@/store/panes', async importOriginal => {
  const actual = await importOriginal<typeof PanesStore>()

  return {
    ...actual,
    setPaneWidthOverride: (id: string, width: number | undefined) => {
      panesMock.setPaneWidthOverride(id, width)

      if (panesMock.throwOnOverride.current) {
        throw new Error('boom')
      }

      actual.setPaneWidthOverride(id, width)
    }
  }
})

vi.mock('@/components/pane-shell/geometry', () => ({ beginSashDrag: vi.fn(), endSashDrag: vi.fn() }))

vi.mock('./tree-node', () => ({
  // The sash math only needs the zone element it measures by
  // [data-tree-group="<id>"]; the real TreeNode pulls in the whole zone chrome
  // (context menus, i18n, drag sessions) that these tests never touch.
  TreeNode: ({ node }: { node: LayoutNode }) =>
    node.type === 'group' ? <div data-tree-group={node.id} /> : null
}))

const SPLIT_ID = 'spl-root'
const SIDEBAR_ID = 'sidebar'
const SIDEBAR_PX = 240
const MAIN_PX = 760

// A row split with a FIXED left sidebar (declared width 240px) and a flex
// main zone, mirroring the classic sidebar | workspace seam.
function makeNode() {
  return split(
    'row',
    [group([SIDEBAR_ID], { id: 'grp-side' }), group(['workspace'], { id: 'grp-main' })],
    [1, 1],
    SPLIT_ID
  )
}

function registerPanes() {
  return [
    registry.register({
      id: SIDEBAR_ID,
      area: 'panes',
      data: { placement: 'left', width: `${SIDEBAR_PX}px` },
      render: () => null,
      title: 'sidebar'
    }),
    registry.register({
      id: 'workspace',
      area: 'panes',
      data: { placement: 'main', uncloseable: true },
      render: () => null,
      title: 'workspace'
    })
  ]
}

// jsdom reports zero rects; the drag math reads live sizes, so pin the
// container, the fixed zone element, and the flex wrapper to known widths.
function stubWidth(el: Element | null, width: number) {
  if (!el) {
    throw new Error('expected element to stub')
  }

  ;(el as HTMLElement).getBoundingClientRect = () =>
    ({
      bottom: 600,
      height: 600,
      left: 0,
      right: width,
      toJSON: () => ({}),
      top: 0,
      width,
      x: 0,
      y: 0
    }) as DOMRect
}

function renderSplit() {
  render(<TreeSplit node={makeNode()} root rootRow />)

  const container = window.document.querySelector<HTMLElement>(`[data-tree-split="${SPLIT_ID}"]`)

  if (!container) {
    throw new Error('split container not rendered')
  }

  const kidA = container.children[0] as HTMLElement
  const kidB = container.children[1] as HTMLElement

  stubWidth(container, 1000)
  stubWidth(container.querySelector('[data-tree-group="grp-side"]'), SIDEBAR_PX)
  stubWidth(kidB, MAIN_PX)

  return { kidA, kidB }
}

let rafCallbacks: FrameRequestCallback[] = []
let disposers: (() => void)[] = []

// rafCoalesce schedules one preview per frame; capture the callbacks so each
// test decides when a frame lands.
function flushRaf() {
  const callbacks = rafCallbacks
  rafCallbacks = []
  callbacks.forEach(cb => cb(0))
}

function pointer(type: string, clientX: number) {
  return new window.PointerEvent(type, { bubbles: true, button: 0, cancelable: true, clientX, pointerId: 1 })
}

function pressSash(clientX: number) {
  fireEvent.pointerDown(screen.getByRole('separator'), { button: 0, clientX, pointerId: 1 })
}

function moveTo(clientX: number) {
  act(() => {
    window.dispatchEvent(pointer('pointermove', clientX))
  })
}

function releaseAt(clientX: number) {
  act(() => {
    window.dispatchEvent(pointer('pointerup', clientX))
  })
}

describe('TreeSplit sash drag cleanup', () => {
  beforeEach(() => {
    window.localStorage.clear()
    disposers = []
    rafCallbacks = []
    panesMock.throwOnOverride.current = false
    panesMock.setPaneWidthOverride.mockClear()
    vi.mocked(beginSashDrag).mockClear()
    vi.mocked(endSashDrag).mockClear()
    vi.spyOn(window, 'requestAnimationFrame').mockImplementation(cb => {
      rafCallbacks.push(cb)

      return rafCallbacks.length
    })
    vi.spyOn(window, 'cancelAnimationFrame').mockImplementation(() => undefined)
  })

  afterEach(() => {
    cleanup()
    disposers.forEach(dispose => dispose())
    $paneStates.set({})
    window.document.body.style.cursor = ''
    window.document.body.style.userSelect = ''
    vi.restoreAllMocks()
  })

  it('restores both seam wrapper styles when the released px is a store no-op', () => {
    disposers = registerPanes()
    // The override already holds the exact px the drag will end at.
    setPaneWidthOverride(SIDEBAR_ID, SIDEBAR_PX)
    panesMock.setPaneWidthOverride.mockClear()

    const { kidA, kidB } = renderSplit()
    const styleA = kidA.getAttribute('style')
    const styleB = kidB.getAttribute('style')

    pressSash(SIDEBAR_PX)

    // Drag out 60px: the rAF preview pins the fixed side's flex-basis inline.
    moveTo(SIDEBAR_PX + 60)
    flushRaf()
    expect(kidA.style.flexBasis).toBe(`${SIDEBAR_PX + 60}px`)

    // Drag back to the origin and release: the commit writes 240px, exactly
    // what the override already holds, so the store early-returns and nothing
    // re-renders. Pre-fix cleanup counted on that re-render to clear the
    // preview, leaving the flex-basis longhand stuck on the wrapper.
    moveTo(SIDEBAR_PX)
    releaseAt(SIDEBAR_PX)

    expect(kidA.getAttribute('style')).toBe(styleA)
    expect(kidB.getAttribute('style')).toBe(styleB)
  })

  it('runs the full drag teardown when the store commit throws', () => {
    disposers = registerPanes()

    const { kidA } = renderSplit()
    const styleA = kidA.getAttribute('style')

    pressSash(SIDEBAR_PX)
    moveTo(SIDEBAR_PX + 60)
    flushRaf()
    expect(kidA.style.flexBasis).toBe(`${SIDEBAR_PX + 60}px`)

    // The released commit throws inside the window-level pointerup listener.
    // jsdom reports a listener exception through an error event instead of
    // propagating it, so swallow that report and assert on teardown effects.
    panesMock.throwOnOverride.current = true

    const swallow = (event: Event) => event.preventDefault()
    window.addEventListener('error', swallow)

    try {
      releaseAt(SIDEBAR_PX + 60)
    } finally {
      window.removeEventListener('error', swallow)
    }

    // Pre-fix, the throwing commit skipped everything below applyShift: the
    // sash-drag depth stayed > 0 and the window listeners stayed attached.
    expect(beginSashDrag).toHaveBeenCalledTimes(1)
    expect(endSashDrag).toHaveBeenCalledTimes(1)
    expect(window.document.body.style.cursor).toBe('')
    expect(window.document.body.style.userSelect).toBe('')
    expect(kidA.getAttribute('style')).toBe(styleA)

    // The pointer listeners are gone: a further move previews nothing.
    moveTo(SIDEBAR_PX + 120)
    flushRaf()
    expect(kidA.getAttribute('style')).toBe(styleA)
  })

})
