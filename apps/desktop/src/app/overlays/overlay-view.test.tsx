// @vitest-environment jsdom
import type { ReactNode } from 'react'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { OverlayView } from './overlay-view'

afterEach(() => {
  cleanup()
})

const TRANSLATE_CLASS = /(?:^|\s)-?translate-[xy](?:-\S+)?(?:\s|$)/
const DRAG_TOKEN = '[-webkit-app-region:drag]'
const NO_DRAG_TOKEN = '[-webkit-app-region:no-drag]'

function classNameOf(node: Element): string {
  return typeof node.className === 'string' ? node.className : String(node.className)
}

function hasTranslateClass(node: Element): boolean {
  return TRANSLATE_CLASS.test(classNameOf(node))
}

function ancestorMatching(node: Element, predicate: (el: Element) => boolean): Element | null {
  let current: Element | null = node
  while (current) {
    if (predicate(current)) return current
    current = current.parentElement
  }
  return null
}

function nearestDragAncestor(node: Element): Element | null {
  return ancestorMatching(node, el => classNameOf(el).includes(DRAG_TOKEN) || classNameOf(el).includes('-webkit-app-region:drag'))
}

function nearestNoDragWrapper(node: Element): Element | null {
  return ancestorMatching(node, el => classNameOf(el).includes(NO_DRAG_TOKEN))
}

function nodesBetweenInclusive(inner: Element, outer: Element): Element[] {
  const chain: Element[] = []
  let current: Element | null = inner
  while (current) {
    chain.push(current)
    if (current === outer) break
    current = current.parentElement
  }
  return chain
}

function renderOverlay(onClose = vi.fn(), extras: { headerContent?: ReactNode; titlebarActions?: ReactNode } = {}) {
  render(
    <OverlayView closeLabel="Close settings" headerContent={extras.headerContent} onClose={onClose} titlebarActions={extras.titlebarActions}>
      <div>overlay body</div>
    </OverlayView>
  )
  return onClose
}

describe('OverlayView close chrome hit target', () => {
  it('keeps the close button on a no-transform no-drag wrapper inside the drag strip', () => {
    renderOverlay()

    const closeButton = screen.getByRole('button', { name: 'Close settings' })
    const dragAncestor = nearestDragAncestor(closeButton)
    const noDragWrapper = nearestNoDragWrapper(closeButton)

    expect(dragAncestor, 'drag strip must remain').not.toBeNull()
    expect(noDragWrapper, 'close X must stay a no-drag hit target').not.toBeNull()
    expect(hasTranslateClass(noDragWrapper!), 'nearest no-drag wrapper must not use translate-*').toBe(false)

    for (const node of nodesBetweenInclusive(closeButton, dragAncestor!)) {
      expect(hasTranslateClass(node), `${classNameOf(node)} must not use translate-* under the drag strip`).toBe(false)
    }
  })

  it('calls onClose when the settings dismiss control is clicked', () => {
    const onClose = renderOverlay()

    fireEvent.click(screen.getByRole('button', { name: 'Close settings' }))

    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it('does not put translate-* on the headerContent no-drag wrapper', () => {
    renderOverlay(vi.fn(), { headerContent: <div>search pill</div> })

    const header = screen.getByText('search pill')
    const dragAncestor = nearestDragAncestor(header)
    const noDragWrapper = nearestNoDragWrapper(header)

    expect(dragAncestor).not.toBeNull()
    expect(noDragWrapper).not.toBeNull()
    expect(hasTranslateClass(noDragWrapper!)).toBe(false)

    for (const node of nodesBetweenInclusive(header, dragAncestor!)) {
      expect(hasTranslateClass(node)).toBe(false)
    }
  })
})
