import { act, cleanup, fireEvent, render } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type * as PanesStore from '@/store/panes'
import { $paneStates, setPaneWidthOverride } from '@/store/panes'

import { MasterDetail } from './master-detail'

vi.mock('@/store/panes', async importOriginal => {
  const actual = await importOriginal<typeof PanesStore>()

  return { ...actual, setPaneWidthOverride: vi.fn() }
})

afterEach(() => {
  cleanup()
  $paneStates.set({})
  vi.clearAllMocks()
})

describe('MasterDetail logical column seam', () => {
  it.each([
    { direction: 'ltr', delta: 30, expected: 270 },
    { direction: 'ltr', delta: -30, expected: 210 },
    { direction: 'rtl', delta: 30, expected: 210 },
    { direction: 'rtl', delta: -30, expected: 270 }
  ])('$direction drag $delta updates the adjacent list column', ({ direction, delta, expected }) => {
    const { getByTestId } = render(
      <MasterDetail resizeId="test-list">
        <aside data-testid="list" />
        <article data-testid="detail" />
      </MasterDetail>
    )

    const list = getByTestId('list')
    const grid = list.parentElement!
    grid.style.direction = direction
    grid.getBoundingClientRect = () => ({ width: 1000 }) as DOMRect
    list.getBoundingClientRect = () => ({ width: 240 }) as DOMRect
    const detail = getByTestId('detail').parentElement!
    const handle = detail.firstElementChild as HTMLElement
    fireEvent.pointerDown(handle, { button: 0, clientX: 500, pointerId: 1 })

    try {
      act(() => {
        window.dispatchEvent(new PointerEvent('pointermove', { clientX: 500 + delta, pointerId: 1 }))
      })
    } finally {
      act(() => {
        window.dispatchEvent(new PointerEvent('pointerup', { pointerId: 1 }))
      })
    }

    expect(handle.style.insetInlineStart).toBe('0px')
    expect(handle.className).not.toMatch(/\bleft-0\b/)
    expect(setPaneWidthOverride).toHaveBeenCalledExactlyOnceWith('test-list', expected)
  })
})
