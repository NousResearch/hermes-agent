import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { SidebarSectionAddButton } from './chrome'

const startNewProjectDrag = vi.hoisted(() => vi.fn())
const startNewSessionDrag = vi.hoisted(() => vi.fn())

vi.mock('@/app/chat/new-session-drag', () => ({ startNewProjectDrag, startNewSessionDrag }))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      sidebar: {
        nav: {
          'drag-hint': 'Drag this onto a tab strip to stack, or onto an edge to split.',
          'new-session': 'New session'
        }
      }
    }
  })
}))

vi.mock('@/lib/keybinds/use-keybind-hint', () => ({ useKeybindHint: () => null }))

afterEach(cleanup)

beforeEach(() => {
  startNewProjectDrag.mockReset()
  startNewSessionDrag.mockReset()
})

describe('SidebarSectionAddButton', () => {
  it('keeps a sub-threshold press an ordinary new-session click', () => {
    const onPlainClick = vi.fn()
    const onNewSessionSplit = vi.fn()

    render(
      <SidebarSectionAddButton
        ariaLabel="New session"
        onNewSessionSplit={onNewSessionSplit}
        onPlainClick={onPlainClick}
      />
    )

    fireEvent.click(screen.getByRole('button', { name: 'New session' }))

    expect(onPlainClick).toHaveBeenCalledOnce()
    expect(onNewSessionSplit).not.toHaveBeenCalled()
  })

  it('starts the new-session drag on pointer-down and creates at the drop placement', () => {
    const onPlainClick = vi.fn()
    const onNewSessionSplit = vi.fn()

    render(
      <SidebarSectionAddButton
        ariaLabel="New session"
        onNewSessionSplit={onNewSessionSplit}
        onPlainClick={onPlainClick}
      />
    )

    fireEvent.pointerDown(screen.getByRole('button', { name: 'New session' }), { button: 0 })

    expect(startNewSessionDrag).toHaveBeenCalledOnce()

    // Simulate the drag committing over a zone (stack at a strip slot).
    const onCreate = startNewSessionDrag.mock.calls[0]?.[0] as (placement: {
      anchor: string
      before?: null | string
      cwd?: null | string
      dir: 'center' | 'right'
    }) => void

    onCreate({ anchor: 'session-tile:abc', before: null, dir: 'right' })

    expect(onNewSessionSplit).toHaveBeenCalledWith('right', { anchor: 'session-tile:abc', before: null })
    expect(onPlainClick).not.toHaveBeenCalled()
  })

  it('reveals the drag affordance in the hover tooltip when the button is a drag source', async () => {
    render(
      <SidebarSectionAddButton
        ariaLabel="New session"
        onNewSessionSplit={vi.fn()}
        onPlainClick={vi.fn()}
      />
    )

    const button = screen.getByRole('button', { name: 'New session' })

    fireEvent.pointerEnter(button)
    fireEvent.pointerMove(button, { pointerType: 'mouse' })

    const tooltip = await screen.findByRole('tooltip')

    // The undiscoverable gesture is now spelled out: drag to stack, drag to split.
    expect(tooltip.textContent).toContain('New session')
    expect(tooltip.textContent).toContain('Drag this onto a tab strip to stack, or onto an edge to split.')
  })

  it('keeps the click-only button tooltip free of the drag hint', async () => {
    render(<SidebarSectionAddButton ariaLabel="New project" onPlainClick={vi.fn()} />)

    const button = screen.getByRole('button', { name: 'New project' })

    fireEvent.pointerEnter(button)
    fireEvent.pointerMove(button, { pointerType: 'mouse' })

    const tooltip = await screen.findByRole('tooltip')

    // No drag handler -> the tooltip must not advertise a gesture that does nothing.
    expect(tooltip.textContent).toBe('New project')
  })

  it('stays click-only without either drag handler', () => {
    const onPlainClick = vi.fn()

    render(<SidebarSectionAddButton ariaLabel="New project" onPlainClick={onPlainClick} />)

    const button = screen.getByRole('button', { name: 'New project' })

    fireEvent.pointerDown(button, { button: 0 })
    fireEvent.click(button)

    expect(startNewProjectDrag).not.toHaveBeenCalled()
    expect(startNewSessionDrag).not.toHaveBeenCalled()
    expect(onPlainClick).toHaveBeenCalledOnce()
  })
})
