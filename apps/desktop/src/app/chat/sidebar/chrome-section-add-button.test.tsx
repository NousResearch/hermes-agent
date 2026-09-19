import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { SidebarSectionAddButton } from './chrome'

const startNewProjectDrag = vi.hoisted(() => vi.fn())
const startNewSessionDrag = vi.hoisted(() => vi.fn())

vi.mock('@/app/chat/new-session-drag', () => ({ startNewProjectDrag, startNewSessionDrag }))

vi.mock('@/i18n', () => ({
  useI18n: () => ({ t: { sidebar: { nav: { 'new-session': 'New session' } } } })
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

  it('drags the project variant when the + creates a project (project-overview mode)', () => {
    const onArm = vi.fn()
    const onPlainClick = vi.fn()
    const onNewSessionSplit = vi.fn()

    render(
      <SidebarSectionAddButton
        ariaLabel="New project"
        onNewProjectDrag={{ onArm }}
        onNewSessionSplit={onNewSessionSplit}
        onPlainClick={onPlainClick}
      />
    )

    fireEvent.pointerDown(screen.getByRole('button', { name: 'New project' }), { button: 0 })

    expect(startNewProjectDrag).toHaveBeenCalledOnce()
    // The arm callback is what the resolver calls with the dropped placement.
    expect(startNewProjectDrag.mock.calls[0]?.[0]).toBe(onArm)
    // The plain click rides along as the drag's tap action (same dialog).
    expect((startNewProjectDrag.mock.calls[0]?.[2] as { onTap?: () => void } | undefined)?.onTap).toBe(onPlainClick)
    expect(startNewSessionDrag).not.toHaveBeenCalled()

    // A committed drop forwards through the arm, not the session path.
    const onArmFromResolver = startNewProjectDrag.mock.calls[0]?.[0] as (placement: unknown) => void

    onArmFromResolver({ anchor: 'workspace', dir: 'center' })
    expect(onArm).toHaveBeenCalledWith({ anchor: 'workspace', dir: 'center' })
    expect(onNewSessionSplit).not.toHaveBeenCalled()
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
