import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuSub,
  DropdownMenuSubTrigger
} from '@/components/ui/dropdown-menu'

import { type FastControl, ModelEditSubmenu } from './model-edit-submenu'

// Radix calls these on open; jsdom doesn't implement them.
beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

// Render the submenu inside an open menu/sub so its content (switches) mounts.
function renderSubmenu(opts: {
  defaultEffort?: string
  effort?: string
  fastControl: FastControl
  isActive?: boolean
  isPinned?: boolean
  onSelectModel?: (model: string) => void
  onSetOptions: (patch: { effort?: string; fast?: boolean }) => void
  onTogglePin?: () => void
  reasoning: boolean
}) {
  return render(
    <DropdownMenu open>
      <DropdownMenuContent>
        <DropdownMenuSub open>
          <DropdownMenuSubTrigger>edit</DropdownMenuSubTrigger>
          <ModelEditSubmenu
            defaultEffort={opts.defaultEffort ?? 'medium'}
            effort={opts.effort ?? 'medium'}
            fastControl={opts.fastControl}
            isActive={opts.isActive ?? true}
            isPinned={opts.isPinned}
            model="m1"
            onSelectModel={opts.onSelectModel ?? vi.fn()}
            onSetOptions={opts.onSetOptions}
            onTogglePin={opts.onTogglePin}
            provider="p1"
            reasoning={opts.reasoning}
          />
        </DropdownMenuSub>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

// The submenu is PURE: it reports edits and never writes to a session, a
// preset store, or the gateway. That's the invariant that lets the same
// component drive a live chat session AND a detached per-task override — if it
// ever writes directly again, picking an effort for a kanban card would reach
// over and change the user's live chat.
describe('ModelEditSubmenu reports edits without performing them', () => {
  it('param fast: reports the toggle', () => {
    const onSetOptions = vi.fn()
    renderSubmenu({ fastControl: { kind: 'param', on: true }, onSetOptions, reasoning: false })

    fireEvent.click(screen.getByRole('switch'))

    expect(onSetOptions).toHaveBeenCalledWith({ fast: false })
  })

  it('thinking: toggling off reports the none level', () => {
    const onSetOptions = vi.fn()
    renderSubmenu({ fastControl: { kind: 'none' }, onSetOptions, reasoning: true })

    // Thinking starts on (medium); toggling it off reports 'none'.
    fireEvent.click(screen.getByRole('switch'))

    expect(onSetOptions).toHaveBeenCalledWith({ effort: 'none' })
  })

  it('thinking: toggling back on restores the row level, not the hardcoded default', () => {
    const onSetOptions = vi.fn()
    renderSubmenu({
      defaultEffort: 'high',
      effort: 'none',
      fastControl: { kind: 'none' },
      onSetOptions,
      reasoning: true
    })

    fireEvent.click(screen.getByRole('switch'))

    expect(onSetOptions).toHaveBeenCalledWith({ effort: 'high' })
  })

  it('variant fast: swaps the model only when the row is active', () => {
    const onSelectModel = vi.fn()
    const onSetOptions = vi.fn()

    renderSubmenu({
      fastControl: { baseId: 'm1', fastId: 'm1-fast', kind: 'variant', on: false },
      isActive: false,
      onSelectModel,
      onSetOptions,
      reasoning: false
    })

    fireEvent.click(screen.getByRole('switch'))

    // Inactive rows stay preference-only — no model switch.
    expect(onSetOptions).toHaveBeenCalledWith({ fast: true })
    expect(onSelectModel).not.toHaveBeenCalled()
  })

  it('variant fast: active row swaps to the -fast sibling', () => {
    const onSelectModel = vi.fn()
    const onSetOptions = vi.fn()

    renderSubmenu({
      fastControl: { baseId: 'm1', fastId: 'm1-fast', kind: 'variant', on: false },
      onSelectModel,
      onSetOptions,
      reasoning: false
    })

    fireEvent.click(screen.getByRole('switch'))

    expect(onSelectModel).toHaveBeenCalledWith('m1-fast')
  })
})

// Pinning is not a capability: lifting a model to the top of the picker has
// nothing to do with what the model can be asked for, so the row reports it
// even when the submenu has nothing else to offer.
describe('ModelEditSubmenu reports pins without performing them', () => {
  it('offers the pin even on a model with no reasoning or fast options', () => {
    const onTogglePin = vi.fn()
    renderSubmenu({ fastControl: { kind: 'none' }, onSetOptions: vi.fn(), onTogglePin, reasoning: false })

    expect(screen.getByText('No options for this model')).toBeTruthy()

    fireEvent.click(screen.getByText('Pin'))

    expect(onTogglePin).toHaveBeenCalled()
  })

  it('names the pin action by state, so unpinning is never a blind click', () => {
    renderSubmenu({
      fastControl: { kind: 'none' },
      isPinned: true,
      onSetOptions: vi.fn(),
      onTogglePin: vi.fn(),
      reasoning: false
    })

    expect(screen.getByText('Unpin')).toBeTruthy()
    expect(screen.queryByText('Pin')).toBeNull()
  })

  it('renders no pin row when the surface passes no toggle', () => {
    renderSubmenu({ fastControl: { kind: 'none' }, onSetOptions: vi.fn(), reasoning: false })

    expect(screen.queryByText('Pin')).toBeNull()
    expect(screen.queryByText('Unpin')).toBeNull()
  })
})
