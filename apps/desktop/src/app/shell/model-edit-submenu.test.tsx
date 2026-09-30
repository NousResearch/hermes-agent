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
  onSelectModel?: (model: string) => void
  onSetOptions: (patch: { effort?: string; fast?: boolean }) => void
  reasoning: boolean
  supportedEfforts?: string[]
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
            model="m1"
            onSelectModel={opts.onSelectModel ?? vi.fn()}
            onSetOptions={opts.onSetOptions}
            provider="p1"
            reasoning={opts.reasoning}
            supportedEfforts={opts.supportedEfforts}
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

describe('ModelEditSubmenu filters the effort ladder to the route vocabulary', () => {
  // i18n labels for the ladder levels (en) — the radio items carry them.
  const levelItem = (label: string) => screen.queryByRole('menuitemradio', { name: label })

  it('offers only the levels the route accepts', () => {
    renderSubmenu({
      effort: 'high',
      fastControl: { kind: 'none' },
      onSetOptions: vi.fn(),
      reasoning: true,
      supportedEfforts: ['none', 'low', 'high', 'max']
    })

    expect(levelItem('Low')).not.toBeNull()
    expect(levelItem('High')).not.toBeNull()
    expect(levelItem('Max')).not.toBeNull()
    // Every level the route does not accept stays hidden (#114029).
    expect(levelItem('Minimal')).toBeNull()
    expect(levelItem('Medium')).toBeNull()
    expect(levelItem('Extra High')).toBeNull()
    expect(levelItem('Ultra')).toBeNull()
  })

  it('offers the full ladder when the route vocabulary is unknown', () => {
    renderSubmenu({
      effort: 'high',
      fastControl: { kind: 'none' },
      onSetOptions: vi.fn(),
      reasoning: true
    })

    for (const label of ['Minimal', 'Low', 'Medium', 'High', 'Extra High', 'Max', 'Ultra']) {
      expect(levelItem(label)).not.toBeNull()
    }
  })

  it('keeps a clamped current pick listed so its radio stays selected', () => {
    renderSubmenu({
      effort: 'ultra',
      fastControl: { kind: 'none' },
      onSetOptions: vi.fn(),
      reasoning: true,
      supportedEfforts: ['none', 'low', 'medium', 'high', 'xhigh', 'max']
    })

    // `ultra` is a Hermes-internal step every route clamps (#61634); hiding the
    // saved pick would deselect the radio showing it. It stays listed.
    const ultra = levelItem('Ultra')
    expect(ultra).not.toBeNull()
    expect(ultra?.getAttribute('aria-checked')).toBe('true')
  })
})
