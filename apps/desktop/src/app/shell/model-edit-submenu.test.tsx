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
  daybreak?: { checked: boolean; required: boolean; onChange: (enabled: boolean) => void }
  defaultEffort?: string
  effort?: string
  fastControl: FastControl
  isActive?: boolean
  serviceTier?: string
  ultrafastSupported?: boolean
  onSelectModel?: (model: string) => void
  onSetOptions: (patch: { effort?: string; fast?: boolean; serviceTier?: string }) => void
  reasoning: boolean
}) {
  return render(
    <DropdownMenu open>
      <DropdownMenuContent>
        <DropdownMenuSub open>
          <DropdownMenuSubTrigger>edit</DropdownMenuSubTrigger>
          <ModelEditSubmenu
            daybreak={opts.daybreak}
            defaultEffort={opts.defaultEffort ?? 'medium'}
            effort={opts.effort ?? 'medium'}
            fastControl={opts.fastControl}
            isActive={opts.isActive ?? true}
            model="m1"
            onSelectModel={opts.onSelectModel ?? vi.fn()}
            onSetOptions={opts.onSetOptions}
            provider="p1"
            reasoning={opts.reasoning}
            serviceTier={opts.serviceTier}
            ultrafastSupported={opts.ultrafastSupported}
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
  it('reports Ultrafast separately from Fast and can return to standard', () => {
    const onSetOptions = vi.fn()
    renderSubmenu({
      fastControl: { kind: 'param', on: true },
      serviceTier: 'ultrafast',
      ultrafastSupported: true,
      onSetOptions,
      reasoning: false
    })
    expect(screen.getByRole('switch', { name: 'Fast' }).getAttribute('aria-checked')).toBe('false')
    expect(screen.getByRole('switch', { name: 'Ultrafast' }).getAttribute('aria-checked')).toBe('true')
    fireEvent.click(screen.getByRole('switch', { name: 'Ultrafast' }))
    expect(onSetOptions).toHaveBeenCalledWith({ serviceTier: 'normal' })
  })

  it('offers only a standard reset for an unsupported carried-over speed', () => {
    const onSetOptions = vi.fn()
    renderSubmenu({
      fastControl: { kind: 'param', on: true, canEnable: false },
      serviceTier: 'priority',
      onSetOptions,
      reasoning: false
    })
    expect(screen.queryByRole('switch')).toBeNull()
    fireEvent.click(screen.getByText('Use standard speed'))
    expect(onSetOptions).toHaveBeenCalledWith({ serviceTier: 'normal' })
  })
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

it('offers Daybreak only on eligible model options and locks required aliases', () => {
  const change = vi.fn()

  const result = renderSubmenu({
    fastControl: { kind: 'none' },
    reasoning: false,
    onSetOptions: vi.fn(),
    daybreak: { checked: false, required: false, onChange: change }
  })

  fireEvent.click(screen.getByRole('switch', { name: 'Daybreak' }))
  expect(change).toHaveBeenCalledWith(true)
  result.unmount()
  renderSubmenu({
    fastControl: { kind: 'none' },
    reasoning: false,
    onSetOptions: vi.fn(),
    daybreak: { checked: true, required: true, onChange: change }
  })
  expect(screen.getByRole('switch', { name: 'Daybreak' }).hasAttribute('disabled')).toBe(true)
})

it('lets an inactive row edit Daybreak like its speed controls', () => {
  const change = vi.fn()
  renderSubmenu({
    fastControl: { kind: 'param', on: false },
    isActive: false,
    reasoning: false,
    onSetOptions: vi.fn(),
    daybreak: { checked: false, required: false, onChange: change }
  })
  const daybreak = screen.getByRole('switch', { name: 'Daybreak' })
  expect(daybreak.hasAttribute('disabled')).toBe(false)
  fireEvent.click(daybreak)
  expect(change).toHaveBeenCalledWith(true)
})
