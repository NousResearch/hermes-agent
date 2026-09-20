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
  canDisableReasoning?: boolean
  reasoningBudget?: { min: number; max: number; dynamic?: boolean }
  defaultEffort?: string
  effort?: string
  fastControl: FastControl
  isActive?: boolean
  modelDefaultEffort?: string
  onSelectModel?: (model: string) => void
  onSetOptions: (patch: { effort?: string; fast?: boolean }) => void
  reasoning: boolean
  reasoningControl?: 'adjustable' | 'default' | 'unsupported' | 'unknown'
  reasoningEfforts?: string[]
}) {
  return render(
    <DropdownMenu open>
      <DropdownMenuContent>
        <DropdownMenuSub open>
          <DropdownMenuSubTrigger>edit</DropdownMenuSubTrigger>
          <ModelEditSubmenu
            canDisableReasoning={opts.canDisableReasoning}
            defaultEffort={opts.defaultEffort ?? 'medium'}
            effort={opts.effort ?? 'medium'}
            fastControl={opts.fastControl}
            isActive={opts.isActive ?? true}
            model="m1"
            modelDefaultEffort={opts.modelDefaultEffort}
            onSelectModel={opts.onSelectModel ?? vi.fn()}
            onSetOptions={opts.onSetOptions}
            provider="p1"
            reasoning={opts.reasoning}
            reasoningControl={opts.reasoningControl}
            reasoningEfforts={opts.reasoningEfforts}
            reasoningBudget={opts.reasoningBudget}
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
  it('validates a token budget separately from effort and reports only Apply', () => {
    const onSetOptions = vi.fn()
    const onSelectModel = vi.fn()
    renderSubmenu({
      reasoning: true,
      reasoningControl: 'adjustable',
      reasoningEfforts: ['none'],
      reasoningBudget: { min: 512, max: 24576, dynamic: true },
      effort: 'auto',
      fastControl: { kind: 'none' },
      onSetOptions,
      onSelectModel
    })
    const input = screen.getByRole('spinbutton', { name: 'Thinking budget (tokens)' })
    fireEvent.change(input, { target: { value: '511' } })
    expect(screen.getByRole('button', { name: 'Apply' }).hasAttribute('disabled')).toBe(true)
    fireEvent.change(input, { target: { value: '4096' } })
    expect(onSetOptions).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: 'Apply' }))
    expect(onSetOptions).toHaveBeenCalledWith({ effort: 'budget:4096' })
    expect(onSelectModel).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('menuitemradio', { name: 'Dynamic' }))
    expect(onSetOptions).toHaveBeenLastCalledWith({ effort: 'budget:-1' })
    expect(screen.queryByRole('switch')).toBeNull() // default is not falsely displayed as On
  })
  it.each([
    ['default', true, [], true],
    ['adjustable', true, ['low', 'high'], true],
    ['unsupported', false, [], false]
  ] as const)('shows the locked thinking state for %s', (reasoningControl, reasoning, levels, checked) => {
    const onSetOptions = vi.fn()
    renderSubmenu({
      reasoningControl,
      reasoning,
      reasoningEfforts: [...levels],
      canDisableReasoning: false,
      effort: 'auto',
      fastControl: { kind: 'none' },
      onSetOptions
    })
    const toggle = screen.getByRole('switch')
    expect(toggle.hasAttribute('disabled')).toBe(true)
    expect(toggle.getAttribute('aria-checked')).toBe(String(checked))
    fireEvent.click(toggle)
    expect(onSetOptions).not.toHaveBeenCalled()
  })

  it('does not present an unknown thinking state as Off', () => {
    renderSubmenu({
      reasoningControl: 'unknown',
      reasoning: false,
      reasoningEfforts: [],
      canDisableReasoning: false,
      fastControl: { kind: 'none' },
      onSetOptions: vi.fn()
    })
    expect(screen.queryByRole('switch')).toBeNull()
    expect(screen.getByText('Thinking controls unverified')).toBeTruthy()
  })

  it('offers only declared effort levels and resolves its checked value against them', () => {
    const onSetOptions = vi.fn()
    renderSubmenu({
      reasoning: true,
      reasoningEfforts: ['low', 'high'],
      effort: 'ultra',
      defaultEffort: 'medium',
      fastControl: { kind: 'none' },
      onSetOptions
    })
    expect(screen.getAllByRole('menuitemradio').map(row => row.textContent)).toEqual(['Low', 'High'])
    expect(screen.getAllByRole('menuitemradio').every(row => row.getAttribute('aria-checked') === 'false')).toBe(true)
    expect(screen.queryByRole('switch')).toBeNull()
    fireEvent.click(screen.getByRole('menuitemradio', { name: 'High' }))
    expect(onSetOptions).toHaveBeenCalledWith({ effort: 'high' })
  })

  it('does not offer a synthetic effort or thinking toggle when the declared list is empty', () => {
    renderSubmenu({ reasoning: true, reasoningEfforts: [], fastControl: { kind: 'none' }, onSetOptions: vi.fn() })
    expect(screen.queryAllByRole('menuitemradio')).toEqual([])
    expect(screen.queryByText('Provider default')).toBeNull()
    expect(screen.queryByRole('switch')).toBeNull()
  })
  it('param fast: reports the toggle', () => {
    const onSetOptions = vi.fn()
    renderSubmenu({ fastControl: { kind: 'param', on: true }, onSetOptions, reasoning: false })

    fireEvent.click(screen.getByRole('switch', { name: 'Fast' }))

    expect(onSetOptions).toHaveBeenCalledWith({ fast: false })
  })

  it('thinking: toggling off reports the none level', () => {
    const onSetOptions = vi.fn()
    renderSubmenu({ fastControl: { kind: 'none' }, onSetOptions, reasoning: true })

    // Thinking starts on (medium); toggling it off reports 'none'.
    fireEvent.click(screen.getByRole('switch', { name: 'Thinking' }))

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

    fireEvent.click(screen.getByRole('switch', { name: 'Thinking' }))

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

    fireEvent.click(screen.getByRole('switch', { name: 'Fast' }))

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

    fireEvent.click(screen.getByRole('switch', { name: 'Fast' }))

    expect(onSelectModel).toHaveBeenCalledWith('m1-fast')
  })

  it('offers only the provider-declared effort levels and locks mandatory thinking', () => {
    renderSubmenu({
      canDisableReasoning: false,
      effort: 'low',
      fastControl: { kind: 'none' },
      modelDefaultEffort: 'low',
      onSetOptions: vi.fn(),
      reasoning: true,
      reasoningControl: 'adjustable',
      reasoningEfforts: ['low', 'high']
    })

    const thinking = screen.getByRole('switch', { name: 'Thinking' })
    const fast = screen.getByRole('switch', { name: 'Fast' })

    expect(thinking.getAttribute('aria-checked')).toBe('true')
    expect(thinking.hasAttribute('disabled')).toBe(true)
    expect(fast.hasAttribute('disabled')).toBe(true)
    expect(screen.getAllByRole('menuitemradio').map(item => item.textContent)).toEqual(['Low', 'High'])
    expect(screen.queryByRole('menuitemradio', { name: 'Medium' })).toBeNull()
  })

  it('shows fixed reasoning as on and unsupported reasoning as off, both disabled', () => {
    const fixed = renderSubmenu({
      canDisableReasoning: false,
      effort: 'auto',
      fastControl: { kind: 'none' },
      onSetOptions: vi.fn(),
      reasoning: true,
      reasoningControl: 'default',
      reasoningEfforts: []
    })

    expect(screen.getByRole('switch', { name: 'Thinking' }).getAttribute('aria-checked')).toBe('true')
    expect(screen.getByRole('switch', { name: 'Thinking' }).hasAttribute('disabled')).toBe(true)
    expect(screen.queryByRole('menuitemradio')).toBeNull()
    fixed.unmount()

    renderSubmenu({
      canDisableReasoning: false,
      fastControl: { kind: 'none' },
      onSetOptions: vi.fn(),
      reasoning: false,
      reasoningControl: 'unsupported',
      reasoningEfforts: []
    })

    expect(screen.getByRole('switch', { name: 'Thinking' }).getAttribute('aria-checked')).toBe('false')
    expect(screen.getByRole('switch', { name: 'Thinking' }).hasAttribute('disabled')).toBe(true)
  })
})

it.each([['low', 'high'], ['low', 'medium', 'high'], []])(
  'declared choices produce only their corresponding edits (%j)',
  (...levels: string[]) => {
    const onSetOptions = vi.fn()
    renderSubmenu({
      reasoningEfforts: levels,
      canDisableReasoning: false,
      effort: 'auto',
      fastControl: { kind: 'none' },
      onSetOptions,
      reasoning: true
    })
    expect(screen.queryByRole('switch')).toBeNull()
    const items = screen.queryAllByRole('menuitemradio')
    expect(items).toHaveLength(levels.length)
    expect(items.every(item => item.getAttribute('aria-checked') === 'false')).toBe(true)
    expect(screen.queryByText('Provider default')).toBeNull()

    for (const [index, value] of levels.entries()) {
      fireEvent.click(items[index])
      expect(onSetOptions).toHaveBeenLastCalledWith({ effort: value })
    }
  }
)

it('does not check a different level for an unsupported saved setting', () => {
  renderSubmenu({
    reasoningEfforts: ['low', 'high'],
    canDisableReasoning: false,
    effort: 'medium',
    defaultEffort: 'high',
    fastControl: { kind: 'none' },
    onSetOptions: vi.fn(),
    reasoning: true
  })

  for (const item of screen.getAllByRole('menuitemradio')) {
    expect(item.getAttribute('aria-checked')).toBe('false')
  }
})
