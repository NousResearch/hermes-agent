import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuSub,
  DropdownMenuSubTrigger
} from '@/components/ui/dropdown-menu'
import { $modelPresets, getModelPreset } from '@/store/model-presets'
import { $activeSessionId, $currentReasoningEffort } from '@/store/session'

import { type FastControl, ModelEditSubmenu } from './model-edit-submenu'

// Radix calls these on open; jsdom doesn't implement them.
beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

beforeEach(() => {
  window.localStorage.clear()
  $modelPresets.set({})
  $activeSessionId.set(null)
  $currentReasoningEffort.set('medium')
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

// Render the submenu inside an open menu/sub so its content (switches) mounts.
function renderSubmenu(opts: {
  effort?: string
  fastControl?: FastControl
  isActive?: boolean
  model?: string
  provider?: string
  reasoning?: boolean
  requestGateway?: () => Promise<unknown>
}) {
  return render(
    <DropdownMenu open>
      <DropdownMenuContent>
        <DropdownMenuSub open>
          <DropdownMenuSubTrigger>edit</DropdownMenuSubTrigger>
          <ModelEditSubmenu
            effort={opts.effort ?? 'medium'}
            fastControl={opts.fastControl ?? { kind: 'none' }}
            isActive={opts.isActive ?? true}
            model={opts.model ?? 'm1'}
            onSelectModel={vi.fn()}
            provider={opts.provider ?? 'p1'}
            reasoning={opts.reasoning ?? true}
            requestGateway={(opts.requestGateway ?? vi.fn().mockResolvedValue({})) as never}
          />
        </DropdownMenuSub>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

const effortRadio = (name: string) => screen.getByRole('menuitemradio', { name })

// Regression: editing the active row before a live session exists must stay
// preset-only — the gateway's config.set falls back to global config when no
// session matches, so it must not be called. (Caught in the second review.)
describe('ModelEditSubmenu no-session guard', () => {
  it('param fast: records the preset but skips the gateway without a session', () => {
    const requestGateway = vi.fn().mockResolvedValue({})
    renderSubmenu({ fastControl: { kind: 'param', on: false }, reasoning: false, requestGateway })

    fireEvent.click(screen.getByRole('switch'))

    expect(getModelPreset('p1', 'm1').fast).toBe(true)
    expect(requestGateway).not.toHaveBeenCalled()
  })

  it('reasoning: records the preset but skips the gateway without a session', () => {
    const requestGateway = vi.fn().mockResolvedValue({})
    renderSubmenu({ fastControl: { kind: 'none' }, reasoning: true, requestGateway })

    // Thinking starts on (medium); toggling it off routes through patchReasoning.
    fireEvent.click(screen.getByRole('switch'))

    expect(getModelPreset('p1', 'm1').effort).toBe('none')
    expect(requestGateway).not.toHaveBeenCalled()
  })

  it('param fast: pushes to the gateway once a session is active', async () => {
    const requestGateway = vi.fn().mockResolvedValue({})
    $activeSessionId.set('sess1')
    renderSubmenu({ fastControl: { kind: 'param', on: false }, reasoning: false, requestGateway })

    fireEvent.click(screen.getByRole('switch'))

    expect(requestGateway).toHaveBeenCalledWith('config.set', { key: 'fast', session_id: 'sess1', value: 'fast' })
  })
})

describe('ModelEditSubmenu reasoning effort parity', () => {
  it('offers distinct enabled effort rows in Hermes order', () => {
    renderSubmenu({ effort: 'medium' })

    expect(screen.getAllByRole('menuitemradio').map((item: HTMLElement) => item.textContent)).toEqual([
      'Minimal',
      'Low',
      'Medium',
      'High',
      'Extra High',
      'Max'
    ])
  })

  it('selects Max when the active effort is max, not Medium', () => {
    renderSubmenu({ effort: 'max' })

    expect(effortRadio('Max').getAttribute('data-state')).toBe('checked')
    expect(effortRadio('Medium').getAttribute('data-state')).toBe('unchecked')
  })

  it('selects Extra High when the active effort is xhigh', () => {
    renderSubmenu({ effort: 'xhigh' })

    expect(effortRadio('Extra High').getAttribute('data-state')).toBe('checked')
  })

  it('keeps Thinking off and no effort radio selected for none', () => {
    renderSubmenu({ effort: 'none' })

    expect(screen.getByRole('switch').getAttribute('data-state')).toBe('unchecked')
    expect(
      screen.getAllByRole('menuitemradio').every((item: HTMLElement) => item.getAttribute('data-state') === 'unchecked')
    ).toBe(true)
  })

  it('sends, stores, and persists xhigh when Extra High is selected in an active session', async () => {
    const requestGateway = vi.fn().mockResolvedValue({})
    $activeSessionId.set('sess1')
    renderSubmenu({ effort: 'medium', requestGateway })

    fireEvent.click(effortRadio('Extra High'))

    await waitFor(() =>
      expect(requestGateway).toHaveBeenCalledWith('config.set', { key: 'reasoning', session_id: 'sess1', value: 'xhigh' })
    )
    expect($currentReasoningEffort.get()).toBe('xhigh')
    expect(getModelPreset('p1', 'm1').effort).toBe('xhigh')
  })

  it('sends, stores, and persists max when Max is selected in an active session', async () => {
    const requestGateway = vi.fn().mockResolvedValue({})
    $activeSessionId.set('sess1')
    renderSubmenu({ effort: 'xhigh', requestGateway })

    fireEvent.click(effortRadio('Max'))

    await waitFor(() =>
      expect(requestGateway).toHaveBeenCalledWith('config.set', { key: 'reasoning', session_id: 'sess1', value: 'max' })
    )
    expect($currentReasoningEffort.get()).toBe('max')
    expect(getModelPreset('p1', 'm1').effort).toBe('max')
  })

  it('rolls back a rejected Max mutation to the exact prior value and preset', async () => {
    const requestGateway = vi.fn().mockRejectedValue(new Error('rejected'))
    $activeSessionId.set('sess1')
    $currentReasoningEffort.set('xhigh')
    renderSubmenu({ effort: 'xhigh', requestGateway })

    fireEvent.click(effortRadio('Max'))

    await waitFor(() => expect(requestGateway).toHaveBeenCalled())
    await waitFor(() => expect($currentReasoningEffort.get()).toBe('xhigh'))
    expect(getModelPreset('p1', 'm1').effort).toBe('xhigh')
  })
})
