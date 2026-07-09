import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuSub,
  DropdownMenuSubTrigger
} from '@/components/ui/dropdown-menu'
import { $modelPresets, getModelPreset } from '@/store/model-presets'
import { $activeSessionId } from '@/store/session'

import { type FastControl, ModelEditSubmenu } from './model-edit-submenu'

// Radix calls these on open; jsdom doesn't implement them.
beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

beforeEach(() => {
  $modelPresets.set({})
  $activeSessionId.set(null)
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

// Render the submenu inside an open menu/sub so its content (switches) mounts.
function renderSubmenu(opts: {
  effort?: string
  fastControl: FastControl
  reasoning: boolean
  requestGateway: () => Promise<unknown>
}) {
  return render(
    <DropdownMenu open>
      <DropdownMenuContent>
        <DropdownMenuSub open>
          <DropdownMenuSubTrigger>edit</DropdownMenuSubTrigger>
          <ModelEditSubmenu
            effort={opts.effort ?? 'medium'}
            fastControl={opts.fastControl}
            isActive
            model="m1"
            onSelectModel={vi.fn()}
            provider="p1"
            reasoning={opts.reasoning}
            requestGateway={opts.requestGateway as never}
          />
        </DropdownMenuSub>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

// Regression: editing the active row before a live session exists must stay
// preset-only — the gateway's config.set falls back to global config when no
// session matches, so it must not be called. (Caught in the second review.)
describe('ModelEditSubmenu no-session guard', () => {
  it('shows and preserves the extended reasoning levels', () => {
    renderSubmenu({
      effort: 'ultra',
      fastControl: { kind: 'none' },
      reasoning: true,
      requestGateway: vi.fn().mockResolvedValue({})
    })

    expect(screen.getByRole('menuitemradio', { name: 'XHigh' }).getAttribute('data-state')).toBe('unchecked')
    expect(screen.getByRole('menuitemradio', { name: 'Max' }).getAttribute('data-state')).toBe('unchecked')
    expect(screen.getByRole('menuitemradio', { name: 'Ultra' }).getAttribute('data-state')).toBe('checked')
  })

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
