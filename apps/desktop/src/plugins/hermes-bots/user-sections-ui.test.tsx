/**
 * `SectionNameDialog`'s Enter-to-submit handling must go through the shared
 * IME-aware predicate (`isSubmitEnter`), not a hand-rolled `isComposing`-only
 * check — a bare check misses the legacy `keyCode === 229` commit-Enter some
 * IMEs (macOS Chinese) still fire after `compositionend`, letting a CJK user's
 * in-flight composition submit/rename with truncated text. See group-chat-parts'
 * identical guard (#93528) for the same class in a sibling hermes-bots surface.
 */

import type * as HermesSdk from '@hermes/plugin-sdk'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { translateBots } from './i18n-test-helper'

// Radix (Dialog) calls these on open; jsdom doesn't implement them.
beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()

  return {
    ...sdk,
    // The plugin bundle normally lands via `ctx.i18n.register` at load, so
    // without this every localized label renders empty.
    usePluginI18n: () => translateBots
  }
})

const { ReorderableRosterRow } = await import('./roster-order-row')
const { $draggingBot, BOT_DRAG_MIME, groupDragKey } = await import('./user-sections')
const { SectionNameDialog, UserSectionHeader } = await import('./user-sections-ui')

afterEach(() => {
  cleanup()
  $draggingBot.set(null)
})

describe('SectionNameDialog Enter handling', () => {
  function mount() {
    const onSubmit = vi.fn()
    render(<SectionNameDialog initialName="" mode="create" onOpenChange={() => undefined} onSubmit={onSubmit} open />)

    return { input: screen.getByRole('textbox'), onSubmit }
  }

  it('submits on a real Enter', () => {
    const { input, onSubmit } = mount()

    fireEvent.change(input, { target: { value: 'Work' } })
    fireEvent.keyDown(input, { key: 'Enter' })

    expect(onSubmit).toHaveBeenCalledWith('Work')
  })

  it('does not submit on an in-flight IME composition Enter', () => {
    const { input, onSubmit } = mount()

    fireEvent.change(input, { target: { value: '中文' } })
    fireEvent.keyDown(input, { isComposing: true, key: 'Enter' })

    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('does not submit on the legacy IME commit Enter (keyCode 229)', () => {
    const { input, onSubmit } = mount()

    fireEvent.change(input, { target: { value: '中文' } })
    fireEvent.keyDown(input, { key: 'Enter', keyCode: 229 })

    expect(onSubmit).not.toHaveBeenCalled()
  })
})

describe('UserSectionHeader custom order reset', () => {
  it('omits actions menu on uncustomized Unassigned and renders Reset custom order when customized', () => {
    const onResetOrder = vi.fn()

    const { rerender } = render(
      <UserSectionHeader
        canMoveDown={false}
        canMoveUp={false}
        collapsed={false}
        count={2}
        hasCustomOrder={false}
        id={null}
        name="Unassigned"
        onDelete={() => undefined}
        onMove={() => undefined}
        onRename={() => undefined}
        onToggle={() => undefined}
      />
    )

    expect(screen.queryByRole('button', { name: /section options/i })).toBeNull()

    rerender(
      <UserSectionHeader
        canMoveDown={false}
        canMoveUp={false}
        collapsed={false}
        count={2}
        hasCustomOrder
        id={null}
        name="Unassigned"
        onDelete={() => undefined}
        onMove={() => undefined}
        onRename={() => undefined}
        onResetOrder={onResetOrder}
        onToggle={() => undefined}
      />
    )

    const trigger = screen.getByRole('button', { name: /unassigned section options/i })
    fireEvent.pointerDown(trigger, { button: 0, ctrlKey: false })
    fireEvent.click(trigger)

    const resetItem = screen.getByRole('menuitem', { name: 'Reset custom order' })
    expect(screen.queryByRole('menuitem', { name: /delete section/i })).toBeNull()

    fireEvent.click(resetItem)
    expect(onResetOrder).toHaveBeenCalledTimes(1)
  })

  it('exposes Reset custom order alongside section actions in both DropdownMenu and ContextMenu on named sections', () => {
    const onResetOrder = vi.fn()
    const onDelete = vi.fn()

    const { container } = render(
      <UserSectionHeader
        canMoveDown
        canMoveUp={false}
        collapsed={false}
        count={3}
        hasCustomOrder
        id="sec-clients"
        name="Clients"
        onDelete={onDelete}
        onMove={() => undefined}
        onRename={() => undefined}
        onResetOrder={onResetOrder}
        onToggle={() => undefined}
      />
    )

    // Right-click ContextMenu parity on the section heading.
    const heading = container.querySelector('[data-slot="bots-section-heading"]')!
    fireEvent.contextMenu(heading)

    expect(screen.getByRole('menuitem', { name: /rename/i })).toBeTruthy()
    expect(screen.getByRole('menuitem', { name: /delete/i })).toBeTruthy()

    const contextReset = screen.getByRole('menuitem', { name: 'Reset custom order' })
    fireEvent.click(contextReset)
    expect(onResetOrder).toHaveBeenCalledTimes(1)
  })
})

describe('ReorderableRosterRow drag reorder', () => {
  function dispatchDragEvent(target: Element, type: 'dragover' | 'drop', payload: string, clientY: number) {
    const event = new MouseEvent(type, { bubbles: true, cancelable: true, clientY })

    Object.defineProperty(event, 'dataTransfer', {
      value: {
        dropEffect: 'none',
        getData: (mime: string) => (mime === BOT_DRAG_MIME ? payload : ''),
        types: [BOT_DRAG_MIME]
      }
    })

    act(() => {
      target.dispatchEvent(event)
    })
  }

  it('reorders bots in the same section and pinned band and clears $draggingBot on drop', () => {
    const onReorder = vi.fn()

    const { container } = render(
      <ReorderableRosterRow
        draggingPinned={false}
        draggingScope="local::sec-a"
        itemKey="local::beta"
        onReorder={onReorder}
        pinned={false}
        scopeKey="local::sec-a"
      >
        <div>Beta</div>
      </ReorderableRosterRow>
    )

    const wrapper = container.querySelector('[data-roster-item-key="local::beta"]')!
    vi.spyOn(wrapper, 'getBoundingClientRect').mockReturnValue({
      bottom: 40,
      height: 40,
      left: 0,
      right: 200,
      toJSON: () => ({}),
      top: 0,
      width: 200,
      x: 0,
      y: 0
    })

    act(() => {
      $draggingBot.set('local::alpha')
    })

    dispatchDragEvent(wrapper, 'dragover', 'local::alpha', 8)
    dispatchDragEvent(wrapper, 'drop', 'local::alpha', 8)

    expect(onReorder).toHaveBeenCalledWith('local::alpha', 'local::beta', 'before')
    expect($draggingBot.get()).toBeNull()

    act(() => {
      $draggingBot.set('local::gamma')
    })

    dispatchDragEvent(wrapper, 'dragover', 'local::gamma', 34)
    dispatchDragEvent(wrapper, 'drop', 'local::gamma', 34)

    expect(onReorder).toHaveBeenLastCalledWith('local::gamma', 'local::beta', 'after')
  })

  it('ignores group-chat drags and cross-band drags so section drops can proceed', () => {
    const onReorder = vi.fn()

    const { container, rerender } = render(
      <ReorderableRosterRow
        draggingPinned={false}
        draggingScope="local::sec-a"
        itemKey="local::beta"
        onReorder={onReorder}
        pinned={false}
        scopeKey="local::sec-a"
      >
        <div>Beta</div>
      </ReorderableRosterRow>
    )

    const wrapper = container.querySelector('[data-roster-item-key="local::beta"]')!

    // Group-chat drag (group:Crew) must not be captured by ReorderableRosterRow.
    const groupPayload = groupDragKey('Crew')
    act(() => {
      $draggingBot.set(groupPayload)
    })
    dispatchDragEvent(wrapper, 'drop', groupPayload, 5)
    expect(onReorder).not.toHaveBeenCalled()

    // Pinned bot dragged onto an unpinned row must not reorder across bands.
    rerender(
      <ReorderableRosterRow
        draggingPinned
        draggingScope="local::sec-a"
        itemKey="local::beta"
        onReorder={onReorder}
        pinned={false}
        scopeKey="local::sec-a"
      >
        <div>Beta</div>
      </ReorderableRosterRow>
    )
    act(() => {
      $draggingBot.set('local::pinned-bot')
    })
    dispatchDragEvent(wrapper, 'drop', 'local::pinned-bot', 5)
    expect(onReorder).not.toHaveBeenCalled()
  })
})
