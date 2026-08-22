import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { DialogPortalContainerContext } from '@/components/ui/dialog-portal-context'
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
  document.querySelectorAll('[data-dialog-portal-container]').forEach(node => node.remove())
  vi.clearAllMocks()
})

// Render the submenu inside an open menu/sub so its content (switches) mounts.
// `container` stands in for a Dialog's content node (what dialog.tsx publishes
// through DialogPortalContainerContext); null means "not inside a dialog".
function renderSubmenu(opts: {
  container?: HTMLElement | null
  defaultEffort?: string
  effort?: string
  fastControl: FastControl
  isActive?: boolean
  onSelectModel?: (model: string) => void
  onSetOptions: (patch: { effort?: string; fast?: boolean }) => void
  reasoning: boolean
}) {
  return render(
    <DialogPortalContainerContext.Provider value={opts.container ?? null}>
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
            />
          </DropdownMenuSub>
        </DropdownMenuContent>
      </DropdownMenu>
    </DialogPortalContainerContext.Provider>
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

// #80798: the thinking/effort panel used to portal to document.body even
// inside a Dialog, landing BEHIND the dialog + parent menu (separate stacking
// siblings) — dimmed and unclickable. The submenu must share the dialog's DOM
// subtree when one encloses it, and keep the bare body portal otherwise (that
// escape is what keeps submenus unclipped by scrollable hosts outside dialogs).
describe('ModelEditSubmenu lands in the right stacking world', () => {
  const SUB_SELECTOR = '[data-slot="dropdown-menu-sub-content"]'

  const baseProps = { fastControl: { kind: 'param', on: true } as FastControl, onSetOptions: vi.fn(), reasoning: true }

  it('inside a dialog: mounts within the dialog container, not as a body-level sibling (#80798)', () => {
    // Stand-in for the DialogContent shell node that dialog.tsx publishes.
    const dialogNode = document.createElement('div')
    dialogNode.setAttribute('data-dialog-portal-container', '')
    document.body.appendChild(dialogNode)

    renderSubmenu({ ...baseProps, container: dialogNode })

    const sub = document.querySelector(SUB_SELECTOR)
    expect(sub).not.toBeNull()
    // A DOM descendant of the dialog: it inherits the dialog's stacking
    // context, so it paints and receives events above the parent menu.
    expect(dialogNode.contains(sub)).toBe(true)
  })

  it('outside a dialog: keeps the document.body portal so overflow-clip escapes still work', () => {
    renderSubmenu(baseProps)

    const sub = document.querySelector(SUB_SELECTOR)
    expect(sub).not.toBeNull()

    // No container was provided, so the portal chain must terminate at
    // document.body exactly as the bare Radix Portal did before.
    let node: Node | null = sub
    while (node && node !== document.body) {
      node = node.parentNode
    }

    expect(node).toBe(document.body)
  })
})
