import { getActiveComposer } from '@/app/chat/composer/focus'
import { RICH_INPUT_SLOT } from '@/app/chat/composer/rich-editor'
import { queryAllVisible } from '@/components/pane-shell/pane-visibility'

import { isEditableTarget } from './combo'
import { composerFocusBlockedBySurface, isActivateOnEnterTarget } from './composer-focus-keys'

let heldStack: HTMLElement | null = null

export function releaseApprovalKey(): void {
  heldStack = null
}

function activeApprovalStack(): HTMLElement | undefined {
  const target = getActiveComposer()

  return queryAllVisible<HTMLElement>('[data-approval-stack]').find(stack => {
    const surface = stack.closest<HTMLElement>('[data-composer-target]')

    return !surface || surface.dataset.composerTarget === target
  })
}

/** One dispatch per keydown, never one listener per card or per session. */
export function handleApprovalKey(event: KeyboardEvent): boolean {
  if (
    event.defaultPrevented ||
    event.isComposing ||
    event.altKey ||
    event.shiftKey ||
    (event.key !== 'Enter' && event.key !== 'Escape') ||
    composerFocusBlockedBySurface()
  ) {
    releaseApprovalKey()
    return false
  }

  const stack = activeApprovalStack()
  const target = event.target instanceof HTMLElement ? event.target : null
  const runTarget = target?.closest<HTMLElement>('[data-approval-run]')
  const emptyComposer = target?.closest(`[data-slot="${RICH_INPUT_SLOT}"]`) && !target.textContent?.trim()

  // A held key must have begun on THIS stack, not in a composer, terminal,
  // dialog, or another session that happened to disappear while it was held.
  if (event.repeat) {
    if (!heldStack) {
      return false
    }
    event.preventDefault()
    event.stopPropagation()
    if (stack !== heldStack || event.key !== 'Enter') {
      return true
    }
  } else {
    if (!stack || (isEditableTarget(target) && !emptyComposer)) {
      return false
    }
    if (event.key === 'Enter' && isActivateOnEnterTarget(target) && !runTarget && !emptyComposer) {
      return false
    }
    if (runTarget && !stack.contains(runTarget)) {
      return false
    }
    heldStack = stack
    event.preventDefault()
    event.stopPropagation()
  }

  const selector = event.key === 'Enter' ? '[data-approval-run]' : '[data-approval-deny]'
  const button = event.key === 'Enter' && !event.repeat && runTarget
    ? runTarget as HTMLButtonElement
    : stack?.querySelector<HTMLButtonElement>(selector)

  // While the first reply is in flight, repeat does nothing. Once the exact
  // card is removed, the next keydown sees the next card. No implicit allow-all.
  if (button && !button.disabled) {
    button.click()
  }
  return true
}
