import { RICH_INPUT_SLOT } from '@/app/chat/composer/rich-editor'

/**
 * Run one undo/redo step for whatever holds focus.
 *
 * The rich composer owns a coalesced stack claimed via `beforeinput`
 * `historyUndo` / `historyRedo`. Ordinary `<input>` / `<textarea>` keep
 * Chromium's native burst undo through `execCommand`.
 */
export function performEditHistory(action: 'redo' | 'undo'): boolean {
  const active = document.activeElement

  if (active instanceof HTMLElement && active.dataset.slot === RICH_INPUT_SLOT) {
    const inputType = action === 'undo' ? 'historyUndo' : 'historyRedo'
    const event = new InputEvent('beforeinput', { bubbles: true, cancelable: true, inputType })
    active.dispatchEvent(event)

    return event.defaultPrevented
  }

  return document.execCommand(action)
}
