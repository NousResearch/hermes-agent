import type { ComposerBusyAction } from './busy-submit-state'

/*
 * A reply paused for its automatic continuation (store/paused-replies.ts) holds a send in the chat's
 * queue until the continuation has completed, also while a restart has the backend away and the composer
 * is otherwise disabled (use-composer-submit.ts). These keep the composer's controls in step: Enter and
 * Send stay usable, the busy action reads Queue, and no Steer is offered (it would land in the
 * continuation).
 */

/** Enter does nothing: the composer is disabled and no send would be held for a paused reply. */
export const enterLocked = (disabled: boolean, replyPaused: boolean) => disabled && !replyPaused

export const sendButtonDisabled = (disabled: boolean, canSubmit: boolean, sendWhileDisabled?: boolean) =>
  (disabled && !sendWhileDisabled) || !canSubmit

export function pausedReplyControls(replyPaused: boolean, busyAction: ComposerBusyAction, onSteer?: () => void) {
  return replyPaused
    ? { busyAction: 'queue' as const, onSteer: undefined, sendWhileDisabled: true }
    : { busyAction, onSteer, sendWhileDisabled: false }
}
