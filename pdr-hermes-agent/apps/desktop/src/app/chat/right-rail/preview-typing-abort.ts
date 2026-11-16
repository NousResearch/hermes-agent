/**
 * PREVIEW TYPING ABORT — the keystroke loop outlives the tool that started it.
 *
 * `drive_preview` blocks on `preview.act` for 45s. When that wait times out, or
 * the turn is interrupted, the backend withdraws the request with
 * `request.cancel`. The renderer used to ignore that for typing and keep
 * sending the rest of the string into whatever still had focus. This map is the
 * only link: the act handler registers the in-flight controller, and cancel
 * aborts it so the loop stops at the next character.
 */

const controllers = new Map<string, AbortController>()

/** Arm a controller for one `preview.act` request. A second arm for the same id
 *  aborts the previous loop — a replayed request must not leave the first one
 *  typing. */
export function trackPreviewTyping(requestId: string): AbortSignal {
  controllers.get(requestId)?.abort('interrupted')

  const controller = new AbortController()

  controllers.set(requestId, controller)

  return controller.signal
}

/** Stop the loop for `requestId`, if one is still running. `reason` is the
 *  cancel's reason (`timeout` or `interrupted`); the loop reports it. */
export function abortPreviewTyping(requestId: string, reason = 'interrupted'): void {
  const controller = controllers.get(requestId)

  if (!controller || controller.signal.aborted) {
    return
  }

  controller.abort(reason)
}

/** Drop the controller once the action has settled, so a late cancel is a no-op
 *  rather than aborting a controller the next action might reuse. An action can
 *  be superseded by a same-id replay (`trackPreviewTyping` aborts the original
 *  and installs a replacement): release only the controller this action owns,
 *  identified by its signal — the original's `finally` must not delete the
 *  replay's entry, or a later cancel finds nothing and the replay keeps typing
 *  (#125408). */
export function releasePreviewTyping(requestId: string, owner?: AbortSignal): void {
  if (owner !== undefined && controllers.get(requestId)?.signal !== owner) {
    return
  }

  controllers.delete(requestId)
}
