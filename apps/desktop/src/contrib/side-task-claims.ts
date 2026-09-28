/**
 * Side-task claims: a plugin that renders a `/background` or `/btw` answer on
 * its own surface claims the task id, and the gateway-event dispatcher then
 * skips the transcript line it would otherwise append for that task.
 *
 * Plugin event listeners run BEFORE app dispatch (`emitGatewayEvent`), so a
 * claim made from the `prompt.background` / `prompt.btw` reply or from inside
 * the plugin's own `background.complete` / `btw.complete` listener is honored
 * for that same event. A claim is consumed on delivery, and a plugin that
 * unloads first takes its unconsumed claims with it — the answer falls back to
 * the transcript rather than vanishing.
 */

const claims = new Map<string, number>()

/** Claim a side task's answer. Returns a disposer that releases an unconsumed claim. */
export function claimSideTask(taskId: string): () => void {
  const id = taskId.trim()

  if (!id) {
    return () => undefined
  }

  claims.set(id, (claims.get(id) ?? 0) + 1)

  let released = false

  return () => {
    if (released) {
      return
    }

    released = true
    const count = claims.get(id)

    if (count === undefined) {
      return
    }

    if (count <= 1) {
      claims.delete(id)
    } else {
      claims.set(id, count - 1)
    }
  }
}

/** Dispatcher side: true (and consumed) when a plugin owns this task's answer. */
export function takeSideTaskClaim(taskId: string): boolean {
  const id = taskId.trim()

  if (!id || !claims.has(id)) {
    return false
  }

  claims.delete(id)

  return true
}
