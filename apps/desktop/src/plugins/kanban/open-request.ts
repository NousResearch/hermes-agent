/**
 * One-shot "open this card's drawer" request, so anything outside the board
 * page (a toast action, another plugin) can land the user ON a card rather
 * than just on /kanban. The caller parks the task id here and navigates; the
 * board page consumes it on arrival (mounted or mounting) and clears it — the
 * same pattern as `$newTaskLane`. Ephemeral: never persisted.
 *
 * Lives in its own module so completion-notify (imported by api.ts) can use
 * it without an import cycle.
 */

import { atom, host } from '@hermes/plugin-sdk'

export const $openTaskRequest = atom<null | string>(null)

/** Park `taskId` for the board page and navigate there. */
export function openTaskOnBoard(taskId: string): void {
  const id = taskId.trim()

  if (id) {
    $openTaskRequest.set(id)
  }

  host.navigate('/kanban')
}
