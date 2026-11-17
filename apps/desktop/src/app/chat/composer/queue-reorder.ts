import type { GatewayRequest } from '@/app/session/hooks/use-prompt-actions/utils'
import { type QueuedPromptEntry, reorderQueuedPrompts } from '@/store/composer-queue'

import { moveQueuedPrompt } from './discard-lost-prompt'

/** A local row always moves; a server row only while queued, and only when its owner serves
 *  `prompt.move_queued` for it (an older owner lists no `move` control). */
export const canMoveQueuedEntry = (entry: Pick<QueuedPromptEntry, 'serverControls' | 'serverStatus'>): boolean =>
  !entry.serverStatus || (entry.serverStatus === 'queued' && Boolean(entry.serverControls?.includes('move')))

/** The id(s) a single drag or ▲/▼ may have moved between two orders of the same ids (an
 *  `arrayMove`): the changed span is a rotation, so the row that moved up into its first slot or
 *  the one that moved down out of it. An adjacent swap reads both ways, so both are returned and
 *  the caller takes the one that can move. Empty when nothing moved. */
export function movedQueueIds(previous: readonly string[], next: readonly string[]): string[] {
  const first = next.findIndex((id, index) => id !== previous[index])

  if (first < 0) {
    return []
  }

  let last = next.length - 1

  while (last > first && next[last] === previous[last]) {
    last -= 1
  }

  const up = next[first] === previous[last] ? [previous[last]!] : []
  const down = next[last] === previous[first] ? [previous[first]!] : []

  return [...up, ...down]
}

/** Where the owner should put a moved server row: ahead of the next still-queued server row in the
 *  new order, or at the back of its FIFO (null). Local rows and lost/started rows are not anchors. */
export function serverMoveAnchor(next: readonly QueuedPromptEntry[], movedId: string): null | string {
  const index = next.findIndex(entry => entry.id === movedId)

  return next.slice(index + 1).find(entry => entry.serverStatus === 'queued')?.id ?? null
}

interface ApplyQueueReorderArgs {
  entries: readonly QueuedPromptEntry[]
  nextIds: readonly string[]
  queueSessionKey: string
  request: GatewayRequest | null
  sessionId: null | string | undefined
  onError: (error: unknown) => void
}

/** Paint the new order at once; a server row's move then goes to its owner, whose pending fanout
 *  is the truth every viewer converges on. A refused move puts the previous order back. */
export async function applyQueueReorder({
  entries,
  nextIds,
  onError,
  queueSessionKey,
  request,
  sessionId
}: ApplyQueueReorderArgs): Promise<void> {
  const previousIds = entries.map(entry => entry.id)

  const moved = movedQueueIds(previousIds, nextIds)
    .map(id => entries.find(entry => entry.id === id))
    .find(entry => entry && canMoveQueuedEntry(entry))

  if (!moved) {
    return
  }

  if (moved.serverStatus && !request) {
    return
  }

  reorderQueuedPrompts(queueSessionKey, nextIds)

  if (!moved.serverStatus || !request) {
    return
  }

  const byId = new Map(entries.map(entry => [entry.id, entry]))
  const nextEntries = nextIds.flatMap(id => byId.get(id) ?? [])

  try {
    await moveQueuedPrompt(sessionId, queueSessionKey, moved.id, serverMoveAnchor(nextEntries, moved.id), request)
  } catch (error) {
    reorderQueuedPrompts(queueSessionKey, previousIds)
    onError(error)
  }
}
