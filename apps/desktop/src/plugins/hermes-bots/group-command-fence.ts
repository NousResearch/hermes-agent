/** Runtime-only generations. Removed entries stay invalid through references
 * held by old turns; no cancellation tombstones or user Stop holds persist. */
export interface GroupCommandFence {
  readonly commandId: string
  readonly roomId: string
  readonly generation: symbol
  readonly leaseValid: () => boolean
  readonly roomValid?: () => boolean
  cancelled: boolean
  epoch: number | null
  thread: string | null
  persistenceFailed?: boolean
}

const MAX_ACTIVE_COMMANDS = 128
const active = new Map<string, GroupCommandFence>()

const key = (fence: Pick<GroupCommandFence, 'roomId' | 'commandId'>) => JSON.stringify([fence.roomId, fence.commandId])

export function beginGroupCommandFence(
  roomId: string,
  commandId: string,
  leaseValid = () => true,
  roomValid = () => true
): GroupCommandFence {
  const id = key({ roomId, commandId })
  const previous = active.get(id)

  if (!previous && active.size >= MAX_ACTIVE_COMMANDS) {
    throw Object.assign(new Error('Too many Group Chat commands are active. Try again shortly.'), { retryable: true })
  }

  if (previous && !previous.cancelled) {
    throw Object.assign(new Error('This Group Chat command is already running on this Desktop.'), { retryable: true })
  }

  const fence = { roomId, commandId, leaseValid, roomValid, generation: Symbol(), cancelled: false, epoch: null, thread: null }
  active.set(id, fence)

  return fence
}

export function groupCommandFenceLive(fence?: GroupCommandFence) {
  if (fence?.roomValid && !fence.roomValid()) {
    fence.cancelled = true
  }

  return !fence || (!fence.cancelled && active.get(key(fence))?.generation === fence.generation && fence.leaseValid())
}

export function groupCommandFenceMatches(
  fence: GroupCommandFence | undefined,
  roomId: null | string | undefined,
  thread: string
) {
  return groupCommandFenceLive(fence) && (!fence || (fence.roomId === roomId && fence.thread === thread))
}

export function bindGroupCommandFence(fence: GroupCommandFence | undefined, thread: string, epoch: number) {
  if (fence && groupCommandFenceLive(fence)) {
    fence.thread = thread
    fence.epoch = epoch
  }
}

export function cancelGroupCommandFence(fence: GroupCommandFence) {
  fence.cancelled = true
}

export function releaseGroupCommandFence(fence: GroupCommandFence) {
  cancelGroupCommandFence(fence)

  if (active.get(key(fence))?.generation === fence.generation) {
    active.delete(key(fence))
  }
}
