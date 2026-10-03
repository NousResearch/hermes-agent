/** Renderer-owned Bot Mode rooms, not gateway Hosted Rooms (groups.*). */
export type GroupChatsStatus = 'unavailable' | 'loading' | 'ready'
export interface GroupChatSend {
  roomId: string
  text: string
  threadId?: string
  /** Window/provider-lifetime deduplication, bounded to the latest 256 accepted submissions. */
  submissionKey?: string
}
export type GroupChatSendResult = { accepted: true; threadId: string } | { accepted: false; error: string }
export interface GroupChatMessageSnapshot {
  readonly at: number
  readonly from: Readonly<{ kind: 'member' | 'user'; name: string; source?: string; gateway?: string }>
  readonly text: string
  readonly id?: string
  readonly thread?: string
  readonly truncated?: boolean
}
export interface GroupChatRoomSnapshot {
  /** Null legacy identities are display-only; never send by name. */
  readonly roomId: string | null
  readonly name: string
  readonly running: boolean
  readonly needsYou: boolean
  readonly log: readonly GroupChatMessageSnapshot[]
  /** Nonsecret source identity: installId matches author.gateway; connectionLabel (or connectionId) matches author.source. */
  readonly members: readonly Readonly<{
    name: string
    title: string
    connectionId?: string
    connectionLabel?: string
    installId?: string
  }>[]
  readonly activity: readonly Readonly<{
    at: number
    kind: string
    member?: string | null
    thread?: string | null
    reason?: string
    preview?: string
  }>[]
  readonly requests: readonly Readonly<{
    at: number
    kind: 'approval' | 'clarify'
    member: string
    thread?: string
    question: string
    choices: readonly string[]
    multiSelect: boolean
    command?: string
  }>[]
}
export interface GroupChatsSnapshot {
  readonly rooms: readonly GroupChatRoomSnapshot[]
}
export interface GroupChatsProvider {
  status(): GroupChatsStatus
  getSnapshot(): GroupChatsSnapshot
  subscribe(listener: () => void): () => void
  send(input: GroupChatSend): GroupChatSendResult
}
let provider: GroupChatsProvider | undefined
let unbind: (() => void) | undefined
const listeners = new Set<() => void>()

const notify = () => {
  for (const listener of listeners) {
    // Consumer rendering must never interrupt the engine's synchronous append.
    try {
      listener()
    } catch (error) {
      console.error('groupChats subscriber failed', error)
    }
  }
}

let sending = false
const empty = Object.freeze({ rooms: Object.freeze([]) })

/** Internal composition seam; deliberately not exported by the public SDK. */
export function registerGroupChatsProvider(next: GroupChatsProvider) {
  unbind?.()
  provider = next
  const token = {}
  generation = token
  unbind = next.subscribe(() => {
    if (generation === token) {
      notify()
    }
  })
  notify()

  return () => {
    if (generation !== token) {
      return
    }

    unbind?.()
    unbind = undefined
    provider = undefined
    generation = undefined
    notify()
  }
}

let generation: object | undefined
export const groupChats = Object.freeze({
  version: 1 as const,
  status: (): GroupChatsStatus => provider?.status() ?? 'unavailable',
  getSnapshot: (): GroupChatsSnapshot => provider?.getSnapshot() ?? empty,
  subscribe(listener: () => void) {
    listeners.add(listener)

    return () => {
      listeners.delete(listener)
    }
  },
  send(input: GroupChatSend): GroupChatSendResult {
    const status = groupChats.status()

    if (status !== 'ready') {
      return { accepted: false, error: status }
    }

    if (sending) {
      return { accepted: false, error: 'submission-in-flight' }
    }

    sending = true

    try {
      return provider!.send(input)
    } finally {
      sending = false
    }
  }
})
