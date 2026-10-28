import { atom, computed } from 'nanostores'

/**
 * Which chats are temporary — a leaf module (no store imports) so every persistence layer can ask
 * `isTemporarySession` before writing without an import cycle through `./session`. Lifecycle and
 * the rest of the temporary-chat state: `./temporary-chat`.
 */
export interface TemporaryChat {
  storedId: string
  runtimeId: null | string
  /** When the chat last became idle while off-screen (null: on screen or running). */
  idleSince: null | number
}

/** The fresh draft on screen will be created temporary on its first send. */
export const $temporaryDraft = atom(false)

/** Live temporary chats this window created, keyed by stored id. */
export const $temporaryChats = atom<Readonly<Record<string, TemporaryChat>>>({})

/** Every id (stored or runtime) that names a live temporary chat. */
export const $temporarySessionIds = computed($temporaryChats, chats => {
  const ids = new Set<string>()

  for (const chat of Object.values(chats)) {
    ids.add(chat.storedId)

    if (chat.runtimeId) {
      ids.add(chat.runtimeId)
    }
  }

  return ids as ReadonlySet<string>
})

export function isTemporarySession(sessionId: null | string | undefined): boolean {
  return Boolean(sessionId) && $temporarySessionIds.get().has(sessionId as string)
}
