/**
 * THE ITEM INDEX — the images, files and links of an owner's recent chats, shared by the Library
 * and the chat items panel (Figma 1198:10627 "Read me": keep results between visits, check only
 * chats that changed, record items as chats happen).
 *
 * - Owner: a connection (`ownerKeyOf`). Each chat inside is keyed by the connection that owns its
 *   row, its profile and its id, so a chat of another machine or profile never shares an entry.
 * - Kept between visits: each owner's snapshot lives in localStorage
 *   (`hermes:item-index:v1:<owner>`) and paints before any request. Only extracted records are
 *   stored, bounded per chat and per snapshot (item-index-snapshot.ts); at most
 *   {@link MAX_STORED_OWNERS} owners are kept, least recently used dropped first.
 * - Incremental refresh: the session list (one cheap request) carries each chat's `last_active` and
 *   `message_count`; only chats whose pair changed since they were read have their transcript read.
 *   Chats that left the list window (deleted, archived, aged out) leave the index.
 * - Live: while a chat streams in this window, its new items enter the index from the gateway
 *   events (`recordChatMessages`), with no transcript read. The chat's marker is left as it was, so
 *   the next refresh still reads that one chat once and the transcript stays the authority.
 * - Windows: a write merges with what another window stored (newest entry per chat wins, a deletion
 *   beats older entries), and a `storage` event from another window is adopted.
 */
import { atom } from 'nanostores'

import {
  type ArtifactLoadFailure,
  type ArtifactRecord,
  artifactSessionTitle,
  collectArtifactsForSession,
  loadArtifactsForSessions
} from '@/app/artifacts/artifact-utils'
import {
  capChatItems,
  chatKey,
  decodeSnapshot,
  emptySnapshot,
  encodeSnapshot,
  type IndexedChat,
  type ItemIndexSnapshot,
  mergeChatItems,
  mergeSnapshots,
  sessionMarker
} from '@/app/artifacts/item-index-snapshot'
import { getSessionMessages, listAllProfileSessions } from '@/hermes'
import { readJson, readKey, writeJson, writeKey } from '@/lib/storage'
import type { SessionInfo, SessionMessage } from '@/types/hermes'

import { $activeConnectionId } from './connections'
import { $removedSessionIds } from './session-removal'

/** The recent chats the index follows (Figma 1163:17707: "Data: 30 latest chats"). */
export const INDEXED_CHAT_WINDOW = 30

export const MAX_STORED_OWNERS = 3

const KEY_PREFIX = 'hermes:item-index:v1:'
const OWNERS_KEY = 'hermes:item-index:owners'

/** Writes coalesce: a streaming turn records many items in a burst. */
const PERSIST_DELAY_MS = 600

export const ownerKeyOf = (connectionId: null | string | undefined) => connectionId?.trim() || 'local'

export const activeItemIndexOwner = () => ownerKeyOf($activeConnectionId.get())

export interface OwnerIndex {
  snapshot: ItemIndexSnapshot
  /** A refresh is running for this owner. */
  refreshing: boolean
  /** The last refresh failed before it could check the list (null once one succeeds). */
  error: unknown
  /** Chats the last refresh could not read (their previous items stay). */
  failures: ArtifactLoadFailure[]
}

/** Every owner this window has opened, by owner key. */
export const $itemIndex = atom<Record<string, OwnerIndex>>({})

const storageKey = (owner: string) => `${KEY_PREFIX}${owner}`

function setOwner(owner: string, patch: Partial<OwnerIndex>): void {
  const current = $itemIndex.get()[owner] ?? { snapshot: emptySnapshot(), refreshing: false, error: null, failures: [] }

  $itemIndex.set({ ...$itemIndex.get(), [owner]: { ...current, ...patch } })
}

/** Snapshots read from storage but not yet published, so a render can paint them without
 *  updating the atom mid-render; the next `ownerIndex` call publishes. */
const hydrated = new Map<string, OwnerIndex>()

/** The owner's index for a render: memory, else storage, without publishing. */
export function peekOwnerIndex(owner: string): OwnerIndex {
  const held = $itemIndex.get()[owner] ?? hydrated.get(owner)

  if (held) {
    return held
  }

  const read: OwnerIndex = {
    snapshot: decodeSnapshot(readKey(storageKey(owner))),
    refreshing: false,
    error: null,
    failures: []
  }

  hydrated.set(owner, read)

  return read
}

/** The owner's index, loaded from storage (and published) the first time it is asked for. */
export function ownerIndex(owner: string): OwnerIndex {
  installListeners()

  const held = $itemIndex.get()[owner]

  if (held) {
    return held
  }

  const read = peekOwnerIndex(owner)

  hydrated.delete(owner)
  $itemIndex.set({ ...$itemIndex.get(), [owner]: read })

  return read
}

function setSnapshot(owner: string, update: (snapshot: ItemIndexSnapshot) => ItemIndexSnapshot): void {
  const snapshot = update(ownerIndex(owner).snapshot)

  setOwner(owner, { snapshot })
  schedulePersist(owner)
}

// ── Persistence ────────────────────────────────────────────────────────────

const pendingPersists = new Map<string, ReturnType<typeof setTimeout>>()

function schedulePersist(owner: string): void {
  if (pendingPersists.has(owner)) {
    return
  }

  pendingPersists.set(
    owner,
    setTimeout(() => {
      pendingPersists.delete(owner)
      persistOwner(owner)
    }, PERSIST_DELAY_MS)
  )
}

/** Write the owner's snapshot now, merged with whatever another window stored meanwhile. */
export function persistOwner(owner: string): void {
  const held = $itemIndex.get()[owner]

  if (!held) {
    return
  }

  const merged = mergeSnapshots(decodeSnapshot(readKey(storageKey(owner))), held.snapshot, Date.now())

  writeKey(storageKey(owner), encodeSnapshot(merged))
  setOwner(owner, { snapshot: merged })
  touchOwner(owner)
}

/** Most recently used first; owners past {@link MAX_STORED_OWNERS} lose their stored snapshot. */
function touchOwner(owner: string): void {
  const owners = readJson<unknown>(OWNERS_KEY)
  const known = Array.isArray(owners) ? owners.filter((entry): entry is string => typeof entry === 'string') : []
  const next = [owner, ...known.filter(entry => entry !== owner)]

  for (const evicted of next.slice(MAX_STORED_OWNERS)) {
    writeKey(storageKey(evicted), null)
  }

  if (next.join('\u0000') !== known.join('\u0000') || next.length > MAX_STORED_OWNERS) {
    writeJson(OWNERS_KEY, next.slice(0, MAX_STORED_OWNERS))
  }
}

/** An owner that can never be dialled again (its connection was removed): drop everything kept. */
export function forgetItemIndexOwner(connectionId: string): void {
  const owner = ownerKeyOf(connectionId)
  const timer = pendingPersists.get(owner)

  if (timer) {
    clearTimeout(timer)
    pendingPersists.delete(owner)
  }

  writeKey(storageKey(owner), null)
  hydrated.delete(owner)

  const { [owner]: _forgotten, ...rest } = $itemIndex.get()

  $itemIndex.set(rest)
}

// ── Removal, other windows ─────────────────────────────────────────────────

/** Chats deleted or archived here leave every loaded owner's index at once, remembered so another
 *  window's older copy cannot bring them back. A rolled-back deletion simply re-reads the chat. */
export function removeIndexedChats(sessionIds: Iterable<string>): void {
  const ids = new Set(sessionIds)
  const now = Date.now()

  for (const owner of Object.keys($itemIndex.get())) {
    const { snapshot } = ownerIndex(owner)
    const doomed = Object.values(snapshot.chats).filter(chat => ids.has(chat.sessionId))

    if (!doomed.length) {
      continue
    }

    setSnapshot(owner, current => {
      const chats = { ...current.chats }
      const removed = { ...current.removed }

      for (const chat of doomed) {
        delete chats[chat.key]
        removed[chat.key] = now
      }

      return { ...current, chats, removed }
    })
  }
}

let listenersInstalled = false

function installListeners(): void {
  if (listenersInstalled || typeof window === 'undefined') {
    return
  }

  listenersInstalled = true

  $removedSessionIds.listen(ids => removeIndexedChats(ids))

  window.addEventListener('storage', event => {
    const owner = event.key?.startsWith(KEY_PREFIX) ? event.key.slice(KEY_PREFIX.length) : null
    const held = owner ? $itemIndex.get()[owner] : undefined

    if (owner) {
      hydrated.delete(owner)
    }

    if (owner && held) {
      setOwner(owner, { snapshot: mergeSnapshots(held.snapshot, decodeSnapshot(event.newValue), Date.now()) })
    }
  })
}

// ── Refresh ────────────────────────────────────────────────────────────────

const inFlight = new Map<string, Promise<void>>()

function chatFromRow(owner: string, row: SessionInfo): Pick<IndexedChat, 'connectionId' | 'key' | 'profile'> {
  const connectionId = ownerKeyOf(row.connection_id ?? owner)
  const profile = row.profile || 'default'

  return { connectionId, key: chatKey(connectionId, profile, row.id), profile }
}

/** Read one chat's transcript into its records, pinned to the connection its row names (an
 *  unstamped row keeps the ambient route, as every other session read does). */
async function readChat(row: SessionInfo) {
  const profile = row.profile || 'default'
  const scope = row.connection_id ? { connectionId: row.connection_id, profile } : profile

  return loadArtifactsForSessions([row], (session, page) =>
    getSessionMessages(session.id, scope, { ...page, includeCompacted: true, order: 'oldest' })
  )
}

function indexedFromRow(owner: string, row: SessionInfo, items: ArtifactRecord[], marker: null | string): IndexedChat {
  return {
    ...chatFromRow(owner, row),
    sessionId: row.id,
    title: artifactSessionTitle(row),
    lastActive: (row.last_active ?? 0) * 1000,
    marker,
    updatedAt: Date.now(),
    items: capChatItems(items)
  }
}

/** Profiles the list could not read this time keep their chats rather than losing them. */
function failedProfiles(errors: undefined | Array<{ profile: string }>): Set<string> {
  return new Set((errors ?? []).map(error => error.profile || 'default'))
}

async function runRefresh(owner: string): Promise<void> {
  const stillOwner = () => activeItemIndexOwner() === owner

  setOwner(owner, { refreshing: true })

  try {
    const list = await listAllProfileSessions(INDEXED_CHAT_WINDOW, 1)

    if (!stillOwner()) {
      return
    }

    const rows = list.sessions
    const listed = new Set(rows.map(row => chatFromRow(owner, row).key))
    const keep = failedProfiles(list.errors)

    // Drop what left the window; carry each listed chat's title and activity; queue changed chats.
    setSnapshot(owner, current => {
      const chats: Record<string, IndexedChat> = {}

      for (const chat of Object.values(current.chats)) {
        if (listed.has(chat.key) || keep.has(chat.profile)) {
          chats[chat.key] = chat
        }
      }

      for (const row of rows) {
        const key = chatFromRow(owner, row).key
        const held = chats[key]
        const title = artifactSessionTitle(row)

        if (held && (held.title !== title || held.lastActive !== row.last_active * 1000)) {
          chats[key] = {
            ...held,
            title,
            lastActive: row.last_active * 1000,
            items: held.items.map(item => ({ ...item, sessionTitle: title })),
            updatedAt: Date.now()
          }
        }
      }

      return { ...current, chats, checkedAt: Date.now() }
    })

    const stale = rows.filter(
      row => ownerIndex(owner).snapshot.chats[chatFromRow(owner, row).key]?.marker !== sessionMarker(row)
    )

    const failures: ArtifactLoadFailure[] = []

    for (const row of stale) {
      // A switch of machine mid-refresh stops reading: later reads would dial the new one.
      if (!stillOwner()) {
        return
      }

      const result = await readChat(row)

      failures.push(...result.failures)

      if (!result.failures.length) {
        setSnapshot(owner, current => {
          const chat = indexedFromRow(owner, row, result.artifacts, sessionMarker(row))

          return { ...current, chats: { ...current.chats, [chat.key]: chat } }
        })
      }
    }

    setOwner(owner, { error: null, failures })
  } catch (error) {
    setOwner(owner, { error })
  } finally {
    setOwner(owner, { refreshing: false })
  }
}

/** Bring the active owner's index up to date, reading only chats that changed. Concurrent calls
 *  share one run. */
export function refreshItemIndex(): Promise<void> {
  const owner = activeItemIndexOwner()

  ownerIndex(owner)

  const running = inFlight.get(owner)

  if (running) {
    return running
  }

  const run = runRefresh(owner).finally(() => inFlight.delete(owner))

  inFlight.set(owner, run)

  return run
}

/** Index one chat now (the items panel opened on a chat outside the Library's window, or one the
 *  index has not read yet) into `owner`'s index. A no-op when its marker is current. */
export async function ensureChatIndexed(row: SessionInfo, owner = activeItemIndexOwner()): Promise<void> {
  const key = chatFromRow(owner, row).key

  if (ownerIndex(owner).snapshot.chats[key]?.marker === sessionMarker(row)) {
    return
  }

  const result = await readChat(row)

  // An unpinned read went to whichever machine was in front; keep it only if that is still `owner`.
  if (result.failures.length || (!row.connection_id && activeItemIndexOwner() !== owner)) {
    return
  }

  setSnapshot(owner, current => {
    const chat = indexedFromRow(owner, row, result.artifacts, sessionMarker(row))

    return { ...current, chats: { ...current.chats, [chat.key]: chat } }
  })
}

// ── Live recording ─────────────────────────────────────────────────────────

export interface LiveChatRef {
  connectionId: string
  profile: string
  sessionId: string
  title?: string
}

/** Records the streamed messages of a chat into its owner's index (no transcript read). The chat's
 *  entry is found by id first, so a profile the event could not name still lands on its row. */
export function recordChatMessages(chat: LiveChatRef, messages: SessionMessage[]): void {
  const owner = ownerKeyOf(chat.connectionId)
  const chats = ownerIndex(owner).snapshot.chats

  const held =
    Object.values(chats).find(entry => entry.sessionId === chat.sessionId && entry.connectionId === owner) ?? null

  const profile = held?.profile ?? (chat.profile || 'default')
  const title = held?.title ?? chat.title ?? ''
  const nowSeconds = Date.now() / 1000

  const session = {
    id: chat.sessionId,
    profile,
    title,
    last_active: nowSeconds,
    started_at: nowSeconds
  } as SessionInfo

  const items = collectArtifactsForSession(session, messages)

  if (!items.length) {
    return
  }

  setSnapshot(owner, current => {
    const key = held?.key ?? chatKey(owner, profile, chat.sessionId)
    const base = current.chats[key]

    const next: IndexedChat = base
      ? { ...base, items: mergeChatItems(base.items, items), lastActive: Date.now(), updatedAt: Date.now() }
      : {
          key,
          sessionId: chat.sessionId,
          profile,
          connectionId: owner,
          title: artifactSessionTitle(session),
          lastActive: Date.now(),
          marker: null,
          updatedAt: Date.now(),
          items: capChatItems(items)
        }

    return { ...current, chats: { ...current.chats, [key]: next } }
  })
}

// ── Reads ──────────────────────────────────────────────────────────────────

/** Every item of the owner's indexed chats, newest first. */
export function ownerItems(snapshot: ItemIndexSnapshot): ArtifactRecord[] {
  return Object.values(snapshot.chats)
    .flatMap(chat => chat.items)
    .sort((left, right) => right.timestamp - left.timestamp)
}

/** The chat a record belongs to, for its owner route. */
export function indexedChatOf(snapshot: ItemIndexSnapshot, item: ArtifactRecord): IndexedChat | undefined {
  return Object.values(snapshot.chats).find(
    chat => chat.sessionId === item.sessionId && chat.profile === (item.profile || 'default')
  )
}

export function _resetItemIndexForTests(): void {
  for (const timer of pendingPersists.values()) {
    clearTimeout(timer)
  }

  pendingPersists.clear()
  inFlight.clear()
  hydrated.clear()
  $itemIndex.set({})
}
