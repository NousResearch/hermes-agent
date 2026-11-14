/**
 * The item index's stored shape and its pure rules: one snapshot per owner (a connection) holding,
 * per chat, the images / files / links extracted from its transcript plus the change marker the
 * chat had when it was read. The Library and the chat items panel both paint from it.
 *
 * Only extracted records are stored, never transcripts. Everything read back is validated: a
 * snapshot of another version, or one that does not parse, is discarded whole; a malformed chat or
 * record is dropped on its own.
 */
import type { ArtifactKind, ArtifactRecord } from './artifact-utils'

export const ITEM_INDEX_VERSION = 1

/** Most chats one owner's snapshot keeps: the Library's 30-chat window plus chats the items panel
 *  indexed on demand or that streamed in this window. */
export const MAX_INDEXED_CHATS = 60

/** Most records one chat keeps (newest first). A chat past it re-reads on every change, never grows. */
export const MAX_ITEMS_PER_CHAT = 400

/** A record whose value is longer than this (an inline `data:` image) stays in memory only; its
 *  chat is stored without a marker, so the next refresh reads it again instead of trusting a
 *  snapshot that lost it. */
export const MAX_STORED_VALUE_CHARS = 2048

/** A snapshot past this many UTF-16 units drops its least recently active chats until it fits. */
export const MAX_SNAPSHOT_CHARS = 1_000_000

/** A deletion is remembered this long, so a window that still holds the chat cannot write it back. */
export const REMOVAL_MEMORY_MS = 24 * 60 * 60 * 1000

export interface IndexedChat {
  /** `chatKey(connection, profile, sessionId)`. */
  key: string
  sessionId: string
  profile: string
  /** The connection that owns the chat (the list row's stamp, or this snapshot's owner). */
  connectionId: string
  title: string
  /** Newest activity the session list reported (ms), for ordering and eviction. */
  lastActive: number
  /** The list's change marker when this chat was last read; null = not read from its transcript
   *  (streamed in live, or stored lossy), so the next refresh reads it. */
  marker: null | string
  /** When this entry last changed (ms): the newer copy wins when two windows merge. */
  updatedAt: number
  items: ArtifactRecord[]
}

export interface ItemIndexSnapshot {
  /** When the owner's list was last checked (ms); null before the first refresh. */
  checkedAt: null | number
  chats: Record<string, IndexedChat>
  /** Chat keys deleted here, with when (ms). */
  removed: Record<string, number>
}

export const emptySnapshot = (): ItemIndexSnapshot => ({ checkedAt: null, chats: {}, removed: {} })

export const chatKey = (connectionId: string, profile: string, sessionId: string) =>
  `${connectionId}\u0000${profile || 'default'}\u0000${sessionId}`

/** What the session list says about a chat's transcript: it changed iff this changed. */
export const sessionMarker = (row: { last_active?: number; message_count?: number }) =>
  `${row.last_active ?? 0}:${row.message_count ?? 0}`

const KINDS: readonly ArtifactKind[] = ['image', 'file', 'link']

interface StoredItem {
  h: string
  k: ArtifactKind
  l: string
  t: number
  v: string
}

interface StoredChat {
  a: number
  c: string
  i: StoredItem[]
  m: null | string
  p: string
  s: string
  ti: string
  u: number
}

interface StoredSnapshot {
  chats: StoredChat[]
  checkedAt: null | number
  removed: Record<string, number>
  v: number
}

const isString = (value: unknown): value is string => typeof value === 'string'
const isNumber = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value)

function decodeItem(raw: unknown, chat: { profile: string; sessionId: string; title: string }): ArtifactRecord | null {
  const item = raw as Partial<StoredItem> | null

  if (!item || !isString(item.v) || !item.v || !isString(item.h) || !isString(item.l) || !isNumber(item.t)) {
    return null
  }

  if (!KINDS.includes(item.k as ArtifactKind)) {
    return null
  }

  return {
    id: `${chat.sessionId}:${item.v}`,
    kind: item.k as ArtifactKind,
    value: item.v,
    href: item.h,
    label: item.l,
    sessionId: chat.sessionId,
    profile: chat.profile,
    sessionTitle: chat.title,
    timestamp: item.t
  }
}

function decodeChat(raw: unknown): IndexedChat | null {
  const chat = raw as Partial<StoredChat> | null

  if (
    !chat ||
    !isString(chat.s) ||
    !chat.s ||
    !isString(chat.p) ||
    !isString(chat.c) ||
    !isString(chat.ti) ||
    !isNumber(chat.a) ||
    !isNumber(chat.u) ||
    !(chat.m === null || isString(chat.m)) ||
    !Array.isArray(chat.i)
  ) {
    return null
  }

  const base = { profile: chat.p, sessionId: chat.s, title: chat.ti }

  return {
    key: chatKey(chat.c, chat.p, chat.s),
    sessionId: chat.s,
    profile: chat.p,
    connectionId: chat.c,
    title: chat.ti,
    lastActive: chat.a,
    marker: chat.m,
    updatedAt: chat.u,
    items: chat.i.map(item => decodeItem(item, base)).filter((item): item is ArtifactRecord => item !== null)
  }
}

/** A stored snapshot, or an empty one when it is absent, of another version, or does not parse. */
export function decodeSnapshot(raw: null | string): ItemIndexSnapshot {
  if (!raw) {
    return emptySnapshot()
  }

  let parsed: Partial<StoredSnapshot> | null

  try {
    parsed = JSON.parse(raw) as Partial<StoredSnapshot>
  } catch {
    return emptySnapshot()
  }

  if (!parsed || parsed.v !== ITEM_INDEX_VERSION || !Array.isArray(parsed.chats)) {
    return emptySnapshot()
  }

  const chats: Record<string, IndexedChat> = {}

  for (const entry of parsed.chats) {
    const chat = decodeChat(entry)

    if (chat) {
      chats[chat.key] = chat
    }
  }

  const removed: Record<string, number> = {}

  if (parsed.removed && typeof parsed.removed === 'object') {
    for (const [key, at] of Object.entries(parsed.removed)) {
      if (isNumber(at)) {
        removed[key] = at
      }
    }
  }

  return { checkedAt: isNumber(parsed.checkedAt) ? parsed.checkedAt : null, chats, removed }
}

function encodeChat(chat: IndexedChat): StoredChat {
  const items = chat.items.filter(item => item.value.length <= MAX_STORED_VALUE_CHARS)

  return {
    a: chat.lastActive,
    c: chat.connectionId,
    i: items.map(item => ({ h: item.href, k: item.kind, l: item.label, t: item.timestamp, v: item.value })),
    // A chat that lost records on the way to storage must be read again after a restart.
    m: items.length === chat.items.length ? chat.marker : null,
    p: chat.profile,
    s: chat.sessionId,
    ti: chat.title,
    u: chat.updatedAt
  }
}

const byActivity = (left: IndexedChat, right: IndexedChat) => right.lastActive - left.lastActive

/** The snapshot as stored: at most {@link MAX_INDEXED_CHATS} chats, the most recently active kept
 *  first, and under {@link MAX_SNAPSHOT_CHARS}. */
export function encodeSnapshot(snapshot: ItemIndexSnapshot): string {
  let chats = Object.values(snapshot.chats).sort(byActivity).slice(0, MAX_INDEXED_CHATS).map(encodeChat)

  const encode = () =>
    JSON.stringify({ v: ITEM_INDEX_VERSION, checkedAt: snapshot.checkedAt, removed: snapshot.removed, chats })

  let text = encode()

  while (text.length > MAX_SNAPSHOT_CHARS && chats.length > 1) {
    chats = chats.slice(0, Math.max(1, Math.floor(chats.length * 0.75)))
    text = encode()
  }

  return text
}

/** Two windows' snapshots of one owner, merged per chat: the newer entry wins, a deletion beats any
 *  entry it postdates, and remembered deletions expire. */
export function mergeSnapshots(left: ItemIndexSnapshot, right: ItemIndexSnapshot, now: number): ItemIndexSnapshot {
  const removed: Record<string, number> = {}

  for (const source of [left.removed, right.removed]) {
    for (const [key, at] of Object.entries(source)) {
      if (now - at < REMOVAL_MEMORY_MS && at > (removed[key] ?? 0)) {
        removed[key] = at
      }
    }
  }

  const chats: Record<string, IndexedChat> = {}

  for (const source of [left.chats, right.chats]) {
    for (const chat of Object.values(source)) {
      const held = chats[chat.key]

      if ((removed[chat.key] ?? -1) >= chat.updatedAt || (held && held.updatedAt >= chat.updatedAt)) {
        continue
      }

      chats[chat.key] = chat
    }
  }

  const checkedAt = Math.max(left.checkedAt ?? 0, right.checkedAt ?? 0) || null

  return { checkedAt, chats, removed }
}

/** Newest first, capped: what one chat keeps. */
export function capChatItems(items: readonly ArtifactRecord[]): ArtifactRecord[] {
  return [...items].sort((left, right) => right.timestamp - left.timestamp).slice(0, MAX_ITEMS_PER_CHAT)
}

/** Records added to a chat without a re-read (the live stream): new ids only, newest first. */
export function mergeChatItems(held: readonly ArtifactRecord[], added: readonly ArtifactRecord[]): ArtifactRecord[] {
  const ids = new Set(held.map(item => item.id))
  const fresh = added.filter(item => !ids.has(item.id))

  return fresh.length ? capChatItems([...held, ...fresh]) : (held as ArtifactRecord[])
}
