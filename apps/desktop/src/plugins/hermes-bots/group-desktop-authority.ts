import { sha256 } from '@noble/hashes/sha2.js'
import { bytesToHex, utf8ToBytes } from '@noble/hashes/utils.js'

import type { GroupChat } from './types'

type AuthorityRecord = {
  desktopAuthorityHash?: unknown
  desktopAuthorityConflict?: unknown
  hosted?: unknown
  roomId?: unknown
  desktopAuthorityToken?: unknown
}

const AUTHORITY_HASH = /^[a-f0-9]{64}$/
const AUTHORITY_TOKEN = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$/

export function classicAuthorityHash(token: string) {
  return bytesToHex(sha256(utf8ToBytes(token)))
}

function authorityToken(value: unknown) {
  const token = typeof value === 'string' ? value.trim() : ''

  return AUTHORITY_TOKEN.test(token) ? token : ''
}

function mintAuthorityToken() {
  if (!globalThis.crypto || typeof globalThis.crypto.getRandomValues !== 'function') {
    throw new Error('Secure Group Chat control is unavailable in this Desktop build.')
  }

  return `authority:${bytesToHex(globalThis.crypto.getRandomValues(new Uint8Array(32)))}`
}

/** A projection key cannot lend another room its mailbox incarnation. */
export function classicProjectionAuthority(source: AuthorityRecord | undefined, key: string) {
  if (key.startsWith('id:') && source?.roomId && source.roomId !== key.slice(3)) {
    return {}
  }

  return classicDesktopAuthority(source)
}

/** Public mailbox incarnation commitment, not a credential or hosted authority receipt. */
export function classicDesktopAuthority(...sources: (AuthorityRecord | null | undefined)[]): {
  desktopAuthorityHash?: string
  desktopAuthorityConflict?: true
} {
  if (sources.some(source => typeof source?.hosted === 'string' && source.hosted.trim())) {
    return {}
  }

  const hashes = new Set<string>()

  for (const source of sources) {
    if (source?.desktopAuthorityConflict === true) {
      return { desktopAuthorityConflict: true }
    }

    const value = source?.desktopAuthorityHash

    if (typeof value === 'string' && AUTHORITY_HASH.test(value.toLowerCase())) {
      hashes.add(value.toLowerCase())
    }
  }

  // The mailbox pins its first commitment. Picking either competing value
  // would silently strand commands on other gateways. Conflict is sticky
  // across revision order and old clients; only a new room can clear it.
  if (hashes.size > 1) {
    return { desktopAuthorityConflict: true }
  }

  const hash = [...hashes][0]

  return hash ? { desktopAuthorityHash: hash } : {}
}

/** Private claim material for the gateway mailbox; never call from projection code. */
export function classicAuthorityClaim(room: GroupChat): null | { authorityHash: string; authorityToken: string } {
  if (room.desktopAuthorityConflict || (typeof room.hosted === 'string' && room.hosted.trim())) {
    return null
  }

  const token = authorityToken(room.desktopAuthorityToken)
  const hash = classicDesktopAuthority(room).desktopAuthorityHash

  return token && hash === classicAuthorityHash(token) ? { authorityHash: hash, authorityToken: token } : null
}

/** Complete private record for plugin storage only. */
export function storedClassicDesktopAuthority(room: GroupChat) {
  const publicState = classicDesktopAuthority(room)
  const claim = classicAuthorityClaim(room)

  return {
    ...publicState,
    ...(claim ? { desktopAuthorityToken: claim.authorityToken } : {})
  }
}

export function ensureClassicDesktopAuthority(
  room: GroupChat,
  previous?: GroupChat
): GroupChat {
  if (room.tombstone || (typeof room.hosted === 'string' && room.hosted.trim())) {
    return room
  }

  const replacement = previous?.tombstone || (previous?.roomId && room.roomId && previous.roomId !== room.roomId)
  const previousToken = replacement ? '' : authorityToken(previous?.desktopAuthorityToken)
  const candidateToken = replacement ? '' : authorityToken(room.desktopAuthorityToken)
  const tokens = new Set([previousToken, candidateToken].filter(Boolean))

  let publicState = replacement ? {} : classicDesktopAuthority(previous, room)

  if (tokens.size > 1 || publicState.desktopAuthorityConflict) {
    return {
      ...room,
      desktopAuthorityHash: undefined,
      desktopAuthorityToken: undefined,
      desktopAuthorityConflict: true
    }
  }

  let token = [...tokens][0] || ''

  // A projected room already carrying a public commitment is owned by the
  // Desktop that has its private token. Never manufacture a competing claim.
  if (!token && publicState.desktopAuthorityHash) {
    if (room.desktopAuthorityHash === publicState.desktopAuthorityHash && room.desktopAuthorityToken === undefined) {
      return room
    }

    return { ...room, desktopAuthorityHash: publicState.desktopAuthorityHash, desktopAuthorityToken: undefined }
  }

  if (!token) {
    token = mintAuthorityToken()
    publicState = {}
  }

  const hash = classicAuthorityHash(token)

  if (room.desktopAuthorityHash === hash && room.desktopAuthorityToken === token && !room.desktopAuthorityConflict) {
    return room
  }

  if (publicState.desktopAuthorityHash && publicState.desktopAuthorityHash !== hash) {
    return {
      ...room,
      desktopAuthorityHash: undefined,
      desktopAuthorityToken: undefined,
      desktopAuthorityConflict: true
    }
  }

  return {
    ...room,
    desktopAuthorityConflict: undefined,
    desktopAuthorityHash: hash,
    desktopAuthorityToken: token
  }
}

/** Stable comparison for fan-out; normal log/revision changes are irrelevant. */
export function classicAuthorityState(rooms: Record<string, GroupChat>): string {
  return JSON.stringify(
    Object.entries(rooms)
      .flatMap(([name, room]) => {
        const value = classicDesktopAuthority(room)

        return Object.keys(value).length ? [[room.roomId || `name:${name}`, value]] : []
      })
      .sort(([left], [right]) => String(left).localeCompare(String(right)))
  )
}
