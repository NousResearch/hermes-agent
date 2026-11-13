/**
 * Friendly @mentions in hosted rooms.
 *
 * A room member's `handle` is immutable: it is what the room was seated with, and
 * every gateway resolves it. A gateway that advertises `room_mention_aliases_v1`
 * also accepts the frozen `display_name`'s normalized forms (slug and collapsed,
 * matching the shared mention-name rules) — but only when exactly one member
 * answers to that form. An exact handle always wins, an ambiguous alias is
 * refused, and the broadcast/reserved words are never derived. The composer
 * inserts a friendly tag only on such an owner; everywhere else it keeps
 * inserting the handle, which older gateways have always understood.
 */

/** What `groups.capabilities.features` lists once a room owner accepts aliases. */
export const ROOM_MENTION_ALIASES_FEATURE = 'room_mention_aliases_v1'

/** The words an alias must never shadow: `@all`/`@everyone` broadcast, and the
 *  identity words the gateway reserves. */
const RESERVED_MENTION_FORMS = new Set(['all', 'everyone', 'user', 'default', 'hermes'])

/** A member as the room reports it: the immutable handle plus the frozen name. */
export interface RoomMentionMember {
  handle: string
  display_name?: string
}

const normalized = (value: unknown) => String(value ?? '').trim().toLowerCase()

/** The mention forms a frozen display name may answer to. ASCII-lower, then the
 *  slug (`scout-homelab`) and the collapsed form (`scouthomelab`) — the same
 *  pair `mentionNameForms` derives for the desktop parser and the gateway
 *  contract. A device-qualified name keeps its device suffix; it is part of the
 *  identity, not decoration. */
export function roomMentionAliasForms(displayName: unknown): string[] {
  const name = normalized(displayName)

  if (!name) {
    return []
  }

  const slug = name.replace(/[^a-z0-9_-]+/g, '-').replace(/^-+|-+$/g, '')
  const collapsed = name.replace(/[^a-z0-9_-]+/g, '')

  return [...new Set([slug, collapsed])].filter(
    form => /^[a-z0-9][a-z0-9_-]*$/.test(form) && !RESERVED_MENTION_FORMS.has(form)
  )
}

/** True when the owner's `groups.capabilities` answer (or its bare `features`
 *  list) advertises friendly mention aliases. An older owner answers neither the
 *  field nor the feature, so the composer keeps the canonical handle. */
export function advertisesRoomMentionAliases(capabilities: unknown): boolean {
  const features = capabilities instanceof Set
    ? capabilities
    : Array.isArray(capabilities)
      ? capabilities
      : capabilities && typeof capabilities === 'object'
        ? (capabilities as Record<string, unknown>).features
        : undefined

  if (features instanceof Set) {
    return features.has(ROOM_MENTION_ALIASES_FEATURE)
  }

  return Array.isArray(features) && features.includes(ROOM_MENTION_ALIASES_FEATURE)
}

/** Every form an @mention may use, mapped to the immutable handle of the one
 *  member it names. Exact handles claim their form first; display-name forms are
 *  added only when that leaves a single claimant, so a form two members share
 *  (or one that spells another member's handle) resolves to nobody and that
 *  member keeps its handle. */
function mentionClaims(members: readonly RoomMentionMember[], aliases: boolean) {
  const claims = new Map<string, Set<string>>()

  const claim = (form: string, handle: string) => {
    const owners = claims.get(form) ?? new Set<string>()

    owners.add(handle)
    claims.set(form, owners)
  }

  for (const member of members) {
    const handle = normalized(member.handle)

    if (handle) {
      claim(handle, handle)
    }
  }

  if (aliases) {
    for (const member of members) {
      const handle = normalized(member.handle)

      if (!handle) {
        continue
      }

      for (const form of roomMentionAliasForms(member.display_name)) {
        claim(form, handle)
      }
    }
  }

  return claims
}

const claimsAlone = (claims: Map<string, Set<string>>, form: string, handle: string) => {
  const owners = claims.get(form)

  return owners?.size === 1 && owners.has(handle)
}

/** The tag the composer inserts for `member`: the first friendly form that
 *  names this member alone, else the immutable handle. With `aliases` false the
 *  answer is always the handle — exactly what an older owner resolves. */
export function roomMentionTag(member: RoomMentionMember, members: readonly RoomMentionMember[], aliases: boolean): string {
  const handle = normalized(member.handle)

  if (!aliases || !handle) {
    return String(member.handle ?? '').trim()
  }

  const claims = mentionClaims(members, true)

  for (const form of roomMentionAliasForms(member.display_name)) {
    if (claimsAlone(claims, form, handle)) {
      return form
    }
  }

  return String(member.handle).trim()
}
