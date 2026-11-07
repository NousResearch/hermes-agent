import { useValue } from '@hermes/plugin-sdk'

import { avatarColor, botAppearance, BotFace } from './avatar'
import { isBackfilledFacePng } from './avatar-image'
import { $botMeta, $lastRoster, botRosterKey } from './data'
import { botRosterMeta } from './routing'
import type { BotMeta, RosterRow } from './types'

/** The meta record a room member's face paints from.
 *
 *  A room carries only a profile name, while bot meta is persisted under the
 *  route-qualified key (`connectionId::profile`, `botMetaKey`) once the v2
 *  migration ran — so a bare `meta[name]` read misses and the face falls back to
 *  the name-derived shape/colour even though the Bots list shows the bot's own
 *  appearance (#148923). Resolve the member to its roster row and read through
 *  `botRosterMeta`, the pipeline `BotRow`/`groupSpeakerLabel` already use: with
 *  the room's connection the row is exact, and when the roster has not loaded
 *  (or no longer lists the owner) the connection alone still keys the record —
 *  and its alias map — so a cached appearance paints instead of borrowing the
 *  face of a same-named bot on another gateway. Without a connection a raw name
 *  resolves only while exactly ONE roster row carries it. A bare-name key
 *  survives as the pre-migration rung, single-source rosters only — once rows
 *  are connection-scoped the bare key cannot say which machine it named. */
export function memberFaceMeta(
  name: string,
  roster: RosterRow[],
  allMeta: Record<string, BotMeta>,
  connectionId?: null | string
): BotMeta | null | undefined {
  const wanted = String(name || '').trim()

  if (!wanted) {
    return null
  }

  const rows = Array.isArray(roster) ? roster.filter(row => row && !row.ghost) : []
  const connection = String(connectionId || '').trim()
  const scopedRow = (row: RosterRow) => Boolean(row.sourceScoped || row.remoteSource)

  if (connection) {
    const keyed = `${connection}::${wanted}`

    const exact =
      rows.find(row => botRosterKey(row) === keyed) ||
      rows.find(row => row.name === wanted && (row.connectionId || row.route?.connectionId) === connection)

    if (exact) {
      return botRosterMeta(exact, allMeta)
    }

    // No row names this member yet: the room's own connection still keys its
    // meta record, and `botRosterMeta` also walks the alias map for it — so a
    // cached appearance paints before the roster loads.
    const owned = botRosterMeta({ connectionId: connection, name: wanted, sourceScoped: true } as RosterRow, allMeta)

    if (owned) {
      return owned
    }
  }

  // `default` names the ACTIVE gateway's primary profile, so a scoped row must
  // never answer for it by bare name — the same rule `groupSpeakerLabel` applies.
  // With the room's connection known, no row scoped to ANOTHER gateway may answer
  // either: only unscoped (pre-migration) rows reach the unique-name rung.
  const isDefault = wanted.toLowerCase() === 'default'

  const named = rows.filter(
    row => row.name === wanted && !(isDefault && scopedRow(row)) && !(connection && scopedRow(row))
  )

  if (named.length === 1) {
    return botRosterMeta(named[0], allMeta)
  }

  const scoped = rows.some(scopedRow)

  return scoped ? undefined : allMeta?.[wanted]
}

/** A bot's face from its Bot Mode appearance (custom image/pet, else its shape + color). */
export function RoomBotAvatar({
  connectionId,
  name,
  size = 24
}: {
  /** The room's own connection; without it the name resolves by a roster-unique match only. */
  connectionId?: null | string
  name: string
  size?: number
}) {
  const allMeta = useValue($botMeta) as Record<string, BotMeta>
  const roster = useValue($lastRoster) as RosterRow[]
  const appearance = botAppearance(name, memberFaceMeta(name, roster, allMeta, connectionId))
  const photo = appearance.image && !isBackfilledFacePng(appearance.image) ? appearance.image : null

  return (
    <BotFace
      color={avatarColor(appearance.color, name)}
      image={photo}
      name={name}
      shape={appearance.shape}
      size={size}
    />
  )
}
