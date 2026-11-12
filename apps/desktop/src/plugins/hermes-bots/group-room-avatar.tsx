import { useValue } from '@hermes/plugin-sdk'

import { avatarColor, botAppearance, BotFace } from './avatar'
import { isBackfilledFacePng } from './avatar-image'
import { $botMeta, $lastRoster, type BotMetaSnapshot, botOwner } from './data'
import { botRosterMeta } from './routing'
import type { BotMeta, RosterRow } from './types'

/** The meta entry a room member's face resolves to, owner-first. A member a
 *  source-scoped roster row claims reads that connection's scoped key (plus
 *  its alias routes) exactly like the roster itself does — botRosterMeta — and
 *  NEVER its bare name: migrated v2 snapshots still carry pre-v2 bare-name
 *  entries as rollback residue, and a stale bare face must not outrank (or,
 *  once the photo filter blanks a 160x160 backfill, stand in for) the real
 *  one. The bare name stays the meta key only where no scoped row claims the
 *  member — legacy single-source installs (data.ts's documented v1 contract),
 *  or a roster snapshot that has not painted yet, read through botOwner.
 *  botOwner's migrated-local remap (migratedLocalRoutes) is only populated in
 *  the session that runs the fresh v1→v2 migration (data.ts:740-743); a
 *  v2-committed restart (data.ts:636-640) never repopulates it, so there the
 *  key degrades back to the bare name. */
function roomBotMeta(name: string, meta: BotMetaSnapshot, roster: RosterRow[]): BotMeta | null | undefined {
  const trimmed = String(name || '').trim()

  if (!trimmed) {
    return null
  }

  const row = roster.find(bot => String(bot?.name || '').trim() === trimmed && (bot?.sourceScoped || bot?.remoteSource))

  return row ? botRosterMeta(row, meta) : meta?.[botOwner(trimmed).key]
}

/** A bot's face from its Bot Mode appearance (custom image/pet, else its shape + color). */
export function RoomBotAvatar({ name, size = 24 }: { name: string; size?: number }) {
  const meta = useValue($botMeta)
  const roster = useValue($lastRoster)
  const appearance = botAppearance(name, roomBotMeta(name, meta, roster))
  const photo = appearance.image && !isBackfilledFacePng(appearance.image) ? appearance.image : null

  return <BotFace color={avatarColor(appearance.color, name)} image={photo} name={name} shape={appearance.shape} size={size} />
}
