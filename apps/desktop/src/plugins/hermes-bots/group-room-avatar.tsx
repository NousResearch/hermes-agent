import { useValue } from '@hermes/plugin-sdk'

import { avatarColor, botAppearance, BotFace } from './avatar'
import { isBackfilledFacePng } from './avatar-image'
import type { CanonicalRoomMember } from './canonical-groups'
import { $botMeta, $lastRoster } from './data'
import { botRosterMeta, resolveBotConnectionRoute } from './routing'

interface RoomBotAvatarProps {
  connectionId?: string
  member?: CanonicalRoomMember
  name: string
  size?: number
}

/** A bot's face from its Bot Mode appearance (custom image/pet, else its shape + color). */
export function RoomBotAvatar({ connectionId, member, name, size = 24 }: RoomBotAvatarProps) {
  const meta = useValue($botMeta)
  const roster = useValue($lastRoster)
  const target = member?.target
  let own: Parameters<typeof botAppearance>[1]

  if (target?.kind === 'peer') {
    // The room's profile/handle can be renamed locally; only its pinned machine and target profile identify the peer.
    const bot = roster.find(row => {
      const route = resolveBotConnectionRoute(row).route

      return route && `install:${row.installId}` === target.installation_id && route.targetProfile === target.profile
    })

    own = bot ? botRosterMeta(bot, meta) : undefined
  } else if (connectionId && member && (!target?.kind || target.kind === 'local')) {
    const profile = typeof target?.profile === 'string' && target.profile ? target.profile : member.profile

    const bot = roster.find(row => {
      const route = resolveBotConnectionRoute(row).route

      return route?.connectionId === connectionId && route.targetProfile === profile
    })

    // Bare-name metadata belongs only to the legacy local store, never to a same-name remote member.
    own = meta[`${connectionId}::${profile}`] ?? (bot ? botRosterMeta(bot, meta) : undefined) ??
      (connectionId === 'local' ? meta[profile] : undefined)
  }

  const appearance = botAppearance(name, own)
  const photo = appearance.image && !isBackfilledFacePng(appearance.image) ? appearance.image : null

  return <BotFace color={avatarColor(appearance.color, name)} image={photo} name={name} shape={appearance.shape} size={size} />
}
