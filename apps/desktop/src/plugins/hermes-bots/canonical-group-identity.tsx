import { avatarColor, botAppearance, BotFace } from './avatar'
import type { CanonicalRoomMember } from './canonical-groups'

/** Identity is supplied by this group chat's owner, never the foreground roster. */
export function canonicalMemberName(member: CanonicalRoomMember | undefined, fallback: string) {
  return member?.display_name?.trim() || member?.handle?.trim() || fallback
}

export function CanonicalMemberFace({ member, seed, name, size = 24 }: {
  member?: CanonicalRoomMember; seed?: string; name: string; size?: number
}) {
  const identity = member?.member_id || seed || name
  const appearance = botAppearance(identity, undefined)

  return <BotFace color={avatarColor(appearance.color, identity)} name={identity} shape={appearance.shape} size={size} />
}
