import { $groupChats, updateGroupChat } from './group-chat'
import type { HostedRoomCapability } from './hosted-room-client'
import { withoutHostedMemberRoute } from './hosted-room-members'
import type { ProfileRoute } from './types'

/** Apply affirmative inventory evidence before awaiting any authority state.
 * A failed room read must not keep a deleted/rebound connection routable. */
export function revokeInvalidHostedMemberRoutes(
  routes: Record<string, ProfileRoute>,
  capabilities: Record<string, HostedRoomCapability>,
  invalidate: (roomId: string) => void
): void {
  for (const [name, room] of Object.entries($groupChats.get())) {
    if (!room.hosted) {
      continue
    }

    let changed = false

    const members = (room.members || []).map(member => {
      const connectionId = member.route?.connectionId || member.connectionId

      if (!connectionId) {
        return member
      }

      const installation = member.hostedIdentity?.installationId
      const currentInstallation = capabilities[connectionId]?.authorityId

      if (routes[connectionId] && !(installation && currentInstallation && installation !== currentInstallation)) {
        return member
      }

      changed = true

      return withoutHostedMemberRoute(member)
    })

    if (changed) {
      invalidate(String(room.roomId || ''))
      updateGroupChat(name, current => ({ ...current, members }), { sync: false })
    }
  }
}
