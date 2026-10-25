import { $groupChats, updateGroupChat } from './group-chat'

/** Flip a group chat's "Pin to top" flag (hermes-agent#89813). The roster
 *  sort already treats `room.pinned` as the outer band for group rows, the
 *  same way `bot-meta.pinned` leads bot rows; this is the write site that flag
 *  never had. Presentation only: the room record is the group's sole durable
 *  identity, so the pin rides the local group-chats persistence like
 *  `rosterOrder` and stays out of the gateway conversation mirror.
 *  Returns the new pinned state, or null when the room is gone. */
export function toggleGroupChatPinned(name: string): boolean | null {
  const room = $groupChats.get()[name]

  if (!room || room.tombstone) {
    return null
  }

  const pinned = !room.pinned
  updateGroupChat(name, current => ({ ...current, pinned }), { sync: false })

  return pinned
}
