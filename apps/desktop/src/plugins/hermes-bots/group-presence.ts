import { computed } from '@hermes/plugin-sdk'

import { botRosterKey } from './data'
import { $groupChats } from './group-chat'

/** All presence consumers share the exact owner captured by the room driver. */
export const $activeGroupMemberKeys = computed(
  $groupChats,
  rooms =>
    new Set(
      Object.values(rooms).flatMap(room =>
        room.running && !room.tombstone && room.turn ? [botRosterKey(room.turn)] : []
      )
    )
)
