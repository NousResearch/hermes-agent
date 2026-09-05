/** The stored-file read path, separate from room execution/control eligibility. */

import { $groupChats } from './group-chat'
import { assertGroupFileIntent } from './group-file-errors'
import { captureGroupFileAccess, guardGroupFileRequest } from './group-files-access'
import { readHostedMessageAttachment } from './hosted-room-attachments-client'
import { requestHostedConnection } from './hosted-room-transport'
import type { Attachment, GroupChat, GroupMessage, ProfileRoute } from './types'

type Resolve = (room: GroupChat, purpose: 'read', signal?: AbortSignal) => Promise<ProfileRoute | null>

export async function readHostedGroupAttachment(
  group: string,
  message: GroupMessage,
  attachment: Attachment,
  resolve: Resolve,
  signal?: AbortSignal
) {
  assertGroupFileIntent(signal)
  const room = $groupChats.get()[group]
  const route = room ? await resolve(room, 'read', signal) : null
  assertGroupFileIntent(signal)
  const roomId = String(room?.roomId || '')
  const eventId = String(message.eventId || message.id || '')
  const current = $groupChats.get()[group]

  if (
    !roomId ||
    !route ||
    (message.roomId && message.roomId !== roomId) ||
    current?.roomId !== roomId ||
    current?.hosted !== room?.hosted ||
    current?.hostedEpoch !== room?.hostedEpoch
  ) {
    throw new Error('This Group Chat attachment is unavailable.')
  }

  return readHostedMessageAttachment(
    guardGroupFileRequest(requestHostedConnection, captureGroupFileAccess(room), signal),
    route,
    roomId,
    eventId,
    attachment
  )
}
