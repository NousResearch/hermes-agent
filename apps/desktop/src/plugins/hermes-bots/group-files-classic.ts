/** Window-local browsing of the same retained data URLs used by transcript chips. */

import { $groupChats, GROUP_CHAT_HISTORY_LIMIT, groupChatHostedGateway, groupSpeakerLabel } from './group-chat'
import { GroupFileError } from './group-file-errors'
import { GROUP_FILES_MAX_QUERY_LENGTH, GROUP_FILES_PAGE_SIZE } from './group-files-client'
import type { GroupFileItem, GroupFilesListInput, GroupFilesPage } from './group-files-client'
import type { GroupChat, GroupMessage } from './types'

export function isClassicFileRoom(room: GroupChat): boolean {
  return !groupChatHostedGateway(room) && (!room.continuityMode || room.continuityMode === 'desktop')
}

export function foldGroupFileSearch(value: string): string {
  return value
    .normalize('NFKD')
    .toLowerCase()
    .split('\u0131')
    .map(part => part.toUpperCase().toLowerCase())
    .join('\u0131')
    .replace(/\p{M}/gu, '')
}

function retainedEntries(room: GroupChat): GroupMessage[] {
  return Array.isArray(room.log) ? room.log.slice(-GROUP_CHAT_HISTORY_LIMIT * 4) : []
}

function localAttachments(entry: GroupMessage) {
  return (Array.isArray(entry.images) ? entry.images.slice(0, 8) : []).filter(
    attachment =>
      Boolean(attachment.classicExport) || (typeof attachment.data === 'string' && attachment.data.startsWith('data:'))
  )
}

export function classicLatestFileKey(room: GroupChat): string {
  const entries = retainedEntries(room)

  for (let index = entries.length - 1; index >= 0; index -= 1) {
    const entry = entries[index]
    const attachments = localAttachments(entry)

    if (attachments.length) {
      return JSON.stringify([
        entry.id || entry.eventId || [entry.at, entry.from.kind, entry.from.name],
        attachments.map(item => [item.uploadId || '', item.name, item.data?.length])
      ])
    }
  }

  return ''
}

export function latestHostedFileSeq(room: GroupChat): number {
  return retainedEntries(room).reduce(
    (latest, entry) =>
      entry.images?.length && Number.isSafeInteger(entry.seq) && Number(entry.seq) > latest
        ? Number(entry.seq)
        : latest,
    0
  )
}

export function createClassicGroupFilesLoader(group: string, roomId: string) {
  const instance = crypto.randomUUID()
  let generation = 0
  let snapshot: GroupFileItem[] = []
  let query = ''
  let snapshotKey = ''

  const load = async (requestedGroup: string, input: GroupFilesListInput = {}): Promise<GroupFilesPage> => {
    const room = $groupChats.get()[group]

    if (requestedGroup !== group || !room || String(room.roomId || '') !== roomId || !isClassicFileRoom(room)) {
      throw new GroupFileError('gone')
    }

    if ([...(input.query || '')].length > GROUP_FILES_MAX_QUERY_LENGTH) {
      throw new GroupFileError('error')
    }

    const requestedQuery = foldGroupFileSearch(input.query || '')
    const limit = input.limit ?? GROUP_FILES_PAGE_SIZE

    if (!Number.isSafeInteger(limit) || limit < 1 || limit > 32) {
      throw new GroupFileError('error')
    }

    let offset = 0

    if (input.cursor) {
      const cursor = input.cursor.split(':')

      if (
        cursor.length !== 4 ||
        cursor[0] !== 'classic' ||
        cursor[1] !== instance ||
        cursor[2] !== String(generation) ||
        query !== requestedQuery
      ) {
        throw Object.assign(new Error('attachment list cursor is invalid'), { code: 4143 })
      }

      offset = Number(cursor[3])

      if (!Number.isSafeInteger(offset) || String(offset) !== cursor[3] || offset < 1 || offset >= snapshot.length) {
        throw new GroupFileError('error')
      }
    } else {
      generation += 1
      query = requestedQuery
      snapshotKey = classicLatestFileKey(room)
      snapshot = []
      const entries = retainedEntries(room)

      for (let index = entries.length - 1; index >= 0; index -= 1) {
        const entry = entries[index]
        const label = groupSpeakerLabel(entry.from.name) + (entry.from.source ? ` (${entry.from.source})` : '')

        for (const [position, original] of localAttachments(entry).entries()) {
          if (query && !foldGroupFileSearch(`${original.name} ${label}`).includes(query)) {
            continue
          }

          const attachment = { ...original }
          const header = /^data:([^;,]{1,127});base64,/.exec(attachment.data || '')

          if (header) {
            attachment.mime ??= header[1]
            attachment.size ??=
              Math.floor(((attachment.data!.length - header[0].length) * 3) / 4) -
              (attachment.data!.endsWith('==') ? 2 : attachment.data!.endsWith('=') ? 1 : 0)
          }

          snapshot.push({
            attachment,
            eventId: entry.eventId || entry.id || '',
            seq: 0,
            sharedAt: entry.at / 1000,
            producer: { identity: entry.from.name, kind: entry.from.kind, label },
            key: `classic:${generation}:${index}:${position}`,
            localMessage: { ...entry, roomId: room.roomId || undefined }
          })
        }
      }
    }

    const items = snapshot.slice(offset, offset + limit)
    const hasMore = offset + items.length < snapshot.length

    return {
      authority: null,
      items,
      hasMore,
      nextCursor: hasMore ? `classic:${instance}:${generation}:${offset + items.length}` : null,
      snapshotSeq: 0,
      localSnapshotKey: snapshotKey
    }
  }

  return Object.assign(load, {
    clear: () => {
      generation += 1
      snapshot = []
      query = ''
      snapshotKey = ''
    }
  })
}
