import type { GroupFilesPage } from './group-files-client'
import type { GroupChat } from './types'

export const FILE_ROOM: GroupChat = {
  continuityMode: 'gateway',
  hosted: 'install:home',
  hostedConnectionId: 'gateway-a',
  hostedEpoch: 1,
  hostedSeq: 20,
  log: [],
  roomId: 'room-1',
  watermarks: {}
}

export const fileItem = (seq = 20, name = `file-${seq}.txt`, id = seq) => ({
  attachment_id: `att_${id.toString(16).padStart(32, '0')}`,
  event_id: `event-${seq}`,
  seq,
  kind: 'file',
  name,
  mime: 'text/plain',
  size: 1,
  producer: { kind: 'member', id: 'builder', label: 'Builder' },
  shared_at: 1_700_000_000 + seq
})

export const filePage = (items = [fileItem()], hasMore = false, snapshotSeq = 20) => ({
  authority: { gateway_id: 'install:home', epoch: 1 },
  has_more: hasMore,
  items,
  next_cursor: hasMore ? `cursor-after-${items.at(-1)?.seq ?? snapshotSeq}` : null,
  snapshot_seq: snapshotSeq
})

export const parsedFilePage = (items = [fileItem()], hasMore = false, snapshotSeq = 20): GroupFilesPage => ({
  authority: { epoch: 1, gatewayId: 'install:home' },
  hasMore,
  items: items.map(item => ({
    attachment: {
      attachmentId: item.attachment_id,
      kind: item.kind as 'file',
      mime: item.mime,
      name: item.name,
      size: item.size
    },
    eventId: item.event_id,
    producer: { identity: item.producer.id, kind: 'member', label: item.producer.label },
    seq: item.seq,
    sharedAt: item.shared_at
  })),
  nextCursor: hasMore ? `cursor-after-${items.at(-1)?.seq ?? snapshotSeq}` : null,
  snapshotSeq
})

export function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason: unknown) => void

  const promise = new Promise<T>((yes, no) => {
    resolve = yes
    reject = no
  })

  return { promise, reject, resolve }
}
