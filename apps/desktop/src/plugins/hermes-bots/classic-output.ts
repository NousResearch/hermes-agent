/** Explicit producer-custodied files; classic Desktop remains the round driver. */
import { host } from '@hermes/plugin-sdk'

import { $groupChats } from './group-chat'
import { GroupFileDeliveryError } from './group-file-delivery'
import { groupMemberKey } from './group-membership'
import { botConnectionRoute, requestForBot } from './routing'
import type { Attachment, GroupChat, GroupMember } from './types'

interface Recipient {
  installation: string
  profile: string
}
export interface ClassicTurn {
  request: { request_id: string; group_id: string; thread_id: string; recipients: Recipient[]; issued_at: number }
  installation: string
  source: GroupMember
  session: string
  epoch: number
  anchor?: string
  exportId?: string
}
export interface ClassicFileRef {
  group: string
  exportId: string
  artifactId: string
  generation: number
  installation: string
  source: GroupMember
  session: string
  sha256: string
  recipients: Recipient[]
}
export interface ClassicReply {
  text: string
  images: Attachment[]
  entryId: string
}
export class ClassicTurnEndedError extends GroupFileDeliveryError {}
export type GroupTurnReply = string | ClassicReply
export const groupTurnText = (reply: GroupTurnReply): string => (typeof reply === 'string' ? reply : reply.text)
const profileFor = (member: GroupMember) => botConnectionRoute(member)?.targetProfile || member.name

async function capability(member: GroupMember): Promise<{ installation: string } | null> {
  try {
    const result = (await requestForBot(member, 'gateway.capabilities', {})) as Record<string, unknown>

    return result?.classic_output_export_v1 === true && typeof result.installation === 'string'
      ? { installation: result.installation }
      : null
  } catch (error: any) {
    if (error?.code === -32601) {
      return null
    }

    throw new GroupFileDeliveryError('The file-capable source could not be checked. Reconnect and retry.')
  }
}

export async function beginClassicTurn(group: string, member: GroupMember, session: string, thread: string) {
  const room = $groupChats.get()[group]

  if (!room?.roomId || room.hosted) {
    return null
  }

  const sourceCapability = await capability(member).catch(() => null)

  if (!sourceCapability) {
    return null
  }

  const recipients: Recipient[] = []

  for (const target of room.members || []) {
    const supported = await capability(target).catch(() => null)

    if (!supported) {
      return null
    }

    recipients.push({ installation: supported.installation, profile: profileFor(target) })
  }

  if (!recipients.length) {
    return null
  }

  return {
    request: {
      request_id: crypto.randomUUID(),
      group_id: room.roomId,
      thread_id: thread,
      recipients,
      issued_at: Date.now() / 1000
    },
    installation: sourceCapability.installation,
    session,
    epoch: room.epoch || 0,
    anchor: room.log.at(-1)?.id,
    source: { name: member.name, connectionId: member.connectionId, remoteSource: member.remoteSource }
  } satisfies ClassicTurn
}

export async function readClassicTurn(
  member: GroupMember,
  runtime: string,
  turn: ClassicTurn
): Promise<ClassicReply | null> {
  const result = (await requestForBot(member, 'session.export.read', {
    session_id: runtime,
    installation: turn.installation,
    group_id: turn.request.group_id,
    ...(turn.exportId ? { export_id: turn.exportId } : { request_id: turn.request.request_id })
  })) as Record<string, any>

  if (result.state === 'running') {
    return null
  }

  if (!['published', 'settled'].includes(result.state)) {
    throw new ClassicTurnEndedError('The file-producing turn did not complete.')
  }

  if (
    result.group_id !== turn.request.group_id ||
    result.thread_id !== turn.request.thread_id ||
    !Number.isSafeInteger(result.generation) ||
    result.generation < 1 ||
    typeof result.export_id !== 'string' ||
    !Array.isArray(result.items) ||
    result.items.length > 8 ||
    JSON.stringify(result.recipients) !== JSON.stringify(turn.request.recipients)
  ) {
    throw new GroupFileDeliveryError('The file export scope could not be verified.')
  }

  const images: Attachment[] = result.items.map((item: any) => {
    if (
      !['file', 'pdf', 'image'].includes(item.kind) ||
      typeof item.name !== 'string' ||
      item.name.length > 255 ||
      typeof item.artifact_id !== 'string' ||
      !Number.isSafeInteger(item.size) ||
      item.size < 1 ||
      item.size > 15_000_000 ||
      !/^[a-f0-9]{64}$/.test(item.sha256) ||
      typeof item.mime !== 'string' ||
      !/^[\w.+-]+\/[\w.+-]+$/.test(item.mime)
    ) {
      throw new GroupFileDeliveryError('The file manifest could not be verified.')
    }

    return {
      kind: item.kind,
      name: item.name,
      mime: item.mime,
      size: item.size,
      classicExport: {
        group: result.group_id,
        exportId: result.export_id,
        artifactId: item.artifact_id,
        generation: result.generation,
        installation: turn.installation,
        source: turn.source,
        session: turn.session,
        sha256: item.sha256,
        recipients: turn.request.recipients
      }
    }
  })

  return { text: String(result.text || ''), images, entryId: `classic-export:${result.export_id}` }
}

async function withSource<T>(source: GroupMember, session: string, fn: (runtime: string) => Promise<T>) {
  const route = botConnectionRoute(source)
  const release = route && typeof host.retainProfile === 'function' ? await host.retainProfile(route) : () => undefined
  let temporary: string | undefined

  try {
    let resumed: { session_id: string }

    try {
      resumed = await requestForBot(source, 'session.resume', {
        session_id: session,
        profile: source.name,
        omit_messages: true
      })
    } catch (error: any) {
      if (error?.code !== 4007) {
        throw error
      }

      // Published custody outlives the producing conversation. This lazy owner
      // session performs only file RPCs; no agent/model or new group is started.
      resumed = await requestForBot(source, 'session.create', { profile: source.name, hidden: true })
      temporary = resumed.session_id
    }

    return await fn(resumed.session_id)
  } finally {
    try {
      if (temporary) {
        await requestForBot(source, 'session.close', { session_id: temporary })
      }
    } finally {
      release()
    }
  }
}

export async function readClassicAttachment(
  group: string,
  attachment: Attachment,
  recipient?: GroupMember
): Promise<Attachment> {
  const ref = attachment.classicExport
  const room = $groupChats.get()[group]
  const current = () => $groupChats.get()[group]?.roomId === ref?.group && !$groupChats.get()[group]?.tombstone

  if (!ref || !room || room.roomId !== ref.group || !current()) {
    throw new GroupFileDeliveryError('File group changed.')
  }

  if (recipient) {
    const target = await capability(recipient)

    if (
      !target ||
      !(room.members || []).some(member => groupMemberKey(member) === groupMemberKey(recipient)) ||
      !ref.recipients.some(
        member => member.installation === target.installation && member.profile === profileFor(recipient)
      )
    ) {
      throw new GroupFileDeliveryError('This member was not a recipient of the shared file.')
    }
  }

  const response = (await withSource(ref.source, ref.session, runtime =>
    requestForBot(ref.source, 'session.export.read', {
      session_id: runtime,
      installation: ref.installation,
      group_id: ref.group,
      export_id: ref.exportId,
      artifact_id: ref.artifactId
    })
  )) as Record<string, any>

  const item = response.item

  if (
    response.generation !== ref.generation ||
    response.group_id !== ref.group ||
    response.export_id !== ref.exportId ||
    !item ||
    item.artifact_id !== ref.artifactId ||
    item.sha256 !== ref.sha256 ||
    item.name !== attachment.name ||
    item.kind !== attachment.kind ||
    item.mime !== attachment.mime ||
    item.size !== attachment.size ||
    typeof response.content_base64 !== 'string' ||
    response.content_base64.length > 20_000_000 ||
    !/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(response.content_base64)
  ) {
    throw new GroupFileDeliveryError('File verification failed.')
  }

  const bytes = Uint8Array.from(atob(response.content_base64), c => c.charCodeAt(0))

  const digest = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), c =>
    c.toString(16).padStart(2, '0')
  ).join('')

  if (
    bytes.length !== attachment.size ||
    digest !== ref.sha256 ||
    !current() ||
    (recipient &&
      !($groupChats.get()[group]?.members || []).some(member => groupMemberKey(member) === groupMemberKey(recipient)))
  ) {
    throw new GroupFileDeliveryError('File bytes or recipient changed.')
  }

  return { ...attachment, data: `data:${attachment.mime};base64,${response.content_base64}` }
}

export async function retireClassicGroup(room: GroupChat) {
  if (!room.roomId || room.hosted) {
    return
  }

  const sources = new Map<string, GroupMember>()
  const references = new Map<string, { session: string; installation: string }>()

  for (const entry of room.log || []) {
    for (const attachment of entry.images || []) {
      if (attachment.classicExport) {
        const ref = attachment.classicExport
        sources.set(groupMemberKey(ref.source), ref.source)
        references.set(groupMemberKey(ref.source), ref)
      }
    }
  }

  for (const marker of Object.values(room.stranded || {})) {
    if (typeof marker === 'object' && marker.classicTurn) {
      const turn = marker.classicTurn
      sources.set(groupMemberKey(turn.source), turn.source)
      references.set(groupMemberKey(turn.source), turn)
    }
  }

  for (const source of sources.values()) {
    const supported = await capability(source)
    const reference = references.get(groupMemberKey(source))

    if (reference && supported?.installation !== reference.installation) {
      throw new GroupFileDeliveryError('Reconnect the original updated producer before retiring this group.')
    }

    if (!supported) {
      continue
    }

    const session = reference?.session || room.sessions?.[groupMemberKey(source)]

    if (typeof session !== 'string') {
      continue
    }

    await withSource(source, session, runtime =>
      requestForBot(source, 'session.export.discard', {
        session_id: runtime,
        installation: supported.installation,
        group_id: room.roomId
      })
    )
  }
}
