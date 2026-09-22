/** Exact retained-file reader from #104199; no producer/session creation. */
import { host } from '@hermes/plugin-sdk'

import { $groupChats } from './group-chat'
import { GroupFileDeliveryError } from './group-file-delivery'
import { groupMemberKey } from './group-membership'
import { botConnectionRoute, requestForBot } from './routing'
import type { Attachment, GroupMember } from './types'

interface Recipient {
  installation: string
  profile: string
}

/** Non-secret, explicitly published producer reference. Raw bytes are never persisted in it. */
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

async function withOriginalSource<T>(
  source: GroupMember,
  session: string,
  fn: (runtime: string) => Promise<T>
): Promise<T> {
  const route = botConnectionRoute(source)
  const release = route && typeof host.retainProfile === 'function' ? await host.retainProfile(route) : () => undefined

  try {
    let resumed: { session_id: string }

    try {
      resumed = await requestForBot(source, 'session.resume', {
        session_id: session,
        profile: source.name,
        omit_messages: true
      })
    } catch (error: any) {
      if (error?.code === 4007) {
        // The accepted reader authorizes the original producer session (or its
        // unique compression lineage). Never mint an unrelated replacement.
        throw new GroupFileDeliveryError('The original producer session is unavailable.')
      }

      throw error
    }

    if (!resumed || typeof resumed.session_id !== 'string' || !resumed.session_id) {
      throw new GroupFileDeliveryError('The original producer session is unavailable.')
    }

    return await fn(resumed.session_id)
  } finally {
    release()
  }
}

/** Read exactly one retained export reference through the pinned producer owner.
 * The original group, session, installation and generation are all mandatory;
 * no foreground route, temporary session or generation inference is permitted. */
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

  if (
    !Number.isSafeInteger(ref.generation) ||
    ref.generation < 1 ||
    !ref.session ||
    !ref.installation ||
    !ref.exportId ||
    !ref.artifactId ||
    !/^[a-f0-9]{64}$/.test(ref.sha256) ||
    !Array.isArray(ref.recipients)
  ) {
    throw new GroupFileDeliveryError('File reference could not be verified.')
  }

  if (recipient) {
    const recipientRoute = botConnectionRoute(recipient)
    const profile = recipientRoute?.targetProfile || recipient.name
    const capability = (await requestForBot(recipient, 'gateway.capabilities', {})) as Record<string, unknown>
    const installation = capability?.classic_output_export_v1 === true ? capability.installation : null

    if (
      typeof installation !== 'string' ||
      !(room.members || []).some(member => groupMemberKey(member) === groupMemberKey(recipient)) ||
      !ref.recipients.some(member => member.installation === installation && member.profile === profile)
    ) {
      throw new GroupFileDeliveryError('This member was not a recipient of the shared file.')
    }
  }

  const response = (await withOriginalSource(ref.source, ref.session, runtime =>
    requestForBot(ref.source, 'session.export.read', {
      session_id: runtime,
      installation: ref.installation,
      group_id: ref.group,
      export_id: ref.exportId,
      artifact_id: ref.artifactId,
      generation: ref.generation
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

  const bytes = Uint8Array.from(atob(response.content_base64), char => char.charCodeAt(0))

  const digest = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), byte =>
    byte.toString(16).padStart(2, '0')
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
