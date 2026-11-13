import { hermesApi } from '@/api/client'
import type { ComposerAttachment } from '@/store/composer'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection } from '@/store/session'

import type { SubmissionDestination } from './submission-destination'

/**
 * Upload an image attachment over HTTP for a server-owned composer queue.
 * Pin HTTP to the same owner before the native byte read yields. Images are
 * immutable admission payloads, never legacy agent.pending_images.
 */
export async function uploadServerOwnedImage(
  attachment: ComposerAttachment,
  destination: SubmissionDestination,
  path: string,
  label: string,
  sessionId: string
): Promise<ComposerAttachment> {
  const owner = destination.owner
  const connectionId = typeof owner === 'object' && owner ? owner.connectionId : $connection.get()?.connectionId

  const profile =
    typeof owner === 'object' && owner ? owner.targetProfile || owner.profile : owner || $activeGatewayProfile.get()

  const dataUrl = attachment.previewUrl?.includes(';base64,')
    ? attachment.previewUrl
    : await window.hermesDesktop?.readFileDataUrl(path)

  if (!dataUrl) {
    throw new Error(`Could not read ${label}`)
  }

  const result = await hermesApi<{ path: string; mime_type: string }>({
    method: 'POST',
    path: '/api/chat/image-upload',
    connectionId: connectionId ?? 'local',
    profile,
    body: { data_url: dataUrl, filename: label }
  })

  return {
    ...attachment,
    path: result.path,
    mime: result.mime_type,
    attachedSessionId: sessionId,
    uploadState: undefined
  }
}
