/** One verified attachment download action shared by transcript chips and Files. */

import { Button, Codicon, host, Tip } from '@hermes/plugin-sdk'
import { useEffect, useRef, useState } from 'react'

import { downloadCanonicalAttachment } from './canonical-attachment-download'
import { useCanonicalFilesLabels } from './canonical-files-labels'
import { withClassicAttachmentSource } from './classic-output'
import { $groupChats } from './group-chat'
import { GroupFileError, groupFileFailure, type GroupFileFailure, withGroupFileDeadline } from './group-file-errors'
import { beginGroupFileDelivery, groupFileAccessCurrent, invalidateGroupFileAccess } from './group-files-access'
import { readHostedGroupChatAttachment } from './hosted-room-runtime'
import type { Attachment, GroupChat, GroupMessage } from './types'

function saveGroupChatAttachment(
  group: string,
  room: GroupChat,
  resolved: Attachment,
  delivery: { current: () => boolean; signal: AbortSignal },
  assertSourceCurrent?: () => void
) {
  if (!delivery.current()) {
    return
  }

  if (!resolved.data) {
    throw new GroupFileError('gone')
  }

  const current = $groupChats.get()[group]

  if (current?.roomId !== room.roomId || current?.hosted !== room.hosted || current?.hostedEpoch !== room.hostedEpoch) {
    throw new Error('Attachment scope changed.')
  }

  const encoded =
    /^data:([^;,]{1,127});base64,((?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?)$/.exec(
      resolved.data
    )

  if (!encoded || (resolved.mime && resolved.mime !== encoded[1])) {
    throw new GroupFileError('verification')
  }

  const bytes = Uint8Array.from(atob(encoded[2]), char => char.charCodeAt(0))

  if (resolved.size !== undefined && bytes.length !== resolved.size) {
    throw new GroupFileError('verification')
  }

  assertSourceCurrent?.()
  downloadCanonicalAttachment(bytes, resolved.name || 'attachment', resolved.mime || encoded[1], delivery.signal)
}

export async function downloadGroupChatAttachment(
  group: string,
  message: GroupMessage,
  attachment: Attachment,
  signal?: AbortSignal
) {
  const room = $groupChats.get()[group]

  if (!room) {
    throw new GroupFileError('gone')
  }

  const delivery = beginGroupFileDelivery(room, signal)

  try {
    if (attachment.classicExport) {
      await withGroupFileDeadline(
        withClassicAttachmentSource(group, attachment, (resolved, assertSourceCurrent) => {
          saveGroupChatAttachment(group, room, resolved, delivery, assertSourceCurrent)
        }),
        delivery.signal
      )

      return
    }

    const resolved = attachment.data
      ? attachment
      : await withGroupFileDeadline(readHostedGroupChatAttachment(group, message, attachment), delivery.signal)

    saveGroupChatAttachment(group, room, resolved, delivery)
  } catch (error) {
    if (groupFileFailure(error) === 'access') {
      invalidateGroupFileAccess(delivery.token)
    }

    if (!signal?.aborted && !groupFileAccessCurrent(delivery.token)) {
      throw new GroupFileError('access')
    }

    throw error
  } finally {
    delivery.cancel()
    delivery.release()
  }
}

interface GroupAttachmentDownloadProps {
  attachment: Attachment
  group: string
  message: GroupMessage
  presentation?: 'chip' | 'icon'
  disabled?: boolean
  onFailure?: (failure: GroupFileFailure) => void
  intentSignal?: AbortSignal
}

export function GroupAttachmentDownload({
  attachment,
  group,
  message,
  presentation = 'chip',
  disabled = false,
  onFailure,
  intentSignal
}: GroupAttachmentDownloadProps) {
  const b = useCanonicalFilesLabels()
  const [pending, setPending] = useState(false)
  const request = useRef<AbortController | null>(null)
  const name = attachment.name || 'attached file'
  const label = `${b.download} ${name}`

  // eslint-disable-next-line no-restricted-syntax -- cancels an in-flight read when its row scope changes
  useEffect(() => {
    setPending(false)

    return () => {
      request.current?.abort()
      request.current = null
    }
  }, [attachment.attachmentId, group, message.eventId, message.roomId])

  const download = async () => {
    if (request.current || intentSignal?.aborted) {
      return
    }

    const controller = new AbortController()
    const abort = () => controller.abort()
    intentSignal?.addEventListener('abort', abort, { once: true })
    request.current = controller
    setPending(true)

    try {
      await downloadGroupChatAttachment(group, message, attachment, controller.signal)
    } catch (error) {
      if (!controller.signal.aborted) {
        const failure = groupFileFailure(error)

        if (onFailure) {
          onFailure(failure)
        } else {
          host.notify({
            kind: 'error',
            message:
              failure === 'verification'
                ? b.fileVerificationFailed
                : failure === 'gone' || failure === 'access'
                  ? b.fileGone
                  : failure === 'timeout'
                    ? b.fileTimeout
                    : b.attachmentDownloadFailed
          })
        }
      }
    } finally {
      intentSignal?.removeEventListener('abort', abort)

      if (request.current === controller) {
        request.current = null
        setPending(false)
      }
    }
  }

  return (
    <Tip label={label}>
      <Button
        aria-busy={pending}
        aria-label={label}
        className={
          presentation === 'chip'
            ? 'max-w-60 gap-1 border border-(--ui-stroke-tertiary) text-[0.65rem] text-(--ui-text-tertiary)'
            : 'text-(--ui-text-tertiary) hover:text-foreground'
        }
        data-file-download="true"
        disabled={pending || disabled}
        onClick={() => void download()}
        size={presentation === 'chip' ? 'sm' : 'icon-xs'}
        variant="ghost"
      >
        {presentation === 'chip' ? (
          <>
            <Codicon
              name={attachment.kind === 'pdf' ? 'file-pdf' : attachment.kind === 'image' ? 'file-media' : 'file'}
            />
            <bdi className="min-w-0 truncate" title={name}>
              {name}
            </bdi>
          </>
        ) : null}
        <Codicon name={pending ? 'loading' : 'cloud-download'} spinning={pending} />
      </Button>
    </Tip>
  )
}
