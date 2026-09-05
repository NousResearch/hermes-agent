/** One verified attachment download action shared by transcript chips and Files. */

import { Button, Codicon, host, Tip } from '@hermes/plugin-sdk'
import { useEffect, useRef, useState } from 'react'

import { readClassicAttachment } from './classic-output'
import { $groupChats } from './group-chat'
import { GroupFileError, groupFileFailure, type GroupFileFailure, withGroupFileDeadline } from './group-file-errors'
import { beginGroupFileDelivery, groupFileAccessCurrent, invalidateGroupFileAccess } from './group-files-access'
import { readHostedGroupChatAttachment } from './hosted-room-runtime'
import { useBots } from './i18n'
import type { Attachment, GroupMessage } from './types'

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
    const resolved = attachment.classicExport
      ? await withGroupFileDeadline(readClassicAttachment(group, attachment), delivery.signal)
      : attachment.data
        ? attachment
        : await withGroupFileDeadline(
            readHostedGroupChatAttachment(group, message, attachment, delivery.signal),
            delivery.signal
          )

    if (!delivery.current()) {
      return
    }

    if (!resolved.data) {
      throw new GroupFileError('gone')
    }

    const current = $groupChats.get()[group]

    if (
      current?.roomId !== room.roomId ||
      current?.hosted !== room.hosted ||
      current?.hostedEpoch !== room.hostedEpoch
    ) {
      throw new Error('Attachment scope changed.')
    }

    const link = document.createElement('a')
    link.href = resolved.data
    link.download = resolved.name || 'attachment'
    link.style.display = 'none'
    document.body.appendChild(link)

    try {
      link.click()
    } finally {
      link.remove()
    }
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
  const b = useBots()
  const [pending, setPending] = useState(false)
  const request = useRef<AbortController | null>(null)
  const name = attachment.name || b.group.attachedFile
  const label = b.group.downloadFile(name)

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
                ? b.group.fileVerificationFailed
                : failure === 'gone' || failure === 'access'
                  ? b.group.fileGone
                  : failure === 'timeout'
                    ? b.group.fileTimeout
                    : b.group.attachmentDownloadFailed
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
