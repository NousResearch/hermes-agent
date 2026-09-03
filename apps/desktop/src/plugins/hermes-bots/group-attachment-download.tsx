/** One verified attachment download action shared by transcript chips and Files. */

import { Button, Codicon, host, Tip } from '@hermes/plugin-sdk'
import { useEffect, useRef, useState } from 'react'

import { $groupChats } from './group-chat'
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
  const resolved = attachment.data ? attachment : await readHostedGroupChatAttachment(group, message, attachment)

  if (signal?.aborted) {
    return
  }

  if (!resolved.data) {
    throw new Error('Attachment data is unavailable.')
  }

  const current = $groupChats.get()[group]

  if (
    room &&
    (current?.roomId !== room.roomId || current?.hosted !== room.hosted || current?.hostedEpoch !== room.hostedEpoch)
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
}

interface GroupAttachmentDownloadProps {
  attachment: Attachment
  group: string
  message: GroupMessage
  presentation?: 'chip' | 'icon'
}

export function GroupAttachmentDownload({
  attachment,
  group,
  message,
  presentation = 'chip'
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
    if (request.current) {
      return
    }

    const controller = new AbortController()
    request.current = controller
    setPending(true)

    try {
      await downloadGroupChatAttachment(group, message, attachment, controller.signal)
    } catch {
      if (!controller.signal.aborted) {
        host.notify({ kind: 'error', message: b.group.attachmentDownloadFailed })
      }
    } finally {
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
        disabled={pending}
        onClick={() => void download()}
        size={presentation === 'chip' ? 'sm' : 'icon-xs'}
        variant="ghost"
      >
        {presentation === 'chip' ? (
          <>
            <Codicon
              name={attachment.kind === 'pdf' ? 'file-pdf' : attachment.kind === 'image' ? 'file-media' : 'file'}
            />
            <span className="min-w-0 truncate" title={name}>
              {name}
            </span>
          </>
        ) : null}
        <Codicon name={pending ? 'loading' : 'cloud-download'} spinning={pending} />
      </Button>
    </Tip>
  )
}
