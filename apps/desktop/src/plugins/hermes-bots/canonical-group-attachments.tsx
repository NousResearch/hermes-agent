import { Button, cn, Codicon, GHOST_ICON_BTN, randomUuid, Tip } from '@hermes/plugin-sdk'
import { type Ref, useImperativeHandle, useLayoutEffect, useRef, useState } from 'react'

import { downloadCanonicalAttachment } from './canonical-attachment-download'
import { useCanonicalGroupLabels } from './canonical-group-labels'
import { type CanonicalGroupBinding, canonicalGroupRequest } from './canonical-groups'
import { ROOM_COMPOSER_AREAS } from './group-room-ui'

export interface CanonicalGroupAttachment {
  attachment_id?: string
  event_id?: string
  kind: string
  name: string
  mime: string
  size?: number
}
type Attachment = CanonicalGroupAttachment
interface DownloadedAttachment extends Attachment {
  data_base64: string
}
export interface CanonicalGroupUploader {
  upload: (files: File[]) => void
}

export async function fetchCanonicalGroupAttachment(
  binding: CanonicalGroupBinding,
  eventId: string,
  attachmentId: string
) {
  const result = await canonicalGroupRequest<DownloadedAttachment>(binding, 'groups.attachment.download', {
    room_id: binding.roomId,
    event_id: eventId,
    attachment_id: attachmentId
  })

  return {
    name: result.name,
    mime: result.mime,
    bytes: Uint8Array.from(atob(result.data_base64), char => char.charCodeAt(0))
  }
}

function kindFor(file: File): string {
  if (file.type.startsWith('image/')) {
    return 'image'
  }

  if (file.type === 'application/pdf') {
    return 'pdf'
  }

  return 'file'
}

export function CanonicalGroupAttachments({
  binding,
  attachments,
  onChange,
  onUploadingChange,
  disabled,
  readOnly = false,
  ref
}: {
  binding: CanonicalGroupBinding
  attachments: Attachment[]
  disabled: boolean
  // C21: the form holds Send while a selected file's manifest is not committed yet.
  onUploadingChange?: (uploading: boolean) => void
  // Paste and drop in the composer reuse this upload path.
  ref?: Ref<CanonicalGroupUploader>
} & ({ readOnly: true; onChange?: never } | { readOnly?: false; onChange: (attachments: Attachment[]) => void })) {
  const labels = useCanonicalGroupLabels()
  const input = useRef<HTMLInputElement>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const lifetime = useRef<AbortController | null>(null)
  // Files still to upload in the running batch; a paste or drop during it joins instead of being lost.
  const queued = useRef<File[] | null>(null)

  useLayoutEffect(() => {
    const controller = new AbortController()
    lifetime.current = controller

    if (disabled) {
      controller.abort()
    }
    setBusy(false)

    return () => controller.abort()
  }, [binding.connectionId, binding.profile, binding.roomId, disabled, readOnly])

  useImperativeHandle(ref, () => ({ upload: files => void upload(files) }))

  async function upload(files: File[]) {
    if (readOnly) {
      return
    }

    if (queued.current) {
      queued.current.push(...files)

      return
    }

    const pending = [...files]
    queued.current = pending
    setBusy(true)
    setError('')
    onUploadingChange?.(true)
    let next = attachments

    try {
      for (let file = pending.shift(); file; file = pending.shift()) {
        const data = await new Promise<string>((resolve, reject) => {
          const reader = new FileReader()
          reader.onload = () => resolve(String(reader.result).split(',', 2)[1])
          reader.onerror = () => reject(reader.error ?? new Error(labels.uploadFailed))
          reader.readAsDataURL(file)
        })

        const result = await canonicalGroupRequest<Attachment>(binding, 'groups.attachment.upload', {
          room_id: binding.roomId,
          upload_id: randomUuid(),
          kind: kindFor(file),
          name: file.name,
          mime: file.type || 'application/octet-stream',
          data_base64: data
        })

        // Upload receipts include storage metadata; Send accepts only the manifest.
        const { attachment_id, kind, name, mime, size } = result
        next = [...next, { attachment_id, kind, name, mime, size }]
        onChange?.(next)
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      queued.current = null
      setBusy(false)
      onUploadingChange?.(false)
    }
  }

  async function download(attachment: Attachment) {
    const signal = lifetime.current?.signal

    if (!signal || signal.aborted || !attachment.attachment_id || !attachment.event_id) {
      return
    }
    setBusy(true)
    setError('')

    try {
      const result = await fetchCanonicalGroupAttachment(binding, attachment.event_id, attachment.attachment_id)

      if (signal.aborted) {
        return
      }
      downloadCanonicalAttachment(result.bytes, result.name, result.mime, signal)
    } catch (e) {
      if (!signal.aborted) {
        setError(e instanceof Error ? e.message : String(e))
      }
    } finally {
      if (!signal.aborted) {
        setBusy(false)
      }
    }
  }

  const chips = (attachments.length > 0 || error) && (
    <div
      className={cn('flex flex-wrap items-center gap-1.5', !readOnly && ROOM_COMPOSER_AREAS.files, readOnly && 'mt-1')}
    >
      {attachments.map(a => (
        <span
          className="inline-flex min-w-0 items-center gap-1 rounded-md border border-(--ui-stroke-tertiary) py-0.5 pl-1.5 pr-0.5 text-[0.6875rem] text-(--ui-text-secondary)"
          key={a.attachment_id ?? a.name}
        >
          <Codicon
            className="shrink-0 text-(--ui-text-tertiary)"
            name={a.kind === 'image' ? 'file-media' : a.kind === 'pdf' ? 'file-pdf' : 'file'}
          />
          <span className="max-w-48 truncate">{a.name}</span>
          <Tip label={labels.download}>
            <Button
              aria-label={labels.download}
              disabled={disabled || busy}
              onClick={() => void download(a)}
              size="icon-xs"
              type="button"
              variant="ghost"
            >
              <Codicon name="cloud-download" />
            </Button>
          </Tip>
          {!readOnly && (
            <Tip label={labels.removeAttachment}>
              <Button
                aria-label={labels.removeAttachment}
                disabled={disabled || busy}
                onClick={() => onChange?.(attachments.filter(item => item !== a))}
                size="icon-xs"
                type="button"
                variant="ghost"
              >
                <Codicon name="close" />
              </Button>
            </Tip>
          )}
        </span>
      ))}
      {error && (
        <span className="text-[0.6875rem] text-destructive" role="alert">
          {labels.uploadFailed}: {error}
        </span>
      )}
    </div>
  )

  if (readOnly) {
    return chips || null
  }

  return (
    <>
      <input
        hidden
        onChange={e => {
          const file = e.target.files?.[0]

          if (file) {
            void upload([file])
          }
          e.currentTarget.value = ''
        }}
        ref={input}
        type="file"
      />
      <Tip label={labels.attachFiles}>
        <Button
          aria-label={labels.attachFiles}
          className={cn(GHOST_ICON_BTN, ROOM_COMPOSER_AREAS.button)}
          disabled={disabled || busy}
          onClick={() => input.current?.click()}
          size="icon"
          type="button"
          variant="ghost"
        >
          <Codicon name="add" size="0.875rem" />
        </Button>
      </Tip>
      {chips}
    </>
  )
}
