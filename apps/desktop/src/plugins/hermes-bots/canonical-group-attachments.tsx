import { Button } from '@hermes/plugin-sdk'
import { useLayoutEffect, useRef, useState } from 'react'

import { downloadCanonicalAttachment } from './canonical-attachment-download'
import { useCanonicalGroupLabels } from './canonical-group-labels'
import { type CanonicalGroupBinding, canonicalGroupRequest } from './canonical-groups'

export interface CanonicalGroupAttachment { attachment_id?: string; event_id?: string; kind: string; name: string; mime: string; size?: number }
type Attachment = CanonicalGroupAttachment
interface DownloadedAttachment extends Attachment { data_base64: string }

function kindFor(file: File): string {
  if (file.type.startsWith('image/')) {return 'image'}

  if (file.type === 'application/pdf') {return 'pdf'}

  return 'file'
}

export function CanonicalGroupAttachments({ binding, attachments, onChange, disabled, readOnly = false }: {
  binding: CanonicalGroupBinding
  attachments: Attachment[]
  disabled: boolean
} & ({ readOnly: true; onChange?: never } | { readOnly?: false; onChange: (attachments: Attachment[]) => void })) {
  const labels = useCanonicalGroupLabels()
  const input = useRef<HTMLInputElement>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const lifetime = useRef<AbortController | null>(null)

  useLayoutEffect(() => {
    const controller = new AbortController()
    lifetime.current = controller

    if (disabled) {controller.abort()}
    setBusy(false)

    return () => controller.abort()
  }, [binding.connectionId, binding.profile, binding.roomId, disabled, readOnly])

  async function upload(file: File) {
    if (readOnly) {return}
    setBusy(true); setError('')

    try {
      const data = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader()
        reader.onload = () => resolve(String(reader.result).split(',', 2)[1])
        reader.onerror = () => reject(reader.error ?? new Error(labels.uploadFailed))
        reader.readAsDataURL(file)
      })

      const result = await canonicalGroupRequest<Attachment>(binding, 'groups.attachment.upload', {
        room_id: binding.roomId, upload_id: crypto.randomUUID(), kind: kindFor(file), name: file.name,
        mime: file.type || 'application/octet-stream', data_base64: data
      })

      // Upload receipts include storage metadata; Send accepts only the manifest.
      const { attachment_id, kind, name, mime, size } = result
      onChange?.([...attachments, { attachment_id, kind, name, mime, size }])
    } catch (e) { setError(e instanceof Error ? e.message : String(e)) }
    finally { setBusy(false) }
  }

  async function download(attachment: Attachment) {
    const signal = lifetime.current?.signal

    if (!signal || signal.aborted || !attachment.attachment_id || !attachment.event_id) {return}
    setBusy(true); setError('')

    try {
      const result = await canonicalGroupRequest<DownloadedAttachment>(binding, 'groups.attachment.download', {
        room_id: binding.roomId, event_id: attachment.event_id, attachment_id: attachment.attachment_id
      })

      if (signal.aborted) {return}
      const bytes = Uint8Array.from(atob(result.data_base64), char => char.charCodeAt(0))
      downloadCanonicalAttachment(bytes, result.name, result.mime, signal)
    } catch (e) { if (!signal.aborted) {setError(e instanceof Error ? e.message : String(e))} }
    finally { if (!signal.aborted) {setBusy(false)} }
  }

  return <div className="flex flex-wrap items-center gap-2">
    {!readOnly && <><input hidden onChange={e => { const file = e.target.files?.[0];

 if (file) {void upload(file);} e.currentTarget.value = '' }} ref={input} type="file" />
    <Button disabled={disabled || busy} onClick={() => input.current?.click()} type="button">{labels.attachFiles}</Button></>}
    {attachments.map(a => <span className="flex items-center gap-1" key={a.attachment_id ?? a.name}>
      <span>{a.name}</span><Button disabled={disabled || busy} onClick={() => void download(a)} type="button">{labels.download}</Button>
      {!readOnly && <Button disabled={disabled || busy} onClick={() => onChange?.(attachments.filter(item => item !== a))} type="button">{labels.removeAttachment}</Button>}
    </span>)}
    {error && <span role="alert">{labels.uploadFailed}: {error}</span>}
  </div>
}
