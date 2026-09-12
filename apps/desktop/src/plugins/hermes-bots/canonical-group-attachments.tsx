import { Button } from '@hermes/plugin-sdk'
import { useRef, useState } from 'react'

import { useCanonicalGroupLabels } from './canonical-group-labels'
import { type CanonicalGroupBinding, canonicalGroupRequest } from './canonical-groups'

interface Attachment { attachment_id?: string; event_id?: string; kind: string; name: string; mime: string; size?: number }
interface DownloadedAttachment extends Attachment { data_base64: string }

function kindFor(file: File): string {
  if (file.type.startsWith('image/')) {return 'image'}

  if (file.type === 'application/pdf') {return 'pdf'}

  return 'file'
}

function extension(mime: string, name: string): string {
  const dot = name.lastIndexOf('.')

  if (dot > 0) {return name.slice(dot)}

  return mime === 'application/pdf' ? '.pdf' : mime.split('/')[1] ? `.${mime.split('/')[1]}` : '.bin'
}

export function CanonicalGroupAttachments({ binding, attachments, onChange, disabled }: {
  binding: CanonicalGroupBinding
  attachments: Attachment[]
  onChange: (attachments: Attachment[]) => void
  disabled: boolean
}) {
  const labels = useCanonicalGroupLabels()
  const input = useRef<HTMLInputElement>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  async function upload(file: File) {
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
      onChange([...attachments, { attachment_id, kind, name, mime, size }])
    } catch (e) { setError(e instanceof Error ? e.message : String(e)) }
    finally { setBusy(false) }
  }

  async function download(attachment: Attachment) {
    if (!attachment.attachment_id || !attachment.event_id || !window.hermesDesktop?.saveImageBuffer) {return}
    setBusy(true); setError('')

    try {
      const result = await canonicalGroupRequest<DownloadedAttachment>(binding, 'groups.attachment.download', {
        room_id: binding.roomId, event_id: attachment.event_id, attachment_id: attachment.attachment_id
      })

      const bytes = Uint8Array.from(atob(result.data_base64), char => char.charCodeAt(0))
      await window.hermesDesktop.saveImageBuffer(bytes, extension(result.mime, result.name), result.name)
    } catch (e) { setError(e instanceof Error ? e.message : String(e)) }
    finally { setBusy(false) }
  }

  return <div className="flex flex-wrap items-center gap-2">
    <input hidden onChange={e => { const file = e.target.files?.[0];

 if (file) {void upload(file);} e.currentTarget.value = '' }} ref={input} type="file" />
    <Button disabled={disabled || busy} onClick={() => input.current?.click()} type="button">{labels.attachFiles}</Button>
    {attachments.map(a => <span className="flex items-center gap-1" key={a.attachment_id ?? a.name}>
      <span>{a.name}</span><Button disabled={disabled || busy} onClick={() => void download(a)} type="button">{labels.download}</Button>
      <Button disabled={disabled || busy} onClick={() => onChange(attachments.filter(item => item !== a))} type="button">{labels.removeAttachment}</Button>
    </span>)}
    {error && <span role="alert">{labels.uploadFailed}: {error}</span>}
  </div>
}
