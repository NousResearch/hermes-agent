import { Button, Codicon, Tip } from '@hermes/plugin-sdk'
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

export function CanonicalGroupAttachments({ binding, attachments, onChange, onUploadingChange, disabled, readOnly = false }: {
  binding: CanonicalGroupBinding
  attachments: Attachment[]
  disabled: boolean
  onUploadingChange?: (uploading: boolean) => void
} & ({ readOnly: true; onChange?: never } | { readOnly?: false; onChange: (attachments: Attachment[]) => void })) {
  const labels = useCanonicalGroupLabels()
  const input = useRef<HTMLInputElement>(null)
  const [error, setError] = useState('')
  const [errorKind, setErrorKind] = useState<'upload' | 'download'>('upload')
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
    if (readOnly || disabled || busy) {return}
    setBusy(true); setError('')
    setErrorKind('upload')
    onUploadingChange?.(true)

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
    finally { setBusy(false); onUploadingChange?.(false) }
  }

  async function download(attachment: Attachment) {
    const signal = lifetime.current?.signal

    if (!signal || signal.aborted || !attachment.attachment_id || !attachment.event_id) {return}
    setBusy(true); setError('')
    setErrorKind('download')

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

  return <div className="flex min-w-0 flex-wrap items-center gap-2">
    {!readOnly && <><input disabled={disabled || busy} hidden onChange={e => { const file = e.target.files?.[0];

 if (file) {void upload(file);} e.currentTarget.value = '' }} ref={input} type="file" />
    <Tip label={labels.attachFiles}><Button aria-label={labels.attachFiles} disabled={disabled || busy} loading={busy}
      onClick={() => input.current?.click()} size="icon-xs" type="button" variant="ghost"><Codicon name="attach" /></Button></Tip></>}
    {attachments.map(a => <div className="flex min-w-0 max-w-full items-center gap-2 rounded-md border border-(--ui-stroke-tertiary) px-2 py-1" key={a.attachment_id ?? a.name}>
      <Codicon className="shrink-0 text-(--ui-text-tertiary)" name={a.kind === 'image' ? 'file-media' : 'file'} />
      <span className="min-w-0 truncate text-xs text-(--ui-text-primary)">{a.name}</span>
      {typeof a.size === 'number' && <span className="shrink-0 text-[length:var(--conversation-tool-font-size)] text-(--ui-text-quaternary)">
        {new Intl.NumberFormat(undefined, { style: 'unit', unit: a.size >= 1_048_576 ? 'megabyte' : 'kilobyte', maximumFractionDigits: 1 })
          .format(a.size / (a.size >= 1_048_576 ? 1_048_576 : 1024))}
      </span>}
      {(readOnly || a.event_id) && <Tip label={labels.download}><Button aria-label={labels.download} disabled={disabled || busy} loading={busy}
        onClick={() => void download(a)} size="icon-xs" type="button" variant="ghost"><Codicon name="cloud-download" /></Button></Tip>}
      {!readOnly && <Tip label={labels.removeAttachment}><Button aria-label={labels.removeAttachment} disabled={disabled || busy}
        onClick={() => onChange?.(attachments.filter(item => item !== a))} size="icon-xs" type="button" variant="ghost"><Codicon name="close" /></Button></Tip>}
    </div>)}
    {error && <div className="w-full text-xs text-destructive" role="alert"><p>{errorKind === 'upload' ? labels.uploadFailed : labels.pendingActionUnconfirmed}</p>
      <details className="mt-1 text-(--ui-text-quaternary)"><summary className="cursor-pointer">{labels.setupDetails}</summary><p className="mt-1 whitespace-pre-wrap break-words">{error}</p></details>
    </div>}
  </div>
}
