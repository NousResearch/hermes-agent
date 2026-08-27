/**
 * Group-chat attachments: turning picked, pasted, or dropped files into the
 * data-URL payloads a room's members are shown.
 *
 * A leaf module — it talks to the DOM and the toast host, nothing else in Bot
 * Mode.
 */

import { host } from '@hermes/plugin-sdk'

import type { Attachment, AttachmentKind } from './types'

// ── group-chat attachments: pick/paste/drop files the room's members see ────

/** Classify a picked file for the group-attachment pipeline. */
function groupAttachmentKind(file: File): AttachmentKind {
  if (/^image\//.test(file.type || '')) {
    return 'image'
  }
  if (file.type === 'application/pdf' || /\.pdf$/i.test(file.name || '')) {
    return 'pdf'
  }
  return 'file'
}

/** File objects → [{ name, data, kind }] (data URLs), oversized files skipped
 *  with a toast. Images are downscaled; PDFs and other files ride as raw data
 *  URLs for the gateway's pdf.attach / file.attach staging. Shared by the
 *  picker button, the composer paste handler, and room drag & drop. */
export async function filesToGroupAttachments(files: File[] | FileList | null | undefined): Promise<Attachment[]> {
  const picked: Attachment[] = []
  for (const file of [...(files || [])]) {
    if (!file) {
      continue
    }
    if (file.size > 15_000_000) {
      host.notify({
        kind: 'error',
        message: `${file.name || 'attachment'}: too large (max 15MB).`
      })
      continue
    }
    const data = await new Promise<null | string>(done => {
      const reader = new FileReader()
      reader.onload = () => done(typeof reader.result === 'string' ? reader.result : null)
      reader.onerror = () => done(null)
      reader.readAsDataURL(file)
    })
    if (!data) {
      continue
    }
    const kind = groupAttachmentKind(file)
    picked.push({
      name: file.name || (kind === 'image' ? 'pasted image' : 'attachment'),
      data: kind === 'image' ? await normalizeGroupAttachment(data) : data,
      kind
    })
  }
  return picked
}

/** Multi-file picker for the group composer — any file type; kind decides
 *  the staging RPC. Resolves to [{ name, data, kind }]. */
export function pickGroupAttachments(): Promise<Attachment[]> {
  return new Promise(resolve => {
    const input = document.createElement('input')
    input.type = 'file'
    input.multiple = true
    input.onchange = () => resolve(filesToGroupAttachments(input.files))
    input.click()
  })
}

/** Bound a group attachment's long edge so room logs (persisted with the
 *  plugin's other durable state) stay light while screenshots keep enough
 *  resolution for vision models to read text. No-op for small images or
 *  anything the canvas can't decode. */
function normalizeGroupAttachment(dataUrl: string, maxEdge = 1568): Promise<string> {
  return new Promise(resolve => {
    const img = new Image()
    img.onload = () => {
      try {
        const long = Math.max(img.width, img.height)
        if (!long || long <= maxEdge) {
          return resolve(dataUrl)
        }
        const scale = maxEdge / long
        const canvas = document.createElement('canvas')
        canvas.width = Math.max(1, Math.round(img.width * scale))
        canvas.height = Math.max(1, Math.round(img.height * scale))
        canvas.getContext('2d')!.drawImage(img, 0, 0, canvas.width, canvas.height)
        resolve(canvas.toDataURL('image/png'))
      } catch {
        resolve(dataUrl)
      }
    }
    img.onerror = () => resolve(dataUrl)
    img.src = dataUrl
  })
}
