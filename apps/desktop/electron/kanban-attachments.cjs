'use strict'

const path = require('node:path')

// Mirror of the backend's per-upload cap (plugins/kanban/dashboard/
// plugin_api.py _MAX_ATTACHMENT_BYTES). Checked before buffering so a huge
// pick fails fast with the same 413 detail the backend would return, instead
// of shipping 2 GB over IPC just to be rejected.
const MAX_KANBAN_ATTACHMENT_BYTES = 25 * 1024 * 1024

function attachmentSizeError() {
  const detail = `attachment exceeds ${MAX_KANBAN_ATTACHMENT_BYTES / (1024 * 1024)} MB limit`
  return new Error(`413: ${JSON.stringify({ detail })}`)
}

// Same reduction the backend applies (_safe_attachment_name): strip directory
// components on both separators, drop control chars and leading dots. The
// server already sanitised the stored filename, but the download target is
// joined under the user's Downloads dir, so never trust it verbatim.
function safeAttachmentBasename(raw) {
  let name = String(raw || '')
    .replace(/\\/g, '/')
    .split('/')
    .pop()
    .trim()
  name = Array.from(name)
    .filter(ch => ch >= ' ' && ch !== '\x7f')
    .join('')
    .trim()
  name = name.replace(/^\.+/, '').trim()
  if (!name) {
    return 'attachment'
  }
  return name.slice(0, 200)
}

// Collision-resolved save path under `dir`: foo.pdf → foo (1).pdf → foo (2).pdf.
// Splits at the FIRST dot to mirror the backend's collision naming, so a file
// uploaded and downloaded twice round-trips to the same shape.
function resolveDownloadTarget(dir, filename, exists) {
  const safe = safeAttachmentBasename(filename)
  const dotIndex = safe.indexOf('.')
  const stem = dotIndex === -1 ? safe : safe.slice(0, dotIndex)
  const ext = dotIndex === -1 ? '' : safe.slice(dotIndex)
  let candidate = safe
  let n = 1
  while (exists(path.join(dir, candidate))) {
    candidate = `${stem} (${n})${ext}`
    n += 1
  }
  return path.join(dir, candidate)
}

// RFC 2388 multipart/form-data body for the kanban upload route
// (POST /tasks/:id/attachments — fields: uploaded_by, file). Built as a
// Buffer so binary payloads survive untouched.
function buildKanbanAttachmentMultipart({ contentType, fileBuffer, filename, uploadedBy }, randomHex) {
  if (!Buffer.isBuffer(fileBuffer)) {
    throw new Error('buildKanbanAttachmentMultipart: fileBuffer must be a Buffer')
  }
  if (fileBuffer.length > MAX_KANBAN_ATTACHMENT_BYTES) {
    throw attachmentSizeError()
  }

  const boundary = `----hermesKanbanAttachment${randomHex || Math.random().toString(16).slice(2)}`
  // Quoted-string escaping for Content-Disposition; CR/LF cannot be escaped
  // portably, so drop them outright.
  const safeName = safeAttachmentBasename(filename).replace(/"/g, '%22')
  const parts = []

  if (uploadedBy) {
    parts.push(
      Buffer.from(
        `--${boundary}\r\nContent-Disposition: form-data; name="uploaded_by"\r\n\r\n${uploadedBy}\r\n`,
        'utf8'
      )
    )
  }

  parts.push(
    Buffer.from(
      `--${boundary}\r\nContent-Disposition: form-data; name="file"; filename="${safeName}"\r\n` +
        `Content-Type: ${contentType || 'application/octet-stream'}\r\n\r\n`,
      'utf8'
    ),
    fileBuffer,
    Buffer.from(`\r\n--${boundary}--\r\n`, 'utf8')
  )

  return {
    body: Buffer.concat(parts),
    boundary,
    contentType: `multipart/form-data; boundary=${boundary}`
  }
}

module.exports = {
  attachmentSizeError,
  buildKanbanAttachmentMultipart,
  MAX_KANBAN_ATTACHMENT_BYTES,
  resolveDownloadTarget,
  safeAttachmentBasename
}
