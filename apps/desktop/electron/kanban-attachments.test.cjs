'use strict'

const test = require('node:test')
const assert = require('node:assert/strict')
const path = require('node:path')

const {
  buildKanbanAttachmentMultipart,
  MAX_KANBAN_ATTACHMENT_BYTES,
  resolveDownloadTarget,
  safeAttachmentBasename
} = require('./kanban-attachments.cjs')

test('safeAttachmentBasename strips directories, control chars, and leading dots', () => {
  assert.equal(safeAttachmentBasename('../../etc/passwd'), 'passwd')
  assert.equal(safeAttachmentBasename('C:\\Users\\evil\\..\\notes.txt'), 'notes.txt')
  assert.equal(safeAttachmentBasename('.hidden'), 'hidden')
  assert.equal(safeAttachmentBasename('re\x00port\n.pdf'), 'report.pdf')
  assert.equal(safeAttachmentBasename(''), 'attachment')
  assert.equal(safeAttachmentBasename('...'), 'attachment')
  assert.equal(safeAttachmentBasename(`${'a'.repeat(300)}.txt`).length, 200)
})

test('resolveDownloadTarget keeps the name when free', () => {
  const target = resolveDownloadTarget('/downloads', 'report.pdf', () => false)
  assert.equal(target, path.join('/downloads', 'report.pdf'))
})

test('resolveDownloadTarget resolves collisions like the backend (first-dot split)', () => {
  const taken = new Set([
    path.join('/downloads', 'archive.tar.gz'),
    path.join('/downloads', 'archive (1).tar.gz')
  ])
  const target = resolveDownloadTarget('/downloads', 'archive.tar.gz', p => taken.has(p))
  assert.equal(target, path.join('/downloads', 'archive (2).tar.gz'))
})

test('resolveDownloadTarget handles extensionless names', () => {
  const taken = new Set([path.join('/downloads', 'Makefile')])
  const target = resolveDownloadTarget('/downloads', 'Makefile', p => taken.has(p))
  assert.equal(target, path.join('/downloads', 'Makefile (1)'))
})

test('buildKanbanAttachmentMultipart emits uploaded_by and file parts with the boundary', () => {
  const payload = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x00, 0xff])
  const { body, boundary, contentType } = buildKanbanAttachmentMultipart(
    {
      contentType: 'image/png',
      fileBuffer: payload,
      filename: 'shot.png',
      uploadedBy: 'desktop'
    },
    'deadbeef'
  )

  assert.equal(boundary, '----hermesKanbanAttachmentdeadbeef')
  assert.equal(contentType, `multipart/form-data; boundary=${boundary}`)

  const text = body.toString('latin1')
  assert.match(text, /Content-Disposition: form-data; name="uploaded_by"\r\n\r\ndesktop\r\n/)
  assert.match(text, /Content-Disposition: form-data; name="file"; filename="shot.png"\r\n/)
  assert.match(text, /Content-Type: image\/png\r\n\r\n/)
  assert.ok(text.endsWith(`\r\n--${boundary}--\r\n`))
  // The binary payload must survive byte-for-byte inside the body.
  assert.notEqual(body.indexOf(payload), -1)
})

test('buildKanbanAttachmentMultipart escapes quotes and strips paths from the filename', () => {
  const { body } = buildKanbanAttachmentMultipart(
    {
      fileBuffer: Buffer.from('x'),
      filename: '../dir/we"ird.txt'
    },
    'cafe'
  )
  const text = body.toString('utf8')
  assert.match(text, /filename="we%22ird.txt"/)
  assert.doesNotMatch(text, /name="uploaded_by"/)
  assert.match(text, /Content-Type: application\/octet-stream\r\n/)
})

test('buildKanbanAttachmentMultipart rejects payloads over the backend cap with a 413-shaped error', () => {
  const big = Buffer.alloc(MAX_KANBAN_ATTACHMENT_BYTES + 1)
  assert.throws(
    () => buildKanbanAttachmentMultipart({ fileBuffer: big, filename: 'big.bin' }),
    /^Error: 413: .*25 MB limit/
  )
})
