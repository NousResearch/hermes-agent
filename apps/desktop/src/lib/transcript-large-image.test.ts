import { crc32 } from 'node:zlib'

import { strFromU8, unzipSync } from 'fflate'
import { expect, it } from 'vitest'

import { DEFAULT_TRANSCRIPT_OPTIONS, type TranscriptDocument } from './transcript-document'
import { transcriptBase64, transcriptImage } from './transcript-media'
import { renderTranscript } from './transcript-render'
const small = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNQLdrxHwAEXgJPUgh8kgAAAABJRU5ErkJggg=='

// Ancillary text is legal PNG content; the original pixel stream remains intact.
function pngWithMetadata(size: number): Uint8Array {
  const raw = Buffer.from(small, 'base64'),
    data = Buffer.alloc(size, 65)

  data.write('Comment\0')

  const type = Buffer.from('tEXt'),
    chunk = Buffer.alloc(data.length + 12)

  chunk.writeUInt32BE(data.length)
  type.copy(chunk, 4)
  data.copy(chunk, 8)
  chunk.writeUInt32BE(crc32(Buffer.concat([type, data])), chunk.length - 4)
  const offset = 8 + 12 + raw.readUInt32BE(8)

  return Buffer.concat([raw.subarray(0, offset), chunk, raw.subarray(offset)])
}

const bytes = pngWithMetadata(3 * 1024 * 1024 + 256)
const dataUrl = 'data:image/png;base64,' + transcriptBase64(bytes)

const doc: TranscriptDocument = {
  version: 1,
  session_id: 'sample',
  title: 'Large image sample',
  exported_at: '2026-11-03T18:00:00Z',
  options: { ...DEFAULT_TRANSCRIPT_OPTIONS },
  warnings: [],
  messages: [
    {
      role: 'user',
      speaker: 'User',
      text: 'Selected image',
      attachments: [{ kind: 'image', name: 'Selected PNG', data_url: dataUrl }]
    }
  ]
}

it('accepts a genuine larger-than-3-MiB PNG within the 16-MiB image budget', () => {
  expect(transcriptImage(dataUrl)?.bytes).toEqual(new Uint8Array(bytes))
})
it.each(['docx', 'odt'] as const)('embeds the selected larger image in %s', format => {
  const value = renderTranscript(doc, format),
    zip = unzipSync(Buffer.from(value.content, 'base64'))

  const names = Object.keys(zip).filter(n => n.startsWith(format === 'docx' ? 'word/media/' : 'Pictures/'))
  expect(names).toHaveLength(1)
  expect(zip[names[0]]).toEqual(new Uint8Array(bytes))
  expect(strFromU8(zip[format === 'docx' ? 'word/document.xml' : 'content.xml'])).not.toContain('image unavailable')
})
it('rejects inline bytes above the explicit per-image limit before decoding', () => {
  expect(
    transcriptImage('data:image/png;base64,' + 'A'.repeat(4 * Math.ceil((16 * 1024 * 1024) / 3) + 4))
  ).toBeUndefined()
})
