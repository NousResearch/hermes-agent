import { strFromU8, unzipSync } from 'fflate'
import { expect, it } from 'vitest'

import { DEFAULT_TRANSCRIPT_OPTIONS, type TranscriptDocument } from './transcript-document'
import { escapeTranscriptXml, transcriptBase64, transcriptImage } from './transcript-media'
import { renderTranscript } from './transcript-render'

const png =
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNQLdrxHwAEXgJPUgh8kgAAAABJRU5ErkJggg=='

// This is the exact reviewed repro: IDAT bytes 41..47 are zeroed without updating its CRC.
const validBytes = Uint8Array.from(atob(png.split(',')[1]), c => c.charCodeAt(0))
const corruptBytes = validBytes.slice()
corruptBytes.fill(0, 41, 48)

it.each(['docx', 'odt'] as const)('rejects the reported corrupt PNG in the actual %s writer', format => {
  const document: TranscriptDocument = {
    version: 1, session_id: 'crc', title: 'CRC', exported_at: '2026-01-01T00:00:00Z',
    options: { ...DEFAULT_TRANSCRIPT_OPTIONS }, warnings: [],
    messages: [{ role: 'user', speaker: 'User', text: 'Image', attachments: [
      { kind: 'image', name: 'pixel', data_url: `data:image/png;base64,${transcriptBase64(corruptBytes)}` }
    ] }]
  }

  const zip = unzipSync(Uint8Array.from(atob(renderTranscript(document, format).content), c => c.charCodeAt(0)))
  const xml = strFromU8(zip[format === 'docx' ? 'word/document.xml' : 'content.xml'])
  expect(Object.keys(zip).filter(name => name.startsWith(format === 'docx' ? 'word/media/' : 'Pictures/'))).toEqual([])
  expect(xml).toContain(format === 'docx' ? 'This image format cannot be embedded' : 'image unavailable or unsupported')
  expect(xml).not.toContain(format === 'docx' ? '<w:drawing' : '<draw:image')
})

it('checks every PNG chunk CRC and framing without changing valid bytes', () => {
  expect(transcriptImage(png)?.bytes).toEqual(validBytes)
  expect(transcriptImage(png, 'odt')?.bytes).toEqual(validBytes)

  for (const offset of [29, 41, 65]) { // IHDR CRC, IDAT data, IEND CRC
    const bad = validBytes.slice()
    bad[offset] ^= 1
    const data = `data:image/png;base64,${transcriptBase64(bad)}`
    expect(transcriptImage(data)).toBeUndefined()
    expect(transcriptImage(data, 'odt')).toBeUndefined()
  }

  for (const bad of [validBytes.subarray(0, validBytes.length - 1), Uint8Array.from([...validBytes, 0])]) {
    const data = `data:image/png;base64,${transcriptBase64(bad)}`
    expect(transcriptImage(data)).toBeUndefined()
    expect(transcriptImage(data, 'odt')).toBeUndefined()
  }
})

it('accepts complete raster bytes but refuses a PNG containing only a header', () => {
  expect(transcriptImage(png)).toMatchObject({ mime: 'image/png', extension: 'png', width: 1, height: 1 })
  const header = Uint8Array.from(atob(png.split(',')[1]), c => c.charCodeAt(0)).subarray(0, 33)
  expect(transcriptImage(`data:image/png;base64,${transcriptBase64(header)}`)).toBeUndefined()
})
it('never treats an SVG, URL or local path as an inline raster', () => {
  for (const bad of [
    'file:///private.png',
    'https://example.com/a.png',
    'data:image/svg+xml;base64,PHN2Zz4=',
    'data:image/png;base64,AAAA'
  ])
    {expect(transcriptImage(bad)).toBeUndefined()}
})
it('keeps Unicode text while removing XML-invalid controls', () => {
  expect(escapeTranscriptXml('Hello 日本語 😀 & < > \u0001')).toBe('Hello 日本語 😀 &amp; &lt; &gt; ')
})

it('reads dimensions from each supported raster and rejects truncated payloads', () => {
  for (const image of [
    'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAAgAEADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDAooor6w+VCiiigAooooAKKKKACiiigAooooAKKKKACiiigD//2Q==',
    'data:image/gif;base64,R0lGODdhQAAgAIEAACByuAAAAAAAAAAAACwAAAAAQAAgAEAISwABCBxIsKDBgwgTKlzIsKHDhxAjSpxIsaLFixgzatzIsaPHjyBDihxJsqTJkyhTqlzJsqXLlzBjypxJs6bNmzhz6tzJs6fPnyADAgA7',
    'data:image/webp;base64,UklGRkgAAABXRUJQVlA4IDwAAACQAwCdASpAACAAPm02l0ikIyIhJAgAgA2JZwB2APwAAEEaSMNAAP7svY//+WfP4bOpz//5SrwWN2bAAAA='
  ]) {
    expect(transcriptImage(image)).toMatchObject({ width: 64, height: 32 })
    const bytes = Uint8Array.from(atob(image.split(',')[1]), c => c.charCodeAt(0))
    expect(transcriptImage(`${image.split(',')[0]},${transcriptBase64(bytes.subarray(0, 24))}`)).toBeUndefined()
  }
})
