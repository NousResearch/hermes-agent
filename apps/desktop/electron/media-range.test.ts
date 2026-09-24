import { mkdtemp, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'

import { describe, expect, it } from 'vitest'

import { buildLocalMediaResponse, mediaMimeFor, parseByteRange } from './media-range'

describe('media-range: parseByteRange', () => {
  it('parses open, closed and suffix ranges', () => {
    expect(parseByteRange('bytes=0-99', 1000)).toEqual({ start: 0, end: 99 })
    expect(parseByteRange('bytes=500-', 1000)).toEqual({ start: 500, end: 999 })
    expect(parseByteRange('bytes=-100', 1000)).toEqual({ start: 900, end: 999 })
    // An end past EOF is clamped (Chromium commonly sends bytes=0-).
    expect(parseByteRange('bytes=990-5000', 1000)).toEqual({ start: 990, end: 999 })
  })

  it('ignores multi-range requests as a whole (RFC 7233 allows answering 200 instead of multipart)', () => {
    expect(parseByteRange('bytes=0-99,500-599', 1000)).toBeNull()
  })

  it('returns null without a usable Range and unsatisfiable when out of bounds', () => {
    expect(parseByteRange(null, 1000)).toBeNull()
    expect(parseByteRange('', 1000)).toBeNull()
    expect(parseByteRange('items=0-1', 1000)).toBeNull()
    expect(parseByteRange('bytes=1000-', 1000)).toBe('unsatisfiable')
    expect(parseByteRange('bytes=50-10', 1000)).toBe('unsatisfiable')
    expect(parseByteRange('bytes=0-', 0)).toBe('unsatisfiable')
  })

  it('maps media extensions to mime types', () => {
    expect(mediaMimeFor('/x/a.MP4')).toBe('video/mp4')
    expect(mediaMimeFor('/x/a.mp3')).toBe('audio/mpeg')
    expect(mediaMimeFor('/x/a.bin')).toBe('application/octet-stream')
  })
})

describe('media-range: buildLocalMediaResponse', () => {
  async function fixture() {
    const dir = await mkdtemp(path.join(tmpdir(), 'media-range-'))
    const file = path.join(dir, 'clip.mp4')
    const bytes = Buffer.from(Array.from({ length: 1000 }, (_, i) => i % 256))

    await writeFile(file, bytes)

    return { bytes, file }
  }

  it('serves the whole file with Accept-Ranges when no Range is given', async () => {
    const { bytes, file } = await fixture()
    const res = await buildLocalMediaResponse(file)

    expect(res.status).toBe(200)
    expect(res.headers.get('accept-ranges')).toBe('bytes')
    expect(res.headers.get('content-length')).toBe('1000')
    expect(res.headers.get('content-type')).toBe('video/mp4')
    expect(Buffer.from(await res.arrayBuffer()).equals(bytes)).toBe(true)
  })

  it('serves a 206 slice with Content-Range for a Range request (seeking)', async () => {
    const { bytes, file } = await fixture()
    const res = await buildLocalMediaResponse(file, { rangeHeader: 'bytes=100-199' })

    expect(res.status).toBe(206)
    expect(res.headers.get('content-range')).toBe('bytes 100-199/1000')
    expect(res.headers.get('content-length')).toBe('100')
    expect(Buffer.from(await res.arrayBuffer()).equals(bytes.subarray(100, 200))).toBe(true)
  })

  it('serves the tail for an open-ended range and 416 when unsatisfiable', async () => {
    const { file } = await fixture()
    const tail = await buildLocalMediaResponse(file, { rangeHeader: 'bytes=900-' })

    expect(tail.status).toBe(206)
    expect(tail.headers.get('content-range')).toBe('bytes 900-999/1000')
    expect((await tail.arrayBuffer()).byteLength).toBe(100)

    const bad = await buildLocalMediaResponse(file, { rangeHeader: 'bytes=5000-' })

    expect(bad.status).toBe(416)
    expect(bad.headers.get('content-range')).toBe('bytes */1000')
  })

  it('serves the whole file (200) for a multi-range request and 404 for a missing file', async () => {
    const { file } = await fixture()
    const multi = await buildLocalMediaResponse(file, { rangeHeader: 'bytes=0-99,500-599' })

    expect(multi.status).toBe(200)
    expect(multi.headers.get('content-length')).toBe('1000')
    expect((await multi.arrayBuffer()).byteLength).toBe(1000)

    const missing = await buildLocalMediaResponse(path.join(path.dirname(file), 'nope.mp4'))

    expect(missing.status).toBe(404)
  })

  it('answers HEAD with headers only', async () => {
    const { file } = await fixture()
    const res = await buildLocalMediaResponse(file, { method: 'HEAD', rangeHeader: 'bytes=0-9' })

    expect(res.status).toBe(206)
    expect(res.headers.get('content-length')).toBe('10')
    expect(res.body).toBeNull()
  })
})
