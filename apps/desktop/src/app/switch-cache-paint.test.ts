// Switch-paint from the render cache (Ace 2026-07-11: "clicking a session for
// the first time doesn't feel snappy"). Contract under test:
//  1. readCachedTranscript returns rows for a cached session, null otherwise;
//     it NEVER throws (fail-open) — a broken preload API is a miss.
//  2. normalizeCachedTranscriptRows handles BOTH writer shapes: raw
//     SessionMessage rows (transcript preloader) convert via toChatMessages;
//     already-converted ChatMessage rows (write-through) pass through.
//  3. Empty/garbage normalizes to [] — the switch falls through to the
//     network, never paints a blank.
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { normalizeCachedTranscriptRows, readCachedTranscript } from './render-cache-hydration'

function installApi(readTranscript: unknown) {
  ;(window as any).hermesDesktop = { renderCache: { readTranscript } }
}

describe('readCachedTranscript (switch-paint)', () => {
  beforeEach(() => {
    delete (window as any).hermesDesktop
  })

  it('returns cached rows for a hit', async () => {
    installApi(vi.fn().mockResolvedValue({ rows: [{ role: 'user', content: 'hi' }] }))
    const rows = await readCachedTranscript('s1')
    expect(rows).toEqual([{ role: 'user', content: 'hi' }])
  })

  it('passes a null gatewayUrl (main resolves it) and the session id', async () => {
    const read = vi.fn().mockResolvedValue({ rows: [1] })
    installApi(read)
    await readCachedTranscript('s42')
    expect(read).toHaveBeenCalledWith(null, 's42')
  })

  it('misses on null payload, empty rows, and missing id', async () => {
    installApi(vi.fn().mockResolvedValue(null))
    expect(await readCachedTranscript('s1')).toBeNull()

    installApi(vi.fn().mockResolvedValue({ rows: [] }))
    expect(await readCachedTranscript('s1')).toBeNull()

    installApi(vi.fn().mockResolvedValue({ rows: [1] }))
    expect(await readCachedTranscript(null)).toBeNull()
    expect(await readCachedTranscript('')).toBeNull()
  })

  it('fail-open: missing bridge, missing method, rejecting IPC are all misses', async () => {
    expect(await readCachedTranscript('s1')).toBeNull() // no hermesDesktop at all

    ;(window as any).hermesDesktop = { renderCache: {} } // no readTranscript method
    expect(await readCachedTranscript('s1')).toBeNull()

    installApi(vi.fn().mockRejectedValue(new Error('ipc down')))
    expect(await readCachedTranscript('s1')).toBeNull()
  })

  it('tolerates a malformed payload (rows not an array)', async () => {
    installApi(vi.fn().mockResolvedValue({ rows: 'garbage' }))
    expect(await readCachedTranscript('s1')).toBeNull()
  })
})

describe('normalizeCachedTranscriptRows (dual writer shapes)', () => {
  it('passes ChatMessage-shaped rows (write-through) straight through', () => {
    const rows = [{ id: 'm1', role: 'user', parts: [{ type: 'text', text: 'hi' }] }]
    expect(normalizeCachedTranscriptRows(rows)).toBe(rows)
  })

  it('converts raw SessionMessage rows (preloader) via toChatMessages', () => {
    const raw = [
      { role: 'user', content: 'hello' },
      { role: 'assistant', content: 'hey there' }
    ]
    const out = normalizeCachedTranscriptRows(raw)
    expect(out.length).toBeGreaterThan(0)
    expect(out[0]).toHaveProperty('parts')
    expect(out[0].role).toBe('user')
  })

  it('empty/garbage input normalizes to [] (a miss, never a blank paint)', () => {
    expect(normalizeCachedTranscriptRows([])).toEqual([])
    expect(normalizeCachedTranscriptRows(null as never)).toEqual([])
    expect(normalizeCachedTranscriptRows('nope' as never)).toEqual([])
  })
})
