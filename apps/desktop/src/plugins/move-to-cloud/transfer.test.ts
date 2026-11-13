import { describe, expect, it } from 'vitest'

import { buildAndRead, LOCAL, type MoveRequestFn, refusalCode, refusalMessage, upload } from './transfer'

// The bridge rejects with `Error("<status>: <json detail>")`; mirror that shape exactly.
const refuse = (status: number, code: string, message = code) =>
  new Error(`${status}: ${JSON.stringify({ detail: { code, message } })}`)

const b64 = (bytes: Uint8Array) => btoa(String.fromCharCode(...bytes))
const unb64 = (data: string) => Uint8Array.from(atob(data), c => c.charCodeAt(0))

/** A receiver with the server's rules: resume by digest, replay-safe chunks, gap refusal. */
function fakeReceiver(opts: { failChunks?: number; corruptOnce?: boolean } = {}) {
  let failures = opts.failChunks ?? 0
  let corrupt = opts.corruptOnce ?? false

  const moves = new Map<
    string,
    { id: string; bytes: number; received: number; sha: string; data: Uint8Array; status: string }
  >()

  const calls: string[] = []

  const request: MoveRequestFn = async <T>(req: { path: string; method?: string; body?: unknown }) => {
    calls.push(`${req.method ?? 'GET'} ${req.path}`)
    const body = (req.body ?? {}) as Record<string, unknown>

    if (req.path === '/api/move/uploads' && req.method === 'POST') {
      const existing = [...moves.values()].find(m => m.sha === body.sha256 && m.status !== 'applied')

      if (existing) {
        return existing as T
      }

      const move = {
        id: `m${moves.size}`,
        bytes: body.size as number,
        received: 0,
        sha: body.sha256 as string,
        data: new Uint8Array(body.size as number),
        status: 'uploading'
      }

      moves.set(move.id, move)

      return move as T
    }

    const [, , , , id, verb] = req.path.split('/')
    const move = moves.get(id)!

    if (verb === 'chunk') {
      if (failures > 0) {
        failures -= 1
        throw new Error('ECONNRESET')
      }

      const offset = body.offset as number
      const data = unb64(body.data as string)

      if (offset > move.received) {
        throw refuse(409, 'offset_gap')
      }

      move.data.set(data, offset)
      move.received = Math.max(move.received, offset + data.length)

      return move as T
    }

    if (verb === 'finish') {
      if (corrupt) {
        corrupt = false
        move.received = 0
        throw refuse(422, 'digest_mismatch')
      }

      move.status = 'ready'

      return { ...move, preview: { sessions: 0 } } as T
    }

    return move as T
  }

  return { request, moves, calls }
}

const payload = (n: number) => Uint8Array.from({ length: n }, (_, i) => i % 251)

describe('upload', () => {
  it('sends every byte and finishes', async () => {
    const r = fakeReceiver()
    const bytes = payload(5 * 1024 * 1024 + 17)
    const state = await upload(r.request, 'cloud-1', bytes, 'a'.repeat(64), () => {})
    expect(state.status).toBe('ready')
    expect([...r.moves.values()][0].data).toEqual(bytes)
  }, 30_000)

  it('retries a dropped chunk and resumes an interrupted upload at the receiver offset', async () => {
    const r = fakeReceiver({ failChunks: 2 })
    const bytes = payload(3 * 1024 * 1024)
    await upload(r.request, 'cloud-1', bytes, 'b'.repeat(64), () => {})
    const chunkCalls = r.calls.filter(c => c.endsWith('/chunk'))
    expect(chunkCalls.length).toBe(4)

    // A second run with the same digest re-attaches to the finished upload instead of resending.
    const again = await upload(r.request, 'cloud-1', bytes, 'b'.repeat(64), () => {})
    expect(again.id).toBe('m0')
    expect(r.calls.filter(c => c.endsWith('/chunk')).length).toBe(4)
  }, 30_000)

  it('restarts once from zero when the bytes arrive damaged', async () => {
    const r = fakeReceiver({ corruptOnce: true })
    const bytes = payload(1024)
    const state = await upload(r.request, 'cloud-1', bytes, 'c'.repeat(64), () => {})
    expect(state.status).toBe('ready')
    expect(r.calls.filter(c => c.endsWith('/finish')).length).toBe(2)
  })

  it('does not retry a refusal a retry cannot change', async () => {
    const request: MoveRequestFn = async () => {
      throw refuse(413, 'too_large', 'The bundle is 600 MB; the limit is 512 MB.')
    }

    await expect(upload(request, 'cloud-1', payload(10), 'd'.repeat(64), () => {})).rejects.toThrow('600 MB')
  })
})

describe('buildAndRead', () => {
  it('reads the built bundle back in chunks', async () => {
    const bytes = payload(9000)
    const seen: string[] = []

    const request: MoveRequestFn = async <T>(req: { connectionId: string; path: string }) => {
      seen.push(req.connectionId)

      if (req.path === '/api/move/bundle') {
        return { name: 'move-0123456789abcdef.zip', bytes: bytes.length } as T
      }

      const offset = Number(new URL(`http://x${req.path}`).searchParams.get('offset'))
      const part = bytes.subarray(offset, offset + 4000)

      return { data: b64(part), done: offset + part.length >= bytes.length } as T
    }

    const out = await buildAndRead(request, true, () => {})
    expect(out.bytes).toEqual(bytes)
    expect(new Set(seen)).toEqual(new Set([LOCAL]))
  })
})

describe('refusal parsing', () => {
  it('reads the code and message the routes answer with', () => {
    const error = refuse(409, 'offset_gap', 'Chunk offset 5 skips ahead of 0 received bytes.')
    expect(refusalCode(error)).toBe('offset_gap')
    expect(refusalMessage(error)).toBe('Chunk offset 5 skips ahead of 0 received bytes.')
    expect(refusalMessage(new Error('500: boom'))).toBe('boom')
  })
})
