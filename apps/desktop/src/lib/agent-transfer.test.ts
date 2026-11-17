import { describe, expect, it } from 'vitest'

import type { HermesApiRequest } from '@/global'

import {
  base64Bytes,
  CHUNK_BYTES,
  cloudEndpoint,
  createUndoLedger,
  failureText,
  runBackup,
  transferAgent,
  type TransferEndpoint,
  type TransferProgress,
  type TransferRequestFn,
  TransferRestoreError,
  undoTransfer
} from './agent-transfer'

const SOURCE: TransferEndpoint = { connectionId: null, profile: 'coder' }
const TARGET = cloudEndpoint('cloud-1')
const instant = { sleep: async () => undefined }

const b64 = (bytes: Uint8Array) => btoa(String.fromCharCode(...bytes))
const unb64 = (data: string) => Uint8Array.from(atob(data), c => c.charCodeAt(0))

interface Call {
  owner: TransferEndpoint
  req: HermesApiRequest
}

interface FakeOptions {
  /** Chunk size the fake source answers with (the real one honours `length`). */
  chunk?: number
  /** Target status polls that fail first. */
  flaky?: number
  /** Answer the import status as failed after reporting the undo point. */
  importFails?: boolean
  keepHost?: boolean
  /** The source chunk reads that fail first. */
  sourceFlaky?: number
  /** The target loses everything past this offset once, answering 409 offset_gap. */
  gapAt?: number
  undoArchive?: null | string
  /** What the cloud agent's finished restore reports it switched off. */
  messagingOff?: string[]
}

/** Both backends with the contract's routes. */
function fakeBackends(archive: Uint8Array, options: FakeOptions = {}) {
  const { chunk = CHUNK_BYTES, keepHost = true, importFails = false } = options
  const undoArchive = options.undoArchive === undefined ? '/cloud/pre-transfer.zip' : options.undoArchive
  const calls: Call[] = []
  let failures = options.flaky ?? 0
  let sourceFailures = options.sourceFlaky ?? 0
  let gapAt = options.gapAt
  const polls: Record<string, number> = {}
  const uploads = new Map<string, Uint8Array>()
  let uploadSeq = 0

  type Handler = (req: HermesApiRequest, params: URLSearchParams, side: string, path: string) => unknown

  const sourceChunk: Handler = (_req, params) => {
    if (sourceFailures > 0) {
      sourceFailures -= 1

      throw new Error('temporary disconnect')
    }

    const offset = Number(params.get('offset'))
    const slice = archive.subarray(offset, offset + Math.min(chunk, Number(params.get('length'))))

    return { data: b64(slice), done: offset + slice.length >= archive.length, offset, size: archive.length }
  }

  const targetChunk: Handler = req => {
    const body = req.body as { data: string; offset: number; upload_id: null | string }
    const id = body.upload_id || `up-${(uploadSeq += 1)}`
    const have = uploads.get(id) ?? new Uint8Array()

    if (gapAt !== undefined && body.offset > gapAt) {
      uploads.set(id, have.subarray(0, gapAt))
      gapAt = undefined

      throw new Error(`409: {"detail": "offset_gap", "received": ${uploads.get(id)!.length}}`)
    }

    if (body.offset > have.length) {
      throw new Error(`409: {"detail": "offset_gap", "received": ${have.length}}`)
    }

    const next = new Uint8Array([...have.subarray(0, body.offset), ...unb64(body.data)])
    uploads.set(id, next)

    return { received: next.length, upload_id: id }
  }

  const restore: Handler = req => {
    const body = req.body as { size: number; upload_id: string }

    if (uploads.get(body.upload_id)?.length !== body.size) {
      throw new Error('409: {"detail": "size mismatch"}')
    }

    return { archive: '/tmp/in.zip', keep_host: true, name: 'import', ok: true, pid: 1 }
  }

  const actionStatus: Handler = (_req, _params, side, path) => {
    if (side !== 'local' && failures > 0) {
      failures -= 1

      throw new Error('502: gateway restarting')
    }

    polls[`${side}${path}`] = (polls[`${side}${path}`] ?? 0) + 1
    const running = polls[`${side}${path}`] < 2
    const undo = side !== 'local' ? { undo_archive: undoArchive } : {}

    if (!running && importFails && side !== 'local') {
      return { exit_code: 1, lines: ['a', 'b', 'c', 'Error: disk full'], running: false, ...undo }
    }

    const off = !running && side !== 'local' && options.messagingOff ? { messaging_off: options.messagingOff } : {}

    return { exit_code: running ? null : 0, lines: ['working'], running, ...undo, ...off }
  }

  const routes: Record<string, Handler> = {
    '/api/ops/backup': () => ({ archive: '/home/u/.hermes/backups/b.zip', name: 'backup' }),
    '/api/ops/backup/chunk': sourceChunk,
    '/api/ops/import': () => ({ name: 'import', ok: true }),
    '/api/ops/transfer/chunk': targetChunk,
    '/api/ops/transfer/restore': restore,
    '/api/status': () => (keepHost ? { import_keep_host: 1, version: 'x' } : { version: 'old' })
  }

  const request: TransferRequestFn = async <T>(owner: TransferEndpoint, req: HermesApiRequest) => {
    calls.push({ owner, req })
    const [path, query = ''] = req.path.split('?')
    const handler = routes[path] ?? (path.endsWith('/status') ? actionStatus : undefined)

    if (!handler) {
      throw new Error(`unexpected ${req.method ?? 'GET'} ${req.path}`)
    }

    return handler(req, new URLSearchParams(query), owner.connectionId ?? 'local', path) as T
  }

  return { calls, request, uploads }
}

const targetCalls = (calls: Call[]) => calls.filter(c => c.owner === TARGET)

describe('transferAgent', () => {
  it('streams the archive chunk by chunk from the source to the cloud agent, then restores it', async () => {
    const archive = Uint8Array.from({ length: 300 }, (_, i) => i % 251)

    const { calls, request, uploads } = fakeBackends(archive, {
      chunk: 90,
      flaky: 3,
      messagingOff: ['Telegram', 'Discord']
    })

    const progress: TransferProgress[] = []

    const result = await transferAgent(request, SOURCE, TARGET, p => progress.push(p), { timing: instant })

    expect(result).toEqual({
      gatewayDown: false,
      messagingOff: ['Telegram', 'Discord'],
      undoArchive: '/cloud/pre-transfer.zip'
    })
    expect(progress[0].phase).toBe('backup')
    expect(progress.at(-1)?.phase).toBe('restore')
    expect(progress.filter(p => p.phase === 'upload').map(p => p.fraction)).toEqual([0, 0.3, 0.6, 0.9, 1])

    const reads = calls.filter(c => c.req.path.startsWith('/api/ops/backup/chunk'))
    expect(reads.every(c => c.owner === SOURCE)).toBe(true)
    expect(reads[0].req.path).toContain(`archive=${encodeURIComponent('/home/u/.hermes/backups/b.zip')}`)
    expect(reads[0].req.path).toContain(`length=${CHUNK_BYTES}`)

    // Each source chunk goes to the target as the same base64 string, interleaved read → write.
    const writes = calls.filter(c => c.req.path === '/api/ops/transfer/chunk')
    expect(writes).toHaveLength(4)
    expect(writes.every(c => c.owner === TARGET && c.req.method === 'POST' && !c.req.upload)).toBe(true)
    const order = calls.map(c => c.req.path.split('?')[0]).filter(p => p.includes('chunk'))
    expect(order).toEqual(Array(4).fill(['/api/ops/backup/chunk', '/api/ops/transfer/chunk']).flat())
    expect(writes.map(c => (c.req.body as { offset: number }).offset)).toEqual([0, 90, 180, 270])
    expect(uploads.get('up-1')).toEqual(archive)

    const restore = calls.find(c => c.req.path === '/api/ops/transfer/restore')!
    expect(restore).toMatchObject({ owner: TARGET, req: { body: { size: 300, upload_id: 'up-1' }, method: 'POST' } })
    expect(calls.some(c => c.owner === TARGET && c.req.path === '/api/actions/import/status?lines=40')).toBe(true)
  })

  it('asks for an update when the cloud agent predates --keep-host, before touching anything', async () => {
    const { calls, request } = fakeBackends(new Uint8Array([1, 2, 3]), { keepHost: false })

    await expect(transferAgent(request, SOURCE, TARGET, () => undefined, { timing: instant })).rejects.toThrow(
      'This cloud agent needs a Hermes update before it can receive a transfer.'
    )
    expect(calls.map(c => c.req.path)).toEqual(['/api/status'])
  })

  it('retries a transient read and re-syncs after the target reports an offset gap', async () => {
    const archive = Uint8Array.from({ length: 250 }, (_, i) => i % 7)
    const { request, uploads } = fakeBackends(archive, { chunk: 100, gapAt: 50, sourceFlaky: 1 })

    await transferAgent(request, SOURCE, TARGET, () => undefined, { timing: instant })

    expect(uploads.get('up-1')).toEqual(archive)
  })

  it('refuses a truncated archive instead of restoring it', async () => {
    const { calls, request } = fakeBackends(new Uint8Array([1, 2, 3]))

    const truncated: TransferRequestFn = async <T>(owner: TransferEndpoint, req: HermesApiRequest) =>
      req.path.startsWith('/api/ops/backup/chunk')
        ? ({ data: b64(new Uint8Array([1])), done: true, offset: 0, size: 2 } as T)
        : request<T>(owner, req)

    await expect(transferAgent(truncated, SOURCE, TARGET, () => undefined, { timing: instant })).rejects.toThrow(
      'The backup changed while it was being read. Try again.'
    )
    expect(calls.some(c => c.req.path === '/api/ops/transfer/restore')).toBe(false)
  })

  it('stops copying when canceled and never asks the cloud agent to restore', async () => {
    const controller = new AbortController()

    const backend = fakeBackends(
      Uint8Array.from({ length: 300 }, (_, i) => i),
      { chunk: 100 }
    )

    const request: TransferRequestFn = async <T>(owner: TransferEndpoint, req: HermesApiRequest) => {
      const response = await backend.request<T>(owner, req)

      if (req.path === '/api/ops/transfer/chunk') {
        controller.abort(new Error('user canceled'))
      }

      return response
    }

    await expect(
      transferAgent(request, SOURCE, TARGET, () => undefined, { signal: controller.signal, timing: instant })
    ).rejects.toThrow('user canceled')
    expect(backend.calls.filter(c => c.req.path === '/api/ops/transfer/chunk')).toHaveLength(1)
    expect(backend.calls.some(c => c.req.path === '/api/ops/transfer/restore')).toBe(false)
  })

  it('reports the undo point as soon as it appears and keeps it on a failed restore', async () => {
    const { request } = fakeBackends(new Uint8Array([1]), { importFails: true })
    const seen: string[] = []

    const error = await transferAgent(request, SOURCE, TARGET, () => undefined, {
      onUndoArchive: archive => seen.push(archive),
      timing: instant
    }).catch(caught => caught)

    expect(error).toBeInstanceOf(TransferRestoreError)
    expect(error).toMatchObject({ message: 'b\nc\nError: disk full', undoArchive: '/cloud/pre-transfer.zip' })
    expect(seen).toEqual(['/cloud/pre-transfer.zip'])
  })

  it('treats a finished restore that never reported its undo point as an error', async () => {
    const { request } = fakeBackends(new Uint8Array([1]), { undoArchive: null })

    await expect(transferAgent(request, SOURCE, TARGET, () => undefined, { timing: instant })).rejects.toThrow(
      'The cloud agent did not report its undo point.'
    )
  })

  it('rides out ten failed restore polls and stops at the eleventh', async () => {
    await expect(
      transferAgent(fakeBackends(new Uint8Array([1]), { flaky: 10 }).request, SOURCE, TARGET, () => undefined, {
        timing: instant
      })
    ).resolves.toEqual({ gatewayDown: false, messagingOff: [], undoArchive: '/cloud/pre-transfer.zip' })

    const { calls, request } = fakeBackends(new Uint8Array([1]), { flaky: 11 })

    await expect(transferAgent(request, SOURCE, TARGET, () => undefined, { timing: instant })).rejects.toThrow(
      '502: gateway restarting'
    )
    expect(targetCalls(calls).filter(c => c.req.path.includes('/status?'))).toHaveLength(11)
  })
})

describe('runBackup', () => {
  it('rejects a backup response without a path before polling', async () => {
    const calls: string[] = []

    const request: TransferRequestFn = async <T>(_owner: TransferEndpoint, req: HermesApiRequest) => {
      calls.push(req.path)

      return { name: 'backup' } as T
    }

    await expect(runBackup(request, SOURCE, undefined, { timing: instant })).rejects.toThrow(
      'The backup did not say where it was written.'
    )
    expect(calls).toEqual(['/api/ops/backup'])
  })
})

describe('base64Bytes', () => {
  it('counts decoded bytes from padding alone', () => {
    for (const length of [0, 1, 2, 3, 4, 5]) {
      expect(base64Bytes(b64(new Uint8Array(length)))).toBe(length)
    }
  })
})

describe('failureText', () => {
  it("names the files the import could not restore, not the log's last lines", () => {
    const lines = [
      'Import incomplete: 355 files restored in 3.2s',
      '  Target: ~/.',
      '',
      '  Warnings (1 files skipped):',
      '  memories/MEMORY.md: [Errno 13] Permission denied',
      'Running in Docker: start the gateway with hermes gateway run',
      'Import incomplete: 1 file(s) were not restored (see Warnings above). Fix the cause and re-run the import.'
    ]

    expect(failureText(lines)).toBe('Warnings (1 files skipped):\nmemories/MEMORY.md: [Errno 13] Permission denied')
    expect(failureText(['a', 'b', 'c', 'Error: bad archive'])).toBe('b\nc\nError: bad archive')
  })
})

describe('createUndoLedger', () => {
  it('keeps the first undo point for an agent: a retry never replaces it', () => {
    const ledger = createUndoLedger()
    ledger.record('agent-1', { archive: '/cloud/first.zip', target: TARGET })
    ledger.record('agent-1', { archive: '/cloud/half-restored.zip', target: TARGET })

    expect(ledger.get('agent-1')?.archive).toBe('/cloud/first.zip')
    ledger.clear('agent-1')
    expect(ledger.get('agent-1')).toBeUndefined()
  })
})

describe('undoTransfer', () => {
  it('restores the cloud agent from its own pre-transfer backup, keeping its host', async () => {
    const { calls, request } = fakeBackends(new Uint8Array())

    await undoTransfer(request, TARGET, '/cloud/pre-transfer.zip', () => undefined, instant)

    expect(calls[0]).toEqual({
      owner: TARGET,
      req: {
        body: { archive: '/cloud/pre-transfer.zip', force: true, keep_host: true },
        method: 'POST',
        path: '/api/ops/import'
      }
    })
    expect(calls.at(-1)?.req.path).toBe('/api/actions/import/status?lines=40')
  })

  it('refuses without an undo archive before sending an import', async () => {
    const { calls, request } = fakeBackends(new Uint8Array())

    await expect(undoTransfer(request, TARGET, '', undefined, instant)).rejects.toThrow(
      'The cloud agent did not keep a backup to undo with.'
    )
    expect(calls).toHaveLength(0)
  })
})
