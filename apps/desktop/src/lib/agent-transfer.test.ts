import { describe, expect, it } from 'vitest'

import type { HermesApiRequest } from '@/global'

import {
  CHUNK_BYTES,
  cloudEndpoint,
  transferAgent,
  type TransferEndpoint,
  type TransferProgress,
  type TransferRequestFn,
  undoTransfer
} from './agent-transfer'

const SOURCE: TransferEndpoint = { connectionId: null, profile: 'coder' }
const TARGET = cloudEndpoint('cloud-1')
const instant = { sleep: async () => undefined }

const b64 = (bytes: Uint8Array) => btoa(String.fromCharCode(...bytes))

interface Call {
  owner: TransferEndpoint
  req: HermesApiRequest
}

/** Both backends with the contract's routes; `flaky` fails that many target status polls first. */
function fakeBackends(archive: Uint8Array, { flaky = 0 } = {}) {
  const calls: Call[] = []
  let failures = flaky
  const polls: Record<string, number> = {}

  const request: TransferRequestFn = async <T>(owner: TransferEndpoint, req: HermesApiRequest) => {
    calls.push({ owner, req })
    const [path, query = ''] = req.path.split('?')
    const params = new URLSearchParams(query)
    const side = owner.connectionId ?? 'local'

    if (path === '/api/ops/backup' && req.method === 'POST') {
      return { archive: '/home/u/.hermes/backups/b.zip', name: 'backup' } as T
    }

    if (path === '/api/ops/backup/chunk') {
      const offset = Number(params.get('offset'))
      const slice = archive.subarray(offset, offset + Number(params.get('length')))

      return { data: b64(slice), done: offset + slice.length >= archive.length, offset, size: archive.length } as T
    }

    if (path === '/api/ops/import-upload' || path === '/api/ops/import') {
      return { archive: '/tmp/in.zip', name: 'import', ok: true, undo_archive: '/cloud/pre-import.zip' } as T
    }

    if (path.endsWith('/status')) {
      if (side !== 'local' && failures > 0) {
        failures -= 1

        throw new Error('502: gateway restarting')
      }

      polls[`${side}${path}`] = (polls[`${side}${path}`] ?? 0) + 1
      const running = polls[`${side}${path}`] < 2

      return { exit_code: running ? null : 0, lines: ['working'], running } as T
    }

    throw new Error(`unexpected ${req.method ?? 'GET'} ${req.path}`)
  }

  return { calls, request }
}

describe('transferAgent', () => {
  it('backs up the source, reads it back, uploads it to the cloud agent and waits for the restore', async () => {
    const archive = Uint8Array.from({ length: 300 }, (_, i) => i % 251)
    const { calls, request } = fakeBackends(archive, { flaky: 3 })
    const phases: TransferProgress['phase'][] = []

    const result = await transferAgent(request, SOURCE, TARGET, p => phases.push(p.phase), { timing: instant })

    expect(result.undoArchive).toBe('/cloud/pre-import.zip')
    expect(phases[0]).toBe('backup')
    expect(phases).toContain('upload')
    expect(phases.at(-1)).toBe('restore')

    // Source calls stay on the source owner; the chunk read names the archive and the 8 MiB cap.
    const chunk = calls.find(c => c.req.path.startsWith('/api/ops/backup/chunk'))!
    expect(chunk.owner).toEqual(SOURCE)
    expect(chunk.req.path).toContain(`archive=${encodeURIComponent('/home/u/.hermes/backups/b.zip')}`)
    expect(chunk.req.path).toContain(`length=${CHUNK_BYTES}`)

    // The upload is one multipart file to the cloud agent's default profile, keeping its host.
    const upload = calls.find(c => c.req.path.startsWith('/api/ops/import-upload'))!
    expect(upload.owner).toEqual({ connectionId: 'cloud-1', profile: 'default' })
    expect(upload.req).toMatchObject({
      method: 'POST',
      path: '/api/ops/import-upload?keep_host=true',
      timeoutMs: 1_800_000,
      upload: { contentType: 'application/zip', filename: 'hermes-transfer.zip' }
    })
    expect(new Uint8Array(upload.req.upload!.bytes)).toEqual(archive)

    // It polled the import on the target, riding out the paused gateway.
    expect(calls.some(c => c.owner === TARGET && c.req.path === '/api/actions/import/status?lines=40')).toBe(true)
  })

  it('surfaces a failed restore with its last lines', async () => {
    const request: TransferRequestFn = async <T>(_owner: TransferEndpoint, req: HermesApiRequest) =>
      (req.path.startsWith('/api/actions/import/status')
        ? { exit_code: 1, lines: ['a', 'b', 'c', 'Error: bad archive'], running: false }
        : req.path.startsWith('/api/actions/')
          ? { exit_code: 0, lines: [], running: false }
          : req.path.startsWith('/api/ops/backup/chunk')
            ? { data: b64(new Uint8Array([1])), done: true, offset: 0, size: 1 }
            : { archive: '/x.zip', name: req.path.includes('import') ? 'import' : 'backup' }) as T

    await expect(transferAgent(request, SOURCE, TARGET, () => undefined, { timing: instant })).rejects.toThrow(
      'b\nc\nError: bad archive'
    )
  })
})

describe('undoTransfer', () => {
  it('restores the cloud agent from its own pre-import backup, keeping its host', async () => {
    const { calls, request } = fakeBackends(new Uint8Array())

    await undoTransfer(request, TARGET, '/cloud/pre-import.zip', () => undefined, instant)

    expect(calls[0]).toEqual({
      owner: TARGET,
      req: {
        body: { archive: '/cloud/pre-import.zip', force: true, keep_host: true },
        method: 'POST',
        path: '/api/ops/import'
      }
    })
    expect(calls.at(-1)?.req.path).toBe('/api/actions/import/status?lines=40')
  })
})
