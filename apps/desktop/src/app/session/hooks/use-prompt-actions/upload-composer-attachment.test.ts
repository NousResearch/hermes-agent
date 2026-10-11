import { JsonRpcGatewayError } from '@hermes/shared'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { uploadComposerAttachment } from '.'

// Per-file copy, matching the convention of the other shards in this directory
// (preview-open.test.tsx, session-tile-actions.test.ts): the id the desktop
// holds is the *runtime* session id from session.create.
const RUNTIME_SESSION_ID = 'rt-abc123'

// Kept byte-identical to index.test.tsx: both specifiers are alias-based and
// this file sits in the same directory, so the mocks resolve to the same
// modules the parent suite mocks.
vi.mock('@/hermes', () => ({
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  getSession: vi.fn(),
  PROMPT_SUBMIT_REQUEST_TIMEOUT_MS: 1_800_000,
  setApiRequestProfile: vi.fn(),
  transcribeAudio: vi.fn()
}))

vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  requestGatewayForAgent: vi.fn()
}))

describe('uploadComposerAttachment remote read failures', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('turns the raw 16MB IPC cap error into a friendly remote-gateway message', async () => {
    // electron/hardening.ts rejects the readFileDataUrl IPC with this exact
    // shape when a file exceeds the configured data-URL read cap.
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {
        readFileDataUrl: vi.fn(async () => {
          throw new Error('File preview failed: file is too large (20971520 bytes; limit 16777216 bytes).')
        })
      }
    })

    const requestGateway = vi.fn(async () => ({}) as never)

    await expect(
      uploadComposerAttachment(
        { id: 'file:big', kind: 'file', label: 'huge.csv', path: '/abs/huge.csv' },
        { remote: true, requestGateway, sessionId: RUNTIME_SESSION_ID }
      )
    ).rejects.toThrow(/huge\.csv.*16 MB/)

    // The cap is hit before any gateway round-trip.
    expect(requestGateway).not.toHaveBeenCalled()
  })
})

describe('uploadComposerAttachment image cache contract', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('always re-reads the image from disk, even when a stale previewUrl is cached', async () => {
    // attachImagePath drops `previewUrl` as soon as a thumbnail exists, so a
    // populated `previewUrl` here does not mean it holds full-resolution
    // bytes (#93324) — the upload must never trust it and must always read
    // the on-disk file for the bytes the model receives.
    const readFileDataUrl = vi.fn(async () => 'data:image/png;base64,ZnJvbS1kaXNr')
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { readFileDataUrl }
    })

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'image.attach_bytes') {
        return { attached: true, path: '/gw/images/shot.png' } as never
      }

      return {} as never
    })

    const uploaded = await uploadComposerAttachment(
      {
        id: 'image:shot.png',
        kind: 'image',
        label: 'shot.png',
        path: '/local/shot.png',
        previewUrl: 'data:image/png;base64,c3RhbGUtY2FjaGVk'
      },
      { remote: true, requestGateway, sessionId: RUNTIME_SESSION_ID }
    )

    expect(readFileDataUrl).toHaveBeenCalledWith('/local/shot.png')
    expect(requestGateway).toHaveBeenCalledWith('image.attach_bytes', {
      content_base64: 'ZnJvbS1kaXNr',
      filename: 'shot.png',
      session_id: RUNTIME_SESSION_ID
    })
    expect(uploaded.path).toBe('/gw/images/shot.png')
  })
})

describe('uploadComposerAttachment local path-only fallback', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('retries once with data_url when a local path-only attach is unresolved gateway-side', async () => {
    // Local gateway, POSIX host path: attachmentPathNeedsUpload() is false, so
    // the first file.attach carries the path only. When the gateway cannot
    // stat that path (5028 "file not found on gateway"), the renderer still
    // holding readable bytes must cross the boundary instead of failing the
    // prompt (#135496).
    const readFileDataUrl = vi.fn(async () => 'data:application/pdf;base64,ZnJhbWVk')
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { readFileDataUrl }
    })

    const requestGateway = vi
      .fn()
      .mockRejectedValueOnce(
        new JsonRpcGatewayError('file not found on gateway and no data_url provided', { code: 5028 })
      )
      .mockResolvedValueOnce({ attached: true, ref_text: '@file:attachments/report.pdf' } as never)

    const uploaded = await uploadComposerAttachment(
      { id: 'file:report', kind: 'file', label: 'report.pdf', path: '/tmp/export/report.pdf' },
      { remote: false, requestGateway, sessionId: RUNTIME_SESSION_ID }
    )

    expect(requestGateway).toHaveBeenCalledTimes(2)
    expect(requestGateway).toHaveBeenNthCalledWith(1, 'file.attach', {
      name: 'report.pdf',
      path: '/tmp/export/report.pdf',
      session_id: RUNTIME_SESSION_ID
    })
    expect(requestGateway).toHaveBeenNthCalledWith(2, 'file.attach', {
      name: 'report.pdf',
      path: '/tmp/export/report.pdf',
      session_id: RUNTIME_SESSION_ID,
      data_url: 'data:application/pdf;base64,ZnJhbWVk'
    })
    expect(uploaded.refText).toBe('@file:attachments/report.pdf')
  })

  it('names the file when neither the gateway nor the renderer can read the path', async () => {
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {
        readFileDataUrl: vi.fn(async () => null)
      }
    })

    const requestGateway = vi.fn(async () => {
      throw new JsonRpcGatewayError('file not found on gateway and no data_url provided', { code: 5028 })
    })

    await expect(
      uploadComposerAttachment(
        { id: 'file:gone', kind: 'file', label: 'vanished.txt', path: '/tmp/export/vanished.txt' },
        { remote: false, requestGateway, sessionId: RUNTIME_SESSION_ID }
      )
    ).rejects.toThrow(/file not found on gateway and no data_url provided: vanished\.txt/)

    // No second round-trip: the renderer could not produce the bytes either.
    expect(requestGateway).toHaveBeenCalledTimes(1)
  })

  it('does not fall back for a 5028 that is not the unresolved-path failure', async () => {
    // The same 5028 code also carries pdftoppm rendering failures; those are
    // deterministic and replaying them with bytes would just re-run the
    // renderer, so the error must surface as-is.
    const readFileDataUrl = vi.fn(async () => 'data:application/pdf;base64,ZnJhbWVk')
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { readFileDataUrl }
    })

    const requestGateway = vi.fn(async () => {
      throw new JsonRpcGatewayError('pdftoppm failed: syntax error', { code: 5028 })
    })

    await expect(
      uploadComposerAttachment(
        { id: 'file:broken', kind: 'file', label: 'broken.pdf', path: '/tmp/export/broken.pdf' },
        { remote: false, requestGateway, sessionId: RUNTIME_SESSION_ID }
      )
    ).rejects.toThrow(/pdftoppm failed/)

    expect(readFileDataUrl).not.toHaveBeenCalled()
    expect(requestGateway).toHaveBeenCalledTimes(1)
  })
})
