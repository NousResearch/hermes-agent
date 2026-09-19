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
    ).rejects.toThrow('huge.csv is too large to upload to the remote gateway (max 16 MB).')

    // The cap is hit before any gateway round-trip.
    expect(requestGateway).not.toHaveBeenCalled()
  })

  it('passes non-cap read errors through unchanged', async () => {
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {
        readFileDataUrl: vi.fn(async () => {
          throw new Error('ENOENT: no such file')
        })
      }
    })

    await expect(
      uploadComposerAttachment(
        { id: 'file:gone', kind: 'file', label: 'gone.csv', path: '/abs/gone.csv' },
        { remote: true, requestGateway: vi.fn(async () => ({}) as never), sessionId: RUNTIME_SESSION_ID }
      )
    ).rejects.toThrow('ENOENT: no such file')
  })
})

describe('uploadComposerAttachment preview reuse', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('reuses the chip previewUrl instead of re-reading the image off disk', async () => {
    // attachImagePath already read the full file for the thumbnail; submit
    // must not pay the disk read + IPC round-trip a second time.
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
        previewUrl: 'data:image/png;base64,ZnJvbS1wcmV2aWV3'
      },
      { remote: true, requestGateway, sessionId: RUNTIME_SESSION_ID }
    )

    expect(readFileDataUrl).not.toHaveBeenCalled()
    expect(requestGateway).toHaveBeenCalledWith('image.attach_bytes', {
      content_base64: 'ZnJvbS1wcmV2aWV3',
      filename: 'shot.png',
      session_id: RUNTIME_SESSION_ID
    })
    expect(uploaded.path).toBe('/gw/images/shot.png')
  })

  it('falls back to the disk read when previewUrl is not a base64 data URL', async () => {
    // A non-data previewUrl (e.g. a gateway media URL) carries no bytes —
    // the upload must still read the real file.
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

    await uploadComposerAttachment(
      {
        id: 'image:shot.png',
        kind: 'image',
        label: 'shot.png',
        path: '/local/shot.png',
        previewUrl: 'https://gateway.example/media/shot.png'
      },
      { remote: true, requestGateway, sessionId: RUNTIME_SESSION_ID }
    )

    expect(readFileDataUrl).toHaveBeenCalledWith('/local/shot.png')
    expect(requestGateway).toHaveBeenCalledWith('image.attach_bytes', {
      content_base64: 'ZnJvbS1kaXNr',
      filename: 'shot.png',
      session_id: RUNTIME_SESSION_ID
    })
  })
})
