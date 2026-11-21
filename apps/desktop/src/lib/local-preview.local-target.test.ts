import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// #101683 pipeline invariant: a typed non-previewable normalization result
// (directory / missing) must NOT fall through to the renderer's blind local
// classification, which fabricates a text-preview tab for a path that cannot
// be previewed. A remote backend's paths are not on this machine, so remote
// mode keeps the fabricated fallback the gateway read resolves later.

const { isDesktopFsRemoteMode, readDesktopFileText } = vi.hoisted(() => ({
  isDesktopFsRemoteMode: vi.fn(() => false),
  readDesktopFileText: vi.fn()
}))

vi.mock('@/lib/desktop-fs', () => ({
  isDesktopFsRemoteMode,
  readDesktopDir: vi.fn(),
  readDesktopFileDataUrl: vi.fn(),
  readDesktopFileText
}))

import { normalizeOrLocalPreviewTarget } from './local-preview'

const previousDesktop = window.hermesDesktop

function stubNormalization(result: unknown) {
  window.hermesDesktop = {
    normalizePreviewTarget: vi.fn(async () => result)
  } as never
}

describe('normalizeOrLocalPreviewTarget non-previewable results', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    isDesktopFsRemoteMode.mockReturnValue(false)
  })

  afterEach(() => {
    window.hermesDesktop = previousDesktop
  })

  it('returns null for a directory in local mode instead of fabricating a text tab', async () => {
    stubNormalization({
      kind: 'file',
      label: 'historical',
      path: '/work/reports/historical',
      previewKind: 'directory',
      source: '/work/reports/historical',
      url: 'file:///work/reports/historical'
    })

    await expect(normalizeOrLocalPreviewTarget('/work/reports/historical')).resolves.toBeNull()
  })

  it('returns null for a missing path in local mode', async () => {
    stubNormalization({
      kind: 'file',
      label: 'gone.md',
      path: '/work/gone.md',
      previewKind: 'missing',
      source: '/work/gone.md',
      url: 'file:///work/gone.md'
    })

    await expect(normalizeOrLocalPreviewTarget('/work/gone.md')).resolves.toBeNull()
  })

  it('treats a null main-process result as authoritative in local mode', async () => {
    stubNormalization(null)

    await expect(normalizeOrLocalPreviewTarget('/work/nope.md')).resolves.toBeNull()
  })

  it('resolves remote-backend paths without the local normalizer', async () => {
    isDesktopFsRemoteMode.mockReturnValue(true)
    readDesktopFileText.mockResolvedValue({ language: 'markdown' })

    // What the main process answers on Windows when this host also has WSL.
    stubNormalization({
      kind: 'file',
      label: 'notes.md',
      path: '\\\\wsl.localhost\\Ubuntu\\srv\\notes.md',
      previewKind: 'text',
      source: '/srv/notes.md',
      url: 'file://wsl.localhost/Ubuntu/srv/notes.md'
    })

    await expect(normalizeOrLocalPreviewTarget('/srv/notes.md')).resolves.toMatchObject({
      path: '/srv/notes.md',
      url: 'file:///srv/notes.md'
    })
    expect(window.hermesDesktop.normalizePreviewTarget).not.toHaveBeenCalled()
    expect(readDesktopFileText).toHaveBeenCalledWith('/srv/notes.md', undefined)
  })
})
