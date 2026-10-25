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

  it('keeps the gateway-backed fallback for remote-backend directories', async () => {
    isDesktopFsRemoteMode.mockReturnValue(true)
    stubNormalization({
      kind: 'file',
      label: 'srv',
      path: '/srv/reports',
      previewKind: 'directory',
      source: '/srv/reports',
      url: 'file:///srv/reports'
    })

    const fabricated = await normalizeOrLocalPreviewTarget('/srv/reports')

    expect(fabricated?.kind).toBe('file')
    expect(fabricated?.path).toBe('/srv/reports')
  })

  it('resolves remote relative paths against the remote cwd without consulting the local host', async () => {
    isDesktopFsRemoteMode.mockReturnValue(true)
    readDesktopFileText.mockResolvedValue({ language: 'markdown', path: '/srv/report.md', text: 'remote report' })
    stubNormalization({
      kind: 'file',
      path: 'C:/srv/report.md',
      previewKind: 'text',
      source: 'report.md'
    })

    const target = await normalizeOrLocalPreviewTarget('report.md', '/srv')

    expect(target?.path).toBe('/srv/report.md')
    expect(readDesktopFileText).toHaveBeenCalledWith('/srv/report.md')
    expect(window.hermesDesktop?.normalizePreviewTarget).not.toHaveBeenCalled()
  })

  it('does not borrow the type of a same-named local file for a remote preview', async () => {
    isDesktopFsRemoteMode.mockReturnValue(true)
    readDesktopFileText.mockResolvedValue({ language: 'markdown', path: '/srv/report.md', text: 'remote report' })
    stubNormalization({ kind: 'file', path: '/srv/report.md', previewKind: 'image' })

    const target = await normalizeOrLocalPreviewTarget('/srv/report.md')

    expect(target?.previewKind).toBe('text')
    expect(readDesktopFileText).toHaveBeenCalledWith('/srv/report.md')
  })

  it('leaves a home-relative remote path for the gateway to expand', async () => {
    isDesktopFsRemoteMode.mockReturnValue(true)
    readDesktopFileText.mockResolvedValue({ language: 'markdown', path: '/home/remote/report.md', text: 'remote report' })
    stubNormalization({ kind: 'file', path: '/client/home/report.md', previewKind: 'text' })

    const target = await normalizeOrLocalPreviewTarget('~/report.md', '/srv')

    expect(target?.path).toBe('~/report.md')
    expect(readDesktopFileText).toHaveBeenCalledWith('~/report.md')
  })
})
