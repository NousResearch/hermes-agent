import { beforeEach, describe, expect, it, vi } from 'vitest'

const { readDesktopFileDataUrl } = vi.hoisted(() => ({ readDesktopFileDataUrl: vi.fn() }))

vi.mock('@/lib/desktop-fs', () => ({
  isDesktopFsRemoteMode: () => true,
  readDesktopFileDataUrl,
  readDesktopFileText: vi.fn()
}))

import {
  localPreviewTarget,
  normalizeOrLocalPreviewTarget,
  openPreviewTargetInBrowser,
  remoteHtmlPreviewDocument,
  validatedRemoteHtmlDataUrl
} from './local-preview'

const remoteTarget = {
  kind: 'file' as const,
  label: 'report.html',
  path: '/srv/report.html',
  previewKind: 'html' as const,
  source: '/srv/report.html',
  url: 'file:///srv/report.html'
}

describe('remote HTML previews', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    window.hermesDesktop = {
      normalizePreviewTarget: vi.fn(async () => remoteTarget)
    } as never
  })

  it('loads bounded authenticated bytes while retaining the canonical file URL', async () => {
    const dataUrl = `data:text/html;base64,${btoa('<h1>remote</h1>')}`
    readDesktopFileDataUrl.mockResolvedValue(dataUrl)

    await expect(normalizeOrLocalPreviewTarget('/srv/report.html')).resolves.toMatchObject({
      ...remoteTarget,
      dataUrl
    })
    expect(readDesktopFileDataUrl).toHaveBeenCalledWith('/srv/report.html')
  })

  it('keeps remote POSIX paths out of the Windows preview normalizer', async () => {
    const normalizePreviewTarget = vi.fn(async () => ({
      ...remoteTarget,
      path: '\\\\wsl.localhost\\Ubuntu-20.04\\srv\\report.html',
      url: 'file://wsl.localhost/Ubuntu-20.04/srv/report.html'
    }))

    window.hermesDesktop = { normalizePreviewTarget } as never
    readDesktopFileDataUrl.mockResolvedValue(`data:text/html;base64,${btoa('<h1>remote</h1>')}`)

    await expect(normalizeOrLocalPreviewTarget('/srv/report.html')).resolves.toMatchObject({
      path: '/srv/report.html',
      url: 'file:///srv/report.html'
    })
    expect(normalizePreviewTarget).not.toHaveBeenCalled()
    expect(readDesktopFileDataUrl).toHaveBeenCalledWith('/srv/report.html')
  })

  it('resolves relative remote paths against a POSIX gateway cwd', async () => {
    const normalizePreviewTarget = vi.fn(async () => remoteTarget)

    window.hermesDesktop = { normalizePreviewTarget } as never
    readDesktopFileDataUrl.mockResolvedValue(`data:text/html;base64,${btoa('<h1>remote</h1>')}`)

    await expect(normalizeOrLocalPreviewTarget('report.html', '/srv')).resolves.toMatchObject({
      path: '/srv/report.html',
      url: 'file:///srv/report.html'
    })
    expect(normalizePreviewTarget).not.toHaveBeenCalled()
    expect(readDesktopFileDataUrl).toHaveBeenCalledWith('/srv/report.html')
  })

  it('keeps remote POSIX file URLs out of the local normalizer', async () => {
    const normalizePreviewTarget = vi.fn(async () => remoteTarget)

    window.hermesDesktop = { normalizePreviewTarget } as never
    readDesktopFileDataUrl.mockResolvedValue(`data:text/html;base64,${btoa('<h1>remote</h1>')}`)

    await expect(normalizeOrLocalPreviewTarget('file:///srv/report.html')).resolves.toMatchObject({
      path: '/srv/report.html',
      url: 'file:///srv/report.html'
    })
    expect(normalizePreviewTarget).not.toHaveBeenCalled()
    expect(readDesktopFileDataUrl).toHaveBeenCalledWith('/srv/report.html')
  })

  it.each([
    ['https://example.test/report.html', '/srv'],
    ['\\\\server\\share\\report.html', '/srv'],
    ['file://server/share/report.html', '/srv'],
    ['C:\\reports\\report.html', '/srv']
  ])('leaves non-POSIX target %s to the Electron normalizer', async (target, cwd) => {
    const normalizePreviewTarget = vi.fn(async () => remoteTarget)

    window.hermesDesktop = { normalizePreviewTarget } as never
    readDesktopFileDataUrl.mockResolvedValue(`data:text/html;base64,${btoa('<h1>remote</h1>')}`)

    await normalizeOrLocalPreviewTarget(target, cwd)

    expect(normalizePreviewTarget).toHaveBeenCalledWith(target, cwd)
  })

  it('falls back to source mode when the transport is not canonical HTML', async () => {
    readDesktopFileDataUrl.mockResolvedValue('data:text/plain;base64,SGVsbG8=')

    await expect(normalizeOrLocalPreviewTarget('/srv/report.html')).resolves.toMatchObject({
      renderMode: 'source',
      transient: true,
      url: 'file:///srv/report.html'
    })
  })

  it('rejects malformed base64', () => {
    expect(validatedRemoteHtmlDataUrl('data:text/html;base64,SGVsbG8=')).toBe('data:text/html;base64,SGVsbG8=')
    expect(validatedRemoteHtmlDataUrl('data:text/html;base64,SGVsbG8')).toBeNull()
    expect(validatedRemoteHtmlDataUrl('data:text/html;base64,SGVsbG8%3D')).toBeNull()
  })

  it('wraps remote HTML in a deny-by-default content policy', () => {
    const html =
      '<meta http-equiv="refresh" content="0;url=https://example.test"><a href="https://example.test">remote</a>'

    const document = remoteHtmlPreviewDocument(`data:text/html;base64,${btoa(html)}`)

    expect(document).toContain(`default-src 'none'`)
    expect(document).toContain(`form-action 'none'`)
    expect(document).toContain('<a>remote</a>')
    expect(document).not.toContain('refresh')
    expect(document).not.toContain('https://example.test')
  })

  it('strips scripts from remote HTML', () => {
    const html =
      '<p>safe</p><script>document.body.textContent = "SCRIPT-RAN"</script><template><script>TEMPLATE-SCRIPT</script></template>'

    const document = remoteHtmlPreviewDocument(`data:text/html;base64,${btoa(html)}`)

    expect(document).toContain('<p>safe</p>')
    expect(document).not.toContain('<script')
    expect(document).not.toContain('SCRIPT-RAN')
    expect(document).not.toContain('TEMPLATE-SCRIPT')
  })

  it('does not create scripts when sanitized HTML is reparsed', () => {
    const html = '<form><math><mtext></form><form><mglyph><style></math><script>MUTATION-SCRIPT</script>'

    const sanitized = remoteHtmlPreviewDocument(`data:text/html;base64,${btoa(html)}`)

    expect(sanitized).not.toContain('<script')
    expect(new DOMParser().parseFromString(sanitized ?? '', 'text/html').querySelector('script')).toBeNull()
  })

  it('decodes remote HTML as UTF-8', () => {
    const html = '<p>café 😀</p>'
    const payload = btoa(String.fromCharCode(...new TextEncoder().encode(html)))

    expect(remoteHtmlPreviewDocument(`data:text/html;base64,${payload}`)).toContain(html)
  })

  it('stages remote HTML before opening it in the system browser', async () => {
    const dataUrl = `data:text/html;base64,${btoa('<h1>remote</h1>')}`
    const saveImageBuffer = vi.fn(async (_data: ArrayBuffer | Uint8Array, _ext: string) => '/tmp/report #1?.html')
    const openPreviewInBrowser = vi.fn(async () => undefined)
    window.hermesDesktop = { openPreviewInBrowser, saveImageBuffer } as never

    await openPreviewTargetInBrowser({ ...remoteTarget, dataUrl })

    expect(new TextDecoder().decode(saveImageBuffer.mock.calls[0]?.[0])).toBe('<h1>remote</h1>')
    expect(saveImageBuffer).toHaveBeenCalledWith(expect.any(Uint8Array), '.html')
    expect(openPreviewInBrowser).toHaveBeenCalledWith('file:///tmp/report%20%231%3F.html')
  })

  it('serializes UNC staging paths as file URLs', async () => {
    const dataUrl = `data:text/html;base64,${btoa('<h1>remote</h1>')}`
    const saveImageBuffer = vi.fn(async () => '\\\\server\\share\\report #1.html')
    const openPreviewInBrowser = vi.fn(async () => undefined)
    window.hermesDesktop = { openPreviewInBrowser, saveImageBuffer } as never

    await openPreviewTargetInBrowser({ ...remoteTarget, dataUrl })

    expect(openPreviewInBrowser).toHaveBeenCalledWith('file://server/share/report%20%231.html')
  })

  it('preserves backslashes in POSIX local file paths', () => {
    expect(localPreviewTarget('/tmp/report\\draft.html')?.url).toBe('file:///tmp/report%5Cdraft.html')
  })

  it('preserves POSIX double-slash file paths', () => {
    expect(localPreviewTarget('//srv/share/report #1?.html')?.url).toBe('file:////srv/share/report%20%231%3F.html')
  })

  it('opens ordinary targets without staging them', async () => {
    const openPreviewInBrowser = vi.fn(async () => undefined)
    const saveImageBuffer = vi.fn()
    window.hermesDesktop = { openPreviewInBrowser, saveImageBuffer } as never

    await openPreviewTargetInBrowser(remoteTarget)

    expect(saveImageBuffer).not.toHaveBeenCalled()
    expect(openPreviewInBrowser).toHaveBeenCalledWith(remoteTarget.url)
  })

  it('keeps local HTML source browser opens on their existing path', async () => {
    const openPreviewInBrowser = vi.fn(async () => undefined)
    window.hermesDesktop = { openPreviewInBrowser } as never

    await openPreviewTargetInBrowser({
      ...remoteTarget,
      renderMode: 'source',
      source: '/tmp/local.html',
      url: 'file:///tmp/local.html'
    })

    expect(openPreviewInBrowser).toHaveBeenCalledWith('file:///tmp/local.html')
  })

  it('does not send failed remote HTML paths to the local browser', async () => {
    const openPreviewInBrowser = vi.fn(async () => undefined)
    window.hermesDesktop = { openPreviewInBrowser } as never

    await expect(
      openPreviewTargetInBrowser({ ...remoteTarget, renderMode: 'source', transient: true })
    ).rejects.toThrow('Remote HTML preview could not be loaded')
    expect(openPreviewInBrowser).not.toHaveBeenCalled()
  })
})

describe('PDF previews', () => {
  it('classifies PDF files as PDF previews', () => {
    expect(localPreviewTarget('/tmp/spec.pdf')).toMatchObject({
      path: '/tmp/spec.pdf',
      previewKind: 'pdf'
    })
  })

  it('keeps ordinary text files on the source-preview path', () => {
    expect(localPreviewTarget('/tmp/spec.md')).toMatchObject({
      language: 'markdown',
      previewKind: 'text'
    })
  })

  it('does not UTF-8-enrich remote PDFs before loading their bytes', async () => {
    vi.clearAllMocks()
    window.hermesDesktop = {
      normalizePreviewTarget: vi.fn(async () => null)
    } as never

    await expect(normalizeOrLocalPreviewTarget('/remote/spec.pdf')).resolves.toMatchObject({
      path: '/remote/spec.pdf',
      previewKind: 'pdf'
    })
    expect(readDesktopFileDataUrl).not.toHaveBeenCalled()
  })
})
