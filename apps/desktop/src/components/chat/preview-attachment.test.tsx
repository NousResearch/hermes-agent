import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { atom, type WritableAtom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { PreviewAttachment } from '@/components/chat/preview-attachment'
import type * as PreviewStore from '@/store/preview'

// #101683 regression coverage: the delivered-file card must branch on the
// target's LOCAL filesystem state — an existing directory opens natively with
// no Download and no broken preview, an existing local file reveals instead of
// re-downloading, a remote-backend file keeps the gateway-backed actions, and a
// missing path reports instead of fabricating a preview tab.

const { isDesktopFsRemoteMode, notifyError, openPreview } = vi.hoisted(() => ({
  isDesktopFsRemoteMode: vi.fn(() => false),
  notifyError: vi.fn(),
  openPreview: vi.fn()
}))

vi.mock('@/lib/desktop-fs', () => ({
  isDesktopFsRemoteMode,
  readDesktopDir: vi.fn(),
  readDesktopFileDataUrl: vi.fn(),
  readDesktopFileText: vi.fn()
}))

vi.mock('@/store/notifications', () => ({ notifyError }))

vi.mock('@/store/preview', async importOriginal => ({
  ...(await importOriginal<typeof PreviewStore>()),
  $previewTabSources: atom<string[]>([]),
  closePreviewForSource: vi.fn(),
  openPreview
}))

const { resolveSessionOwner } = vi.hoisted(() => ({ resolveSessionOwner: vi.fn() }))

// The session view's ids decide which backend a download pins to. The box is
// (re)built in beforeEach; tests that exercise download routing set values on
// it, everything else keeps the stock null ids.
const sessionBox = vi.hoisted(() => ({
  atoms: null as null | {
    $cwd: WritableAtom<string | null>
    $runtimeId: WritableAtom<string | null>
    $storedId: WritableAtom<string | null>
  }
}))

vi.mock('@/app/chat/session-view', () => ({
  useSessionView: () => sessionBox.atoms
}))

vi.mock('@/app/session/hooks/use-session-actions/utils', () => ({ resolveSessionOwner }))

const { downloadGatewayMediaFile } = vi.hoisted(() => ({
  downloadGatewayMediaFile: vi.fn(async () => ({ saved: true }))
}))

vi.mock('@/lib/media', () => ({ downloadGatewayMediaFile }))

const previousDesktop = window.hermesDesktop

function fileTarget(path: string, previewKind = 'text') {
  return {
    kind: 'file' as const,
    label: path.split('/').filter(Boolean).pop() || path,
    path,
    previewKind,
    source: `file://${path}`,
    url: `file://${path}`
  }
}

function mountDesktopStub(normalizePreviewTarget: (target: string, baseDir?: string) => unknown) {
  const openDir = vi.fn(async () => ({ ok: true }))
  const revealPath = vi.fn(async () => true)

  window.hermesDesktop = {
    normalizePreviewTarget: vi.fn(normalizePreviewTarget),
    openDir,
    revealPath
  } as never

  return { openDir, revealPath }
}

async function buttons() {
  return (await screen.findAllByRole('button')).map(button => button.textContent)
}

describe('PreviewAttachment local target classification (#101683)', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    isDesktopFsRemoteMode.mockReturnValue(false)
    resolveSessionOwner.mockResolvedValue(undefined)
    sessionBox.atoms = {
      $cwd: atom<string | null>(null),
      $runtimeId: atom<string | null>(null),
      $storedId: atom<string | null>(null)
    }
  })

  afterEach(() => {
    cleanup()
    window.hermesDesktop = previousDesktop
  })

  it('opens a Windows file:///C:/... directory link natively with no Download or preview action', async () => {
    const { openDir, revealPath } = mountDesktopStub(() => fileTarget('C:/Users/E/reports/historical', 'directory'))

    render(<PreviewAttachment target="file:///C:/Users/E/reports/historical" />)

    const open = await screen.findByRole('button', { name: 'Open containing folder' })
    expect(await buttons()).toEqual(['Open containing folder'])

    await act(async () => {
      fireEvent.click(open)
    })

    expect(openDir).toHaveBeenCalledWith('C:/Users/E/reports/historical')
    expect(revealPath).not.toHaveBeenCalled()
    expect(openPreview).not.toHaveBeenCalled()
  })

  it('opens a POSIX directory link through the native folder action', async () => {
    const { openDir } = mountDesktopStub(() => fileTarget('/work/reports/historical', 'directory'))

    render(<PreviewAttachment target="/work/reports/historical" />)

    const open = await screen.findByRole('button', { name: 'Open containing folder' })

    await act(async () => {
      fireEvent.click(open)
    })

    expect(openDir).toHaveBeenCalledWith('/work/reports/historical')
    expect(openPreview).not.toHaveBeenCalled()
  })

  it('reveals an existing local file in the file manager instead of offering Download', async () => {
    const { openDir, revealPath } = mountDesktopStub(() => fileTarget('/work/report.md'))

    render(<PreviewAttachment target="/work/report.md" />)

    await screen.findByRole('button', { name: 'Open containing folder' })
    expect(await buttons()).toEqual(['Open containing folder', 'Open preview'])

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Open containing folder' }))
    })

    expect(revealPath).toHaveBeenCalledWith('/work/report.md')
    expect(openDir).not.toHaveBeenCalled()
  })

  it('keeps Download and the preview action for a remote-backend file', async () => {
    isDesktopFsRemoteMode.mockReturnValue(true)
    const { openDir, revealPath } = mountDesktopStub(() => fileTarget('/srv/report.md'))

    render(<PreviewAttachment target="/srv/report.md" />)

    expect(await buttons()).toEqual(['Download', 'Open preview'])

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Open preview' }))
    })

    // The remote path is not on this machine: no native folder action fires.
    expect(openDir).not.toHaveBeenCalled()
    expect(revealPath).not.toHaveBeenCalled()
  })

  // A download must pin to the SESSION's own route: a local-route chat can be
  // shown over a secondary route while the window's published connection is a
  // remote primary — an unpinned save then 404s ("Session not found") on the
  // remote for a path that only exists on this machine.
  it("pins a file download to the session's own route, not the window connection", async () => {
    isDesktopFsRemoteMode.mockReturnValue(true)
    mountDesktopStub(() => fileTarget('/srv/Atkar-Gateway-Full-Picture.pptx'))
    resolveSessionOwner.mockResolvedValue({ connectionId: 'local', profile: 'laptop-it-suppot' })
    sessionBox.atoms?.$storedId.set('local-83b228174b7cc37fc572e031e35a216981b250065da6ddf488ffdc4c9d3bb413')

    render(<PreviewAttachment target="/srv/Atkar-Gateway-Full-Picture.pptx" />)

    const download = await screen.findByRole('button', { name: 'Download' })

    await act(async () => {
      fireEvent.click(download)
    })

    expect(resolveSessionOwner).toHaveBeenCalledWith('local-83b228174b7cc37fc572e031e35a216981b250065da6ddf488ffdc4c9d3bb413')
    expect(downloadGatewayMediaFile).toHaveBeenCalledWith('/srv/Atkar-Gateway-Full-Picture.pptx', {
      sessionId: 'local-83b228174b7cc37fc572e031e35a216981b250065da6ddf488ffdc4c9d3bb413',
      owner: { connectionId: 'local', profile: 'laptop-it-suppot' }
    })
  })

  it('keeps the ambient fallback when no session owner resolves', async () => {
    isDesktopFsRemoteMode.mockReturnValue(true)
    mountDesktopStub(() => fileTarget('/srv/orphan.bin'))
    sessionBox.atoms?.$storedId.set('unknown-session')

    render(<PreviewAttachment target="/srv/orphan.bin" />)

    const download = await screen.findByRole('button', { name: 'Download' })

    await act(async () => {
      fireEvent.click(download)
    })

    expect(downloadGatewayMediaFile).toHaveBeenCalledWith('/srv/orphan.bin', { sessionId: 'unknown-session' })
  })

  it('reports a missing path instead of opening a broken preview tab', async () => {
    mountDesktopStub(() => fileTarget('/work/gone.md', 'missing'))

    render(<PreviewAttachment target="/work/gone.md" />)

    const preview = await screen.findByRole('button', { name: 'Preview unavailable' })

    await act(async () => {
      fireEvent.click(preview)
    })

    expect(notifyError).toHaveBeenCalled()
    expect(openPreview).not.toHaveBeenCalled()
  })
})
