import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { PreviewAttachment } from '@/components/chat/preview-attachment'
import type * as PreviewStore from '@/store/preview'
import { $activeSessionId, $connection } from '@/store/session'

// #101683 regression coverage: the delivered-file card must branch on the
// target's LOCAL filesystem state — an existing directory opens natively with
// no Download and no broken preview, an existing local file reveals instead of
// re-downloading, a remote-backend file keeps the gateway-backed actions, and a
// missing path reports instead of fabricating a preview tab.

const { isDesktopFsRemoteMode, notifyError, openPreview, sessionFileOwnerScope } = vi.hoisted(() => ({
  isDesktopFsRemoteMode: vi.fn(() => false),
  notifyError: vi.fn(),
  openPreview: vi.fn(),
  sessionFileOwnerScope: vi.fn()
}))

vi.mock('@/lib/desktop-fs', () => ({
  isDesktopFsRemoteMode,
  readDesktopDir: vi.fn(),
  readDesktopFileDataUrl: vi.fn(),
  readDesktopFileText: vi.fn()
}))

vi.mock('@/api/client', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  sessionFileOwnerScope
}))

vi.mock('@/store/notifications', () => ({ notifyError }))

vi.mock('@/store/preview', async importOriginal => ({
  ...(await importOriginal<typeof PreviewStore>()),
  $previewTabSources: atom<string[]>([]),
  closePreviewForSource: vi.fn(),
  openPreview
}))

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

describe('PreviewAttachment download owner scope', () => {
  const previousConnection = $connection.get()

  beforeEach(() => {
    vi.clearAllMocks()
    isDesktopFsRemoteMode.mockReturnValue(true)
  })

  afterEach(() => {
    cleanup()
    window.hermesDesktop = previousDesktop
    $activeSessionId.set(null)
    $connection.set(previousConnection)
  })

  function mountSaveStub() {
    const saveGatewayFile = vi.fn(async () => ({ saved: true }))

    window.hermesDesktop = { saveGatewayFile } as never

    return saveGatewayFile
  }

  async function clickDownload() {
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Download' }))
    })
  }

  it('rides the owner profile and connection when the session belongs to another profile', async () => {
    const saveGatewayFile = mountSaveStub()

    sessionFileOwnerScope.mockReturnValue({ connectionId: 'gateway-1', profile: 'agent-profile' })
    $activeSessionId.set('20261125_142659_000000')

    render(<PreviewAttachment target="/srv/report.md" />)

    await clickDownload()

    expect(sessionFileOwnerScope).toHaveBeenCalledWith('20261125_142659_000000')
    expect(saveGatewayFile).toHaveBeenCalledWith(
      expect.objectContaining({
        path: '/srv/report.md',
        connectionId: 'gateway-1',
        profile: 'agent-profile',
        sessionId: '20261125_142659_000000'
      })
    )
  })

  it('keeps a profile-only session owner on the window connection', async () => {
    const saveGatewayFile = mountSaveStub()

    $connection.set({ connectionId: 'window-gateway', mode: 'remote', profile: 'foreground-profile' } as never)
    sessionFileOwnerScope.mockReturnValue({ profile: 'agent-profile' })
    $activeSessionId.set('20261125_142659_000000')

    render(<PreviewAttachment target="/srv/report.md" />)

    await clickDownload()

    expect(saveGatewayFile).toHaveBeenCalledWith(
      expect.objectContaining({
        connectionId: 'window-gateway',
        path: '/srv/report.md',
        profile: 'agent-profile',
        sessionId: '20261125_142659_000000'
      })
    )
  })

  it('names no session when the session owner profile is unknown', async () => {
    const saveGatewayFile = mountSaveStub()

    sessionFileOwnerScope.mockReturnValue(null)
    $activeSessionId.set('20261125_142659_000000')

    render(<PreviewAttachment target="/srv/report.md" />)

    await clickDownload()

    expect(sessionFileOwnerScope).toHaveBeenCalledWith('20261125_142659_000000')
    expect(saveGatewayFile).toHaveBeenCalledWith(expect.not.objectContaining({ sessionId: expect.any(String) }))
    expect(saveGatewayFile).toHaveBeenCalledWith(expect.objectContaining({ path: '/srv/report.md' }))
  })
})
