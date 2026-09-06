import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { PRIMARY_SESSION_VIEW, SessionViewProvider } from '@/app/chat/session-view'
import { $connection, _resetSessionOwnerHintsForTests, setSessionOwnerHint } from '@/store/session'

import { PreviewAttachment } from '../chat/preview-attachment'
import { ZoomableImage } from '../chat/zoomable-image'

import { MarkdownTextContent } from './markdown-text'

// Regression: a MEDIA-delivered non-media file (pdf, zip, ...) renders as a
// preview card. Right-clicking its filename must offer
// reveal-in-file-manager + Copy Path — the transcript's "where is that file?"
// door — so the user can find the artifact on disk without re-asking the agent.
describe('MEDIA file preview card actions', () => {
  const revealPath = vi.fn().mockResolvedValue(undefined)
  const saveGatewayFile = vi.fn().mockResolvedValue({ path: '/tmp/saved-report.pdf', saved: true })
  const writeText = vi.fn().mockResolvedValue(undefined)
  let originalDesktop: typeof window.hermesDesktop
  let originalClipboard: PropertyDescriptor | undefined

  beforeEach(() => {
    revealPath.mockClear()
    saveGatewayFile.mockClear()
    writeText.mockClear()
    originalDesktop = window.hermesDesktop
    originalClipboard = Object.getOwnPropertyDescriptor(navigator, 'clipboard')
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { revealPath, saveGatewayFile }
    })
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText } })

    $connection.set(null)
  })

  afterEach(() => {
    cleanup()
    _resetSessionOwnerHintsForTests({ storage: true })
    $connection.set(null)
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: originalDesktop
    })

    if (originalClipboard) {
      Object.defineProperty(navigator, 'clipboard', originalClipboard)
    }
  })

  it('right-clicking the preview filename exposes working copy and reveal actions', async () => {
    render(<MarkdownTextContent isRunning={false} text="Done: [report.pdf](#media:%2Ftmp%2Freport.pdf)" />)

    const filename = await screen.findByText('report.pdf')

    await waitFor(() => expect(filename.closest('[data-hermes-context-menu-trigger]')).not.toBeNull())

    fireEvent.contextMenu(filename)

    expect(await screen.findByRole('menuitem', { name: /Open Containing Folder|Reveal in/i })).toBeTruthy()
    fireEvent.click(screen.getByRole('menuitem', { name: /Copy Path/i }))
    await waitFor(() => expect(writeText).toHaveBeenCalledWith('/tmp/report.pdf'))
    await waitFor(() => expect(screen.queryByRole('menuitem', { name: /Copy Path/i })).toBeNull())

    fireEvent.contextMenu(filename)
    fireEvent.click(await screen.findByRole('menuitem', { name: /Open Containing Folder|Reveal in/i }))

    await waitFor(() => expect(revealPath).toHaveBeenCalledWith('/tmp/report.pdf'))
  })

  it('downloads a local file instead of opening it in its associated app', async () => {
    render(<MarkdownTextContent isRunning={false} text="Done: [report.pdf](#media:%2Ftmp%2Freport.pdf)" />)

    fireEvent.click(await screen.findByRole('button', { name: 'Download' }))

    await waitFor(() =>
      expect(saveGatewayFile).toHaveBeenCalledWith({
        connectionId: null,
        path: '/tmp/report.pdf',
        profile: null,
        suggestedName: 'report.pdf'
      })
    )
  })

  it('downloads through the viewed remote session while the ambient connection is local', async () => {
    setSessionOwnerHint('stored-remote', {
      connectionId: 'session-ssh',
      mode: 'remote',
      profile: 'desktop-name',
      targetProfile: 'gateway-name'
    })
    $connection.set({ connectionId: 'ambient-local', mode: 'local', profile: 'default' } as never)

    render(
      <SessionViewProvider
        value={{ ...PRIMARY_SESSION_VIEW, $cwd: atom('/srv/work'), $storedId: atom('stored-remote') }}
      >
        <MarkdownTextContent isRunning={false} text="Done: [report.pdf](#media:report.pdf)" />
      </SessionViewProvider>
    )

    const filename = await screen.findByText('report.pdf')
    expect(filename.closest('[data-hermes-context-menu-trigger]')).toBeNull()
    expect(screen.queryByRole('button', { name: 'File actions' })).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Download' }))
    await waitFor(() =>
      expect(saveGatewayFile).toHaveBeenCalledWith({
        connectionId: 'session-ssh',
        path: '/srv/work/report.pdf',
        profile: 'gateway-name',
        suggestedName: 'report.pdf'
      })
    )
  })

  it.each([
    ['Windows drive path', 'C:\\Users\\a\\report.pdf', 'C:\\Users\\a\\report.pdf'],
    ['Windows file URL', 'file:///C:/Users/a/report.pdf', 'C:/Users/a/report.pdf'],
    ['home-relative path', '~/report.pdf', '~/report.pdf']
  ])('downloads a %s without prepending the viewed session cwd', async (_kind, target, expectedPath) => {
    $connection.set({ connectionId: 'ambient-local', mode: 'local', profile: 'default' } as never)

    render(
      <SessionViewProvider
        value={{ ...PRIMARY_SESSION_VIEW, $cwd: atom('/srv/work'), $storedId: atom<null | string>(null) }}
      >
        <PreviewAttachment target={target} />
      </SessionViewProvider>
    )

    fireEvent.click(await screen.findByRole('button', { name: 'Download' }))
    await waitFor(() =>
      expect(saveGatewayFile).toHaveBeenCalledWith({
        connectionId: 'ambient-local',
        path: expectedPath,
        profile: 'default',
        suggestedName: 'report.pdf'
      })
    )
  })

  it.each([
    ['backslash UNC', '%5C%5Cserver%5Cshare%5Creport.pdf'],
    ['forward-slash UNC', '//server/share/report.pdf']
  ])('does not reveal a transcript-controlled %s path', async (_kind, mediaPath) => {
    render(<MarkdownTextContent isRunning={false} text={`Done: [report.pdf](#media:${mediaPath})`} />)

    const filename = await screen.findByText('report.pdf')
    fireEvent.click(screen.getByRole('button', { name: 'Download' }))
    fireEvent.contextMenu(filename)
    fireEvent.click(await screen.findByRole('menuitem', { name: /Open Containing Folder|Reveal in/i }))

    expect(revealPath).not.toHaveBeenCalled()
    expect(saveGatewayFile).not.toHaveBeenCalled()
  })

  it('does not add transcript file actions to images without a local reveal path', () => {
    render(<ZoomableImage alt="remote image" src="https://example.com/image.png" />)

    const image = screen.getByAltText('remote image')
    expect(image.closest('[data-hermes-context-menu-trigger]')).toBeNull()
    expect(screen.queryByRole('button', { name: 'File actions' })).toBeNull()
  })
})
