import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $connection } from '@/store/session'

import { ZoomableImage } from '../chat/zoomable-image'

import { MarkdownTextContent } from './markdown-text'

// Regression: a MEDIA-delivered non-media file (pdf, zip, ...) renders as an
// "Open <name>" fallback link. Right-clicking it must offer
// reveal-in-file-manager + Copy Path — the transcript's "where is that file?"
// door — so the user can find the artifact on disk without re-asking the agent.
describe('MEDIA file fallback link context menu', () => {
  const revealPath = vi.fn().mockResolvedValue(undefined)
  const writeText = vi.fn().mockResolvedValue(undefined)
  let originalDesktop: typeof window.hermesDesktop
  let originalClipboard: PropertyDescriptor | undefined

  beforeEach(() => {
    revealPath.mockClear()
    writeText.mockClear()
    originalDesktop = window.hermesDesktop
    originalClipboard = Object.getOwnPropertyDescriptor(navigator, 'clipboard')
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { revealPath }
    })
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText } })

    $connection.set(null)
  })

  afterEach(() => {
    cleanup()
    $connection.set(null)
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: originalDesktop
    })

    if (originalClipboard) {
      Object.defineProperty(navigator, 'clipboard', originalClipboard)
    }
  })

  it('right-clicking the "Open <file>" fallback link reveals it in the file manager', async () => {
    render(<MarkdownTextContent isRunning={false} text="Done: [report.pdf](#media:%2Ftmp%2Freport.pdf)" />)

    const link = await screen.findByText('Open report.pdf')

    // The fallback anchor must be wrapped in a Radix context-menu trigger.
    await waitFor(() => expect(link.closest('[data-hermes-context-menu-trigger]')).not.toBeNull())

    fireEvent.contextMenu(link)

    const revealItem = await screen.findByRole('menuitem', { name: /Open Containing Folder|Reveal in/i })
    expect(screen.getByRole('menuitem', { name: /Copy Path/i })).toBeTruthy()

    fireEvent.click(revealItem)

    await waitFor(() => expect(revealPath).toHaveBeenCalledWith('/tmp/report.pdf'))
  })

  it('renders bare children after switching to a remote connection', async () => {
    $connection.set({ mode: 'remote' } as never)
    render(<MarkdownTextContent isRunning={false} text="Done: [report.pdf](#media:%2Ftmp%2Freport.pdf)" />)

    const link = await screen.findByText('Open report.pdf')
    expect(link.closest('[data-hermes-context-menu-trigger]')).toBeNull()
    expect(screen.queryByRole('button', { name: 'File actions' })).toBeNull()
  })

  it('copies the artifact path from the keyboard-accessible actions menu', async () => {
    render(<MarkdownTextContent isRunning={false} text="Done: [report.pdf](#media:%2Ftmp%2Freport.pdf)" />)

    const actions = await screen.findByRole('button', { name: 'File actions' })
    fireEvent.pointerDown(actions, { button: 0, ctrlKey: false, pointerType: 'mouse' })
    fireEvent.click(actions)
    fireEvent.click(await screen.findByRole('menuitem', { name: 'Copy Path' }))

    await waitFor(() => expect(writeText).toHaveBeenCalledWith('/tmp/report.pdf'))
  })

  it('does not reveal a transcript-controlled UNC path', async () => {
    render(<MarkdownTextContent isRunning={false} text="Done: [report.pdf](#media:%5C%5Cserver%5Cshare%5Creport.pdf)" />)

    const link = await screen.findByText('Open report.pdf')
    fireEvent.contextMenu(link)
    fireEvent.click(await screen.findByRole('menuitem', { name: /Open Containing Folder|Reveal in/i }))

    expect(revealPath).not.toHaveBeenCalled()
  })

  it('does not add transcript file actions to images without a local reveal path', () => {
    render(<ZoomableImage alt="remote image" src="https://example.com/image.png" />)

    const image = screen.getByAltText('remote image')
    expect(image.closest('[data-hermes-context-menu-trigger]')).toBeNull()
    expect(screen.queryByRole('button', { name: 'File actions' })).toBeNull()
  })
})
