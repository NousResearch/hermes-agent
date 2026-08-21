import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $connection } from '@/store/session'

import { MarkdownTextContent } from './markdown-text'

// Regression: a MEDIA-delivered non-media file (pdf, zip, ...) renders as an
// "Open <name>" fallback link. Right-clicking it must offer
// reveal-in-file-manager + Copy Path — the transcript's "where is that file?"
// door — so the user can find the artifact on disk without re-asking the agent.
describe('MEDIA file fallback link context menu', () => {
  const revealPath = vi.fn().mockResolvedValue(undefined)
  let originalDesktop: typeof window.hermesDesktop

  beforeEach(() => {
    revealPath.mockClear()
    originalDesktop = window.hermesDesktop
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { revealPath }
    })
    $connection.set(null)
  })

  afterEach(() => {
    cleanup()
    $connection.set(null)
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: originalDesktop
    })
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
})
