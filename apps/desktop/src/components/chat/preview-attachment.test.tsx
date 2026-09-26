import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { setAlwaysExternalLinks } from '@/store/external-links'
import { $previewTabs, closeRightRail } from '@/store/preview'
import { $connection } from '@/store/session'

import { PreviewAttachment } from './preview-attachment'

afterEach(() => {
  cleanup()
  setAlwaysExternalLinks(false)
  closeRightRail()
  $connection.set(null)
})

describe('transcript preview URL on a remote gateway', () => {
  it('never launches a failed SSH loopback forward on the local machine', async () => {
    const url = 'http://localhost:5173'
    const openPreviewInBrowser = vi.fn(async () => undefined)
    $connection.set({ mode: 'remote' } as never)
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {
        normalizePreviewTarget: vi.fn(async () => ({ kind: 'url', label: url, source: url, url })),
        reachPreviewUrl: vi.fn(async () => url),
        openPreviewInBrowser
      }
    })
    setAlwaysExternalLinks(true)

    render(<PreviewAttachment target={url} />)
    fireEvent.click(screen.getByRole('button', { name: 'Open in browser' }))

    await waitFor(() => expect($previewTabs.get()).toHaveLength(1))
    expect(openPreviewInBrowser).not.toHaveBeenCalled()
  })
})
