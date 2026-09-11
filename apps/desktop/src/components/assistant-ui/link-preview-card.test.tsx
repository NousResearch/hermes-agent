import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { LinkPreviewChipCard } from './link-preview-card'

const desktopWindow = window as unknown as { hermesDesktop?: unknown }
const initialHermesDesktop = desktopWindow.hermesDesktop

const URL = 'https://example.com/article'

function installBridge(fetchLinkPreview: unknown) {
  desktopWindow.hermesDesktop = {
    fetchLinkTitle: vi.fn().mockResolvedValue(''),
    openExternal: vi.fn().mockResolvedValue(undefined),
    fetchLinkPreview
  }
}

afterEach(() => {
  cleanup()
  desktopWindow.hermesDesktop = initialHermesDesktop
})

describe('LinkPreviewChipCard', () => {
  it('renders a collapsed chip and makes NO bridge call until clicked', () => {
    const bridge = vi.fn()
    installBridge(bridge)

    render(<LinkPreviewChipCard href={URL} />)

    const chip = screen.getByRole('button')

    expect(chip).toBeTruthy()
    expect(chip.getAttribute('data-link-preview')).toBe('chip')
    expect(chip.textContent).toContain('example.com')
    expect(bridge).not.toHaveBeenCalled()
  })

  it('click calls the bridge once and expands into the loaded card', async () => {
    const bridge = vi.fn().mockResolvedValue({
      ok: true,
      meta: { url: URL, title: 'Example Article', description: 'A fine read.', imageUrl: '', fetchedAt: 1_000 }
    })

    installBridge(bridge)

    render(<LinkPreviewChipCard href={URL} />)
    fireEvent.click(screen.getByRole('button'))

    await waitFor(() => expect(screen.getAllByText('Example Article').length).toBeGreaterThan(0))
    await waitFor(() => expect(screen.getByText('A fine read.')).toBeTruthy())

    expect(bridge).toHaveBeenCalledTimes(1)
    expect(bridge).toHaveBeenCalledWith(URL)

    const card = screen.getAllByText('Example Article')[0].closest('[data-link-preview]')

    expect(card?.getAttribute('data-link-preview')).toBe('card')
  })

  it('is sticky: re-render after load does not refetch', async () => {
    const bridge = vi.fn().mockResolvedValue({
      ok: true,
      meta: { url: URL, title: 'T', description: '', imageUrl: '', fetchedAt: 1_000 }
    })

    installBridge(bridge)

    const view = render(<LinkPreviewChipCard href={URL} />)
    fireEvent.click(screen.getByRole('button'))

    await waitFor(() => expect(screen.getAllByText('T').length).toBeGreaterThan(0))
    view.rerender(<LinkPreviewChipCard href={URL} />)

    expect(bridge).toHaveBeenCalledTimes(1)
  })

  it('error failure leg renders an honest message (never silent)', async () => {
    const bridge = vi.fn().mockResolvedValue({ ok: false, reason: 'error' })
    installBridge(bridge)

    render(<LinkPreviewChipCard href={URL} />)
    fireEvent.click(screen.getByRole('button'))

    await waitFor(() => expect(screen.getByText(/Preview unavailable/)).toBeTruthy())
    expect(screen.queryByText(/private address/i)).toBeNull()
  })

  it('private-url failure leg says so specifically', async () => {
    const bridge = vi.fn().mockResolvedValue({ ok: false, reason: 'private-url' })
    installBridge(bridge)

    render(<LinkPreviewChipCard href={URL} />)
    fireEvent.click(screen.getByRole('button'))

    await waitFor(() => expect(screen.getByText(/private address/i)).toBeTruthy())
  })

  it('a rejected bridge promise degrades to the error leg, not a throw', async () => {
    const bridge = vi.fn().mockRejectedValue(new Error('bridge down'))
    installBridge(bridge)

    render(<LinkPreviewChipCard href={URL} />)
    fireEvent.click(screen.getByRole('button'))

    await waitFor(() => expect(screen.getByText(/Preview unavailable/)).toBeTruthy())
  })

  it('with no desktop bridge at all, clicking renders the unavailable leg', async () => {
    desktopWindow.hermesDesktop = undefined

    render(<LinkPreviewChipCard href={URL} />)
    fireEvent.click(screen.getByRole('button'))

    await waitFor(() => expect(screen.getByText(/Preview unavailable/)).toBeTruthy())
  })

  it('meta with no readable fields renders the thumbnail data URL inside a loaded card', async () => {
    // The renderer paints the main-process-validated data URL — it NEVER
    // GETs meta.imageUrl itself (that would bypass every SSRF guard).
    const bridge = vi.fn().mockResolvedValue({
      ok: true,
      meta: { url: URL, title: '', description: '', imageUrl: 'https://cdn.example.com/x.png', image: 'data:image/png;base64,AAAA', fetchedAt: 1_000 }
    })

    installBridge(bridge)

    render(<LinkPreviewChipCard href={URL} />)
    fireEvent.click(screen.getByRole('button'))

    await waitFor(() => {
      const img = document.querySelector('img[data-link-preview-image], img[src^="data:image/png;base64,"]')

      expect(img).toBeTruthy()
    })

    expect(document.querySelector('img[src="https://cdn.example.com/x.png"]')).toBeNull()
  })

  it('an unprovable thumbnail (image: \'\') renders the unavailable note, not a remote img', async () => {
    const bridge = vi.fn().mockResolvedValue({
      ok: true,
      meta: { url: URL, title: '', description: '', imageUrl: 'https://cdn.example.com/x.png', image: '', fetchedAt: 1_000 }
    })

    installBridge(bridge)

    render(<LinkPreviewChipCard href={URL} />)
    fireEvent.click(screen.getByRole('button'))

    await waitFor(() => expect(screen.getByText(/Preview unavailable/)).toBeTruthy())
    expect(document.querySelector('img')).toBeNull()
  })
})
