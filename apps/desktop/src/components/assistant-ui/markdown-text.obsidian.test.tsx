import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { MarkdownTextContent } from './markdown-text'

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const initialHermesDesktop = desktopWindow.hermesDesktop

const OBSIDIAN_URL =
  'obsidian://open?vault=PG%20Vault&file=Social%20Media%2FDrafts%2F2026-08-21-americana-vin-fiz-parts-train.md'

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()

  if (initialHermesDesktop) {
    desktopWindow.hermesDesktop = initialHermesDesktop
  } else {
    delete desktopWindow.hermesDesktop
  }
})

describe('MarkdownTextContent Obsidian links', () => {
  it('preserves an Obsidian URI and hands a user click to the desktop shell', async () => {
    const openExternal = vi.fn().mockResolvedValue(undefined)

    desktopWindow.hermesDesktop = { openExternal } as unknown as Window['hermesDesktop']

    render(<MarkdownTextContent isRunning={false} text={`[Open the canonical draft](${OBSIDIAN_URL})`} />)

    const link = await screen.findByRole('link', { name: 'Open the canonical draft' })

    expect(link.getAttribute('href')).toBe(OBSIDIAN_URL)
    expect(screen.queryByText('[blocked]')).toBeNull()

    fireEvent.click(link)

    expect(openExternal).toHaveBeenCalledOnce()
    expect(openExternal).toHaveBeenCalledWith(OBSIDIAN_URL)
  })
})
