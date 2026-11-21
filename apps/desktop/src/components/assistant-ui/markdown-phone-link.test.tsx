import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { MarkdownTextContent } from './markdown-text'

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const initialHermesDesktop = desktopWindow.hermesDesktop

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()

  if (initialHermesDesktop) {
    desktopWindow.hermesDesktop = initialHermesDesktop
  } else {
    delete desktopWindow.hermesDesktop
  }
})

describe('assistant phone links', () => {
  it('opens a linked phone number through the desktop bridge', async () => {
    const openExternal = vi.fn().mockResolvedValue(undefined)

    desktopWindow.hermesDesktop = {
      fetchLinkTitle: vi.fn().mockResolvedValue(''),
      openExternal
    } as unknown as Window['hermesDesktop']

    render(<MarkdownTextContent isRunning={false} text="Viktoria Pettersson: `010-495 64 04`" />)

    const link = await screen.findByRole('link', { name: '010-495 64 04' })

    expect(link.getAttribute('href')).toBe('tel:0104956404')
    expect(link.querySelector('code')).not.toBeNull()
    expect(openExternal).not.toHaveBeenCalled()
    fireEvent.click(link)
    expect(openExternal).toHaveBeenCalledWith('tel:0104956404')
  })
})
