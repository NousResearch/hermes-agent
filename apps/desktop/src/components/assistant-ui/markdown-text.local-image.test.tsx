import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { MarkdownTextContent } from './markdown-text'

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const initialHermesDesktop = desktopWindow.hermesDesktop

afterEach(() => {
  vi.restoreAllMocks()
  cleanup()

  if (initialHermesDesktop) {
    desktopWindow.hermesDesktop = initialHermesDesktop
  } else {
    delete desktopWindow.hermesDesktop
  }
})

describe('MarkdownTextContent local images', () => {
  it('resolves file image markdown through the desktop data-url bridge', async () => {
    const readFileDataUrl = vi.fn().mockResolvedValue('data:image/png;base64,AAAA')
    desktopWindow.hermesDesktop = {
      readFileDataUrl
    } as unknown as Window['hermesDesktop']

    render(<MarkdownTextContent isRunning={false} text="![Iceland](file:///home/user/Downloads/iceland.png)" />)

    await waitFor(() => {
      expect(readFileDataUrl).toHaveBeenCalledWith('/home/user/Downloads/iceland.png')
    })

    const image = await screen.findByRole('img', { name: 'Iceland' })
    await waitFor(() => {
      expect(image.getAttribute('src')).toBe('data:image/png;base64,AAAA')
    })
  })
})
