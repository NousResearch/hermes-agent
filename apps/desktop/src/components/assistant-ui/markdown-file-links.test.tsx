import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $connection } from '@/store/session'

import { MarkdownTextContent } from './markdown-text'

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const initialHermesDesktop = desktopWindow.hermesDesktop

afterEach(() => {
  cleanup()
  $connection.set(null)
  vi.restoreAllMocks()

  if (initialHermesDesktop) {
    desktopWindow.hermesDesktop = initialHermesDesktop
  } else {
    delete desktopWindow.hermesDesktop
  }
})

describe('MarkdownTextContent file links', () => {
  it('opens file links through the remote download endpoint', () => {
    const openExternal = vi.fn().mockResolvedValue(undefined)
    desktopWindow.hermesDesktop = { openExternal } as unknown as Window['hermesDesktop']
    $connection.set({ baseUrl: 'https://gw', mode: 'remote', token: 's e/cret' } as never)

    render(<MarkdownTextContent isRunning={false} text="[Report](file:///tmp/a%20b.pdf)" />)

    fireEvent.click(screen.getByRole('link', { name: 'Open a b.pdf' }))

    expect(openExternal).toHaveBeenCalledWith(
      'https://gw/api/files/download?path=%2Ftmp%2Fa%20b.pdf&token=s%20e%2Fcret'
    )
  })
})
