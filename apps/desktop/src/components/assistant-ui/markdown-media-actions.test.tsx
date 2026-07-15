// @vitest-environment jsdom
import { fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $connection } from '@/store/session'

import { MarkdownTextContent } from './markdown-text'

vi.mock('@/components/chat/image-generation-placeholder', () => ({ DiffusionCanvas: () => null }))
vi.mock('@/components/chat/shiki-highlighter', () => ({
  chunkByLines: (text: string) => [{ text }],
  SyntaxHighlighter: () => null
}))

describe('MEDIA image actions', () => {
  afterEach(() => {
    $connection.set(null)
    vi.unstubAllGlobals()
    vi.clearAllMocks()
  })

  it('renders a native Open full file button for a local MEDIA image', async () => {
    const openExternal = vi.fn(async () => true)
    const readFileDataUrl = vi.fn(async () => 'data:image/png;base64,ZHVtbXk=')

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { openExternal, readFileDataUrl }
    })
    $connection.set({ mode: 'local' } as never)

    const source = '/tmp/generated-image.png'
    render(
      <MarkdownTextContent
        isRunning={false}
        text={`[Image: generated-image.png](#media:${encodeURIComponent(source)})`}
      />
    )

    const button = await screen.findByRole('button', { name: 'Open full file' })
    fireEvent.click(button)

    expect(openExternal).toHaveBeenCalledWith('file:///tmp/generated-image.png')
  })
})
