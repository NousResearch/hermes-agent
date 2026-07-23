import { describe, it, expect, beforeAll, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { MarkdownTextContent } from '@/components/assistant-ui/markdown-text'
import { renderMediaTags } from '@/lib/chat-messages'

// Integration proof: the runtime converts a model-emitted `MEDIA:` tag into a
// `#media:` markdown link (renderMediaTags), then MarkdownTextContent renders
// that link as an inline <img> thumbnail via the desktop bridge. We exercise
// both halves with the real wrapped-path shapes that were previously broken.
const PNG_DATA_URL =
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M8AAAMBAQDJ/pLvAAAAAElFTkSuQmCC'

describe('MEDIA image thumbnail render', () => {
  beforeAll(() => {
    // @ts-expect-error test bridge
    window.hermesDesktop = {
      readFileDataUrl: vi.fn(async () => PNG_DATA_URL),
      readFileText: vi.fn(),
      api: vi.fn(),
    }
  })

  it('leading-emphasis MEDIA tag renders as a thumbnail', async () => {
    const md = renderMediaTags('MEDIA:** `/tmp/shot.png`')
    render(<MarkdownTextContent isRunning={false} text={md as string} />)

    const img = await waitFor(() => screen.getByRole('img'))
    expect(img).toBeTruthy()
    expect(img.getAttribute('src')).toBe(PNG_DATA_URL)
  })

  it('trailing-emphasis MEDIA tag renders as a thumbnail', async () => {
    const md = renderMediaTags('MEDIA:/tmp/shot.png**')
    render(<MarkdownTextContent isRunning={false} text={md as string} />)

    const img = await waitFor(() => screen.getByRole('img'))
    expect(img).toBeTruthy()
    expect(img.getAttribute('src')).toBe(PNG_DATA_URL)
  })
})
