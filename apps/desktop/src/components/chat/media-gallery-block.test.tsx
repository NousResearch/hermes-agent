import { describe, expect, it, beforeAll, vi } from 'vitest'
import { render, waitFor } from '@testing-library/react'
import { MarkdownTextContent } from '@/components/assistant-ui/markdown-text'
import { renderMediaTags } from '@/lib/chat-messages'

// Regression: a `#gallery:` link expands to <MediaCarousel> (a <div>). Streamdown
// wraps link text in a <p>, and a <div> cannot be a child of <p> — React drops
// the nesting and the carousel never paints. The markdown pipeline must hoist
// the carousel out of the <p> (see the `p` override in markdown-text.tsx) so it
// actually renders.
const PNG_DATA_URL =
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M8AAAMBAQDJ/pLvAAAAAElFTkSuQmCC'

describe('MEDIA-GALLERY renders as a block-level carousel (not inside <p>)', () => {
  beforeAll(() => {
    // @ts-expect-error test bridge
    window.hermesDesktop = {
      readFileDataUrl: vi.fn(async () => PNG_DATA_URL),
      readFileText: vi.fn(),
      api: vi.fn()
    }
  })

  it('renders one stage frame per gallery image and is not wrapped in a <p>', async () => {
    const raw = [
      'MEDIA-GALLERY: Demo reel (1500ms)',
      'MEDIA:/tmp/login-1.png',
      'MEDIA:/tmp/login-2.png',
      'MEDIA:/tmp/login-3.png',
      '<!-- /MEDIA-GALLERY -->'
    ].join('\n')

    const md = renderMediaTags(raw)
    const { container } = render(<MarkdownTextContent isRunning={false} text={md as string} />)

    const frames = await waitFor(() =>
      container.querySelectorAll(
        '[data-slot="aui_media-carousel-stage"] [data-slot="aui_media-carousel-frame"]'
      )
    )
    expect(frames).toHaveLength(3)

    const carousel = container.querySelector('[data-slot="aui_media-carousel"]')
    expect(carousel).toBeTruthy()
    // The bug: carousel root was a direct child of <p>. Assert it is not.
    expect(carousel?.parentElement?.tagName).not.toBe('P')
  })
})
