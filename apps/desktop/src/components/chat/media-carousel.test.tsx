import { describe, expect, it, beforeAll, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'
import { MediaCarousel } from '@/components/chat/media-carousel'
import { galleryMarkdownHref } from '@/lib/media'

const PNG_DATA_URL =
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M8AAAMBAQDJ/pLvAAAAAElFTkSuQmCC'

describe('MediaCarousel', () => {
  beforeAll(() => {
    // @ts-expect-error test bridge
    window.hermesDesktop = {
      readFileDataUrl: vi.fn(async () => PNG_DATA_URL),
      readFileText: vi.fn(),
      api: vi.fn()
    }
  })

  afterEach(() => cleanup())

  it('renders one stage frame per gallery source', async () => {
    const href = galleryMarkdownHref({
      title: 'Login flow',
      intervalMs: 1200,
      images: [{ src: '/tmp/login-1.png' }, { src: '/tmp/login-2.png' }, { src: '/tmp/login-3.png' }]
    })

    const payload = JSON.parse(decodeURIComponent(href.slice('#gallery:'.length)))

    const { container } = render(
      <MediaCarousel images={payload.images} intervalMs={payload.intervalMs} title={payload.title} />
    )

    // Three stage frames (the filmstrip thumbnails are a second set of images,
    // so we scope by the stage slot rather than counting every <img>).
    const frames = container.querySelectorAll(
      '[data-slot="aui_media-carousel-stage"] [data-slot="aui_media-carousel-frame"]'
    )
    expect(frames).toHaveLength(3)

    // The bridge resolves each path to a data URL asynchronously.
    await waitFor(() => {
      frames.forEach(frame => {
        const img = frame.querySelector('img')
        expect(img?.getAttribute('src')).toBe(PNG_DATA_URL)
      })
    })
  })

  it('exposes next/previous controls that move the active slide', async () => {
    const payload = {
      images: [
        { src: '/tmp/a.png', title: 'A' },
        { src: '/tmp/b.png', title: 'B' }
      ]
    }

    const { container } = render(<MediaCarousel images={payload.images} />)

    const next = container.querySelector('[aria-label="Next image"]') as HTMLButtonElement
    const prev = container.querySelector('[aria-label="Previous image"]') as HTMLButtonElement

    expect(screen.getByText('1 / 2')).toBeTruthy()

    fireEvent.click(next)
    expect(screen.getByText('2 / 2')).toBeTruthy()

    fireEvent.click(prev)
    expect(screen.getByText('1 / 2')).toBeTruthy()
  })

  it('toggles playback with the pause/play button', async () => {
    const payload = { images: [{ src: '/tmp/a.png' }, { src: '/tmp/b.png' }] }

    const { container } = render(<MediaCarousel images={payload.images} />)

    const toggle = container.querySelector('[aria-label="Pause slideshow"]') as HTMLButtonElement
    fireEvent.click(toggle)
    expect(container.querySelector('[aria-label="Play slideshow"]')).toBeTruthy()
  })
})
