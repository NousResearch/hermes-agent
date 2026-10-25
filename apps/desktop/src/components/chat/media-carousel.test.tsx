import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { MediaCarousel } from '@/components/chat/media-carousel'
import { galleryMarkdownHref } from '@/lib/media'

const PNG_DATA_URL =
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M8AAAMBAQDJ/pLvAAAAAElFTkSuQmCC'

const activeDotIndex = (container: HTMLElement): number =>
  Array.from(container.querySelectorAll('[data-slot="aui_media-carousel-dot"]')).findIndex(
    dot => dot.getAttribute('aria-current') === 'true'
  )

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

    const frames = container.querySelectorAll(
      '[data-slot="aui_media-carousel-stage"] [data-slot="aui_media-carousel-frame"]'
    )

    expect(frames).toHaveLength(3)

    // One dot per image, and the bridge resolves each path to a data URL.
    expect(container.querySelectorAll('[data-slot="aui_media-carousel-dot"]')).toHaveLength(3)

    await waitFor(() => {
      frames.forEach(frame => {
        const img = frame.querySelector('img')
        expect(img?.getAttribute('src')).toBe(PNG_DATA_URL)
      })
    })
  })

  it('moves the active slide with the next/previous controls', () => {
    const images = [
      { src: '/tmp/a.png', title: 'A' },
      { src: '/tmp/b.png', title: 'B' },
      { src: '/tmp/c.png', title: 'C' }
    ]

    const { container } = render(<MediaCarousel images={images} />)

    expect(activeDotIndex(container)).toBe(0)

    fireEvent.click(container.querySelector('[aria-label="Next image"]') as HTMLButtonElement)
    expect(activeDotIndex(container)).toBe(1)

    fireEvent.click(container.querySelector('[aria-label="Previous image"]') as HTMLButtonElement)
    expect(activeDotIndex(container)).toBe(0)
  })

  it('toggles playback with the pause/play control', () => {
    const images = [
      { src: '/tmp/a.png', title: 'A' },
      { src: '/tmp/b.png', title: 'B' }
    ]

    const { container } = render(<MediaCarousel images={images} />)

    fireEvent.click(container.querySelector('[aria-label="Pause slideshow"]') as HTMLButtonElement)
    expect(container.querySelector('[aria-label="Play slideshow"]')).toBeTruthy()
  })

  it('selects a slide from its dot without opening the lightbox', () => {
    const images = [
      { src: '/tmp/a.png', title: 'A' },
      { src: '/tmp/b.png', title: 'B' }
    ]

    const { container } = render(<MediaCarousel images={images} />)

    expect(activeDotIndex(container)).toBe(0)

    const dotB = container.querySelector('[data-slot="aui_media-carousel-dot"][aria-label="B"]') as HTMLButtonElement
    expect(dotB).toBeTruthy()
    fireEvent.click(dotB)

    // The dot selects its slide …
    expect(activeDotIndex(container)).toBe(1)
    // … and must NOT open the zoom lightbox: dots hold no image, so there is no
    // nested interactive control that could fire on the same click.
    expect(screen.queryByRole('dialog')).toBeNull()
  })
})
