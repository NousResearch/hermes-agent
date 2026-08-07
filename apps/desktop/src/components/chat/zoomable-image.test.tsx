import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n/context'

import { ZoomableImage } from './zoomable-image'

const SRC = 'https://example.com/long-detail.png'
const ALT = 'long detail image'

async function renderImage() {
  let result!: ReturnType<typeof render>
  await act(async () => {
    result = render(
      <I18nProvider configClient={{ getConfig: async () => ({}), saveConfig: async () => ({ ok: true }) }}>
        <ZoomableImage alt={ALT} src={SRC} />
      </I18nProvider>
    )
  })

  return result
}

async function openLightbox() {
  const trigger = screen.getByAltText(ALT).closest('button')
  expect(trigger).toBeTruthy()
  await act(async () => {
    fireEvent.click(trigger!)
  })
}

function lightboxImg(): HTMLImageElement {
  const dialog = screen.getByRole('dialog')

  return within(dialog).getByRole('img') as HTMLImageElement
}

function scaleOf(img: HTMLElement): number {
  const match = img.style.transform.match(/scale\(([0-9.]+)\)/)

  return match ? parseFloat(match[1]) : 1
}

function translateX(img: HTMLElement): number {
  const match = img.style.transform.match(/translate\(([-0-9.]+)px/)

  return match ? parseFloat(match[1]) : 0
}

function percentageText(): string {
  const node = screen.getByText(/\d+%/)

  return node.textContent ?? ''
}

describe('ZoomableImage lightbox', () => {
  afterEach(cleanup)

  it('opens the lightbox when the inline image is activated', async () => {
    await renderImage()
    await openLightbox()

    expect(screen.getByRole('dialog')).toBeTruthy()
    expect(lightboxImg()).toBeTruthy()
  })

  it('wheel zooms toward the cursor and prevents the page from scrolling', async () => {
    await renderImage()
    await openLightbox()
    const img = lightboxImg()

    const wheel = new WheelEvent('wheel', { bubbles: true, cancelable: true, clientX: 10, clientY: 10, deltaY: -100 })
    const preventDefault = vi.spyOn(wheel, 'preventDefault')

    await act(async () => {
      img.dispatchEvent(wheel)
    })

    expect(preventDefault).toHaveBeenCalled()
    expect(scaleOf(img)).toBeGreaterThan(1)
  })

  it('zoom buttons update the zoom percentage', async () => {
    await renderImage()
    await openLightbox()

    const zoomIn = screen.getByRole('button', { name: /zoom in/i })
    const reset = screen.getByRole('button', { name: /reset/i })
    const zoomOut = screen.getByRole('button', { name: /zoom out/i })

    await act(async () => {
      fireEvent.click(zoomIn)
    })
    expect(percentageText()).toBe('130%')

    await act(async () => {
      fireEvent.click(reset)
    })
    expect(percentageText()).toBe('100%')

    await act(async () => {
      fireEvent.click(zoomOut)
    })
    // 1 / 1.3 ≈ 0.769 → 77%
    expect(percentageText()).toBe('77%')
  })

  it('pans after zoom but does not close the lightbox', async () => {
    await renderImage()
    await openLightbox()
    const img = lightboxImg()

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /zoom in/i }))
    })
    expect(scaleOf(img)).toBeCloseTo(1.3, 5)

    await act(async () => {
      fireEvent.pointerDown(img, { clientX: 10, clientY: 10, pointerId: 1 })
    })
    await act(async () => {
      fireEvent.pointerMove(img, { clientX: 60, clientY: 10, pointerId: 1 })
    })
    await act(async () => {
      fireEvent.pointerUp(img, { clientX: 60, clientY: 10, pointerId: 1 })
    })

    expect(translateX(img)).not.toBe(0)
    // A pan must NOT close the lightbox.
    expect(screen.getByRole('dialog')).toBeTruthy()
  })

  it('closes the lightbox on a clean click (no pan/pinch)', async () => {
    await renderImage()
    await openLightbox()
    const img = lightboxImg()

    await act(async () => {
      fireEvent.click(img)
    })

    expect(screen.queryByRole('dialog')).toBeNull()
  })

  it('pinch zooms with two pointers', async () => {
    await renderImage()
    await openLightbox()
    const img = lightboxImg()

    await act(async () => {
      fireEvent.pointerDown(img, { clientX: 100, clientY: 100, pointerId: 1 })
    })
    await act(async () => {
      fireEvent.pointerDown(img, { clientX: 200, clientY: 100, pointerId: 2 })
    })
    await act(async () => {
      fireEvent.pointerMove(img, { clientX: 50, clientY: 100, pointerId: 1 })
    })
    await act(async () => {
      fireEvent.pointerMove(img, { clientX: 250, clientY: 100, pointerId: 2 })
    })

    expect(scaleOf(img)).toBeGreaterThan(1)

    await act(async () => {
      fireEvent.pointerUp(img, { clientX: 50, clientY: 100, pointerId: 1 })
    })
    await act(async () => {
      fireEvent.pointerUp(img, { clientX: 250, clientY: 100, pointerId: 2 })
    })
  })

  it('recovers from pointercancel and treats the next single-pointer gesture as a pan, not a pinch', async () => {
    await renderImage()
    await openLightbox()
    const img = lightboxImg()

    // Zoom in so panning engages (pan only applies above scale 1).
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /zoom in/i }))
    })
    expect(scaleOf(img)).toBeCloseTo(1.3, 5)

    // Start a gesture with pointer 1, then the browser cancels it (e.g. it
    // steals the gesture for a scroll). Without cleanup the stale pointer
    // lingers and the next pointerdown is misread as a pinch.
    await act(async () => {
      fireEvent.pointerDown(img, { clientX: 100, clientY: 100, pointerId: 1 })
    })
    await act(async () => {
      fireEvent.pointerCancel(img, { clientX: 100, clientY: 100, pointerId: 1 })
    })

    // A fresh single pointer (id 2) should pan cleanly by 60px (100 → 160).
    await act(async () => {
      fireEvent.pointerDown(img, { clientX: 100, clientY: 100, pointerId: 2 })
    })
    await act(async () => {
      fireEvent.pointerMove(img, { clientX: 160, clientY: 100, pointerId: 2 })
    })
    await act(async () => {
      fireEvent.pointerUp(img, { clientX: 160, clientY: 100, pointerId: 2 })
    })

    // Scale is unchanged (no pinch zoom), and the image panned the expected 60px.
    expect(scaleOf(img)).toBeCloseTo(1.3, 5)
    expect(translateX(img)).toBeCloseTo(60, 5)
  })
})
