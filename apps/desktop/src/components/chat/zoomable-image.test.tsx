import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n/context'
import { resetTranscriptLightbox } from '@/store/transcript-lightbox'

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
  afterEach(() => {
    cleanup()
    resetTranscriptLightbox()
  })

  it('opens the lightbox when the inline image is activated', async () => {
    await renderImage()
    await openLightbox()

    expect(screen.getByRole('dialog')).toBeTruthy()
    expect(lightboxImg()).toBeTruthy()
  })

  // The lightbox shell must be an invisible frame. styles.css paints every
  // [data-slot='dialog-content'] with the themed elevated background, and that
  // paint shows as a stray box whenever a zoomed/panned image no longer covers
  // the whole shell (#99066).
  it('renders the lightbox shell transparent, not as a painted box', async () => {
    await renderImage()
    await openLightbox()

    // The dialog role lands on Radix's Dialog.Content — the same node DialogContent styles.
    const shell = screen.getByRole('dialog') as HTMLElement

    expect(shell.getAttribute('data-slot')).toBe('dialog-content')
    // The variant opts the shell out of the shared dialog paint.
    expect(shell.getAttribute('data-variant')).toBe('media-lightbox')
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
    expect(percentageText()).toBe('125%')

    await act(async () => {
      fireEvent.click(reset)
    })
    expect(percentageText()).toBe('100%')

    await act(async () => {
      fireEvent.click(zoomOut)
    })
    // 1 / 1.25 = 0.8 → 80%
    expect(percentageText()).toBe('80%')
  })

  it('pans after zoom but does not close the lightbox', async () => {
    await renderImage()
    await openLightbox()
    const img = lightboxImg()

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /zoom in/i }))
    })
    expect(scaleOf(img)).toBeCloseTo(1.25, 5)

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
    expect(scaleOf(img)).toBeCloseTo(1.25, 5)

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
    expect(scaleOf(img)).toBeCloseTo(1.25, 5)
    expect(translateX(img)).toBeCloseTo(60, 5)
  })
})

async function renderWithI18n(ui: ReactNode) {
  let result: ReturnType<typeof render>
  await act(async () => {
    result = render(
      <I18nProvider configClient={{ getConfig: async () => ({}), saveConfig: async () => ({ ok: true }) }}>
        {ui}
      </I18nProvider>
    )
  })

  return result!
}

const THUMB = 'data:image/png;base64,dGh1bWJuYWls'
const FULL = 'data:image/png;base64,ZnVsbHJlc29sdXRpb24='

describe('ZoomableImage zoomSrc', () => {
  afterEach(() => {
    cleanup()
    resetTranscriptLightbox()
  })

  it('paints the bounded src inline but enlarges the full-resolution zoomSrc (#93204)', async () => {
    await renderWithI18n(<ZoomableImage alt="shot" src={THUMB} zoomSrc={FULL} />)

    // Inline stays the cheap thumbnail — no multi-MB paint.
    const inline = screen.getByAltText('shot') as HTMLImageElement
    expect(inline.getAttribute('src')).toBe(THUMB)

    // Click to zoom: the lightbox must show the original, not the thumbnail.
    fireEvent.click(inline)

    await waitFor(() => {
      const full = screen.getAllByAltText('shot').find(img => img.getAttribute('src') === FULL)
      expect(full).toBeDefined()
    })

    expect(screen.queryAllByAltText('shot').some(img => img.getAttribute('src') === THUMB)).toBe(true)
  })

  it('falls back to src for the lightbox when no zoomSrc is given', async () => {
    await renderWithI18n(<ZoomableImage alt="shot" src={THUMB} />)

    fireEvent.click(screen.getByAltText('shot'))

    await waitFor(() => {
      // Both inline and lightbox use src — backward compatible with callers that
      // pass a single source.
      expect(screen.getAllByAltText('shot').every(img => img.getAttribute('src') === THUMB)).toBe(true)
    })
  })

  // #123018: transcript rows remount routinely while a turn streams (render-
  // budget slice recycling, markdown AST re-parse). The open flag lives in a
  // store keyed by source identity, so a remounted row re-presents its own
  // open lightbox instead of dropping it.
  it('re-presents an open lightbox after the row unmounts and remounts (#123018)', async () => {
    const { unmount } = await renderWithI18n(<ZoomableImage alt="shot" src={THUMB} zoomSrc={FULL} />)

    fireEvent.click(screen.getByAltText('shot'))

    await waitFor(() => {
      expect(screen.getAllByAltText('shot').some(img => img.getAttribute('src') === FULL)).toBe(true)
    })

    // The row recycles — component-local state dies with it.
    unmount()
    await renderWithI18n(<ZoomableImage alt="shot" src={THUMB} zoomSrc={FULL} />)

    await waitFor(() => {
      expect(screen.getAllByAltText('shot').some(img => img.getAttribute('src') === FULL)).toBe(true)
    })
  })

  it('keeps only one preview open at a time and only the owning row can close it', async () => {
    await renderWithI18n(
      <>
        <ZoomableImage alt="one" src={THUMB} zoomSrc={FULL} />
        <ZoomableImage alt="two" src={THUMB} zoomSrc={THUMB} />
      </>
    )

    fireEvent.click(screen.getAllByAltText('one')[0]!)

    await waitFor(() => {
      expect(screen.getAllByAltText('one').some(img => img.getAttribute('src') === FULL)).toBe(true)
    })

    // Opening the second preview replaces the first — same one-at-a-time UX
    // as the component-local flag.
    fireEvent.click(screen.getAllByAltText('two')[0]!)

    await waitFor(() => {
      expect(screen.getAllByAltText('two').some(img => img.getAttribute('src') === THUMB)).toBe(true)
      expect(screen.queryAllByAltText('one').every(img => img.getAttribute('src') !== FULL)).toBe(true)
    })
  })
})
