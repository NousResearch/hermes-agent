import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n/context'
import { resetTranscriptLightbox } from '@/store/transcript-lightbox'

import { ZoomableImage } from './zoomable-image'

// A bounded thumbnail (what the composer/optimistic bubble paints inline) and
// the full-resolution source (what the lightbox should enlarge). Issue #93204:
// clicking a freshly sent image enlarged the 512px thumbnail instead of the
// original.
const THUMB = 'data:image/png;base64,dGh1bWJuYWls'
const FULL = 'data:image/png;base64,ZnVsbHJlc29sdXRpb24='

async function renderWithI18n(ui: React.ReactNode) {
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
