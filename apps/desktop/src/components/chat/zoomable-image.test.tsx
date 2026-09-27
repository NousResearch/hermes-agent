import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n/context'

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
})
