import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n/context'

import { ZoomableImage } from './zoomable-image'

describe('ZoomableImage', () => {
  afterEach(cleanup)

  it('applies an action class to inline and lightbox download controls', () => {
    render(
      <I18nProvider>
        <ZoomableImage
          actionButtonClassName="touch-target"
          alt="Artifact preview"
          src="data:image/png;base64,c21hbGw="
        />
      </I18nProvider>
    )

    expect(screen.getByRole('button', { name: 'Download image' }).className).toContain('touch-target')

    fireEvent.click(screen.getByRole('button', { name: 'Artifact preview' }))

    const downloadButtons = document.querySelectorAll<HTMLButtonElement>('button[aria-label="Download image"]')
    expect(downloadButtons).toHaveLength(2)
    expect([...downloadButtons].every(button => button.className.includes('touch-target'))).toBe(true)
  })
})
