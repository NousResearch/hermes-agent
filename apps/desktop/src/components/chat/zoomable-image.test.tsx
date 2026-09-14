import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { ImageGallery, ZoomableImage } from './zoomable-image'

afterEach(cleanup)

describe('ZoomableImage gallery navigation', () => {
  it('moves between images in the same gallery with arrow controls', () => {
    render(
      <ImageGallery>
        <ZoomableImage alt="First screenshot" src="data:image/png;base64,first" />
        <ZoomableImage alt="Second screenshot" src="data:image/png;base64,second" />
      </ImageGallery>
    )

    fireEvent.click(screen.getByRole('img', { name: 'First screenshot' }))

    const dialog = screen.getByRole('dialog')
    expect(within(dialog).getByText('1 / 2')).toBeTruthy()

    fireEvent.click(within(dialog).getByRole('button', { name: 'Next image' }))

    expect(within(dialog).getByRole('img', { name: 'Second screenshot' })).toBeTruthy()
    expect(within(dialog).getByText('2 / 2')).toBeTruthy()
    expect((within(dialog).getByRole('button', { name: 'Next image' }) as HTMLButtonElement).disabled).toBe(true)
  })

  it('supports left and right arrow keys while the gallery is open', () => {
    render(
      <ImageGallery>
        <ZoomableImage alt="First screenshot" src="data:image/png;base64,first" />
        <ZoomableImage alt="Second screenshot" src="data:image/png;base64,second" />
      </ImageGallery>
    )

    fireEvent.click(screen.getByRole('img', { name: 'First screenshot' }))
    const dialog = screen.getByRole('dialog')

    fireEvent.keyDown(dialog, { key: 'ArrowRight' })
    expect(within(dialog).getByRole('img', { name: 'Second screenshot' })).toBeTruthy()

    fireEvent.keyDown(dialog, { key: 'ArrowLeft' })
    expect(within(dialog).getByRole('img', { name: 'First screenshot' })).toBeTruthy()
  })

  it('keeps standalone images free of gallery controls', () => {
    render(<ZoomableImage alt="Only screenshot" src="data:image/png;base64,only" />)

    fireEvent.click(screen.getByRole('img', { name: 'Only screenshot' }))

    const dialog = screen.getByRole('dialog')
    expect(within(dialog).queryByRole('button', { name: 'Next image' })).toBeNull()
    expect(within(dialog).queryByText(/\/ 1$/)).toBeNull()
  })
})
