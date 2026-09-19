import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { ImageLightbox } from './zoomable-image'

afterEach(cleanup)
it.each(['data:image/png;base64,AAAA', ''])('has a visible Close action with loaded or unavailable image: %s', src => {
  const close = vi.fn()
  render(
    <ImageLightbox
      alt="Test image"
      copy={{ downloadImage: 'Download image', savingImage: 'Saving image' }}
      onClick={vi.fn()}
      onOpenChange={close}
      open
      placeholder={<span>Image unavailable</span>}
      saving={false}
      src={src}
    />
  )
  fireEvent.click(screen.getByRole('button', { name: 'Close' }))
  expect(close).toHaveBeenCalledWith(false)
})
