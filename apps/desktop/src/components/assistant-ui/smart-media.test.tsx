import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { SmartMedia } from './smart-media'

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      desktop: {
        resumeRetry: 'Retry',
        openImage: 'Open image',
        downloadImage: 'Download image',
        savingImage: 'Saving image'
      }
    }
  })
}))

vi.mock('@/hooks/use-image-download', () => ({
  useImageDownload: () => ({ download: vi.fn(), saving: false })
}))

describe('SmartMedia', () => {
  afterEach(cleanup)

  it('forwards native image props onto the rendered <img>', () => {
    render(
      <SmartMedia
        alt="shot"
        data-testid="smart-img"
        kind="image"
        name="shot.png"
        src="https://example.com/shot.png"
        title="preview"
      />
    )

    const image = screen.getByRole('img', { name: 'shot' })
    expect(image.getAttribute('title')).toBe('preview')
    expect(image.getAttribute('data-testid')).toBe('smart-img')
  })

  it('shows ErrorFallback + retry for image load failures', () => {
    render(<SmartMedia alt="broken" kind="image" name="broken.png" src="https://example.com/broken.png" />)

    fireEvent.error(screen.getByRole('img', { name: 'broken' }))

    expect(screen.getByText(/Couldn't load broken.png/i)).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Retry' })).toBeTruthy()
    expect(screen.queryByRole('img')).toBeNull()
  })

  it('remounts the image on retry after a failure', () => {
    const { container } = render(
      <SmartMedia alt="broken" kind="image" name="broken.png" src="https://example.com/broken.png" />
    )

    fireEvent.error(screen.getByRole('img', { name: 'broken' }))
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }))

    const image = screen.getByRole('img', { name: 'broken' })
    expect(image).toBeTruthy()
    expect(container.querySelector('img')?.getAttribute('src')).toBe('https://example.com/broken.png')
  })

  it('renders an inline video player for video kind', () => {
    const { container } = render(
      <SmartMedia kind="video" name="clip.mp4" src="hermes-media://stream/clip.mp4" />
    )

    expect(container.querySelector('video')).not.toBeNull()
    expect(container.querySelector('img')).toBeNull()
  })

  it('shows ErrorFallback for video load failures and supports open external', () => {
    const onOpenExternal = vi.fn()
    const { container } = render(
      <SmartMedia
        kind="video"
        name="clip.mp4"
        onOpenExternal={onOpenExternal}
        src="hermes-media://stream/clip.mp4"
      />
    )

    const video = container.querySelector('video')
    expect(video).not.toBeNull()
    fireEvent.error(video as HTMLVideoElement)

    expect(screen.getByText(/Couldn't load clip.mp4/i)).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Open' }))
    expect(onOpenExternal).toHaveBeenCalledTimes(1)
  })
})
