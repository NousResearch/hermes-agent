import { render } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { $backgroundPreference, $backgroundRuntime } from '@/store/background'

import { Backdrop } from './Backdrop'

vi.mock('leva', () => ({
  Leva: () => null,
  useControls: (name: string) =>
    name === 'UI / Shape'
      ? { radiusScalar: 0.2 }
      : {
          blendMode: 'difference',
          brightness: 1,
          enabled: true,
          invert: true,
          objectPosition: 'top left',
          opacity: 0.025,
          saturate: 1,
          scale: 160
        }
}))

const basePreference = {
  intervalMinutes: 30 as const,
  mode: 'none' as const,
  rotationAnchorMs: 0,
  shuffleSeed: 0,
  sourcePath: null,
  strength: 10
}

describe('Backdrop', () => {
  beforeEach(() => {
    $backgroundPreference.set(basePreference)
    $backgroundRuntime.set({
      current: null,
      error: null,
      previous: null,
      resolving: false,
      sourceKey: '',
      truncated: false
    })
  })

  it('renders no artwork in none mode', () => {
    const { container } = render(<Backdrop />)
    expect(container.querySelector('img')).toBeNull()
  })

  it('renders the built-in Hermes artwork only in Hermes mode', () => {
    $backgroundPreference.set({ ...basePreference, mode: 'hermes' })
    const { container } = render(<Backdrop />)
    expect(container.querySelector('img')?.getAttribute('src')).toContain('ds-assets/filler-bg0.jpg')
  })

  it('renders the resolved custom image at the selected strength', () => {
    $backgroundPreference.set({ ...basePreference, mode: 'image', sourcePath: '/wallpaper.png', strength: 60 })
    $backgroundRuntime.set({
      current: {
        fingerprint: '1:10',
        id: 'wallpaper',
        name: 'wallpaper.png',
        url: 'hermes-background://image/token'
      },
      error: null,
      previous: null,
      resolving: false,
      sourceKey: 'default:image:/wallpaper.png',
      truncated: false
    })
    const { container } = render(<Backdrop />)

    const image = container.querySelector('img')
    const background = container.querySelector('[data-chat-background]')
    expect(image).not.toBeNull()
    expect(image?.getAttribute('src')).toBe('hermes-background://image/token')
    expect(background?.getAttribute('style')).toContain('opacity: 0.6')
  })
})
