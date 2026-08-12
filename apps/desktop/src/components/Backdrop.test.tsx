import { cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { DEFAULT_WALLPAPER_PREFERENCES } from '@/lib/wallpaper'
import { $wallpaper } from '@/store/wallpaper'

import { Backdrop, HiddenBackdropScope, WorkspaceWallpaperBackdrop, WorkspaceWallpaperScope } from './Backdrop'

const initialWallpaper = $wallpaper.get()

afterEach(() => {
  cleanup()
  $wallpaper.set(initialWallpaper)
})

describe('workspace wallpaper backdrop', () => {
  it('paints one wallpaper and one mask for the workspace and nested chat surfaces', () => {
    $wallpaper.set({
      ...initialWallpaper,
      asset: { url: 'hermes-wallpaper://default/wallpaper.jpg' },
      preferences: { ...DEFAULT_WALLPAPER_PREFERENCES, enabled: true },
      status: 'ready',
      supported: true
    })

    const { container } = render(
      <WorkspaceWallpaperScope>
        <WorkspaceWallpaperBackdrop />
        <Backdrop />
      </WorkspaceWallpaperScope>
    )

    expect(container.querySelectorAll('[data-wallpaper-mode]')).toHaveLength(1)
    expect(container.querySelectorAll('[data-wallpaper-mask-shape]')).toHaveLength(1)
  })

  it('suppresses image backdrops inside surfaces such as the HUD', () => {
    $wallpaper.set({
      ...initialWallpaper,
      asset: { url: 'hermes-wallpaper://default/wallpaper.jpg' },
      preferences: { ...DEFAULT_WALLPAPER_PREFERENCES, enabled: true },
      status: 'ready',
      supported: true
    })

    const { container } = render(
      <HiddenBackdropScope>
        <Backdrop />
      </HiddenBackdropScope>
    )

    expect(container.querySelector('[data-wallpaper-mode]')).toBeNull()
    expect(container.querySelector('img')).toBeNull()
  })
})
