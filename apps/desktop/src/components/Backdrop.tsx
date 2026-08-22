import { useStore } from '@nanostores/react'
import { type CSSProperties, useEffect, useState } from 'react'

import { isFileMediaPath, resolveMediaDisplaySrc } from '@/lib/media'
import { $backdrop } from '@/store/backdrop'
import { useTheme } from '@/themes/context'

const assetPath = (path: string) => `${import.meta.env.BASE_URL}${path.replace(/^\/+/, '')}`

export function Backdrop() {
  const on = useStore($backdrop)
  const { theme } = useTheme()
  const wallpaper = (theme.backgroundImage ?? '').trim()
  const [wallpaperSrc, setWallpaperSrc] = useState('')

  useEffect(() => {
    let cancelled = false

    if (!wallpaper) {
      setWallpaperSrc('')

      return
    }

    // A backend skin may be remote. Keep the renderer from making arbitrary
    // outbound requests: files use the existing authenticated media path and
    // inline data must identify itself as an image.
    if (!isFileMediaPath(wallpaper) && !/^data:image\//i.test(wallpaper)) {
      setWallpaperSrc('')

      return
    }

    void resolveMediaDisplaySrc(wallpaper)
      .then(src => {
        if (!cancelled) {
          setWallpaperSrc(src)
        }
      })
      .catch(() => {
        if (!cancelled) {
          setWallpaperSrc('')
        }
      })

    return () => {
      cancelled = true
    }
  }, [wallpaper])

  if (!on && !wallpaperSrc) {
    return null
  }

  return (
    <>
      {wallpaperSrc ? (
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0 z-1 overflow-hidden"
          data-hermes-skin-wallpaper
        >
          <img
            alt=""
            className="h-full w-full"
            fetchPriority="low"
            src={wallpaperSrc}
            style={{
              objectFit: (theme.backgroundImageFit || 'cover') as CSSProperties['objectFit'],
              objectPosition: theme.backgroundImagePosition || 'center'
            }}
          />
          {theme.backgroundOverlay ? (
            <div
              className="absolute inset-0"
              data-hermes-skin-wallpaper-overlay
              style={{ background: theme.backgroundOverlay }}
            />
          ) : null}
        </div>
      ) : null}

      {on && !wallpaperSrc ? (
        <div aria-hidden className="pointer-events-none absolute inset-0 z-2 opacity-[0.025] mix-blend-difference">
          <img
            alt=""
            className="h-[160dvh] w-auto min-w-dvw object-cover object-left-top [filter:invert(var(--backdrop-invert-mul,1))]"
            fetchPriority="low"
            src={assetPath('ds-assets/filler-bg0.jpg')}
          />
        </div>
      ) : null}
    </>
  )
}
