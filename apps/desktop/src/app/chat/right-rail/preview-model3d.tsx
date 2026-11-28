import { useEffect, useRef, useState } from 'react'

import { PageLoader } from '@/components/page-loader'
import { onThemeRepaint } from '@/hooks/use-theme-epoch'
import { useI18n } from '@/i18n'

/** Resolve a CSS color (any syntax, incl. oklch/var()) to 0-255 RGB via a 1px canvas. */
function resolveRgb(el: HTMLElement, cssColor: string): [number, number, number] {
  const probe = document.createElement('span')

  probe.style.color = cssColor
  el.appendChild(probe)

  const resolved = getComputedStyle(probe).color

  probe.remove()

  const ctx = document.createElement('canvas').getContext('2d', { willReadFrequently: true })

  if (!ctx) {
    return [32, 32, 32]
  }

  ctx.fillStyle = resolved
  ctx.fillRect(0, 0, 1, 1)

  const [r = 32, g = 32, b = 32] = ctx.getImageData(0, 0, 1, 1).data

  return [r, g, b]
}

/** 3D model preview (stl/obj/glb/gltf/ply/3mf) via Online3DViewer's embedded
 *  viewer. The library is loaded lazily so it stays out of the main bundle. */
export function Model3dPreview({ dataUrl, name }: { dataUrl: string; name: string }) {
  const { t } = useI18n()
  const hostRef = useRef<HTMLDivElement>(null)
  const [error, setError] = useState<string>()
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const host = hostRef.current
    let viewer: { Destroy: () => void } | undefined
    let stopTheme: (() => void) | undefined
    let active = true

    setError(undefined)
    setLoading(true)

    void (async () => {
      try {
        const [o3dv, blob] = await Promise.all([import('online-3d-viewer'), fetch(dataUrl).then(r => r.blob())])

        if (!active || !host) {
          return
        }

        const themeColor = () => new o3dv.RGBAColor(...resolveRgb(host, 'var(--background)'), 255)

        const embedded = new o3dv.EmbeddedViewer(host, {
          backgroundColor: themeColor(),
          onModelLoadFailed: () => active && (setLoading(false), setError(t.preview.model3dFailed)),
          onModelLoaded: () => active && setLoading(false)
        })

        viewer = embedded
        stopTheme = onThemeRepaint(() => embedded.GetViewer().SetBackgroundColor(themeColor()))
        embedded.LoadModelFromFileList([new File([blob], name)])
      } catch (e) {
        if (active) {
          setLoading(false)
          setError(e instanceof Error ? e.message : String(e))
        }
      }
    })()

    return () => {
      active = false
      stopTheme?.()
      viewer?.Destroy()
    }
  }, [dataUrl, name, t])

  return (
    <div className="relative h-full w-full overflow-hidden">
      <div className="h-full w-full" ref={hostRef} />
      {loading && <PageLoader label={t.preview.model3dLoading} />}
      {error && <div className="absolute inset-0 grid place-items-center text-sm text-muted-foreground">{error}</div>}
    </div>
  )
}
