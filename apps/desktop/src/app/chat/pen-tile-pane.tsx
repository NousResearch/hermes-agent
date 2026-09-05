import { useEffect, useRef } from 'react'

import { $canvasTabs } from './canvas-tile'
import { ensurePenWebview, layoutPenWebview } from './pen-webview'

/** Host box for the singleton editor guest. Unmount hides it; it does not die. */
export function PenTilePane() {
  const hostRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    const host = hostRef.current
    const tab = $canvasTabs.get().find(t => t.provider === 'pen')

    if (!host || !tab) {
      return
    }

    ensurePenWebview(tab.url)
    layoutPenWebview(host)

    const sync = () => layoutPenWebview(hostRef.current)
    const ro = new ResizeObserver(sync)

    ro.observe(host)
    window.addEventListener('resize', sync)

    return () => {
      ro.disconnect()
      window.removeEventListener('resize', sync)
      layoutPenWebview(null)
    }
  }, [])

  return <div className="size-full min-h-0 min-w-0 bg-(--ui-editor-surface-background)" ref={hostRef} />
}
