import { type ReactNode, useEffect, useState } from 'react'

import { Loader2 } from '@/lib/icons'
import { cn } from '@/lib/utils'

/**
 * A site's icon, resolved the thorough way.
 *
 * The work happens in the main process (`electron/favicon.ts`): read the
 * page, collect every declared `<link rel="icon">` and web-app-manifest
 * icon, rank them, then fall back to the well-known paths. None of that is
 * reachable from here — cross-origin HTML is unreadable under CORS — so this
 * component is just the request, the wait, and the placeholder.
 *
 * `fallback` holds the box before and after: a spinner marks the lookup, and
 * a site with no readable icon keeps the caller's own glyph rather than
 * flashing a broken image.
 */

/** url → data URL, or '' for a site we've already failed to read. The main
 *  process caches across restarts; this keeps a remount from crossing the
 *  IPC boundary at all. */
const resolved = new Map<string, string>()

export function Favicon({
  className,
  fallback,
  title,
  url
}: {
  className?: string
  fallback: ReactNode
  title?: string
  url: null | string | undefined
}) {
  const [icon, setIcon] = useState(() => (url ? (resolved.get(url) ?? '') : ''))
  const [pending, setPending] = useState(false)

  useEffect(() => {
    const resolveFavicon = window.hermesDesktop?.resolveFavicon

    if (!url || !resolveFavicon) {
      setIcon('')
      setPending(false)

      return
    }

    const cached = resolved.get(url)

    if (cached !== undefined) {
      setIcon(cached)
      setPending(false)

      return
    }

    let live = true

    setPending(true)

    void resolveFavicon(url)
      .catch(() => '')
      .then(found => {
        resolved.set(url, found)

        if (live) {
          setIcon(found)
          setPending(false)
        }
      })

    return () => {
      live = false
    }
  }, [url])

  return (
    <span className={cn('inline-grid size-full place-items-center', className)} title={title}>
      {icon ? (
        <img alt="" aria-hidden className="size-full object-contain" src={icon} />
      ) : pending ? (
        <Loader2 aria-hidden className="h-1/2 w-1/2 animate-spin opacity-60" />
      ) : (
        fallback
      )}
    </span>
  )
}
