import { useEffect } from 'react'
import { useSearchParams } from 'react-router-dom'

interface DeepLinkHighlightOptions {
  param: string
  ready: (target: string) => boolean
  elementId: (target: string) => string
  onResolve?: (target: string) => void
  block?: ScrollLogicalPosition
}

// Deep-link from the command palette (?<param>=<id>): once the target row is
// renderable, scroll it into view and flash it, then drop the param so it
// doesn't re-fire. Returns the pending target (null once consumed) so callers
// can force the row open before it mounts.
export function useDeepLinkHighlight({
  param,
  ready,
  elementId,
  onResolve,
  block = 'center'
}: DeepLinkHighlightOptions): null | string {
  const [searchParams, setSearchParams] = useSearchParams()
  const target = searchParams.get(param)

  useEffect(() => {
    if (!target || !ready(target)) {
      return
    }

    onResolve?.(target)

    let cancelled = false
    let timer = 0

    // onResolve may flip view state that mounts the row a few frames later, so
    // poll briefly for it and only drop the param AFTER a successful scroll —
    // deleting up front would lose the deep link when the target mounts late.
    let attempts = 0

    const attempt = () => {
      if (cancelled) {
        return
      }

      const element = document.getElementById(elementId(target))

      if (element) {
        element.scrollIntoView({ behavior: 'smooth', block })
        element.classList.add('setting-field-highlight')
        window.setTimeout(() => element.classList.remove('setting-field-highlight'), 1600)

        setSearchParams(
          previous => {
            const next = new URLSearchParams(previous)
            next.delete(param)

            return next
          },
          { replace: true }
        )

        return
      }

      if (attempts++ < 20) {
        timer = window.setTimeout(attempt, 80)
      }
    }

    timer = window.setTimeout(attempt, 80)

    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [block, elementId, onResolve, param, ready, setSearchParams, target])

  return target
}
