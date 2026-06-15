import { type RefObject, useLayoutEffect, useRef } from 'react'

/**
 * Observe element resizes. The callback receives the ResizeObserver entries
 * (empty on the initial synchronous call and in non-RO environments) so
 * callers can read the observed size off the entry instead of forcing a
 * fresh layout read.
 */
export function useResizeObserver(
  onResize: (entries: readonly ResizeObserverEntry[]) => void,
  ...refs: readonly RefObject<Element | null>[]
) {
  const refsRef = useRef(refs)
  refsRef.current = refs

  useLayoutEffect(() => {
    if (typeof ResizeObserver === 'undefined') {
      onResize([])

      return
    }

    const observer = new ResizeObserver(entries => onResize(entries))
    let observed = false

    for (const ref of refsRef.current) {
      const element = ref.current

      if (!element) {
        continue
      }

      observer.observe(element)
      observed = true
    }

    if (!observed) {
      observer.disconnect()

      return
    }

    onResize([])

    return () => observer.disconnect()
  }, [onResize])
}
