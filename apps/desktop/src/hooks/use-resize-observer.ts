import { type RefObject, useLayoutEffect, useRef } from 'react'

/**
 * Observe element resizes. The callback receives the ResizeObserver entries
 * (empty only in non-RO environments) so callers can read the observed size
 * off the entry instead of forcing a fresh layout read.
 *
 * The initial measurement rides the observer's spec-guaranteed first delivery
 * (same frame, after layout, before paint) instead of a synchronous call from
 * the layout effect. A sync call here runs while the commit's layout is still
 * dirty, so any size read in the callback forces a full reflow — and with many
 * instances mounting at once (every user bubble on a session switch), the
 * interleaved read→write→read pattern cascades into seconds of layout thrash.
 * Inside RO timing, layout is already clean and the same reads are ~free.
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

    return () => observer.disconnect()
  }, [onResize])
}
