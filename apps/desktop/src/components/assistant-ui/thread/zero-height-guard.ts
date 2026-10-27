import { type RefObject, useCallback, useEffect, useRef } from 'react'

import { useResizeObserver } from '@/hooks/use-resize-observer'

/** One frame is enough. The nudge's listeners read layout synchronously, and
 *  the probe's own rect read flushes whatever is still dirty. */
const PROBE_DELAY_MS = 16

interface ZeroHeightViewportGuardOptions {
  /** The transcript scroller (`[data-slot="aui_thread-viewport"]`). */
  viewportRef: RefObject<HTMLElement | null>
  /** The pane is on screen AND the transcript has turns to paint. A hidden
   *  pane or an empty draft must never raise this. */
  armed: boolean
  /** Called once per collapse episode. */
  onCollapse: () => void
}

/**
 * A transcript can be fully loaded and still paint nothing: when the pane
 * around it resolves to zero height, every row below measures 0px, so the
 * scroller is invisible with no error anywhere (#141084). Nothing in the data
 * path can see that. The RPC succeeded and the message nodes exist, which is
 * why the blank pane reads as "the gateway is connected but chats don't load".
 *
 * So watch the scroller's own box. A delivery that reports height 0 while the
 * pane is visible and turns are present is a layout failure, not an empty
 * transcript. Nudge the pane shell's measurements once (geometry.ts re-reads
 * split geometry on window resize), then report if a fresh read is still 0.
 * One report per episode, re-armed by the first non-zero delivery, so a
 * re-render cannot stack toasts.
 *
 * Only real observer deliveries arm this. A non-RO environment hands back an
 * empty entry list, and claiming a collapse from a measurement nobody took
 * would be a lie.
 */
export function useZeroHeightViewportGuard({
  viewportRef,
  armed,
  onCollapse
}: ZeroHeightViewportGuardOptions) {
  const armedRef = useRef(armed)
  armedRef.current = armed

  const onCollapseRef = useRef(onCollapse)
  onCollapseRef.current = onCollapse

  // Cleared by the first healthy delivery, so a pane that recovers and
  // collapses again reports again.
  const reportedRef = useRef(false)
  const probeRef = useRef<null | ReturnType<typeof setTimeout>>(null)

  const cancelProbe = useCallback(() => {
    if (probeRef.current !== null) {
      clearTimeout(probeRef.current)
      probeRef.current = null
    }
  }, [])

  const probe = useCallback(() => {
    probeRef.current = null

    const element = viewportRef.current

    if (!armedRef.current || !element) {
      return
    }

    // The nudge may have been the fix. Measure before accusing it.
    if (element.getBoundingClientRect().height > 0) {
      return
    }

    reportedRef.current = true
    onCollapseRef.current()
  }, [viewportRef])

  const onResize = useCallback(
    (entries: readonly ResizeObserverEntry[]) => {
      const entry = entries[0]

      // No entry means nothing was measured, which is not evidence of a
      // collapse.
      if (!entry) {
        return
      }

      if (entry.contentRect.height > 0) {
        reportedRef.current = false
        cancelProbe()

        return
      }

      if (!armedRef.current || reportedRef.current || probeRef.current !== null) {
        return
      }

      // A minimized window measures every pane at 0. A toast raised from there
      // is one nobody can see, for a layout that is probably fine.
      if (typeof document !== 'undefined' && document.visibilityState === 'hidden') {
        return
      }

      if (typeof window !== 'undefined') {
        window.dispatchEvent(new Event('resize'))
      }

      probeRef.current = setTimeout(probe, PROBE_DELAY_MS)
    },
    [cancelProbe, probe]
  )

  useResizeObserver(onResize, viewportRef)

  useEffect(() => cancelProbe, [cancelProbe])
}
