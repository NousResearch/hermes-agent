import { useAuiState } from '@assistant-ui/react'
import { type RefObject, useCallback, useEffect, useRef, useState } from 'react'

import {
  chatSurfaceRoot,
  clearSurfaceVar,
  COMPOSER_HEIGHT_VAR,
  COMPOSER_SURFACE_HEIGHT_VAR,
  setSurfaceVar,
  THREAD_SETTLED_CLEARANCE_VAR
} from '@/app/chat/surface-vars'
import { useResizeObserver } from '@/hooks/use-resize-observer'

import {
  COMPOSER_COMPACT_PILL_PX,
  COMPOSER_FOLD_VOICE_PX,
  COMPOSER_MINIMAL_PX,
  COMPOSER_SINGLE_LINE_MAX_PX,
  COMPOSER_STACK_BREAKPOINT_PX
} from '../composer-utils'

interface UseComposerMetricsArgs {
  composerDockRef: RefObject<HTMLDivElement | null>
  composerRef: RefObject<HTMLFormElement | null>
  composerSurfaceRef: RefObject<HTMLDivElement | null>
  editorRef: RefObject<HTMLDivElement | null>
  poppedOut: boolean
  running: boolean
  surfaceKey?: string | null
}

/** Every width-driven collapse stage, resolved from the composer's own width. */
export interface ComposerFit {
  compactPill: boolean
  foldVoice: boolean
  minimal: boolean
  tight: boolean
}

const ROOMY: ComposerFit = { compactPill: false, foldVoice: false, minimal: false, tight: false }

const fitForWidth = (width: number): ComposerFit => ({
  compactPill: width < COMPOSER_COMPACT_PILL_PX,
  foldVoice: width < COMPOSER_FOLD_VOICE_PX,
  minimal: width < COMPOSER_MINIMAL_PX,
  tight: width < COMPOSER_STACK_BREAKPOINT_PX
})

const sameFit = (a: ComposerFit, b: ComposerFit) =>
  a.compactPill === b.compactPill && a.foldVoice === b.foldVoice && a.minimal === b.minimal && a.tight === b.tight

interface UseComposerMetricsResult extends ComposerFit {
  stacked: boolean
}

/**
 * Owns the composer's *sizing* engine: the stacked-vs-inline layout decision
 * and the measured-height CSS vars the thread reads for bottom clearance. All
 * work is edge-gated — the ResizeObserver only fires on real size changes, the
 * height vars are 8px-bucketed so per-keystroke growth never invalidates the
 * tree's computed style, and the fit only re-renders when it crosses a stage.
 */
export function useComposerMetrics({
  composerDockRef,
  composerRef,
  composerSurfaceRef,
  editorRef,
  poppedOut,
  running,
  surfaceKey
}: UseComposerMetricsArgs): UseComposerMetricsResult {
  const [expanded, setExpanded] = useState(false)
  const [fit, setFit] = useState<ComposerFit>(ROOMY)

  // Edge signals, not the live text: these only re-render when emptiness / the
  // presence of a non-trailing newline actually flips, so typing within a line
  // costs nothing here.
  const isEmpty = useAuiState(s => s.composer.text.length === 0)
  const hasHardNewline = useAuiState(s => s.composer.text.trimEnd().includes('\n'))

  // Expansion (input on its own full-width row, controls below) is driven by
  // the editor's *actual* rendered height via the ResizeObserver in
  // syncComposerMetrics — it only fires when the text genuinely wraps to a
  // second line, so the layout flips exactly at the wrap point rather than at
  // a guessed character count. We only handle the two cases the observer
  // can't: an explicit newline (expand before layout settles) and an emptied
  // draft (collapse back). We never read scrollHeight per keystroke.
  useEffect(() => {
    if (isEmpty) {
      setExpanded(false)

      return
    }

    if (expanded) {
      return
    }

    // Only a non-trailing newline forces an immediate expand. A trailing newline
    // (or phantom \n from contenteditable junk) is left to the ResizeObserver,
    // which expands only when the editor's real height actually grows.
    if (hasHardNewline) {
      setExpanded(true)
    }
  }, [expanded, hasHardNewline, isEmpty])

  // Bucket measured heights so we only invalidate the global CSS var when
  // the size crosses a meaningful threshold. Without bucketing, the editor
  // grows ~1px per character → setProperty fires every keystroke → entire
  // tree's computed style is invalidated → next paint forces a full
  // recalculate-style pass. With an 8px bucket, the invalidation rate drops
  // ~8× and small char-by-char typing produces no style invalidation at all
  // until a wrap or row change actually happens.
  const lastBucketedHeightRef = useRef(0)
  const lastBucketedSurfaceHeightRef = useRef(0)
  const lastRunningDockHeightRef = useRef(0)
  const lastSettledClearanceRef = useRef(0)
  const lastSyncedRunningRef = useRef(running)
  const lastSyncedSurfaceKeyRef = useRef(surfaceKey)
  const lastFitRef = useRef(ROOMY)
  // Mirrored into a ref so `syncComposerMetrics` stays referentially stable —
  // it's the shared ResizeObserver's handler, and a new identity every render
  // would re-register the observation.
  const poppedOutRef = useRef(poppedOut)
  poppedOutRef.current = poppedOut
  const runningRef = useRef(running)
  runningRef.current = running
  const surfaceKeyRef = useRef(surfaceKey)
  surfaceKeyRef.current = surfaceKey

  const syncComposerMetrics = useCallback(() => {
    const composer = composerRef.current
    // The dock is the full docked footprint — strips, status stack, composer —
    // so it, not the composer alone, is what the thread has to clear.
    const dock = composerDockRef.current

    if (!composer || !dock) {
      return
    }

    const startedRun = runningRef.current && !lastSyncedRunningRef.current
    const changedSurface = surfaceKeyRef.current !== lastSyncedSurfaceKeyRef.current
    lastSyncedRunningRef.current = runningRef.current
    lastSyncedSurfaceKeyRef.current = surfaceKeyRef.current

    if (startedRun || changedSurface) {
      lastRunningDockHeightRef.current = 0
    }

    if (changedSurface && lastSettledClearanceRef.current !== 0) {
      lastSettledClearanceRef.current = 0
      setSurfaceVar(composer, THREAD_SETTLED_CLEARANCE_VAR, '0px')
    }

    // Floating composer is out of the thread's flow — it must not reserve any
    // bottom clearance. Zero the measured vars so the thread reclaims the space.
    // Read through a ref so the callback stays stable, and read THIS surface's
    // own state: pop-out is per layout zone, so a float in the left split must
    // not zero the right split's clearance.
    if (poppedOutRef.current) {
      lastBucketedHeightRef.current = 0
      lastBucketedSurfaceHeightRef.current = 0
      lastRunningDockHeightRef.current = 0
      lastSettledClearanceRef.current = 0
      setSurfaceVar(composer, COMPOSER_HEIGHT_VAR, '0px')
      setSurfaceVar(composer, COMPOSER_SURFACE_HEIGHT_VAR, '0px')
      setSurfaceVar(composer, THREAD_SETTLED_CLEARANCE_VAR, '0px')

      return
    }

    const { height } = dock.getBoundingClientRect()
    const { width } = composer.getBoundingClientRect()
    const surfaceHeight = composerSurfaceRef.current?.getBoundingClientRect().height

    if (width > 0) {
      const nextFit = fitForWidth(width)

      if (!sameFit(nextFit, lastFitRef.current)) {
        lastFitRef.current = nextFit
        setFit(nextFit)
      }
    }

    // Expand once the input has actually wrapped past a single line. The
    // observer only fires on real size changes, so this reads scrollHeight at
    // most once per wrap (not per keystroke). One line ≈ 28px (1.625rem
    // min-height + padding); a second line clears ~36px. We only ever expand
    // here — collapse is handled by the emptied-draft effect to avoid
    // oscillating across the wrap boundary as the input switches widths.
    const editor = editorRef.current

    if (editor && editor.scrollHeight > COMPOSER_SINGLE_LINE_MAX_PX) {
      setExpanded(true)
    }

    if (height > 0) {
      const bucket = Math.round(height / 8) * 8

      if (bucket !== lastBucketedHeightRef.current) {
        lastBucketedHeightRef.current = bucket
        setSurfaceVar(composer, COMPOSER_HEIGHT_VAR, `${bucket}px`)
      }

      // The in-flow status stack can be almost half a viewport tall. Keep the
      // largest dock reached by this run. If transient status UI disappears
      // before the terminal running=false render, publish that maximum
      // immediately: otherwise the active layout contracts first and settlement
      // reintroduces the old maximum as an inverse scroll jump.
      const previousRunningDockHeight = lastRunningDockHeightRef.current

      if (runningRef.current) {
        lastRunningDockHeightRef.current = Math.max(previousRunningDockHeight, bucket)
      }

      const clearanceFloor = runningRef.current
        ? !startedRun && bucket < previousRunningDockHeight
          ? previousRunningDockHeight
          : 0
        : lastRunningDockHeightRef.current

      if (clearanceFloor !== lastSettledClearanceRef.current) {
        lastSettledClearanceRef.current = clearanceFloor
        setSurfaceVar(composer, THREAD_SETTLED_CLEARANCE_VAR, `${clearanceFloor}px`)
      }
    }

    if (surfaceHeight && surfaceHeight > 0) {
      const bucket = Math.round(surfaceHeight / 8) * 8

      if (bucket !== lastBucketedSurfaceHeightRef.current) {
        lastBucketedSurfaceHeightRef.current = bucket
        setSurfaceVar(composer, COMPOSER_SURFACE_HEIGHT_VAR, `${bucket}px`)
      }
    }
  }, [composerDockRef, composerRef, composerSurfaceRef, editorRef])

  useResizeObserver(syncComposerMetrics, composerDockRef, composerRef, composerSurfaceRef, editorRef)

  // Toggling pop-out changes whether the composer reserves thread clearance.
  // Starting a new run releases the previous turn's clearance floor; the
  // thread's run-start handler owns the deliberate re-anchor to the new bottom.
  // Neither transition necessarily resizes the dock, so re-sync explicitly.
  useEffect(() => {
    syncComposerMetrics()
  }, [poppedOut, running, surfaceKey, syncComposerMetrics])

  // A hard newline expands through React before layout settles, and emptying the
  // draft collapses through the same path. Neither transition depends on a
  // ResizeObserver delivery, so measure the committed layout explicitly. Once
  // an idle user deliberately reshapes the composer, the previous run's floor
  // no longer owns the geometry; release it instead of preserving stale space.
  const syncExpandedComposerMetrics = useCallback(() => {
    if (!runningRef.current) {
      lastRunningDockHeightRef.current = 0

      if (lastSettledClearanceRef.current !== 0) {
        lastSettledClearanceRef.current = 0
        setSurfaceVar(composerRef.current, THREAD_SETTLED_CLEARANCE_VAR, '0px')
      }
    }

    syncComposerMetrics()
  }, [composerRef, syncComposerMetrics])

  useEffect(() => {
    syncExpandedComposerMetrics()
  }, [expanded, syncExpandedComposerMetrics])

  useEffect(() => {
    // Resolve the owning surface while the composer is still attached; the
    // unmount cleanup runs after React detached the node, where closest() can
    // no longer find [data-chat-surface].
    const root = chatSurfaceRoot(composerRef.current)

    return () => {
      clearSurfaceVar(root, COMPOSER_HEIGHT_VAR)
      clearSurfaceVar(root, COMPOSER_SURFACE_HEIGHT_VAR)
      clearSurfaceVar(root, THREAD_SETTLED_CLEARANCE_VAR)
    }
  }, [composerRef])

  // Every decision comes from the composer's OWN measured width, never the
  // viewport's. There used to be a `(max-width: 30rem)` media query in here as
  // well, and it quietly outranked everything: any window under 480px stacked
  // the row AND compacted the pill in the same instant, regardless of how much
  // room the composer actually had. That collapsed the whole progressive ladder
  // into one step for small windows — HUD mode is ~470px, so it never saw the
  // ladder at all — and it disagreed with the measured breakpoints (320 to
  // stack) by 160px. The ResizeObserver knows the real width; the viewport is
  // not a proxy for it.
  //
  // The ladder is monotonic: each stage implies the ones above it, so the pill
  // is always compact by the time the row stacks, and the voice controls are
  // always folded before minimal drops them.
  return {
    compactPill: fit.compactPill || fit.tight,
    foldVoice: fit.foldVoice || fit.minimal,
    minimal: fit.minimal,
    stacked: expanded || fit.tight,
    tight: fit.tight
  }
}
