import { useAuiState } from '@assistant-ui/react'
import { type RefObject, useCallback, useEffect, useRef, useState } from 'react'

import { useMediaQuery } from '@/hooks/use-media-query'
import { useResizeObserver } from '@/hooks/use-resize-observer'
import { $composerPoppedOut } from '@/store/composer-popout'
import { isSecondaryWindow } from '@/store/windows'

import { COMPOSER_SINGLE_LINE_MAX_PX, COMPOSER_STACK_BREAKPOINT_PX } from '../composer-utils'

interface UseComposerMetricsArgs {
  composerRef: RefObject<HTMLFormElement | null>
  composerSurfaceRef: RefObject<HTMLDivElement | null>
  editorRef: RefObject<HTMLDivElement | null>
  poppedOut: boolean
}

/**
 * Owns the composer's *sizing* engine: the stacked-vs-inline layout decision
 * and the measured-height CSS vars the thread reads for bottom clearance. All
 * work is edge-gated — the ResizeObserver only fires on real size changes, the
 * height vars are 8px-bucketed so per-keystroke growth never invalidates the
 * tree's computed style, and `tight` only flips when it crosses the breakpoint.
 * Returns `stacked` (the only value the render needs).
 */
export function useComposerMetrics({ composerRef, composerSurfaceRef, editorRef, poppedOut }: UseComposerMetricsArgs): {
  stacked: boolean
} {
  const [expanded, setExpanded] = useState(false)
  const [tight, setTight] = useState(false)
  const narrow = useMediaQuery('(max-width: 30rem)')

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
  const lastTightRef = useRef<boolean | null>(null)

  const syncComposerMetrics = useCallback(() => {
    const composer = composerRef.current

    if (!composer) {
      return
    }

    // Floating composer is out of the thread's flow — it must not reserve any
    // bottom clearance. Zero the measured vars so the thread reclaims the space.
    // (Read globals here so the callback stays stable; mirror the popoutAllowed
    // gate since secondary windows are forced docked.)
    if ($composerPoppedOut.get() && !isSecondaryWindow()) {
      const root = document.documentElement
      lastBucketedHeightRef.current = 0
      lastBucketedSurfaceHeightRef.current = 0
      root.style.setProperty('--composer-measured-height', '0px')
      root.style.setProperty('--composer-surface-measured-height', '0px')

      return
    }

    const { height, width } = composer.getBoundingClientRect()
    const surfaceHeight = composerSurfaceRef.current?.getBoundingClientRect().height
    const root = document.documentElement

    if (width > 0) {
      const nextTight = width < COMPOSER_STACK_BREAKPOINT_PX

      if (nextTight !== lastTightRef.current) {
        lastTightRef.current = nextTight
        setTight(nextTight)
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
        root.style.setProperty('--composer-measured-height', `${bucket}px`)
      }
    }

    if (surfaceHeight && surfaceHeight > 0) {
      const bucket = Math.round(surfaceHeight / 8) * 8

      if (bucket !== lastBucketedSurfaceHeightRef.current) {
        lastBucketedSurfaceHeightRef.current = bucket
        root.style.setProperty('--composer-surface-measured-height', `${bucket}px`)
      }
    }
  }, [composerRef, composerSurfaceRef, editorRef])

  useResizeObserver(syncComposerMetrics, composerRef, composerSurfaceRef, editorRef)

  // Toggling pop-out changes whether the composer reserves thread clearance.
  // The ResizeObserver may not fire (the box can keep the same box size), so
  // re-sync explicitly: docked republishes the measured height, floating zeroes
  // it so the thread reclaims the bottom space.
  useEffect(() => {
    syncComposerMetrics()
  }, [poppedOut, syncComposerMetrics])

  useEffect(() => {
    return () => {
      const root = document.documentElement
      root.style.removeProperty('--composer-measured-height')
      root.style.removeProperty('--composer-surface-measured-height')
    }
  }, [])

  return { stacked: expanded || narrow || tight }
}
