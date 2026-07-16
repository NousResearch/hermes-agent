import { useStore } from '@nanostores/react'
import { type ReactNode, useEffect, useMemo, useRef } from 'react'

import {
  $annotations,
  beginAnnotation,
  documentReviewContext,
  reconcileAnnotationAnchors,
  type ReviewContext,
  type TextAnnotationAnchor
} from '@/store/annotations'

import { captureTextPosition, contentFingerprint, restoreTextPosition } from './anchors'
import { ANNOTATION_NAVIGATE_EVENT } from './navigation'

interface AnnotatableTextProps {
  children: ReactNode
  kind: TextAnnotationAnchor['kind']
  path: string
  reviewContext?: ReviewContext
  text: string
}

interface HighlightRegistry {
  delete: (name: string) => void
  set: (name: string, highlight: unknown) => void
}

let nextHighlightId = 0

export function AnnotatableText({ children, kind, path, reviewContext, text }: AnnotatableTextProps) {
  const rootRef = useRef<HTMLDivElement>(null)
  const highlightNameRef = useRef<string | null>(null)

  highlightNameRef.current ??= `hermes-annotations-${(nextHighlightId += 1)}`
  const highlightName = highlightNameRef.current
  const annotations = useStore($annotations)
  const contentHash = useMemo(() => contentFingerprint(text), [text])

  const matching = useMemo(
    () =>
      annotations.filter(
        (item): item is typeof item & { anchor: TextAnnotationAnchor } =>
          item.anchor.path === path && item.anchor.kind === kind
      ),
    [annotations, kind, path]
  )

  useEffect(() => {
    const root = rootRef.current

    if (!root) {
      return
    }

    const ranges = matching
      .map(item => ({ id: item.id, range: restoreTextPosition(root, item.anchor) }))
      .filter((item): item is { id: string; range: Range } => item.range !== null)

    reconcileAnnotationAnchors(new Set(ranges.map(item => item.id)), path)

    const registry = (CSS as unknown as { highlights?: HighlightRegistry }).highlights

    const HighlightConstructor = (globalThis as unknown as { Highlight?: new (...ranges: Range[]) => unknown })
      .Highlight

    if (registry && HighlightConstructor) {
      registry.set(highlightName, new HighlightConstructor(...ranges.map(item => item.range)))

      return () => registry.delete(highlightName)
    }
  }, [highlightName, matching, path])

  useEffect(() => {
    const navigate = (event: Event) => {
      const detail = (event as CustomEvent<{ anchor?: TextAnnotationAnchor }>).detail
      const anchor = detail?.anchor
      const root = rootRef.current

      if (!root || !anchor || anchor.path !== path || anchor.kind !== kind) {
        return
      }

      const range = restoreTextPosition(root, anchor)

      if (!range) {
        return
      }

      const element = range.startContainer.parentElement
      element?.scrollIntoView({ behavior: 'smooth', block: 'center' })
      const browserSelection = window.getSelection()
      browserSelection?.removeAllRanges()
      browserSelection?.addRange(range)
    }

    window.addEventListener(ANNOTATION_NAVIGATE_EVENT, navigate)

    return () => window.removeEventListener(ANNOTATION_NAVIGATE_EVENT, navigate)
  }, [kind, path])

  const captureSelection = () => {
    const root = rootRef.current
    const browserSelection = window.getSelection()

    if (!root || !browserSelection || browserSelection.rangeCount === 0 || browserSelection.isCollapsed) {
      return
    }

    const range = browserSelection.getRangeAt(0).cloneRange()

    if (!root.contains(range.commonAncestorContainer)) {
      return
    }

    const position = captureTextPosition(root, range)

    if (!position) {
      return
    }

    const rect = range.getBoundingClientRect()

    const boundary =
      root.closest<HTMLElement>('[data-annotation-surface]')?.getBoundingClientRect() ?? root.getBoundingClientRect()

    beginAnnotation(
      { ...position, contentHash, kind, path },
      {
        boundary: { height: boundary.height, width: boundary.width, x: boundary.left, y: boundary.top },
        height: rect.height,
        width: rect.width,
        x: rect.left,
        y: rect.top
      },
      reviewContext ?? documentReviewContext(path, contentHash)
    )
  }

  return (
    <div className="relative h-full" onMouseUp={captureSelection} ref={rootRef}>
      {/* Kept runtime-local because the production CSS optimizer does not yet
          recognize the standards-based Custom Highlight pseudo-element. */}
      <style>{`::highlight(${highlightName}) { background-color: color-mix(in srgb, var(--ui-accent) 24%, transparent); }`}</style>
      {children}
    </div>
  )
}
