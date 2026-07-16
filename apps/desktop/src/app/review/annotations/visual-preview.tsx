import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'

import { captureTextPosition, restoreTextPosition } from '@/app/review/annotations/anchors'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'
import { defaultOcrLanguage, type OcrLanguage, type OcrWord, recognizeImageText } from '@/lib/ocr-runtime'
import { cn } from '@/lib/utils'
import {
  $annotationDraft,
  $annotationDraftAnchor,
  $annotations,
  type AnnotationEditorAnchor,
  beginAnnotation,
  editAnnotation,
  type ReviewContext,
  type TextPosition,
  updateAnnotationDraft,
  type VisualAnnotationAnchor,
  type VisualAnnotationMark,
  type VisualPoint
} from '@/store/annotations'

type VisualTool = 'arrow' | 'pan' | 'pen' | 'pin' | 'rectangle' | 'select' | 'text'

export const VISUAL_ZOOM_MIN = 25
export const VISUAL_ZOOM_MAX = 400
export const VISUAL_ZOOM_STEP = 25

export function clampVisualZoom(zoom: number): number {
  return Math.max(VISUAL_ZOOM_MIN, Math.min(VISUAL_ZOOM_MAX, zoom))
}

function VisualZoomControls({ onZoomChange, zoom }: { onZoomChange: (zoom: number) => void; zoom: number }) {
  const { t } = useI18n()
  const copy = t.desktop.annotations.preview.image

  return (
    <div
      aria-label={copy.zoom}
      className="flex h-6 shrink-0 items-stretch overflow-hidden rounded border border-(--ui-stroke-secondary) bg-(--ui-surface-secondary) text-[0.625rem] tabular-nums text-foreground"
      role="group"
    >
      <Button
        aria-label={copy.decrease}
        className="grid w-6 place-items-center hover:bg-accent disabled:opacity-35"
        disabled={zoom <= VISUAL_ZOOM_MIN}
        onClick={() => onZoomChange(clampVisualZoom(zoom - VISUAL_ZOOM_STEP))}
        size="icon"
        title={copy.decrease}
        type="button"
        variant="ghost"
      >
        −
      </Button>
      <Button
        aria-label={copy.reset(zoom)}
        className="min-w-10 border-x border-border/70 px-1 hover:bg-accent"
        onClick={() => onZoomChange(100)}
        size="sm"
        title={copy.reset(zoom)}
        type="button"
        variant="ghost"
      >
        {zoom}%
      </Button>
      <Button
        aria-label={copy.increase}
        className="grid w-6 place-items-center hover:bg-accent disabled:opacity-35"
        disabled={zoom >= VISUAL_ZOOM_MAX}
        onClick={() => onZoomChange(clampVisualZoom(zoom + VISUAL_ZOOM_STEP))}
        size="icon"
        title={copy.increase}
        type="button"
        variant="ghost"
      >
        +
      </Button>
    </div>
  )
}

interface VisualAnnotationPreviewProps {
  contentHash?: string
  dataUrl: string
  filePath: string
  label: string
  mediaKind: VisualAnnotationAnchor['mediaKind']
  reviewContext: ReviewContext
  svgSource?: string
}

function id(): string {
  return globalThis.crypto?.randomUUID?.() ?? `mark-${Date.now()}-${Math.random().toString(36).slice(2)}`
}

function clamp(value: number): number {
  return Math.max(0, Math.min(1, value))
}

function translateMark(mark: VisualAnnotationMark, dx: number, dy: number): VisualAnnotationMark {
  const move = (point: VisualPoint) => ({ x: clamp(point.x + dx), y: clamp(point.y + dy) })

  if (mark.tool === 'pen') {
    return { ...mark, points: mark.points.map(move) }
  }

  if (mark.tool === 'pin') {
    return { ...mark, point: move(mark.point) }
  }

  return { ...mark, end: move(mark.end), start: move(mark.start) }
}

function markPath(mark: Extract<VisualAnnotationMark, { tool: 'pen' }>, width: number, height: number): string {
  return mark.points.map((point, index) => `${index ? 'L' : 'M'} ${point.x * width} ${point.y * height}`).join(' ')
}

function MarkShape({
  height,
  mark,
  number,
  selected,
  width
}: {
  height: number
  mark: VisualAnnotationMark
  number: number
  selected: boolean
  width: number
}) {
  const common = {
    className: cn('pointer-events-auto vector-effect-non-scaling-stroke', selected && 'drop-shadow-[0_0_3px_white]'),
    stroke: 'var(--ui-accent)',
    strokeWidth: selected ? 2.5 : 2
  }

  if (mark.tool === 'pen') {
    return (
      <path {...common} d={markPath(mark, width, height)} fill="none" strokeLinecap="round" strokeLinejoin="round" />
    )
  }

  if (mark.tool === 'rectangle') {
    const x = Math.min(mark.start.x, mark.end.x) * width
    const y = Math.min(mark.start.y, mark.end.y) * height

    return (
      <rect
        {...common}
        fill="color-mix(in srgb, var(--ui-accent) 12%, transparent)"
        height={Math.abs(mark.end.y - mark.start.y) * height}
        width={Math.abs(mark.end.x - mark.start.x) * width}
        x={x}
        y={y}
      />
    )
  }

  if (mark.tool === 'arrow') {
    return (
      <line
        {...common}
        markerEnd="url(#visual-arrowhead)"
        x1={mark.start.x * width}
        x2={mark.end.x * width}
        y1={mark.start.y * height}
        y2={mark.end.y * height}
      />
    )
  }

  return (
    <g className="pointer-events-auto">
      <circle cx={mark.point.x * width} cy={mark.point.y * height} fill="var(--ui-accent)" r="9" />
      <text
        dominantBaseline="central"
        fill="var(--ui-accent-foreground)"
        fontSize="11"
        fontWeight="700"
        textAnchor="middle"
        x={mark.point.x * width}
        y={mark.point.y * height}
      >
        {number}
      </text>
    </g>
  )
}

export function svgHasExternalResources(source: string): boolean {
  return /(?:href|xlink:href)\s*=\s*["']\s*(?!#|data:|blob:)|url\(\s*["']?\s*(?!#|data:|blob:)/i.test(source)
}

export function buildSvgSandboxDocument(source: string): string {
  return `<!doctype html>
<html><head>
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data: blob:; font-src data:; style-src 'unsafe-inline'; connect-src 'none'; media-src 'none'; object-src 'none'; frame-src 'none'; form-action 'none'; base-uri 'none'">
<style>html,body{margin:0;width:100%;height:100%;overflow:hidden;background:transparent}body{display:grid;place-items:center}svg{display:block;width:100%;height:100%;max-width:100%;max-height:100%;user-select:text}a{cursor:default}::highlight(hermes-svg-saved){background:rgba(90,130,255,.24)}::highlight(hermes-svg-draft){background:rgba(90,130,255,.38)}::highlight(hermes-svg-pending){background:rgba(90,130,255,.48)}</style>
</head><body>${source}</body></html>`
}

function svgNaturalSize(source: string): { height: number; width: number } {
  const parsed = new DOMParser().parseFromString(source, 'image/svg+xml')
  const root = parsed.documentElement

  const viewBox = root
    .getAttribute('viewBox')
    ?.trim()
    .split(/[\s,]+/)
    .map(Number)

  if (viewBox?.length === 4 && viewBox[2] > 0 && viewBox[3] > 0) {
    return { height: viewBox[3], width: viewBox[2] }
  }

  const width = Number.parseFloat(root.getAttribute('width') ?? '')
  const height = Number.parseFloat(root.getAttribute('height') ?? '')

  return { height: height > 0 ? height : 600, width: width > 0 ? width : 800 }
}

export function VisualAnnotationPreview({
  contentHash,
  dataUrl,
  filePath,
  label,
  mediaKind,
  reviewContext,
  svgSource
}: VisualAnnotationPreviewProps) {
  const { t } = useI18n()
  const previewCopy = t.desktop.annotations.preview
  const annotations = useStore($annotations)
  const draft = useStore($annotationDraft)
  const draftAnchor = useStore($annotationDraftAnchor)
  const viewportRef = useRef<HTMLDivElement>(null)
  const surfaceRef = useRef<HTMLDivElement>(null)
  const imageRef = useRef<HTMLImageElement>(null)
  const svgFrameRef = useRef<HTMLIFrameElement>(null)
  const svgSize = useMemo(() => (svgSource ? svgNaturalSize(svgSource) : { height: 600, width: 800 }), [svgSource])
  const [dimensions, setDimensions] = useState({ height: 1, width: 1 })
  const [renderedDimensions, setRenderedDimensions] = useState({ height: 1, width: 1 })
  const [svgRevision, setSvgRevision] = useState(0)

  const [svgSelection, setSvgSelection] = useState<null | {
    anchor: TextPosition
    editorAnchor: AnnotationEditorAnchor
    range: Range
  }>(null)

  const [tool, setTool] = useState<VisualTool>(mediaKind === 'svg' ? 'text' : 'pan')
  const [zoom, setZoom] = useState(100)
  const [language, setLanguage] = useState<OcrLanguage>(() => defaultOcrLanguage())
  const [ocrWords, setOcrWords] = useState<OcrWord[]>([])
  const [ocrState, setOcrState] = useState<'error' | 'idle' | 'loading' | 'ready'>('idle')
  const [selectedMark, setSelectedMark] = useState<string | null>(null)
  const [history, setHistory] = useState<VisualAnnotationMark[][]>([])
  const [liveMark, setLiveMark] = useState<VisualAnnotationMark | null>(null)

  const languages = (['eng', 'jpn', 'chi_sim', 'chi_tra'] as const).map(value => ({
    label: previewCopy.ocr.languages[value],
    value
  }))

  const pointerFrame = useRef<number | null>(null)
  const pendingPointer = useRef<{ clientX: number; clientY: number } | null>(null)
  const cancelGestureRef = useRef<() => void>(() => undefined)

  const gesture = useRef<null | {
    mark: VisualAnnotationMark
    origin: VisualPoint
    original?: VisualAnnotationMark
    snapshot?: VisualAnnotationMark[]
  }>(null)

  const panGesture = useRef<null | {
    clientX: number
    clientY: number
    scrollLeft: number
    scrollTop: number
  }>(null)

  const svgDocument = useMemo(() => buildSvgSandboxDocument(svgSource ?? ''), [svgSource])
  const hasExternalSvgResources = useMemo(() => Boolean(svgSource && svgHasExternalResources(svgSource)), [svgSource])

  const visualItems = annotations.filter(
    item =>
      item.anchor.kind === 'visual' &&
      item.anchor.path === filePath &&
      item.status !== 'stale' &&
      item.id !== draft?.editingId
  )

  const visualDraft =
    draft?.contextId === reviewContext.id && draft.anchor.kind === 'visual' && draft.anchor.path === filePath
      ? draft.anchor
      : null

  const marks = visualDraft?.marks ?? []

  useEffect(() => {
    const surface = surfaceRef.current

    if (!surface || typeof ResizeObserver === 'undefined') {
      return
    }

    const update = () => {
      const rect = surface.getBoundingClientRect()

      if (rect.width > 0 && rect.height > 0) {
        setRenderedDimensions(current =>
          current.width === rect.width && current.height === rect.height
            ? current
            : { height: rect.height, width: rect.width }
        )
      }
    }

    const observer = new ResizeObserver(update)
    observer.observe(surface)
    update()

    return () => observer.disconnect()
  }, [mediaKind])

  useEffect(() => {
    if (mediaKind !== 'svg') {
      return
    }

    setDimensions(svgSize)
  }, [mediaKind, svgSize])

  useEffect(() => {
    if (mediaKind !== 'svg' || svgRevision === 0) {
      return
    }

    const frame = svgFrameRef.current
    const frameWindow = frame?.contentWindow
    const frameDocument = frame?.contentDocument

    if (!frame || !frameWindow || !frameDocument) {
      return
    }

    const preventNavigation = (event: Event) => {
      if ((event.target as Element | null)?.closest?.('a')) {
        event.preventDefault()
      }
    }

    const captureSelection = () => {
      if (tool !== 'text') {
        return
      }

      const selection = frameWindow.getSelection()

      if (!selection || selection.isCollapsed || selection.rangeCount === 0) {
        return
      }

      const range = selection.getRangeAt(0).cloneRange()
      const position = captureTextPosition(frameDocument.documentElement, range)

      if (!position) {
        return
      }

      const rangeRect = range.getBoundingClientRect()
      const frameRect = frame.getBoundingClientRect()
      const boundary = surfaceRef.current?.closest<HTMLElement>('[data-annotation-surface]')?.getBoundingClientRect()

      setSvgSelection({
        anchor: position,
        editorAnchor: {
          boundary: boundary
            ? { height: boundary.height, width: boundary.width, x: boundary.left, y: boundary.top }
            : undefined,
          height: rangeRect.height,
          width: rangeRect.width,
          x: frameRect.left + rangeRect.left,
          y: frameRect.top + rangeRect.top
        },
        range
      })
    }

    frameDocument.addEventListener('click', preventNavigation, true)
    frameDocument.addEventListener('mouseup', captureSelection)

    return () => {
      frameDocument.removeEventListener('click', preventNavigation, true)
      frameDocument.removeEventListener('mouseup', captureSelection)
    }
  }, [mediaKind, svgRevision, tool])

  useEffect(() => {
    if (mediaKind !== 'svg' || svgRevision === 0) {
      return
    }

    const frameWindow = svgFrameRef.current?.contentWindow as
      | (Window & {
          CSS?: { highlights?: { delete: (name: string) => void; set: (name: string, value: unknown) => void } }
          Highlight?: new (...ranges: Range[]) => unknown
        })
      | null

    const root = svgFrameRef.current?.contentDocument?.documentElement
    const registry = frameWindow?.CSS?.highlights
    const HighlightConstructor = frameWindow?.Highlight

    if (!root || !registry || !HighlightConstructor) {
      return
    }

    const savedRanges = annotations
      .flatMap(item =>
        item.anchor.kind === 'svg' && item.anchor.path === filePath ? [restoreTextPosition(root, item.anchor)] : []
      )
      .filter((range): range is Range => range !== null)

    const draftRange =
      draftAnchor?.kind === 'svg' && draftAnchor.path === filePath ? restoreTextPosition(root, draftAnchor) : null

    registry.set('hermes-svg-saved', new HighlightConstructor(...savedRanges))

    if (draftRange) {
      registry.set('hermes-svg-draft', new HighlightConstructor(draftRange))
    } else {
      registry.delete('hermes-svg-draft')
    }

    if (svgSelection) {
      registry.set('hermes-svg-pending', new HighlightConstructor(svgSelection.range))
    } else {
      registry.delete('hermes-svg-pending')
    }

    return () => {
      registry.delete('hermes-svg-saved')
      registry.delete('hermes-svg-draft')
      registry.delete('hermes-svg-pending')
    }
  }, [annotations, draftAnchor, filePath, mediaKind, svgRevision, svgSelection])

  useEffect(() => {
    if (mediaKind === 'svg' || !dataUrl || !contentHash) {
      return
    }

    let active = true

    const timer = window.setTimeout(() => {
      setOcrState('loading')
      void recognizeImageText(dataUrl, contentHash, language)
        .then(words => {
          if (active) {
            setOcrWords(words)
            setOcrState('ready')
          }
        })
        .catch(() => active && setOcrState('error'))
    }, 250)

    return () => {
      active = false
      window.clearTimeout(timer)
    }
  }, [contentHash, dataUrl, language, mediaKind])

  const pointFromClient = (clientX: number, clientY: number): VisualPoint => {
    const rect = surfaceRef.current!.getBoundingClientRect()

    return { x: clamp((clientX - rect.left) / rect.width), y: clamp((clientY - rect.top) / rect.height) }
  }

  const pointFromEvent = (event: React.PointerEvent): VisualPoint => pointFromClient(event.clientX, event.clientY)

  const setMarks = (next: VisualAnnotationMark[], previous = marks) => {
    if (!visualDraft) {
      const rect = surfaceRef.current?.getBoundingClientRect()
      beginAnnotation(
        {
          contentHash,
          kind: 'visual',
          marks: next,
          mediaKind,
          naturalHeight: dimensions.height,
          naturalWidth: dimensions.width,
          path: filePath
        },
        rect ? { boundary: rect, height: 28, width: 28, x: rect.right - 40, y: rect.top + 40 } : null,
        reviewContext
      )
    } else {
      setHistory(value => [...value.slice(-30), previous])
      updateAnnotationDraft({ anchor: { ...visualDraft, marks: next } })
    }
  }

  const onPointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    if (tool === 'text') {
      return
    }

    event.currentTarget.setPointerCapture(event.pointerId)

    if (tool === 'pan') {
      const viewport = viewportRef.current

      if (viewport) {
        panGesture.current = {
          clientX: event.clientX,
          clientY: event.clientY,
          scrollLeft: viewport.scrollLeft,
          scrollTop: viewport.scrollTop
        }
        event.currentTarget.style.cursor = 'grabbing'
        event.preventDefault()
      }

      return
    }

    const point = pointFromEvent(event)

    if (tool === 'select' && selectedMark) {
      const original = marks.find(mark => mark.id === selectedMark)

      if (original) {
        gesture.current = { mark: original, origin: point, original, snapshot: marks }
      }

      return
    }

    if (tool === 'select') {
      return
    }

    const mark: VisualAnnotationMark =
      tool === 'pen'
        ? { id: id(), points: [point], tool }
        : tool === 'pin'
          ? { id: id(), point, tool }
          : { end: point, id: id(), start: point, tool }

    gesture.current = { mark, origin: point }
    setLiveMark(mark)

    if (tool === 'pin') {
      setMarks([...marks, mark])
      setSelectedMark(mark.id)
      gesture.current = null
      setLiveMark(null)
    }
  }

  const applyPointerMove = (clientX: number, clientY: number) => {
    const pan = panGesture.current

    if (pan) {
      const viewport = viewportRef.current

      if (viewport) {
        viewport.scrollLeft = pan.scrollLeft + pan.clientX - clientX
        viewport.scrollTop = pan.scrollTop + pan.clientY - clientY
      }

      return
    }

    const active = gesture.current

    if (!active) {
      return
    }

    const point = pointFromClient(clientX, clientY)

    if (active.original) {
      const moved = translateMark(active.original, point.x - active.origin.x, point.y - active.origin.y)
      updateAnnotationDraft({
        anchor: { ...visualDraft!, marks: marks.map(mark => (mark.id === moved.id ? moved : mark)) }
      })

      return
    }

    if (active.mark.tool === 'pen') {
      const previous = active.mark.points.at(-1)
      const movedEnough = !previous || Math.hypot(point.x - previous.x, point.y - previous.y) >= 0.001

      if (movedEnough && active.mark.points.length < 2048) {
        active.mark = { ...active.mark, points: [...active.mark.points, point] }
      }
    } else if (active.mark.tool !== 'pin') {
      active.mark = { ...active.mark, end: point }
    }

    setLiveMark(active.mark)
  }

  const flushPointerMove = () => {
    if (pointerFrame.current !== null) {
      window.cancelAnimationFrame(pointerFrame.current)
      pointerFrame.current = null
    }

    const pending = pendingPointer.current
    pendingPointer.current = null

    if (pending) {
      applyPointerMove(pending.clientX, pending.clientY)
    }
  }

  const onPointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    pendingPointer.current = { clientX: event.clientX, clientY: event.clientY }

    if (pointerFrame.current !== null) {
      return
    }

    pointerFrame.current = window.requestAnimationFrame(() => {
      pointerFrame.current = null
      const pending = pendingPointer.current
      pendingPointer.current = null

      if (pending) {
        applyPointerMove(pending.clientX, pending.clientY)
      }
    })
  }

  const onPointerUp = (event: React.PointerEvent<HTMLDivElement>) => {
    flushPointerMove()

    if (panGesture.current) {
      panGesture.current = null
      event.currentTarget.style.cursor = 'grab'

      return
    }

    const active = gesture.current
    gesture.current = null
    setLiveMark(null)

    if (!active) {
      return
    }

    if (active.original) {
      if (active.snapshot) {
        setHistory(value => [...value.slice(-30), active.snapshot!])
      }

      return
    }

    setMarks([...marks, active.mark])
    setSelectedMark(active.mark.id)
  }

  const cancelPointerGesture = () => {
    pendingPointer.current = null

    if (pointerFrame.current !== null) {
      window.cancelAnimationFrame(pointerFrame.current)
    }

    pointerFrame.current = null
    panGesture.current = null
    const active = gesture.current
    gesture.current = null
    setLiveMark(null)

    if (active?.original && active.snapshot && visualDraft) {
      updateAnnotationDraft({ anchor: { ...visualDraft, marks: active.snapshot } })
    }
  }

  cancelGestureRef.current = cancelPointerGesture

  useEffect(() => {
    const cancel = () => cancelGestureRef.current()
    window.addEventListener('blur', cancel)

    return () => {
      window.removeEventListener('blur', cancel)
      cancel()
    }
  }, [])

  const captureText = () => {
    if (tool !== 'text' || mediaKind === 'svg') {
      return
    }

    const selection = window.getSelection()
    const quote = selection?.toString().trim()

    if (!quote || !surfaceRef.current?.contains(selection?.anchorNode ?? null)) {
      return
    }

    beginAnnotation(
      {
        contentHash,
        kind: 'visual',
        marks: [],
        mediaKind,
        naturalHeight: dimensions.height,
        naturalWidth: dimensions.width,
        path: filePath,
        quote
      },
      null,
      reviewContext
    )
  }

  const allMarks = [
    ...visualItems.flatMap(item =>
      item.anchor.kind === 'visual' ? item.anchor.marks.map(mark => ({ item, mark })) : []
    ),
    ...marks.map(mark => ({ item: null, mark })),
    ...(liveMark ? [{ item: null, mark: liveMark }] : [])
  ]

  return (
    <div className="flex h-full min-h-0 flex-col bg-transparent">
      <div className="flex h-9 shrink-0 items-center gap-0.5 overflow-x-auto border-b border-border/60 px-2 [scrollbar-width:none]">
        <VisualZoomControls onZoomChange={setZoom} zoom={zoom} />
        <span className="mx-1 h-4 w-px shrink-0 bg-border" />
        {(
          [
            ['pan', 'grabber'],
            ['select', 'move'],
            ['text', 'symbol-key'],
            ['pen', 'edit'],
            ['rectangle', 'primitive-square'],
            ['arrow', 'arrow-right'],
            ['pin', 'pin']
          ] as const
        ).map(([value, icon]) => (
          <Button
            aria-label={previewCopy.tools[value]}
            className={cn(
              'grid size-7 place-items-center rounded hover:bg-accent',
              tool === value && 'bg-accent text-foreground'
            )}
            key={value}
            onClick={() => setTool(value)}
            size="icon"
            title={previewCopy.tools[value]}
            type="button"
            variant="ghost"
          >
            <Codicon name={icon} />
          </Button>
        ))}
        <span className="mx-1 h-4 w-px bg-border" />
        <Button
          aria-label={previewCopy.image.undoMarkup}
          className="grid size-7 place-items-center rounded hover:bg-accent disabled:opacity-35"
          disabled={!visualDraft || history.length === 0}
          onClick={() => {
            const previous = history.at(-1)

            if (previous && visualDraft) {
              setHistory(value => value.slice(0, -1))
              updateAnnotationDraft({ anchor: { ...visualDraft, marks: previous } })
            }
          }}
          size="icon"
          title={previewCopy.image.undoMarkup}
          type="button"
          variant="ghost"
        >
          <Codicon name="discard" />
        </Button>
        <Button
          aria-label={previewCopy.image.deleteMarkup}
          className="grid size-7 place-items-center rounded hover:bg-accent disabled:opacity-35"
          disabled={!visualDraft || !selectedMark}
          onClick={() => {
            if (visualDraft && selectedMark) {
              setMarks(marks.filter(mark => mark.id !== selectedMark))
              setSelectedMark(null)
            }
          }}
          size="icon"
          title={previewCopy.image.deleteMarkup}
          type="button"
          variant="ghost"
        >
          <Codicon name="trash" />
        </Button>
        {mediaKind !== 'svg' && (
          <>
            <span className="ml-1 text-[0.62rem] text-muted-foreground">
              {ocrState === 'loading'
                ? previewCopy.ocr.recognizing
                : ocrState === 'error'
                  ? previewCopy.ocr.unavailable
                  : previewCopy.ocr.label}
            </span>
            <select
              aria-label={previewCopy.ocr.language}
              className="h-7 rounded border border-border bg-background px-1 text-[0.62rem]"
              onChange={event => setLanguage(event.target.value as OcrLanguage)}
              value={language}
            >
              {languages.map(item => (
                <option key={item.value} value={item.value}>
                  {item.label}
                </option>
              ))}
            </select>
          </>
        )}
      </div>

      <div className="min-h-0 flex-1 overflow-auto overscroll-contain p-4" ref={viewportRef}>
        {hasExternalSvgResources && (
          <Alert className="mx-auto mb-2 w-fit grid-cols-1 px-2 py-1 text-[0.65rem]" variant="warning">
            <AlertDescription className="col-start-1">{previewCopy.image.externalResourcesBlocked}</AlertDescription>
          </Alert>
        )}
        <div
          className={cn(
            'relative mx-auto w-fit max-w-full overflow-hidden rounded-lg shadow-sm',
            tool === 'text'
              ? 'cursor-text select-text'
              : tool === 'pan'
                ? 'cursor-grab select-none'
                : tool === 'select'
                  ? 'cursor-default'
                  : 'cursor-crosshair'
          )}
          onLostPointerCapture={cancelPointerGesture}
          onMouseUp={captureText}
          onPointerCancel={cancelPointerGesture}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          ref={surfaceRef}
          style={{
            zoom: zoom / 100,
            ...(mediaKind === 'svg'
              ? {
                  aspectRatio: `${svgSize.width} / ${svgSize.height}`,
                  width: `min(${svgSize.width}px, 100%, calc((100vh - 11rem) * ${svgSize.width / svgSize.height}))`
                }
              : {})
          }}
        >
          {mediaKind === 'svg' ? (
            <iframe
              aria-label={label}
              className={cn(
                'block size-full border-0',
                tool === 'text' ? 'pointer-events-auto' : 'pointer-events-none'
              )}
              onLoad={() => setSvgRevision(value => value + 1)}
              ref={svgFrameRef}
              sandbox="allow-same-origin"
              srcDoc={svgDocument}
              title={label}
            />
          ) : (
            <img
              alt={label}
              className="block max-h-[calc(100vh-11rem)] max-w-full select-none object-contain"
              draggable={false}
              onLoad={event =>
                setDimensions({ height: event.currentTarget.naturalHeight, width: event.currentTarget.naturalWidth })
              }
              ref={imageRef}
              src={dataUrl}
            />
          )}
          {mediaKind !== 'svg' && ocrState === 'ready' && (
            <div
              className={cn(
                'absolute inset-0 z-10',
                tool === 'text' ? 'pointer-events-auto select-text' : 'pointer-events-none'
              )}
            >
              {ocrWords.map((word, index) => (
                <span
                  className="absolute overflow-hidden text-transparent selection:bg-(--ui-accent)/35 selection:text-transparent"
                  key={`${word.x}:${word.y}:${index}`}
                  style={{
                    fontSize: `${Math.max(8, (word.height / dimensions.height) * renderedDimensions.height)}px`,
                    height: `${(word.height / dimensions.height) * 100}%`,
                    left: `${(word.x / dimensions.width) * 100}%`,
                    lineHeight: 1,
                    top: `${(word.y / dimensions.height) * 100}%`,
                    whiteSpace: 'nowrap',
                    width: `${(word.width / dimensions.width) * 100}%`
                  }}
                >
                  {word.text}{' '}
                </span>
              ))}
            </div>
          )}
          <svg
            className="pointer-events-none absolute inset-0 z-20 size-full overflow-visible"
            preserveAspectRatio="none"
            viewBox={`0 0 ${renderedDimensions.width} ${renderedDimensions.height}`}
          >
            <defs>
              <marker id="visual-arrowhead" markerHeight="6" markerWidth="6" orient="auto" refX="5" refY="3">
                <path d="M0,0 L0,6 L6,3 z" fill="var(--ui-accent)" />
              </marker>
            </defs>
            {allMarks.map(({ item, mark }, index) => (
              <g
                key={`${item?.id ?? 'draft'}:${mark.id}`}
                onClick={event => {
                  event.stopPropagation()

                  if (item) {
                    editAnnotation(item.id)
                  }

                  setSelectedMark(mark.id)
                  setTool('select')
                }}
              >
                <MarkShape
                  height={renderedDimensions.height}
                  mark={mark}
                  number={index + 1}
                  selected={selectedMark === mark.id}
                  width={renderedDimensions.width}
                />
              </g>
            ))}
          </svg>
          {svgSelection &&
            createPortal(
              <Button
                aria-label={previewCopy.annotateSelection}
                className="fixed z-[99] -translate-x-1/2 rounded-md border border-(--ui-stroke-secondary) bg-(--ui-surface-background) px-2.5 py-1.5 text-xs font-semibold text-(--ui-text-primary) shadow-lg [-webkit-app-region:no-drag] hover:bg-(--ui-surface-secondary)"
                onClick={() => {
                  beginAnnotation(
                    {
                      ...svgSelection.anchor,
                      contentHash,
                      kind: 'svg',
                      path: filePath
                    },
                    svgSelection.editorAnchor,
                    reviewContext
                  )
                  svgFrameRef.current?.contentWindow?.getSelection()?.removeAllRanges()
                  setSvgSelection(null)
                }}
                onMouseDown={event => event.preventDefault()}
                style={{
                  left: Math.max(
                    76,
                    Math.min(svgSelection.editorAnchor.x + svgSelection.editorAnchor.width / 2, window.innerWidth - 76)
                  ),
                  top: Math.max(8, svgSelection.editorAnchor.y - 38)
                }}
                type="button"
                variant="outline"
              >
                {previewCopy.annotateSelection}
              </Button>,
              document.body
            )}
        </div>
      </div>
    </div>
  )
}
