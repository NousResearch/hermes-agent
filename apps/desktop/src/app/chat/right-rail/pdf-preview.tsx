import 'pdfjs-dist/legacy/web/pdf_viewer.css'

import { useStore } from '@nanostores/react'
import type { PDFDocumentLoadingTask, PDFDocumentProxy, PDFPageProxy, RenderTask } from 'pdfjs-dist'
import { type RefObject, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'

import { captureTextPosition, restoreTextPosition } from '@/app/review/annotations/anchors'
import { ANNOTATION_NAVIGATE_EVENT } from '@/app/review/annotations/navigation'
import { PageLoader } from '@/components/page-loader'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import type { HermesPdfDocument } from '@/global'
import { useI18n } from '@/i18n'
import { readPdfDocumentRange } from '@/lib/document-preview'
import { defaultOcrLanguage, recognizeImageText } from '@/lib/ocr-runtime'
import { loadPdfRuntime, pdfWorkerUrl } from '@/lib/pdf-runtime'
import { cn } from '@/lib/utils'
import {
  $annotationDraft,
  $annotationDraftAnchor,
  $annotationEditorCollapsed,
  $annotations,
  type AnnotationEditorAnchor,
  beginAnnotation,
  documentReviewContext,
  type PdfAnnotationAnchor,
  reconcileAnnotationAnchors,
  reopenAnnotationEditor,
  type ReviewContext
} from '@/store/annotations'

interface PendingPdfSelection {
  anchor: PdfAnnotationAnchor
  editorAnchor: AnnotationEditorAnchor
  range: Range
}

function SelectionPopover({ onAnnotate, selection }: { onAnnotate: () => void; selection: PendingPdfSelection }) {
  const { t } = useI18n()
  const [rect, setRect] = useState(() => selection.range.getBoundingClientRect())

  useEffect(() => {
    const update = () => setRect(selection.range.getBoundingClientRect())

    document.addEventListener('scroll', update, true)
    window.addEventListener('resize', update)

    return () => {
      document.removeEventListener('scroll', update, true)
      window.removeEventListener('resize', update)
    }
  }, [selection])

  const left = Math.max(8, Math.min(rect.left + rect.width / 2, window.innerWidth - 148))
  const top = Math.max(8, rect.top - 38)

  return createPortal(
    <Button
      aria-label={t.desktop.annotations.preview.annotateSelection}
      className="fixed z-[99] -translate-x-1/2 rounded-md border border-(--ui-stroke-secondary) bg-(--ui-surface-background) px-2.5 py-1.5 text-xs font-semibold text-(--ui-text-primary) shadow-lg [-webkit-app-region:no-drag] hover:bg-(--ui-surface-secondary)"
      onClick={onAnnotate}
      onMouseDown={event => event.preventDefault()}
      style={{ left, top }}
      type="button"
      variant="outline"
    >
      {t.desktop.annotations.preview.annotateSelection}
    </Button>,
    document.body
  )
}

function expectedCancellation(reason: unknown) {
  return (
    reason instanceof Error &&
    (reason.name === 'RenderingCancelledException' || reason.message === 'Transport destroyed')
  )
}

function PdfPage({
  annotationPath,
  contentHash,
  document,
  documentKind,
  onError,
  pageNumber,
  reviewContext,
  scale
}: {
  annotationPath: string
  contentHash: string
  document: PDFDocumentProxy
  documentKind: 'pdf' | 'tex'
  onError: (reason: unknown) => void
  pageNumber: number
  reviewContext?: ReviewContext
  scale: number
}) {
  const { t } = useI18n()
  const hostRef = useRef<HTMLDivElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const textRef = useRef<HTMLDivElement | null>(null)
  const [visible, setVisible] = useState(pageNumber <= 2)
  const [page, setPage] = useState<PDFPageProxy | null>(null)
  const [canvasRevision, setCanvasRevision] = useState(0)
  const [textLayerRevision, setTextLayerRevision] = useState(0)
  const [ocrStatus, setOcrStatus] = useState<'idle' | 'loading' | 'ready'>('idle')
  const [pendingSelection, setPendingSelection] = useState<PendingPdfSelection | null>(null)
  const annotations = useStore($annotations)
  const draftAnchor = useStore($annotationDraftAnchor)

  const highlightNames = useMemo(() => {
    const id = `${pageNumber}-${Math.random().toString(36).slice(2)}`

    return {
      draft: `hermes-pdf-draft-${id}`,
      pending: `hermes-pdf-pending-${id}`,
      saved: `hermes-pdf-saved-${id}`
    }
  }, [pageNumber])

  const matching = useMemo(
    () =>
      annotations.filter(
        (item): item is typeof item & { anchor: PdfAnnotationAnchor } =>
          item.anchor.kind === 'pdf' && item.anchor.path === annotationPath && item.anchor.page === pageNumber
      ),
    [annotationPath, annotations, pageNumber]
  )

  useEffect(() => {
    const host = hostRef.current

    if (!host || typeof IntersectionObserver === 'undefined') {
      setVisible(true)

      return
    }

    const observer = new IntersectionObserver(
      entries => {
        if (entries.some(entry => entry.isIntersecting)) {
          setVisible(true)
        }
      },
      { rootMargin: '800px 0px' }
    )

    observer.observe(host)

    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    if (!visible) {
      return
    }

    let active = true
    void document
      .getPage(pageNumber)
      .then(next => {
        if (active) {
          setPage(next)
        }
      })
      .catch(reason => {
        if (active && !expectedCancellation(reason)) {
          onError(reason)
        }
      })

    return () => {
      active = false
    }
  }, [document, onError, pageNumber, visible])

  const viewport = useMemo(() => page?.getViewport({ scale }), [page, scale])

  useEffect(() => {
    const canvas = canvasRef.current

    if (!page || !viewport || !canvas) {
      return
    }

    const ratio = Math.min(window.devicePixelRatio || 1, 2)
    canvas.width = Math.floor(viewport.width * ratio)
    canvas.height = Math.floor(viewport.height * ratio)
    canvas.style.width = `${viewport.width}px`
    canvas.style.height = `${viewport.height}px`
    const context = canvas.getContext('2d', { alpha: false })

    if (!context) {
      return
    }

    let task: RenderTask

    try {
      task = page.render({
        canvas,
        canvasContext: context,
        transform: ratio === 1 ? undefined : [ratio, 0, 0, ratio, 0, 0],
        viewport
      })
    } catch (reason) {
      onError(reason)

      return
    }

    void task.promise
      .then(() => setCanvasRevision(value => value + 1))
      .catch(reason => {
        if (!expectedCancellation(reason)) {
          onError(reason)
        }
      })

    return () => {
      task.cancel()
    }
  }, [onError, page, viewport])

  useEffect(() => {
    const container = textRef.current

    if (!page || !viewport || !container) {
      return
    }

    let active = true
    let layer: { cancel: () => void; render: () => Promise<unknown> } | null = null
    setOcrStatus('idle')
    container.replaceChildren()
    void loadPdfRuntime()
      .then(({ TextLayer }) => {
        if (!active) {
          return
        }

        layer = new TextLayer({ container, textContentSource: page.streamTextContent(), viewport })

        return layer.render().then(() => {
          if (active) {
            setTextLayerRevision(value => value + 1)
          }
        })
      })
      .catch(reason => {
        if (active && !expectedCancellation(reason)) {
          onError(reason)
        }
      })

    return () => {
      active = false
      layer?.cancel()
      container.replaceChildren()
    }
  }, [onError, page, viewport])

  useEffect(() => {
    const container = textRef.current
    const canvas = canvasRef.current

    if (
      !container ||
      !canvas ||
      canvasRevision === 0 ||
      textLayerRevision === 0 ||
      ocrStatus !== 'idle' ||
      container.textContent?.trim()
    ) {
      return
    }

    let active = true
    setOcrStatus('loading')
    void recognizeImageText(canvas, `${contentHash}:page:${pageNumber}`, defaultOcrLanguage())
      .then(words => {
        if (!active || container.textContent?.trim()) {
          return
        }

        for (const word of words) {
          const span = globalThis.document.createElement('span')
          span.textContent = word.text
          span.style.left = `${(word.x / canvas.width) * 100}%`
          span.style.top = `${(word.y / canvas.height) * 100}%`
          span.style.width = `${(word.width / canvas.width) * 100}%`
          span.style.height = `${(word.height / canvas.height) * 100}%`
          span.style.fontSize = `${Math.max(8, (word.height / canvas.height) * viewport!.height)}px`
          span.style.lineHeight = '1'
          span.style.position = 'absolute'
          span.style.color = 'transparent'
          span.style.overflow = 'hidden'
          span.style.userSelect = 'text'
          span.style.whiteSpace = 'nowrap'
          span.append(globalThis.document.createTextNode(' '))
          container.append(span)
        }

        setOcrStatus('ready')
        setTextLayerRevision(value => value + 1)
      })
      .catch(() => {
        if (active) {
          setOcrStatus('idle')
        }
      })

    return () => {
      active = false
    }
  }, [canvasRevision, contentHash, ocrStatus, pageNumber, textLayerRevision, viewport])

  useEffect(() => {
    const container = textRef.current

    if (!container || textLayerRevision === 0) {
      return
    }

    const savedRanges = matching
      .map(item => ({ id: item.id, range: restoreTextPosition(container, item.anchor) }))
      .filter((item): item is { id: string; range: Range } => item.range !== null)

    reconcileAnnotationAnchors(new Set(savedRanges.map(item => item.id)), annotationPath)

    const draftRange =
      draftAnchor?.kind === 'pdf' && draftAnchor.path === annotationPath && draftAnchor.page === pageNumber
        ? restoreTextPosition(container, draftAnchor)
        : null

    const registry = (
      CSS as unknown as {
        highlights?: { delete: (name: string) => void; set: (name: string, highlight: unknown) => void }
      }
    ).highlights

    const HighlightConstructor = (globalThis as unknown as { Highlight?: new (...ranges: Range[]) => unknown })
      .Highlight

    if (registry && HighlightConstructor) {
      registry.set(highlightNames.saved, new HighlightConstructor(...savedRanges.map(item => item.range)))

      if (draftRange) {
        registry.set(highlightNames.draft, new HighlightConstructor(draftRange))
      } else {
        registry.delete(highlightNames.draft)
      }

      if (pendingSelection) {
        registry.set(highlightNames.pending, new HighlightConstructor(pendingSelection.range))
      } else {
        registry.delete(highlightNames.pending)
      }

      return () => {
        registry.delete(highlightNames.saved)
        registry.delete(highlightNames.draft)
        registry.delete(highlightNames.pending)
      }
    }
  }, [annotationPath, draftAnchor, highlightNames, matching, pageNumber, pendingSelection, textLayerRevision])

  useEffect(() => {
    const navigate = (event: Event) => {
      const detail = (event as CustomEvent<{ anchor?: PdfAnnotationAnchor }>).detail
      const anchor = detail.anchor
      const container = textRef.current

      if (!container || anchor?.kind !== 'pdf' || anchor.path !== annotationPath || anchor.page !== pageNumber) {
        return
      }

      hostRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' })
      const range = restoreTextPosition(container, anchor)

      if (range) {
        const selection = window.getSelection()
        selection?.removeAllRanges()
        selection?.addRange(range)
      }
    }

    window.addEventListener(ANNOTATION_NAVIGATE_EVENT, navigate)

    return () => window.removeEventListener(ANNOTATION_NAVIGATE_EVENT, navigate)
  }, [annotationPath, pageNumber])

  const captureSelection = () => {
    const container = textRef.current
    const selection = window.getSelection()

    if (!container || !selection || selection.isCollapsed || selection.rangeCount === 0) {
      return
    }

    const range = selection.getRangeAt(0).cloneRange()

    if (!container.contains(range.commonAncestorContainer)) {
      return
    }

    const position = captureTextPosition(container, range)

    if (!position) {
      return
    }

    const rect = range.getBoundingClientRect()
    const boundary = hostRef.current?.closest<HTMLElement>('[data-annotation-surface]')?.getBoundingClientRect()

    const anchor: PdfAnnotationAnchor = {
      ...position,
      contentHash,
      documentKind,
      kind: 'pdf',
      page: pageNumber,
      path: annotationPath
    }

    const editorAnchor: AnnotationEditorAnchor = {
      boundary: boundary
        ? { height: boundary.height, width: boundary.width, x: boundary.left, y: boundary.top }
        : undefined,
      height: rect.height,
      width: rect.width,
      x: rect.left,
      y: rect.top
    }

    setPendingSelection({ anchor, editorAnchor, range })
  }

  useEffect(() => {
    if (!pendingSelection) {
      return
    }

    const dismiss = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setPendingSelection(null)
      }
    }

    window.addEventListener('keydown', dismiss)

    return () => window.removeEventListener('keydown', dismiss)
  }, [pendingSelection])

  return (
    <div
      className="relative mx-auto bg-white shadow-nous"
      data-pdf-page={pageNumber}
      ref={hostRef}
      style={{ minHeight: viewport?.height ?? 900 * scale, width: viewport?.width ?? 640 * scale }}
    >
      <style>{`
        ::highlight(${highlightNames.saved}) { background-color: color-mix(in srgb, var(--ui-accent) 24%, transparent); }
        ::highlight(${highlightNames.draft}) { background-color: color-mix(in srgb, var(--ui-accent) 38%, transparent); }
        ::highlight(${highlightNames.pending}) { background-color: color-mix(in srgb, var(--ui-accent) 48%, transparent); }
      `}</style>
      <canvas
        aria-label={`${t.desktop.annotations.preview.pdf.page} ${pageNumber}`}
        className="block"
        ref={canvasRef}
      />
      <div
        className="textLayer z-10 cursor-text select-text pointer-events-auto"
        data-selectable-text="true"
        onMouseUp={captureSelection}
        ref={textRef}
      />
      <span className="pointer-events-none absolute bottom-1 right-2 text-[0.6rem] text-black/45">{pageNumber}</span>
      {ocrStatus === 'loading' && (
        <span className="pointer-events-none absolute bottom-1 left-2 text-[0.55rem] text-black/45">
          {t.desktop.annotations.preview.ocr.recognizing}
        </span>
      )}
      {pendingSelection && (
        <SelectionPopover
          onAnnotate={() => {
            beginAnnotation(
              pendingSelection.anchor,
              pendingSelection.editorAnchor,
              reviewContext ?? documentReviewContext(annotationPath, contentHash)
            )
            window.getSelection()?.removeAllRanges()
            setPendingSelection(null)
          }}
          selection={pendingSelection}
        />
      )}
    </div>
  )
}

function PdfDocumentAnnotationAction({
  annotationPath,
  contentHash,
  reviewContext,
  surfaceRef
}: {
  annotationPath: string
  contentHash: string
  reviewContext: ReviewContext
  surfaceRef: RefObject<HTMLDivElement | null>
}) {
  const { t } = useI18n()
  const annotationDraft = useStore($annotationDraft)
  const annotationCollapsed = useStore($annotationEditorCollapsed)

  const draftForDocument =
    annotationDraft?.contextId === reviewContext.id && annotationDraft.anchor.path === annotationPath

  return (
    <Button
      className="rounded px-1.5 py-1 font-semibold text-muted-foreground hover:bg-accent hover:text-foreground"
      onClick={event => {
        if (draftForDocument && annotationCollapsed) {
          reopenAnnotationEditor()

          return
        }

        const rect = event.currentTarget.getBoundingClientRect()
        const boundary = surfaceRef.current?.getBoundingClientRect()
        beginAnnotation(
          { contentHash, kind: 'file', path: annotationPath },
          {
            boundary: boundary
              ? { height: boundary.height, width: boundary.width, x: boundary.left, y: boundary.top }
              : undefined,
            height: rect.height,
            width: rect.width,
            x: rect.left,
            y: rect.top
          },
          reviewContext
        )
      }}
      type="button"
      variant="ghost"
    >
      {draftForDocument && annotationCollapsed
        ? t.desktop.annotations.reopen
        : t.desktop.annotations.preview.documentComment}
    </Button>
  )
}

export function PdfPreview({
  annotationPath: annotationPathProp,
  descriptor,
  documentKind = 'pdf',
  label,
  onReload,
  reviewContext
}: {
  annotationPath?: string
  descriptor: HermesPdfDocument
  documentKind?: 'pdf' | 'tex'
  label: string
  onReload?: () => void
  reviewContext?: ReviewContext
}) {
  const { t } = useI18n()
  const annotationPath = annotationPathProp ?? label
  const scrollerRef = useRef<HTMLDivElement | null>(null)
  const [document, setDocument] = useState<PDFDocumentProxy | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [zoom, setZoom] = useState(100)
  const [fitWidth, setFitWidth] = useState(true)
  const [availableWidth, setAvailableWidth] = useState(0)
  const [pageWidth, setPageWidth] = useState(612)
  const [page, setPage] = useState(1)
  const [query, setQuery] = useState('')
  const [matches, setMatches] = useState<number[]>([])
  const [password, setPassword] = useState<null | { incorrect: boolean; submit: (value: string) => void }>(null)
  const recoveryRef = useRef<string | null>(null)
  const onReloadRef = useRef(onReload)
  onReloadRef.current = onReload
  const activeReviewContext = reviewContext ?? documentReviewContext(annotationPath, descriptor.revision)
  const reviewHash = activeReviewContext.contentHash ?? descriptor.revision

  const fail = useCallback(
    (reason: unknown) => {
      if (expectedCancellation(reason)) {
        return
      }

      const message = reason instanceof Error ? reason.message : String(reason)

      if (
        onReloadRef.current &&
        recoveryRef.current !== descriptor.id &&
        /closed or unavailable|PDF_CHANGED/i.test(message)
      ) {
        recoveryRef.current = descriptor.id
        onReloadRef.current()

        return
      }

      setError(message)
      setDocument(null)
      setLoading(false)
    },
    [descriptor.id]
  )

  useEffect(() => {
    let active = true
    let task: PDFDocumentLoadingTask | null = null
    setLoading(true)
    setError(null)
    setDocument(null)
    void loadPdfRuntime()
      .then(({ getDocument, GlobalWorkerOptions, PDFDataRangeTransport }) => {
        if (!active) {
          return
        }

        GlobalWorkerOptions.workerSrc = pdfWorkerUrl

        class HermesPdfRangeTransport extends PDFDataRangeTransport {
          override requestDataRange(begin: number, end: number) {
            void readPdfDocumentRange(descriptor, begin, end)
              .then(data => {
                if (active) {
                  this.onDataRange(begin, data)
                }
              })
              .catch(reason => {
                if (active) {
                  fail(reason)
                }
              })
          }

          override abort() {
            // The parent owns the opaque descriptor. PDF.js can abort this
            // transport without invalidating the handle reused by StrictMode.
          }
        }

        const range = new HermesPdfRangeTransport(descriptor.byteLength, descriptor.initialData)
        task = getDocument({ length: descriptor.byteLength, range, rangeChunkSize: 64 * 1024 })

        task.onPassword = (updatePassword: (password: string) => void, reason: number) => {
          if (active) {
            setPassword({ incorrect: reason === 2, submit: updatePassword })
          }
        }

        return task.promise.then(async next => {
          if (!active) {
            await next.destroy()

            return
          }

          const first = await next.getPage(1)
          setPageWidth(first.getViewport({ scale: 1 }).width)
          setDocument(next)
          setLoading(false)
          setPassword(null)
        })
      })
      .catch(reason => {
        if (active) {
          fail(reason)
        }
      })

    return () => {
      active = false

      if (task) {
        void task.destroy().catch(() => undefined)
      }
    }
  }, [descriptor, fail])

  useLayoutEffect(() => {
    const scroller = scrollerRef.current

    if (!scroller) {
      return
    }

    const commitWidth = (width: number) => {
      const next = Math.round(width)
      setAvailableWidth(current => (current === next ? current : next))
    }

    commitWidth(scroller.getBoundingClientRect().width || scroller.clientWidth || 800)

    if (typeof ResizeObserver === 'undefined') {
      return
    }

    const observer = new ResizeObserver(([entry]) => commitWidth(entry.contentRect.width))
    observer.observe(scroller)

    return () => observer.disconnect()
  }, [document])

  const scale = fitWidth ? Math.max(0.25, (availableWidth - 40) / pageWidth) : zoom / 100

  const pages = useMemo(
    () => (document ? Array.from({ length: document.numPages }, (_, index) => index + 1) : []),
    [document]
  )

  const navigate = (next: number) => {
    if (!document) {
      return
    }

    const clamped = Math.max(1, Math.min(document.numPages, next))
    setPage(clamped)
    scrollerRef.current
      ?.querySelector(`[data-pdf-page="${clamped}"]`)
      ?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }

  const search = async () => {
    if (!document || !query.trim()) {
      setMatches([])

      return
    }

    const needle = query.trim().toLocaleLowerCase()
    const found: number[] = []

    for (let number = 1; number <= document.numPages; number += 1) {
      const content = await (await document.getPage(number)).getTextContent()

      const text = content.items
        .map(item => ('str' in item ? item.str : ''))
        .join(' ')
        .toLocaleLowerCase()

      if (text.includes(needle)) {
        found.push(number)
      }
    }

    setMatches(found)

    if (found[0]) {
      navigate(found[0])
    }
  }

  if (password) {
    return (
      <form
        className="grid h-full place-items-center bg-background/75 p-6"
        onSubmit={event => {
          event.preventDefault()
          const value = new FormData(event.currentTarget).get('password')
          password.submit(String(value || ''))
        }}
      >
        <div className="w-full max-w-xs rounded-lg border border-border bg-background p-4 shadow-nous">
          <div className="text-sm font-semibold">{t.desktop.annotations.preview.pdf.passwordProtected}</div>
          {password.incorrect && (
            <div className="mt-1 text-xs text-destructive">{t.desktop.annotations.preview.pdf.incorrectPassword}</div>
          )}
          <Input
            autoFocus
            className="mt-3 h-8 w-full rounded border border-border bg-transparent px-2 text-sm"
            name="password"
            type="password"
          />
          <Button className="mt-3" size="sm" type="submit">
            {t.desktop.annotations.preview.pdf.open}
          </Button>
        </div>
      </form>
    )
  }

  if (loading) {
    return <PageLoader label={`${t.preview.loading}: ${label}`} />
  }

  if (error || !document) {
    return (
      <div className="grid h-full place-items-center p-6 text-center text-xs text-destructive">
        {error || t.desktop.annotations.preview.pdf.couldNotOpen}
      </div>
    )
  }

  return (
    <div className="flex h-full min-h-0 flex-col bg-transparent">
      <div className="flex h-8 shrink-0 items-center gap-1 border-b border-border/50 px-2 text-[0.65rem]">
        <button
          className="rounded px-1.5 py-1 hover:bg-accent disabled:opacity-35"
          disabled={page <= 1}
          onClick={() => navigate(page - 1)}
          type="button"
        >
          ‹
        </button>
        <Input
          aria-label={t.desktop.annotations.preview.pdf.page}
          className="h-5 w-10 rounded border border-border bg-transparent px-1 text-center tabular-nums"
          min={1}
          onChange={event => setPage(Number(event.target.value) || 1)}
          onKeyDown={event => {
            if (event.key === 'Enter') {
              navigate(page)
            }
          }}
          type="number"
          value={page}
        />
        <span className="text-muted-foreground">/ {document.numPages}</span>
        <button
          className="rounded px-1.5 py-1 hover:bg-accent disabled:opacity-35"
          disabled={page >= document.numPages}
          onClick={() => navigate(page + 1)}
          type="button"
        >
          ›
        </button>
        <div className="mx-1 h-4 w-px bg-border" />
        <button
          className="rounded px-1.5 py-1 hover:bg-accent"
          onClick={() => {
            setFitWidth(false)
            setZoom(value => Math.max(50, value - 10))
          }}
          type="button"
        >
          −
        </button>
        <button
          className={cn('rounded px-1.5 py-1 hover:bg-accent', fitWidth && 'bg-accent')}
          onClick={() => setFitWidth(true)}
          type="button"
        >
          {t.desktop.annotations.preview.pdf.fit}
        </button>
        <button
          className="rounded px-1.5 py-1 hover:bg-accent"
          onClick={() => {
            setFitWidth(false)
            setZoom(value => Math.min(250, value + 10))
          }}
          type="button"
        >
          +
        </button>
        <span className="min-w-9 text-muted-foreground">
          {fitWidth ? t.desktop.annotations.preview.pdf.width : `${zoom}%`}
        </span>
        <div className="ml-auto flex items-center gap-1">
          <PdfDocumentAnnotationAction
            annotationPath={annotationPath}
            contentHash={reviewHash}
            reviewContext={activeReviewContext}
            surfaceRef={scrollerRef}
          />
          <input
            aria-label={t.desktop.annotations.preview.pdf.search}
            className="h-5 w-32 rounded border border-border bg-transparent px-1.5"
            onChange={event => setQuery(event.target.value)}
            onKeyDown={event => {
              if (event.key === 'Enter') {
                void search()
              }
            }}
            placeholder={t.desktop.annotations.preview.pdf.search}
            value={query}
          />
          <button className="rounded px-1.5 py-1 hover:bg-accent" onClick={() => void search()} type="button">
            {t.desktop.annotations.preview.pdf.find}
          </button>
          {matches.length > 0 && <span className="text-muted-foreground">{matches.length}</span>}
        </div>
      </div>
      <div
        className="min-h-0 flex-1 overflow-auto bg-black/8 p-5 dark:bg-black/25"
        data-annotation-surface
        ref={scrollerRef}
        style={{ scrollbarGutter: 'stable' }}
      >
        <div className="grid gap-5">
          {availableWidth > 0 &&
            pages.map(number => (
              <PdfPage
                annotationPath={annotationPath}
                contentHash={reviewHash}
                document={document}
                documentKind={documentKind}
                key={number}
                onError={fail}
                pageNumber={number}
                reviewContext={reviewContext}
                scale={scale}
              />
            ))}
        </div>
      </div>
    </div>
  )
}
