'use client'

import { TextMessagePartProvider, useMessagePartText } from '@assistant-ui/react'
import {
  type StreamdownTextComponents,
  StreamdownTextPrimitive,
  type SyntaxHighlighterProps
} from '@assistant-ui/react-streamdown'
import { code } from '@streamdown/code'
import { type ComponentProps, memo, type ReactNode, useDeferredValue, useEffect, useMemo, useRef, useState } from 'react'

import { PreviewAttachment } from '@/components/chat/preview-attachment'
import { SyntaxHighlighter } from '@/components/chat/shiki-highlighter'
import { ZoomableImage } from '@/components/chat/zoomable-image'
import { normalizeExternalUrl, openExternalLink, PrettyLink } from '@/lib/external-link'
import { createMemoizedMathPlugin } from '@/lib/katex-memo'
import { preprocessMarkdown } from '@/lib/markdown-preprocess'
import {
  filePathFromMediaPath,
  gatewayMediaDataUrl,
  isRemoteGateway,
  mediaExternalUrl,
  mediaKind,
  mediaName,
  mediaPathFromMarkdownHref,
  mediaStreamUrl
} from '@/lib/media'
import { previewTargetFromMarkdownHref } from '@/lib/preview-targets'
import { cn } from '@/lib/utils'

// Math rendering plugin (KaTeX). Configured once at module scope — the
// plugin is stateless beyond its internal cache so re-creating per-render
// would needlessly thrash. We use a memoizing wrapper around rehype-katex
// (see lib/katex-memo.ts) so that during streaming we re-katex only the
// equations whose source actually changed since the last token. With the
// stock @streamdown/math plugin every equation re-renders on every token,
// which throttles UI updates badly for math-heavy responses; the memoized
// plugin keeps the steady-state work proportional to "new equations
// arriving" rather than "equations × tokens-per-second".
//
// `singleDollarTextMath: true` enables `$x^2$` for inline math (de-facto
// LLM convention). The default false-setting only accepts `$$...$$`.
const mathPlugin = createMemoizedMathPlugin({ singleDollarTextMath: true })

async function mediaSrc(path: string): Promise<string> {
  if (/^(?:https?|data):/i.test(path)) {
    return path
  }

  // Stream audio/video through the custom protocol: data URLs are capped and
  // load the whole file into memory, which broke playback for larger videos.
  if (window.hermesDesktop && ['audio', 'video'].includes(mediaKind(path))) {
    return mediaStreamUrl(path)
  }

  // Remote gateway: the image lives on the gateway machine, so read it over the
  // authenticated API rather than this machine's disk.
  if (window.hermesDesktop && isRemoteGateway()) {
    return gatewayMediaDataUrl(path)
  }

  if (!window.hermesDesktop?.readFileDataUrl) {
    return mediaExternalUrl(path)
  }

  return window.hermesDesktop.readFileDataUrl(filePathFromMediaPath(path))
}

function OpenMediaButton({ kind, path }: { kind: 'audio' | 'video'; path: string }) {
  return (
    <button
      className="mt-2 bg-transparent text-xs font-medium text-muted-foreground underline underline-offset-4 decoration-current/20 hover:text-foreground"
      onClick={() => void window.hermesDesktop?.openExternal(mediaExternalUrl(path))}
      type="button"
    >
      Open {kind} file
    </button>
  )
}

function MediaAttachment({ path }: { path: string }) {
  const [src, setSrc] = useState('')
  const [failed, setFailed] = useState(false)
  const kind = mediaKind(path)
  const name = mediaName(path)

  useEffect(() => {
    let cancelled = false
    let objectUrl = ''

    setFailed(false)
    setSrc('')
    void mediaSrc(path)
      .then(value => {
        if (value.startsWith('blob:')) {
          objectUrl = value
        }

        if (!cancelled) {
          setSrc(value)
        } else if (objectUrl) {
          URL.revokeObjectURL(objectUrl)
        }
      })
      .catch(() => {
        if (!cancelled) {
          setFailed(true)
        }
      })

    return () => {
      cancelled = true

      if (objectUrl) {
        URL.revokeObjectURL(objectUrl)
      }
    }
  }, [path])

  if (kind === 'image' && src) {
    return (
      <span className="block">
        <MarkdownImage alt={name} src={src} />
      </span>
    )
  }

  if (kind === 'audio' && src) {
    return (
      <span className="my-3 block max-w-md rounded-xl border border-border bg-muted/35 p-3">
        <span className="mb-2 block truncate text-xs font-medium text-muted-foreground">{name}</span>
        <audio className="block w-full" controls onError={() => setFailed(true)} preload="metadata" src={src} />
        {failed && <OpenMediaButton kind="audio" path={path} />}
      </span>
    )
  }

  if (kind === 'video' && src) {
    return (
      <span className="my-3 block max-w-2xl rounded-xl border border-border bg-muted/35 p-3">
        <span className="mb-2 block truncate text-xs font-medium text-muted-foreground">{name}</span>
        <video
          className="block max-h-112 w-full rounded-lg bg-black"
          controls
          onError={() => setFailed(true)}
          src={src}
        />
        {failed && <OpenMediaButton kind="video" path={path} />}
      </span>
    )
  }

  return (
    <a
      className="font-semibold text-foreground underline underline-offset-4 decoration-current/20 wrap-anywhere"
      href="#"
      onClick={event => {
        event.preventDefault()
        openExternalLink(mediaExternalUrl(path))
      }}
    >
      {failed ? `Open ${name}` : `Loading ${name}...`}
    </a>
  )
}

function childrenToText(children: unknown): string {
  if (typeof children === 'string' || typeof children === 'number') {
    return String(children).trim()
  }

  if (Array.isArray(children) && children.every(c => typeof c === 'string' || typeof c === 'number')) {
    return children.join('').trim()
  }

  return ''
}

function MarkdownLink({ children, className, href, ...props }: ComponentProps<'a'>) {
  const mediaPath = mediaPathFromMarkdownHref(href)

  if (mediaPath) {
    return <MediaAttachment path={mediaPath} />
  }

  const previewTarget = previewTargetFromMarkdownHref(href)

  if (previewTarget) {
    return <PreviewAttachment source="explicit-link" target={previewTarget} />
  }

  const target = href ? normalizeExternalUrl(href) : href

  if (!target || !/^https?:\/\//i.test(target)) {
    return (
      <a
        className={cn(
          'font-semibold text-foreground underline underline-offset-4 decoration-current/20 wrap-anywhere',
          className
        )}
        href={href}
        rel="noopener noreferrer"
        target="_blank"
        {...props}
      >
        {children}
      </a>
    )
  }

  const text = childrenToText(children)
  const fallbackLabel = text && normalizeExternalUrl(text) !== target ? text : undefined

  return (
    <PrettyLink className={cn('wrap-anywhere', className)} fallbackLabel={fallbackLabel} href={target} {...props} />
  )
}

function MarkdownImage({ className, src, alt, ...props }: ComponentProps<'img'>) {
  return (
    <ZoomableImage
      alt={alt}
      className={cn(
        'm-0 block h-auto w-auto max-h-(--image-preview-height) max-w-[min(100%,var(--image-preview-max-width))] rounded-lg object-contain shadow-[0_0.0625rem_0.125rem_color-mix(in_srgb,#000_4%,transparent),0_0.625rem_1.5rem_color-mix(in_srgb,#000_5%,transparent)]',
        className
      )}
      containerClassName="my-2 block w-fit max-w-full"
      slot="aui_markdown-image"
      src={src}
      {...props}
    />
  )
}

// Steady character-reveal for streaming text: decouples visible cadence from
// bursty arrival so text flows instead of popping (cf. assistant-ui's useSmooth,
// reimplemented for a tunable rate). Proportional drain — each frame reveals a
// slice of the backlog so the reveal converges within ~REVEAL_DRAIN_MS whatever
// the size; the per-frame cap stops a huge dump rendering as one slab. The loop
// is gated on backlog, not isRunning, so a stream that completes mid-reveal
// keeps draining its tail instead of snapping.
const REVEAL_DRAIN_MS = 500
const REVEAL_MAX_CHARS_PER_FRAME = 30

function useSmoothReveal(text: string, isRunning: boolean): string {
  const [displayed, setDisplayed] = useState(isRunning ? '' : text)
  const targetRef = useRef(text)
  const shownRef = useRef(displayed)
  const frameRef = useRef<number | null>(null)
  const lastTickRef = useRef(0)

  shownRef.current = displayed
  targetRef.current = text

  useEffect(() => {
    if (typeof window === 'undefined') {
      return
    }

    // Non-extending change (regenerate / branch / history swap): restart from
    // empty while streaming, else snap to the replacement.
    if (!text.startsWith(shownRef.current)) {
      shownRef.current = isRunning ? '' : text
      setDisplayed(shownRef.current)
    }

    if (shownRef.current.length >= text.length || frameRef.current !== null) {
      return
    }

    lastTickRef.current = performance.now()

    const tick = () => {
      const now = performance.now()
      const dt = now - lastTickRef.current
      lastTickRef.current = now

      const remaining = targetRef.current.length - shownRef.current.length
      const add = Math.min(remaining, REVEAL_MAX_CHARS_PER_FRAME, Math.max(1, Math.ceil((remaining * dt) / REVEAL_DRAIN_MS)))
      shownRef.current = targetRef.current.slice(0, shownRef.current.length + add)
      setDisplayed(shownRef.current)

      frameRef.current = shownRef.current.length < targetRef.current.length ? requestAnimationFrame(tick) : null
    }

    frameRef.current = requestAnimationFrame(tick)
  }, [text, isRunning])

  useEffect(
    () => () => {
      if (frameRef.current !== null && typeof window !== 'undefined') {
        cancelAnimationFrame(frameRef.current)
      }
    },
    []
  )

  return displayed
}

// Re-publish the part context with a smooth character-reveal, above
// DeferStreamingText so the reveal feeds the deferred markdown pipeline. Status
// stays running while revealing so the caret persists past the underlying part
// settling.
function SmoothStreamingText({ children }: { children: ReactNode }) {
  const { text, status } = useMessagePartText()
  const isRunning = status.type === 'running'
  const revealed = useSmoothReveal(text, isRunning)

  return (
    <TextMessagePartProvider isRunning={isRunning || revealed !== text} text={revealed}>
      {children}
    </TextMessagePartProvider>
  )
}

/**
 * Re-publish the active message-part context with React's `useDeferredValue`
 * applied to the streaming text and status. The outer wrapper still re-renders
 * on every token, but the work it does is trivial (one hook, one provider).
 *
 * The expensive subtree (Streamdown → micromark → mdast → hast → React) lives
 * inside `<TextMessagePartProvider>` and reads the deferred text via the
 * normal `useMessagePartText` hook. React's concurrent scheduler then has
 * permission to:
 *   - skip intermediate token states when the next token arrives mid-render
 *     (it abandons the in-flight deferred render and starts over)
 *   - deprioritize the markdown render when the main thread is busy with an
 *     urgent task (typing, scrolling, layout work elsewhere)
 *
 * Net effect: per-token CPU is unchanged but the *blocking* part of that work
 * goes away — typing-while-streaming stays a single-frame paint, scroll
 * stutter disappears, and the longtask histogram tightens because long
 * commits can be interrupted and discarded.
 *
 * Industry standard (Streamdown's own block-array setState already uses
 * `useTransition`); this just lifts the deferral up to the consumer text
 * boundary so it covers the whole pipeline, not just the inner setState.
 */
function DeferStreamingText({ children }: { children: ReactNode }) {
  const { text, status } = useMessagePartText()
  const deferredText = useDeferredValue(text)
  const isRunning = status.type === 'running'

  return (
    <TextMessagePartProvider isRunning={isRunning} text={deferredText}>
      {children}
    </TextMessagePartProvider>
  )
}

interface MarkdownTextSurfaceProps {
  containerClassName?: string
  containerProps?: ComponentProps<'div'>
}

// Headings shrink to chat scale rather than the prose default (h1≈xl). Kept
// table-driven so adding/tweaking levels is one row.
const HEADING_SIZES: Record<'h1' | 'h2' | 'h3' | 'h4', string> = {
  h1: 'text-[1rem] tracking-tight',
  h2: 'text-[0.9375rem] tracking-tight',
  h3: 'text-[0.875rem]',
  h4: 'text-[0.8125rem]'
}

const MARKDOWN_CONTAINER_CLASS_NAME = cn(
  'aui-md prose w-full max-w-none overflow-hidden text-[length:var(--conversation-text-font-size)] leading-(--dt-line-height) text-foreground',
  'prose-p:leading-(--dt-line-height) prose-li:leading-(--dt-line-height)',
  'prose-headings:text-foreground prose-strong:text-foreground',
  'prose-a:break-words prose-p:[overflow-wrap:anywhere]',
  'prose-li:marker:text-muted-foreground/70',
  'prose-code:rounded-[0.25rem] prose-code:px-[0.1875rem] prose-code:py-px prose-code:font-mono prose-code:text-[0.9em] prose-code:font-normal prose-code:before:content-none prose-code:after:content-none',
  '[&>*:first-child]:mt-0 [&>*:last-child]:mb-0 [&>*+*]:mt-(--paragraph-gap)'
)

function MarkdownTextSurface({ containerClassName, containerProps }: MarkdownTextSurfaceProps) {
  const { status } = useMessagePartText()
  const isStreaming = status.type === 'running'

  // Keep code parsing enabled while streaming so incomplete fenced blocks still
  // render as code cards. The expensive Shiki pass is deferred by
  // `SyntaxHighlighter` below when `isStreaming` is true.
  const plugins = useMemo(() => ({ math: mathPlugin, code }), [])

  const components = useMemo(
    () =>
      ({
        h1: ({ className, ...props }: ComponentProps<'h1'>) => (
          <h1 className={cn('my-1 font-semibold', HEADING_SIZES.h1, className)} {...props} />
        ),
        h2: ({ className, ...props }: ComponentProps<'h2'>) => (
          <h2 className={cn('my-1 font-semibold', HEADING_SIZES.h2, className)} {...props} />
        ),
        h3: ({ className, ...props }: ComponentProps<'h3'>) => (
          <h3 className={cn('my-1 font-semibold', HEADING_SIZES.h3, className)} {...props} />
        ),
        h4: ({ className, ...props }: ComponentProps<'h4'>) => (
          <h4 className={cn('my-1 font-semibold', HEADING_SIZES.h4, className)} {...props} />
        ),
        p: ({ className, ...props }: ComponentProps<'p'>) => (
          // Vertical rhythm is owned by styles.css (`--paragraph-gap`), which
          // must out-specify Tailwind Typography's `prose` margins — so no
          // `my-*` here on purpose.
          <p className={cn('wrap-anywhere leading-(--dt-line-height)', className)} {...props} />
        ),
        a: MarkdownLink,
        // `---` as quiet spacing, not a heavy full-width rule.
        hr: (_props: ComponentProps<'hr'>) => <div aria-hidden className="my-3" />,
        blockquote: ({ className, ...props }: ComponentProps<'blockquote'>) => (
          <blockquote
            className={cn('border-l-2 border-border pl-3 text-muted-foreground italic', className)}
            {...props}
          />
        ),
        ul: ({ className, ...props }: ComponentProps<'ul'>) => (
          <ul className={cn('my-1 gap-0', className)} {...props} />
        ),
        ol: ({ className, ...props }: ComponentProps<'ol'>) => (
          <ol className={cn('my-1 gap-0', className)} {...props} />
        ),
        li: ({ className, ...props }: ComponentProps<'li'>) => (
          <li className={cn('leading-(--dt-line-height)', className)} {...props} />
        ),
        table: ({ className, ...props }: ComponentProps<'table'>) => (
          <div className="aui-md-table my-2 max-w-full overflow-x-auto rounded-[0.375rem] border border-border">
            <table
              className={cn(
                'm-0 w-full min-w-[18rem] border-collapse text-[0.8125rem] [&_tr]:border-b [&_tr]:border-border last:[&_tr]:border-0',
                className
              )}
              {...props}
            />
          </div>
        ),
        thead: ({ className, ...props }: ComponentProps<'thead'>) => (
          <thead className={cn('m-0 bg-muted/35 text-muted-foreground', className)} {...props} />
        ),
        th: ({ className, ...props }: ComponentProps<'th'>) => (
          <th
            className={cn(
              'whitespace-nowrap px-2.5 py-1.5 text-left align-middle text-[0.75rem] font-medium text-muted-foreground',
              className
            )}
            {...props}
          />
        ),
        td: ({ className, ...props }: ComponentProps<'td'>) => (
          <td className={cn('px-2.5 py-1.5 align-top text-[0.8125rem] leading-snug', className)} {...props} />
        ),
        img: MarkdownImage,
        SyntaxHighlighter: (props: SyntaxHighlighterProps) => <SyntaxHighlighter {...props} defer={isStreaming} />
      }) as StreamdownTextComponents,
    [isStreaming]
  )

  return (
    <StreamdownTextPrimitive
      components={components}
      containerClassName={cn(MARKDOWN_CONTAINER_CLASS_NAME, containerClassName)}
      containerProps={containerProps}
      lineNumbers={false}
      mode="streaming"
      // Always auto-close incomplete fences — even during streaming.
      // Without this, an unclosed ```python ... ``` whose body contains
      // `$` (very common: shell snippets, JS template strings, dollar
      // amounts) leaks those dollars out to the math parser and they
      // get rendered as broken inline math until the closing fence
      // arrives. Shiki is independently deferred via `defer={isStreaming}`
      // on the SyntaxHighlighter component, so we don't pay code-block
      // tokenization on every token even with this set.
      parseIncompleteMarkdown
      plugins={plugins}
      preprocess={preprocessMarkdown}
    />
  )
}

interface MarkdownTextContentProps extends MarkdownTextSurfaceProps {
  isRunning: boolean
  text: string
}

export function MarkdownTextContent({ isRunning, text, ...surfaceProps }: MarkdownTextContentProps) {
  return (
    <TextMessagePartProvider isRunning={isRunning} text={text}>
      <SmoothStreamingText>
        <DeferStreamingText>
          <MarkdownTextSurface {...surfaceProps} />
        </DeferStreamingText>
      </SmoothStreamingText>
    </TextMessagePartProvider>
  )
}

const MarkdownTextImpl = () => {
  return (
    <SmoothStreamingText>
      <DeferStreamingText>
        <MarkdownTextSurface />
      </DeferStreamingText>
    </SmoothStreamingText>
  )
}

export const MarkdownText = memo(MarkdownTextImpl)
