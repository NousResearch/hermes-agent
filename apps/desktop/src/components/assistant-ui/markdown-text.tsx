'use client'

import { TextMessagePartProvider, useMessagePartText } from '@assistant-ui/react'
import {
  parseMarkdownIntoBlocks,
  type StreamdownTextComponents,
  StreamdownTextPrimitive,
  type SyntaxHighlighterProps
} from '@assistant-ui/react-streamdown'
import { code } from '@streamdown/code'
import {
  Children,
  type ComponentProps,
  type FC,
  isValidElement,
  memo,
  type ReactNode,
  useDeferredValue,
  useEffect,
  useMemo,
  useRef,
  useState
} from 'react'

import { ExpandableBlock } from '@/components/chat/expandable-block'
import { PreviewAttachment } from '@/components/chat/preview-attachment'
import { chunkByLines, SyntaxHighlighter } from '@/components/chat/shiki-highlighter'
import { ZoomableImage } from '@/components/chat/zoomable-image'
import { MediaCarousel } from '@/components/chat/media-carousel'
import { normalizeExternalUrl, openExternalLink, PrettyLink } from '@/lib/external-link'
import { createMemoizedMathPlugin } from '@/lib/katex-memo'
import { preprocessMarkdown } from '@/lib/markdown-preprocess'
import {
  downloadGatewayMediaFile,
  filePathFromMediaPath,
  galleryPayloadFromHref,
  gatewayMediaDataUrl,
  isRemoteGateway,
  mediaExternalUrl,
  mediaKind,
  mediaName,
  mediaPathFromMarkdownHref,
  mediaStreamUrl
} from '@/lib/media'
import { previewTargetFromMarkdownHref } from '@/lib/preview-targets'
import { tailBoundedRemend } from '@/lib/remend-tail'
import { cn } from '@/lib/utils'

import { detectEmbed, extractAlert, MarkdownAlert, RichCodeBlock, UrlEmbed } from './embeds'

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

// Replaces Streamdown's `parseIncompleteMarkdown` (full-text remend per
// flush) with a tail-bounded repair — see lib/remend-tail.ts. Must stay
// module-scope so the prop identity is stable across renders.
function preprocessWithTailRepair(text: string): string {
  try {
    return tailBoundedRemend(preprocessMarkdown(text))
  } catch {
    return text
  }
}

// Memoized block splitter. Streamdown calls `parseMarkdownIntoBlocks` (a full
// `marked` lex of the entire message, ~1.6ms per 28KB) inside a useMemo keyed
// on the text — but the same text is re-lexed every time a message REMOUNTS
// (virtualizer scroll, session switch) and whenever multiple surfaces render
// the same content (deferred + smooth reveal republish). A small module-level
// LRU keyed by the exact source string removes all of those repeat parses
// with zero correctness risk (same input → same output). Streaming tail
// growth misses the cache by design (every flush is a new string) — that
// single lex is the irreducible cost.
const BLOCK_CACHE_MAX = 64
const BLOCK_CACHE_MIN_LENGTH = 1024
const blockCache = new Map<string, string[]>()

function parseMarkdownIntoBlocksCached(markdown: string): string[] {
  if (markdown.length < BLOCK_CACHE_MIN_LENGTH) {
    return parseMarkdownIntoBlocks(markdown)
  }

  const hit = blockCache.get(markdown)

  if (hit) {
    // Refresh recency (Map iteration order is insertion order).
    blockCache.delete(markdown)
    blockCache.set(markdown, hit)

    return hit
  }

  const blocks = parseMarkdownIntoBlocks(markdown)
  blockCache.set(markdown, blocks)

  if (blockCache.size > BLOCK_CACHE_MAX) {
    blockCache.delete(blockCache.keys().next().value as string)
  }

  return blocks
}

// mediaSrc now lives in @/lib/media to avoid a circular import between this
// module and the carousel that consumes it. Imported here (not just
// re-exported) so local callers like MediaAttachment keep working unchanged.
import { mediaSrc } from '@/lib/media'
export { mediaSrc }

function useOpenMediaFile(path: string) {
  const [openFailed, setOpenFailed] = useState(false)

  const open = () => {
    if (window.hermesDesktop && isRemoteGateway()) {
      setOpenFailed(false)
      void downloadGatewayMediaFile(path).catch(() => setOpenFailed(true))
    } else {
      openExternalLink(mediaExternalUrl(path))
    }
  }

  return { open, openFailed }
}

function OpenMediaFailedNote({ name }: { name: string }) {
  return (
    <span className="mt-1 block text-xs text-muted-foreground">
      Couldn&apos;t fetch {name} from the gateway (missing, unreadable, or too large).
    </span>
  )
}

function OpenMediaButton({ kind, path }: { kind: 'audio' | 'video'; path: string }) {
  const { open, openFailed } = useOpenMediaFile(path)

  return (
    <span className="block">
      <button
        className="mt-2 bg-transparent text-xs font-medium text-muted-foreground underline underline-offset-4 decoration-current/20 hover:text-foreground"
        onClick={open}
        type="button"
      >
        Open {kind} file
      </button>
      {openFailed && <OpenMediaFailedNote name={mediaName(path)} />}
    </span>
  )
}

function MediaAttachment({ path }: { path: string }) {
  const [src, setSrc] = useState('')
  const [failed, setFailed] = useState(false)
  const { open, openFailed } = useOpenMediaFile(path)
  const kind = mediaKind(path)
  const name = mediaName(path)

  useEffect(() => {
    let cancelled = false
    let objectUrl = ''

    setFailed(false)
    setSrc('')

    if (kind === 'file') {
      setFailed(true)

      return () => {
        cancelled = true
      }
    }

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
  }, [kind, path])

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
    <span className="wrap-anywhere">
      <a
        className="font-semibold text-foreground underline underline-offset-4 decoration-current/20 wrap-anywhere"
        href="#"
        onClick={event => {
          event.preventDefault()
          open()
        }}
      >
        {failed ? `Open ${name}` : `Loading ${name}...`}
      </a>
      {openFailed && <OpenMediaFailedNote name={name} />}
    </span>
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

// A `#gallery:` link renders <MediaCarousel> — a <div>. Streamdown wraps link
// text in a <p>, and a <div> cannot be a child of <p> (React drops/errors on
// the nesting, so the carousel never paints). Walk the subtree and detect any
// carousel so the paragraph override can escape the <p> for that case only.
function subtreeContainsGallery(node: ReactNode): boolean {
  if (!node) return false
  if (Array.isArray(node)) return node.some(subtreeContainsGallery)
  if (typeof node === 'string' || typeof node === 'number' || typeof node === 'boolean') return false
  if (!isValidElement(node)) return false

  const slot = (node.props as { 'data-slot'?: string })?.[`data-slot`]
  if (slot === 'aui_media-carousel') return true
  // Also catch the raw link element whose href is a gallery href.
  const href = (node.props as { href?: string } | undefined)?.href
  if (typeof href === 'string' && href.startsWith('#gallery:')) return true

  return subtreeContainsGallery((node.props as { children?: ReactNode }).children)
}

function MarkdownLink({ children, className, href, ...props }: ComponentProps<'a'>) {
  const galleryPayload = galleryPayloadFromHref(href)

  if (galleryPayload) {
    return (
      <MediaCarousel
        images={galleryPayload.images}
        intervalMs={galleryPayload.intervalMs}
        title={galleryPayload.title}
      />
    )
  }

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

  // Bare autolink → inline rich embed when a provider matches. Labeled links
  // (`[watch](url)`) stay plain. Desktop only (webview / iframe renderers).
  if (window.hermesDesktop && text && normalizeExternalUrl(text) === target) {
    const embed = detectEmbed(target)

    if (embed) {
      return <UrlEmbed descriptor={embed} />
    }
  }

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
// Floor between reveal commits. Each commit republishes the text context and
// re-runs the whole Streamdown pipeline (preprocess → remend → lex → micromark
// on the open block) over the full accumulated text — at raw rAF cadence
// that's 60 full parses/second and was the dominant streaming cost for
// reasoning text. ~33ms keeps the reveal visually fluid (2 frames) while
// halving the parse work.
const REVEAL_MIN_COMMIT_MS = 33

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

      // Skip this frame if the floor hasn't elapsed — the backlog math below
      // is dt-proportional, so delayed commits reveal proportionally more.
      if (dt < REVEAL_MIN_COMMIT_MS) {
        frameRef.current = requestAnimationFrame(tick)

        return
      }

      lastTickRef.current = now

      const remaining = targetRef.current.length - shownRef.current.length

      const add = Math.min(
        remaining,
        // dt-scaled so the per-commit cap stays equivalent to the old
        // per-frame cap at any commit cadence.
        Math.ceil((REVEAL_MAX_CHARS_PER_FRAME * dt) / 16.7),
        Math.max(1, Math.ceil((remaining * dt) / REVEAL_DRAIN_MS))
      )

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

const MAX_MARKDOWN_CHARS = 200_000

function HugeTextFallback({ containerClassName, text }: { containerClassName?: string; text: string }) {
  const chunks = useMemo(() => chunkByLines(text, 200), [text])

  return (
    <div
      className={cn(
        'aui-md w-full max-w-none overflow-hidden rounded-[0.625rem] border border-border font-mono text-[0.7rem] leading-relaxed text-foreground/90',
        containerClassName
      )}
    >
      <ExpandableBlock className="p-2">
        {chunks.map((chunk, index) => (
          <div
            className="[content-visibility:auto]"
            key={index}
            style={{ containIntrinsicSize: `auto ${chunk.lines * 16}px` }}
          >
            {chunk.text}
          </div>
        ))}
      </ExpandableBlock>
    </div>
  )
}

function MarkdownTextSurface({ containerClassName, containerProps }: MarkdownTextSurfaceProps) {
  const { status, text } = useMessagePartText()
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
        p: ({ className, children, ...props }: ComponentProps<'p'>) => {
          // Vertical rhythm is owned by styles.css (`--paragraph-gap`), which
          // must out-specify Tailwind Typography's `prose` margins — so no
          // `my-*` here on purpose.
          //
          // A `#gallery:` link expands to <MediaCarousel> (a <div>), which is
          // invalid inside a <p>. When the paragraph's content contains a
          // carousel, render it in a <div> instead so the block-level carousel
          // is valid HTML and actually paints.
          if (subtreeContainsGallery(children)) {
            return (
              <div className={cn('wrap-anywhere leading-(--dt-line-height)', className)} {...props}>
                {children}
              </div>
            )
          }

          return (
            <p className={cn('wrap-anywhere leading-(--dt-line-height)', className)} {...props}>
              {children}
            </p>
          )
        },
        a: MarkdownLink,
        // Inline code must not vote when an ancestor resolves `dir="auto"`
        // (HTML's algorithm skips descendants that carry their own dir),
        // mirroring the CSS isolate that already keeps it out of the
        // plaintext scan. Fenced code never reaches this override; it goes
        // through the code plugin's CodeCard path.
        inlineCode: ({ className, ...props }: ComponentProps<'code'>) => (
          <code className={className} dir="ltr" {...props} />
        ),
        // `---` as quiet spacing, not a heavy full-width rule.
        hr: (_props: ComponentProps<'hr'>) => <div aria-hidden className="my-3" />,
        // Lists and blockquotes have chrome that sits *beside* the text
        // (markers, the quote border), and that side is driven by the CSS
        // `direction` of the box, which `unicode-bidi: plaintext` never
        // touches — an RTL list otherwise renders its numbers stranded at
        // the far left. `dir="auto"` lets the browser resolve the box
        // direction from content; the plaintext rules in styles.css keep
        // owning per-line text direction. Inline code carries `dir="ltr"`
        // (see the `code` override) so it doesn't vote here either, same
        // contract as the CSS isolate.
        // A `> [!NOTE]`/`[!WARNING]`/... blockquote renders as a GFM alert
        // callout; everything else stays a plain quote.
        blockquote: ({ children, className, ...props }: ComponentProps<'blockquote'>) => {
          const alert = extractAlert(children)

          if (alert) {
            return <MarkdownAlert type={alert.type}>{alert.body}</MarkdownAlert>
          }

          return (
            <blockquote
              className={cn('border-s-2 border-border ps-3 text-muted-foreground italic', className)}
              dir="auto"
              {...props}
            >
              {children}
            </blockquote>
          )
        },
        ul: ({ className, ...props }: ComponentProps<'ul'>) => (
          <ul className={cn('my-1 gap-0', className)} dir="auto" {...props} />
        ),
        ol: ({ className, ...props }: ComponentProps<'ol'>) => (
          <ol className={cn('my-1 gap-0', className)} dir="auto" {...props} />
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
        // ```mermaid / ```svg fences route to their lazy renderers; every other
        // language falls back to the Shiki-highlighted code block.
        SyntaxHighlighter: (props: SyntaxHighlighterProps) => (
          <RichCodeBlock
            code={props.code}
            fallback={<SyntaxHighlighter {...props} defer={isStreaming} />}
            language={props.language}
            streaming={isStreaming}
          />
        )
      }) as StreamdownTextComponents,
    [isStreaming]
  )

  if (text.length > MAX_MARKDOWN_CHARS) {
    return <HugeTextFallback containerClassName={containerClassName} text={text} />
  }

  return (
    <StreamdownTextPrimitive
      components={components}
      containerClassName={cn(MARKDOWN_CONTAINER_CLASS_NAME, containerClassName)}
      containerProps={containerProps}
      lineNumbers={false}
      mode="streaming"
      // Incomplete-markdown repair is handled by `preprocessWithTailRepair`
      // below (tail-bounded remend) instead of Streamdown's built-in pass,
      // which re-runs remend over the ENTIRE message on every flush — ~18%
      // of streaming script time on 50KB+ messages. The repair itself stays
      // always-on (even between flushes / for completed messages): an
      // unclosed ```python ... ``` whose body contains `$` (shell snippets,
      // JS template strings, dollar amounts) would otherwise leak those
      // dollars to the math parser and render broken inline math. Shiki is
      // independently deferred via `defer={isStreaming}` on the
      // SyntaxHighlighter component.
      parseIncompleteMarkdown={false}
      parseMarkdownIntoBlocksFn={parseMarkdownIntoBlocksCached}
      plugins={plugins}
      preprocess={preprocessWithTailRepair}
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
    <DeferStreamingText>
      <MarkdownTextSurface />
    </DeferStreamingText>
  )
}

export const MarkdownText = memo(MarkdownTextImpl)
