'use client'

import { TextMessagePartProvider, useMessagePartText } from '@assistant-ui/react'
import {
  type StreamdownTextComponents,
  StreamdownTextPrimitive,
  type SyntaxHighlighterProps,
  tailBoundedRemend
} from '@assistant-ui/react-streamdown'
import { code } from '@streamdown/code'
import { type ComponentProps, memo, useEffect, useMemo, useState } from 'react'

import { ExpandableBlock } from '@/components/chat/expandable-block'
import { PreviewAttachment } from '@/components/chat/preview-attachment'
import { chunkByLines, SyntaxHighlighter } from '@/components/chat/shiki-highlighter'
import { ZoomableImage } from '@/components/chat/zoomable-image'
import { normalizeExternalUrl, openExternalLink, PrettyLink } from '@/lib/external-link'
import { createMemoizedMathPlugin } from '@/lib/katex-memo'
import { parseMarkdownIntoBlocksCached } from '@/lib/markdown-blocks'
import { preprocessMarkdown } from '@/lib/markdown-preprocess'
import {
  downloadGatewayMediaFile,
  isInlineMediaSrc,
  isRemoteGateway,
  mediaExternalUrl,
  mediaKind,
  mediaName,
  mediaPathFromMarkdownHref,
  mediaStreamUrl,
  resolveMediaDisplaySrc
} from '@/lib/media'
import { previewTargetFromMarkdownHref } from '@/lib/preview-targets'
import { cn } from '@/lib/utils'

import { detectEmbed, extractAlert, MarkdownAlert, RichCodeBlock, UrlEmbed } from './embeds'

const mathPlugin = createMemoizedMathPlugin({ singleDollarTextMath: true })

function preprocessWithTailRepair(text: string): string {
  try {
    return tailBoundedRemend(preprocessMarkdown(text))
  } catch {
    return text
  }
}

async function mediaSrc(path: string): Promise<string> {
  if (window.hermesDesktop && ['audio', 'video'].includes(mediaKind(path))) {
    return mediaStreamUrl(path)
  }
  return resolveMediaDisplaySrc(path)
}

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
    <span className=\"mt-1 block text-xs text-muted-foreground\">
      Couldn&apos;t fetch {name} from the gateway (missing, unreadable, or too large).
    </span>
  )
}

function OpenMediaButton({ kind, path }: { kind: 'audio' | 'video'; path: string }) {
  const { open, openFailed } = useOpenMediaFile(path)
  return (
    <span className=\"block\">
      <button
        className=\"mt-2 bg-transparent text-xs font-medium text-muted-foreground underline underline-offset-4 decoration-current/20 hover:text-foreground\"
        onClick={open}
        type=\"button\"
      >
        Open {kind} file
      </button>
      {openFailed && <OpenMediaFailedNote name={mediaName(path)} />}\n    </span>
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
      return () => { cancelled = true }
    }
    void mediaSrc(path)
      .then(value => {
        if (value.startsWith('blob:')) { objectUrl = value }
        if (!cancelled) { setSrc(value) } else if (objectUrl) { URL.revokeObjectURL(objectUrl) }
      })
      .catch(() => {
        if (!cancelled) { setFailed(true) }
      })
    return () => {
      cancelled = true
      if (objectUrl) { URL.revokeObjectURL(objectUrl) }
    }
  }, [kind, path])

  if (kind === 'image' && src) {
    return (
      <span className=\"block\">
        <MarkdownImage alt={name} src={src} />
      </span>
    )
  }
  if (kind === 'audio' && src) {
    return (
      <span className=\"my-3 block max-w-md rounded-xl border border-border bg-muted/35 p-3\">
        <span className=\"mb-2 block truncate text-xs font-medium text-muted-foreground\">{name}</span>
        <audio className=\"block w-full\" controls onError={() => setFailed(true)} preload=\"metadata\" src={src} />
        {failed && <OpenMediaButton kind=\"audio\" path={path} />}\n      </span>
    )
  }
  if (kind === 'video' && src) {
    return (
      <span className=\"my-3 block max-w-2xl rounded-xl border border-border bg-muted/35 p-3\">
        <span className=\"mb-2 block truncate text-xs font-medium text-muted-foreground\">{name}</span>
        <video
          className=\"block max-h-112 w-full rounded-lg bg-black\"
          controls
          onError={() => setFailed(true)}
          src={src}
        />
        {failed && <OpenMediaButton kind=\"video\" path={path} />}\n      </span>
    )
  }
  return (
    <span className=\"wrap-anywhere\">
      <a
        className=\"font-semibold text-foreground underline underline-offset-4 decoration-current/20 wrap-anywhere\"
        href=\"#\"
        onClick={event => {
          event.preventDefault()
          open()
        }}
      >
        {failed ? `Open ${name}` : `Loading ${name}...`}\n      </a>
      {openFailed && <OpenMediaFailedNote name={name} />}\n    </span>
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
  if (mediaPath) { return <MediaAttachment path={mediaPath} /> }
  const previewTarget = previewTargetFromMarkdownHref(href)
  if (previewTarget) { return <PreviewAttachment source=\"explicit-link\" target={previewTarget} /> }
  const target = href ? normalizeExternalUrl(href) : href
  if (!target || !/^https?:\\/\\//i.test(target)) {
    return (
      <a
        className={cn('font-semibold text-foreground underline underline-offset-4 decoration-current/20 wrap-anywhere', className)}
        href={href}
        rel=\"noopener noreferrer\"
        target=\"_blank\"
        {...props}
      >
        {children}
      </a>
    )
  }
  const text = childrenToText(children)
  if (window.hermesDesktop && text && normalizeExternalUrl(text) === target) {
    const embed = detectEmbed(target)
    if (embed) { return <UrlEmbed descriptor={embed} /> }
  }
  const fallbackLabel = text && normalizeExternalUrl(text) !== target ? text : undefined
  return <PrettyLink className={cn('wrap-anywhere', className)} fallbackLabel={fallbackLabel} href={target} {...props} />
}

function MarkdownImage({ className, src, alt, ...props }: ComponentProps<'img'>) {
  const rawSrc = typeof src === 'string' ? src : ''
  const [resolvedSrc, setResolvedSrc] = useState(() => (rawSrc && isInlineMediaSrc(rawSrc) ? rawSrc : ''))
  const [failed, setFailed] = useState(false)
  const { open, openFailed } = useOpenMediaFile(rawSrc)
  const name = mediaName(rawSrc || String(alt || 'image'))

  useEffect(() => {
    let cancelled = false
    setFailed(false)
    setResolvedSrc(rawSrc && isInlineMediaSrc(rawSrc) ? rawSrc : '')
    if (!rawSrc || isInlineMediaSrc(rawSrc)) {
      return () => { cancelled = true }
    }
    void resolveMediaDisplaySrc(rawSrc)
      .then(value => {
        if (!cancelled) { setResolvedSrc(value) }
      })
      .catch(() => {
        if (!cancelled) { setFailed(true) }
      })
    return () => { cancelled = true }
  }, [rawSrc])

  if (!rawSrc) { return null }
  if (failed) {
    return (
      <span className=\"my-2 block text-sm text-muted-foreground\">
        Couldn&apos;t load {name}.{' '}
        <button
          className=\"bg-transparent font-medium text-foreground underline underline-offset-4 decoration-current/20 hover:text-foreground\"
          onClick={open}
          type=\"button\"
        >
          Open image
        </button>
        {openFailed && <OpenMediaFailedNote name={name} />}\n      </span>
    )
  }
  if (!resolvedSrc) { return <span className=\"my-2 block text-sm text-muted-foreground\">Loading {name}...</span> }
  return (
    <ZoomableImage
      alt={alt}
      className={cn('m-0 block h-auto w-auto max-h-(--image-preview-height) max-w-[min(100%,var(--image-preview-max-width))] rounded-lg object-contain shadow-[0_0.0625rem_0.125rem_color-mix(in_srgb,#000_4%,transparent),0_0.625rem_1.5rem_color-mix(in_srgb,#000_5%,transparent)]', className)}
      containerClassName=\"my-2 block w-fit max-w-full\"
      slot=\"aui_markdown-image\"
      src={resolvedSrc}
      {...props}
    />
  )
}

interface MarkdownTextSurfaceProps {
  containerClassName?: string
  containerProps?: ComponentProps<'div'>
  defer?: boolean
}

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
  'prose-a:break-words prose-p:[overflow-wrap:anywhere] prose-p:whitespace-pre-wrap',
  'prose-li:marker:text-muted-foreground/70',
  'prose-code:rounded-[0.25rem] prose-code:px-[0.1875rem] prose-code:py-px prose-code:font-mono prose-code:text-[0.9em] prose-code:font-normal prose-code:before:content-none prose-code:after:content-none',
  '[&>*:first-child]:mt-0 [&>*:last-child]:mb-0 [&>*+*]:mt-(--paragraph-gap)'
)

const MAX_MARKDOWN_CHARS = 200_000

function HugeTextFallback({ containerClassName, text }: { containerClassName?: string; text: string }) {
  const chunks = useMemo(() => chunkByLines(text, 200), [text])
  return (
    <div className={cn('aui-md w-full max-w-none overflow-hidden rounded-[0.625rem] border border-border font-mono text-[0.7rem] leading-relaxed text-foreground/90', containerClassName)}>
      <ExpandableBlock className=\"p-2\">
        {chunks.map((chunk, index) => (
          <div className=\"[content-visibility:auto]\" key={index} style={{ containIntrinsicSize: `auto ${chunk.lines * 16}px` }}>
            {chunk.text}
          </div>
        ))}
      </ExpandableBlock>
    </div>
  )
}

function MarkdownTextSurface({ containerClassName, containerProps, defer }: MarkdownTextSurfaceProps) {
  const { status, text } = useMessagePartText()
  const isStreaming = status.type === 'running'
  const plugins = useMemo(() => ({ math: mathPlugin, code }), [])
  const components = useMemo(
    () => ({
      h1: ({ className, ...props }: ComponentProps<'h1'>) => <h1 className={cn('my-1 font-semibold', HEADING_SIZES.h1, className)} {...props} />,
      h2: ({ className, ...props }: ComponentProps<'h2'>) => <h2 className={cn('my-1 font-semibold', HEADING_SIZES.h2, className)} {...props} />,
      h3: ({ className, ...props }: ComponentProps<'h3'>) => <h3 className={cn('my-1 font-semibold', HEADING_SIZES.h3, className)} {...props} />,
      h4: ({ className, ...props }: ComponentProps<'h4'>) => <h4 className={cn('my-1 font-semibold', HEADING_SIZES.h4, className)} {...props} />,
      p: ({ className, ...props }: ComponentProps<'p'>) => <p className={cn('wrap-anywhere leading-(--dt-line-height)', className)} {...props} />,
      a: MarkdownLink,
      inlineCode: ({ className, ...props }: ComponentProps<'code'>) => <code className={className} dir=\"ltr\" {...props} />,
      hr: (_props: ComponentProps<'hr'>) => <div aria-hidden className=\"my-3\" />,
      blockquote: ({ children, className, ...props }: ComponentProps<'blockquote'>) => {
        const alert = extractAlert(children)
        if (alert) { return <MarkdownAlert type={alert.type}>{alert.body}</MarkdownAlert> }
        return <blockquote className={cn('border-s-2 border-border ps-3 text-muted-foreground italic', className)} dir=\"auto\" {...props}>{children}</blockquote>
      },
      ul: ({ className, ...props }: ComponentProps<'ul'>) => <ul className={cn('my-1 gap-0', className)} dir=\"auto\" {...props} />,
      ol: ({ className, ...props }: ComponentProps<'ol'>) => <ol className={cn('my-1 gap-0', className)} dir=\"auto\" {...props} />,
      li: ({ className, ...props }: ComponentProps<'li'>) => <li className={cn('leading-(--dt-line-height)', className)} {...props} />,
      table: ({ className, ...props }: ComponentProps<'table'>) => (
        <div className=\"aui-md-table my-2 max-w-full overflow-x-auto rounded-[0.375rem] border border-border\">
          <table className={cn('m-0 w-full min-w-[18rem] border-collapse text-[0.8125rem] [&_tr]:border-b [&_tr]:border-border last:[&_tr]:border-0', className)} {...props} />
        </div>
      ),
      thead: ({ className, ...props }: ComponentProps<'thead'>) => <thead className={cn('m-0 bg-muted/35 text-muted-foreground', className)} {...props} />,
      th: ({ className, ...props }: ComponentProps<'th'>) => <th className={cn('whitespace-nowrap px-2.5 py-1.5 text-left align-middle text-[0.75rem] font-medium text-muted-foreground', className)} {...props} />,
      td: ({ className, ...props }: ComponentProps<'td'>) => <td className={cn('px-2.5 py-1.5 align-top text-[0.8125rem] leading-snug', className)} {...props} />,
      img: MarkdownImage,
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

  if (text.length > MAX_MARKDOWN_CHARS) { return <HugeTextFallback containerClassName={containerClassName} text={text} /> }

  return (
    <StreamdownTextPrimitive
      components={components}
      containerClassName={cn(MARK MarkdownCONTAINER_CLASS_NAME, containerClassName)}
      containerProps={containerProps}
      defer={defer}
      lineNumbers={false}
      mode=\"streaming\"
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
      <MarkdownTextSurface defer {...surfaceProps} />
    </TextMessagePartProvider>
  )
}

const MarkdownTextImpl = () => {
  return <MarkdownTextSurface defer />
}

export const MarkdownText = memo(MarkdownTextImpl)
