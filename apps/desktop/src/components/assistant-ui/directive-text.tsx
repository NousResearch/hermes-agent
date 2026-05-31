'use client'

import type { Unstable_DirectiveFormatter, Unstable_DirectiveSegment, Unstable_TriggerItem } from '@assistant-ui/core'
import type { TextMessagePartComponent, TextMessagePartProps } from '@assistant-ui/react'
import type { FC } from 'react'
import { Fragment, useEffect, useMemo, useState } from 'react'

import { ZoomableImage } from '@/components/chat/zoomable-image'
import { extractEmbeddedImages } from '@/lib/embedded-images'

const HERMES_REF_TYPES = ['file', 'folder', 'url', 'image', 'tool', 'line', 'terminal'] as const
type HermesRefType = (typeof HERMES_REF_TYPES)[number]

/** Single source of truth for chip icon glyphs (Tabler outline @ 24×24).
 * Used both by the rendered <DirectiveIcon> and the raw SVG markup the
 * contenteditable composer embeds via `directiveIconSvg`. */
const ICON_PATHS: Record<HermesRefType, string[]> = {
  file: [
    'M14 3v4a1 1 0 0 0 1 1h4',
    'M17 21h-10a2 2 0 0 1 -2 -2v-14a2 2 0 0 1 2 -2h7l5 5v11a2 2 0 0 1 -2 2',
    'M9 9l1 0',
    'M9 13l6 0',
    'M9 17l6 0'
  ],
  folder: [
    'M5 19l2.757 -7.351a1 1 0 0 1 .936 -.649h12.307a1 1 0 0 1 .986 1.164l-.996 5.211a2 2 0 0 1 -1.964 1.625h-14.026a2 2 0 0 1 -2 -2v-11a2 2 0 0 1 2 -2h4l3 3h7a2 2 0 0 1 2 2v2'
  ],
  url: [
    'M9 15l6 -6',
    'M11 6l.463 -.536a5 5 0 0 1 7.071 7.072l-.534 .464',
    'M13 18l-.397 .534a5.068 5.068 0 0 1 -7.127 0a4.972 4.972 0 0 1 0 -7.071l.524 -.463'
  ],
  image: [
    'M15 8h.01',
    'M3 6a3 3 0 0 1 3 -3h12a3 3 0 0 1 3 3v12a3 3 0 0 1 -3 3h-12a3 3 0 0 1 -3 -3v-12',
    'M3 16l5 -5c.928 -.893 2.072 -.893 3 0l5 5',
    'M14 14l1 -1c.928 -.893 2.072 -.893 3 0l3 3'
  ],
  tool: ['M7 10h3v-3l-3.5 -3.5a6 6 0 0 1 8 8l6 6a2 2 0 0 1 -3 3l-6 -6a6 6 0 0 1 -8 -8l3.5 3.5'],
  line: ['M5 9l14 0', 'M5 15l14 0', 'M11 4l-4 16', 'M17 4l-4 16'],
  terminal: ['M5 7l5 5l-5 5', 'M12 19l7 0']
}

const ICON_FALLBACK = ['M8 12a4 4 0 1 0 8 0a4 4 0 1 0 -8 0', 'M16 12v1.5a2.5 2.5 0 0 0 5 0v-1.5a9 9 0 1 0 -5.5 8.28']

const SVG_ATTRS =
  'xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"'

const iconPathsFor = (type: string) => ICON_PATHS[type as HermesRefType] ?? ICON_FALLBACK

/** SVG markup string for embedding directly in HTML (composer contenteditable). */
export function directiveIconSvg(type: string) {
  const inner = iconPathsFor(type)
    .map(d => `<path d="${d}"/>`)
    .join('')

  return `<svg ${SVG_ATTRS} class="size-3 shrink-0 opacity-80">${inner}</svg>`
}

export function directiveIconElement(type: string) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg')
  svg.setAttribute('class', 'size-3 shrink-0 opacity-80')
  svg.setAttribute('fill', 'none')
  svg.setAttribute('stroke', 'currentColor')
  svg.setAttribute('stroke-linecap', 'round')
  svg.setAttribute('stroke-linejoin', 'round')
  svg.setAttribute('stroke-width', '2')
  svg.setAttribute('viewBox', '0 0 24 24')
  svg.setAttribute('xmlns', 'http://www.w3.org/2000/svg')

  for (const d of iconPathsFor(type)) {
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path')
    path.setAttribute('d', d)
    svg.append(path)
  }

  return svg
}

const DirectiveIcon: FC<{ type: string }> = ({ type }) => (
  <svg
    className="size-3 shrink-0 opacity-80"
    fill="none"
    stroke="currentColor"
    strokeLinecap="round"
    strokeLinejoin="round"
    strokeWidth={2}
    viewBox="0 0 24 24"
    xmlns="http://www.w3.org/2000/svg"
  >
    {iconPathsFor(type).map(d => (
      <path d={d} key={d} />
    ))}
  </svg>
)

/** Shared chip styling — used by both the rendered <DirectiveChip> and the
 * raw HTML composer chips in `rich-editor.ts`. Neutral subtle wash + plain
 * muted-foreground text so chips read as quiet tags on any bubble color. */
export const DIRECTIVE_CHIP_CLASS =
  'mx-0.5 inline-flex max-w-56 items-center gap-1 rounded px-1.5 py-0.5 align-[0.02em] text-[0.86em] font-normal leading-none bg-[color-mix(in_srgb,currentColor_8%,transparent)] text-muted-foreground'

/**
 * Parses our composer's `@type:value` references into directive segments
 * so they render as inline chips in user messages instead of raw text.
 *
 * Supported types: file, folder, url, image. Anything else stays plain text.
 *
 * Mirrors the Python `agent/context_references.REFERENCE_PATTERN` syntax:
 * the value may be wrapped in backticks, single quotes, or double quotes so
 * paths with spaces/parens/etc. survive parsing intact.
 */
const CANONICAL_DIRECTIVE_RE = /:([\w-]{1,64})\[([^\]\n]{1,1024})\](?:\{name=([^}\n]{1,1024})\})?/g

const HERMES_DIRECTIVE_RE = new RegExp(
  '@(file|folder|url|image|tool|line|terminal):(' + '`[^`\\n]+`' + '|"[^"\\n]+"' + "|'[^'\\n]+'" + '|\\S+' + ')',
  'g'
)

const TRAILING_PUNCTUATION_RE = /[,.;!?]+$/

function unwrapRefValue(raw: string): string {
  if (raw.length < 2) {
    return raw
  }

  const head = raw[0]
  const tail = raw[raw.length - 1]

  if ((head === '`' && tail === '`') || (head === '"' && tail === '"') || (head === "'" && tail === "'")) {
    return raw.slice(1, -1)
  }

  return raw.replace(TRAILING_PUNCTUATION_RE, '')
}

function needsQuoting(value: string): boolean {
  return /[\s()[\]{}<>"'`]/.test(value)
}

export function formatRefValue(value: string): string {
  if (!needsQuoting(value)) {
    return value
  }

  if (!value.includes('`')) {
    return `\`${value}\``
  }

  if (!value.includes('"')) {
    return `"${value}"`
  }

  if (!value.includes("'")) {
    return `'${value}'`
  }

  return value
}

export const hermesDirectiveFormatter: Unstable_DirectiveFormatter = {
  serialize(item: Unstable_TriggerItem): string {
    const metadata = item.metadata as { rawText?: unknown; insertId?: unknown } | undefined
    const rawText = typeof metadata?.rawText === 'string' ? metadata.rawText : null
    const insertId = typeof metadata?.insertId === 'string' ? metadata.insertId : null

    // Live-completion items carry the gateway's original `text` field via metadata.
    if (rawText) {
      // Palette starters (`@file:` with empty value) — insert verbatim so the
      // user can keep typing the path inline.
      if (rawText.endsWith(':') && !insertId) {
        return rawText
      }

      // Simple references like `@diff` / `@staged`.
      if (!insertId) {
        return rawText
      }

      // Typed references with a value — quote when needed.
      const kindMatch = rawText.match(/^@([^:]+):/)
      const kind = kindMatch?.[1] ?? item.type

      return `@${kind}:${formatRefValue(insertId)}`
    }

    // Fallback for legacy callers that pass raw `id` strings.
    if (item.id === `${item.type}:`) {
      return `@${item.id}`
    }

    return `@${item.type}:${formatRefValue(item.id)}`
  },
  parse(text: string): readonly Unstable_DirectiveSegment[] {
    return parseDirectiveText(text)
  }
}

function parseDirectiveText(text: string): Unstable_DirectiveSegment[] {
  const matches = [
    ...Array.from(text.matchAll(CANONICAL_DIRECTIVE_RE)).map(match => ({
      start: match.index ?? 0,
      end: (match.index ?? 0) + match[0].length,
      type: match[1] || 'tool',
      label: match[2] || match[3] || '',
      id: match[3] || match[2] || ''
    })),
    ...Array.from(text.matchAll(HERMES_DIRECTIVE_RE)).map(match => {
      const id = unwrapRefValue(match[2] || '')

      return {
        start: match.index ?? 0,
        end: (match.index ?? 0) + match[0].length,
        type: match[1] || 'file',
        label: shortLabel(match[1] as HermesRefType, id),
        id
      }
    })
  ]
    .filter(match => match.id)
    .sort((a, b) => a.start - b.start)

  const segments: Unstable_DirectiveSegment[] = []
  let cursor = 0

  for (const match of matches) {
    if (match.start < cursor) {
      continue
    }

    if (match.start > cursor) {
      segments.push({ kind: 'text', text: text.slice(cursor, match.start) })
    }

    segments.push({
      kind: 'mention',
      type: match.type,
      label: match.label,
      id: match.id
    })
    cursor = match.end
  }

  if (cursor < text.length) {
    segments.push({ kind: 'text', text: text.slice(cursor) })
  }

  return segments
}

function shortLabel(type: HermesRefType, id: string): string {
  if (type === 'terminal') {
    return id || 'terminal'
  }

  if (type === 'url') {
    try {
      const parsed = new URL(id)

      return parsed.hostname || id
    } catch {
      return id
    }
  }

  const tail = id.split(/[\\/]/).filter(Boolean).pop()

  return tail || id
}

/**
 * Renders text containing Hermes directives (`@file:...`, `@image:...`) as
 * inline chips. Embedded MEDIA images render below as a thumbnail row.
 */
export function DirectiveContent({ text }: { text: string }) {
  const { cleanedText, images } = useMemo(() => extractEmbeddedImages(text ?? ''), [text])
  const segments = useMemo(() => hermesDirectiveFormatter.parse(cleanedText), [cleanedText])

  return (
    <span className="whitespace-pre-line" data-slot="aui_directive-text">
      {segments.map((segment, index) =>
        segment.kind === 'text' ? (
          <Fragment key={`t-${index}`}>{segment.text}</Fragment>
        ) : segment.type === 'image' ? (
          <DirectiveImage id={segment.id} key={`img-${index}-${segment.id}`} label={segment.label} />
        ) : (
          <DirectiveChip id={segment.id} key={`m-${index}-${segment.id}`} label={segment.label} type={segment.type} />
        )
      )}
      {images.length > 0 && (
        <span className="mt-2 flex flex-wrap gap-2" data-slot="aui_embedded-images">
          {images.map((src, index) => (
            <ZoomableImage
              alt=""
              className="max-h-48 max-w-full rounded-lg border border-border/60 object-contain"
              draggable={false}
              key={`img-${index}`}
              slot="aui_embedded-image"
              src={src}
            />
          ))}
        </span>
      )}
    </span>
  )
}

/** assistant-ui adapter: same renderer, exposed as a TextMessagePartComponent. */
export const DirectiveText: TextMessagePartComponent = ({ text }: TextMessagePartProps) => (
  <DirectiveContent text={text ?? ''} />
)

/** Image refs render as a thumbnail rather than a chip — matches how persisted
 * messages render after the backend embeds the data URL, so the UX is stable
 * across initial send and refresh. */
const DirectiveImage: FC<{ id: string; label: string }> = ({ id, label }) => {
  const remote = /^(?:https?|data):/i.test(id)
  const [src, setSrc] = useState<string | null>(remote ? id : null)
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    if (remote || !id) {
      return
    }

    let alive = true
    void window.hermesDesktop
      ?.readFileDataUrl(id)
      .then(url => alive && setSrc(url))
      .catch(() => alive && setFailed(true))

    return () => {
      alive = false
    }
  }, [id, remote])

  if (failed) {
    return <DirectiveChip id={id} label={label} type="image" />
  }

  if (!src) {
    return (
      <span
        aria-hidden
        className="inline-block size-12 shrink-0 animate-pulse rounded-md bg-[color-mix(in_srgb,currentColor_8%,transparent)]"
      />
    )
  }

  return (
    <ZoomableImage
      alt={label}
      className="max-h-32 max-w-48 rounded-md border border-border/40 object-contain"
      draggable={false}
      slot="aui_directive-image"
      src={src}
    />
  )
}

const DirectiveChip: FC<{
  type: string
  label: string
  id: string
}> = ({ type, label, id }) => (
  <span
    className={DIRECTIVE_CHIP_CLASS}
    data-directive-id={id}
    data-directive-type={type}
    data-slot="aui_directive-chip"
    title={id}
  >
    <DirectiveIcon type={type} />
    <span className="truncate">{label}</span>
  </span>
)
