'use client'

import type { Unstable_DirectiveFormatter, Unstable_DirectiveSegment, Unstable_TriggerItem } from '@assistant-ui/core'
import type { TextMessagePartComponent, TextMessagePartProps } from '@assistant-ui/react'
import type { FC } from 'react'
import { Fragment, useEffect, useMemo, useState } from 'react'

import { ZoomableImage } from '@/components/chat/zoomable-image'
import { extractEmbeddedImages } from '@/lib/embedded-images'
import { triggerHaptic } from '@/lib/haptics'
import { gatewayMediaDataUrl, isRemoteGateway } from '@/lib/media'
import { useSessionLinkTitle } from '@/lib/session-link-title'
import { parseSessionRefValue, sessionRefFallbackLabel } from '@/lib/session-refs'
import { cn } from '@/lib/utils'

import { referenceKind, referenceRe, referenceStyle, WIRE_REFERENCE_KINDS } from './reference-kinds'

const HERMES_REF_TYPES = WIRE_REFERENCE_KINDS
type HermesRefType = (typeof HERMES_REF_TYPES)[number]

/** Icon glyphs come from the shared reference vocabulary, so the popover row
 *  and the chip can never drift apart. */
const iconPathsFor = (type: string) => referenceStyle(type).paths

const SVG_ATTRS =
  'xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"'

/**
 * The class + attributes that make any element an inline reference. Pair with
 * the `.ref` rules in styles.css, which own the per-kind accent — pass the kind
 * and the theme decides the colour.
 *
 * One helper for every surface: the composer's contenteditable chips, a sent
 * message's mentions, a markdown link, a completion row's glyph. If it points
 * at something from inside text, it goes through here.
 */
export function refAttrs(kind?: string, extra?: string): { className: string; 'data-ref'?: string } {
  const className = extra ? `ref ${extra}` : 'ref'

  return kind ? { className, 'data-ref': referenceKind(kind) } : { className }
}

/** The same thing as a raw attribute string, for HTML built by hand. */
export function refAttrsHtml(kind?: string): string {
  return kind ? `class="ref" data-ref="${referenceKind(kind)}"` : 'class="ref"'
}

/** SVG markup string for embedding directly in HTML (composer contenteditable). */
export function directiveIconSvg(type: string) {
  const inner = iconPathsFor(type)
    .map(d => `<path d="${d}"/>`)
    .join('')

  return `<svg ${SVG_ATTRS}>${inner}</svg>`
}

function iconElementFromPaths(paths: string[]) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg')
  svg.setAttribute('fill', 'none')
  svg.setAttribute('stroke', 'currentColor')
  svg.setAttribute('stroke-linecap', 'round')
  svg.setAttribute('stroke-linejoin', 'round')
  svg.setAttribute('stroke-width', '2')
  svg.setAttribute('viewBox', '0 0 24 24')
  svg.setAttribute('xmlns', 'http://www.w3.org/2000/svg')

  for (const d of paths) {
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path')
    path.setAttribute('d', d)
    svg.append(path)
  }

  return svg
}

export function directiveIconElement(type: string) {
  return iconElementFromPaths(iconPathsFor(type))
}

/** Commands, skills, and themes are three more reference kinds — no separate
 *  pill styling, just the shared `.ref` treatment with their own accent. */
export type SlashChipKind = 'command' | 'skill' | 'theme'

export function slashIconElement(kind: SlashChipKind) {
  return iconElementFromPaths(iconPathsFor(kind))
}

/** The glyph for a reference kind. Size, spacing, and opacity come from the
 *  `.ref > svg` rules — the icon only has to say which shape it is. */
const DirectiveIcon: FC<{ type: string; className?: string }> = ({ type, className }) => (
  <svg
    className={className}
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

const HERMES_DIRECTIVE_RE = referenceRe()

// A skill referenced in a sent message — either the invocation that opens it
// (`/work fix the leak`, which is all a skill turn ever renders as) or one
// named mid-prose (`clean this up with /clean`). The composer inserts both as
// pills, so the sent message renders them as pills too rather than flattening
// back to raw text.
//
// #71664 deliberately excluded a LEADING slash, and was right then: a command
// only ever executed, so it never reached a rendered message as text. Skill
// turns now project back onto their invocation, so that precondition is gone
// and `^` joins the lookbehind.
//
// Unlike the composer's caret-anchored trigger, this scans finished text, so
// it must reject a token that continues into a path: `/usr/local/bin` would
// otherwise chip as `/usr`. `(?![\w-]*\/)` requires the token to end at
// something other than another slash.
const SLASH_SKILL_RE = /(?<=^|\s)\/([a-zA-Z][\w-]*)(?![\w-]*\/)/g

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
        label: refChipLabel(match[1] || 'file', id),
        id
      }
    }),
    ...Array.from(text.matchAll(SLASH_SKILL_RE)).map(match => ({
      start: match.index ?? 0,
      end: (match.index ?? 0) + match[0].length,
      type: 'skill',
      label: match[1],
      id: `/${match[1]}`
    }))
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

/** The single display label for a `@kind:value` reference — used by the `@`
 *  popover row, the composer chip, and the sent-message chip alike, so a
 *  reference reads the same everywhere. Upstream keeps one label on the
 *  directive node and hands it to every consumer verbatim; our wire format
 *  (`@kind:value`) can't carry a label, so this is the shared deriver that
 *  holds the same invariant.
 *
 *  Paths keep their directory for the reason links keep theirs: a bare
 *  basename can't tell two references apart (`src`, `index.ts`, `main.tsx`
 *  repeat all over a repo), and browsing into `apps/desktop/` only to be
 *  handed a chip reading `desktop` throws away the context you navigated for.
 *  The chip's `truncate` cuts the overflow. */
export function refChipLabel(type: string, id: string): string {
  if (type === 'terminal') {
    return id || 'terminal'
  }

  if (type === 'session') {
    return sessionRefFallbackLabel(id)
  }

  if (type === 'url') {
    try {
      const { hostname, pathname, search } = new URL(id)
      const path = `${pathname}${search}`.replace(/\/$/, '')

      return `${hostname.replace(/^www\./i, '')}${path}` || id
    } catch {
      return id
    }
  }

  // `./` is noise the completer emits, not part of the reference. A trailing
  // slash is kept — it's what distinguishes a folder from a file.
  return id.replace(/^\.\//, '') || id
}

function safeEmbeddedImages(text: string) {
  try {
    return extractEmbeddedImages(text)
  } catch {
    return { cleanedText: text, images: [] as string[] }
  }
}

function safeDirectiveSegments(text: string): Unstable_DirectiveSegment[] {
  try {
    return [...hermesDirectiveFormatter.parse(text)]
  } catch {
    return [{ kind: 'text', text }]
  }
}

/**
 * Renders text containing Hermes directives (`@file:...`, `@image:...`) as
 * inline chips. Embedded MEDIA images render below as a thumbnail row.
 */
export function DirectiveContent({ text }: { text: string }) {
  const { cleanedText, images } = useMemo(() => safeEmbeddedImages(text ?? ''), [text])
  const segments = useMemo(() => safeDirectiveSegments(cleanedText), [cleanedText])

  // `@image:<path>` directives render as a block-level thumbnail row (like
  // embedded base64 images below), not inline mid-text — otherwise a large
  // thumbnail gets wedged between words and breaks the text's line flow.
  const imageSegments = useMemo(
    () =>
      segments.filter(
        (segment): segment is Extract<Unstable_DirectiveSegment, { kind: 'mention' }> =>
          segment.kind === 'mention' && segment.type === 'image'
      ),
    [segments]
  )

  return (
    <span className="whitespace-pre-line" data-slot="aui_directive-text">
      {segments.map((segment, index) =>
        segment.kind === 'text' ? (
          <Fragment key={`t-${index}`}>{segment.text}</Fragment>
        ) : segment.type === 'image' ? null : segment.type === 'session' ? (
          <SessionRefChip key={`m-${index}-${segment.id}`} label={segment.label} value={segment.id} />
        ) : segment.type === 'skill' ? (
          <SlashChip key={`m-${index}-${segment.id}`} kind="skill" label={segment.label} value={segment.id} />
        ) : (
          <DirectiveChip id={segment.id} key={`m-${index}-${segment.id}`} label={segment.label} type={segment.type} />
        )
      )}
      {(imageSegments.length > 0 || images.length > 0) && (
        <span className="mt-2 flex flex-wrap gap-2" data-slot="aui_embedded-images">
          {imageSegments.map((segment, index) => (
            <DirectiveImage id={segment.id} key={`img-ref-${index}-${segment.id}`} label={segment.label} />
          ))}
          {images.map((src, index) => (
            <ZoomableImage
              alt=""
              className="max-h-48 max-w-full rounded-lg border border-(--ui-stroke-tertiary) object-contain"
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
  const isUrl = /^(?:https?|data):/i.test(id)
  const [src, setSrc] = useState<string | null>(isUrl ? id : null)
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    if (isUrl || !id) {
      return
    }

    let alive = true

    // Remote gateway: the image lives on the gateway's disk, not ours — fetch
    // it over the authenticated API. Local: read it straight off this disk.
    const load =
      window.hermesDesktop && isRemoteGateway() ? gatewayMediaDataUrl(id) : window.hermesDesktop?.readFileDataUrl(id)

    void Promise.resolve(load)
      .then(url => alive && url && setSrc(url))
      .catch(() => alive && setFailed(true))

    return () => {
      alive = false
    }
  }, [id, isUrl])

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
      className="max-h-48 max-w-full rounded-lg border border-(--ui-stroke-tertiary) object-contain"
      draggable={false}
      slot="aui_directive-image"
      src={src}
    />
  )
}

/** Opens the referenced session the way a sidebar ⌘-click would: jump to it if
 *  it's already a tile/main, otherwise open a stacked tab (never steals main
 *  from under the chat you're reading). Lazy-imports so the composer's rich
 *  editor can pull this module in without booting the profile/REST stack. */
function openSessionRef(value: string) {
  const { sessionId } = parseSessionRefValue(value)

  if (!sessionId) {
    return
  }

  triggerHaptic('selection')
  // navigate is unused for the `tab` intent (focus-or-tile only).
  void import('@/app/open-session').then(({ openSession }) => openSession(sessionId, () => undefined, 'tab'))
}

/** A `@session:<profile>/<id>` reference in the user transcript (directive
 *  segments), rendered as a chip like the other composer refs. Clicking it
 *  opens the session as a tab. */
export const SessionRefChip: FC<{
  label?: string
  value: string
}> = ({ label, value }) => {
  const resolved = useSessionLinkTitle(value, label)

  return <DirectiveChip id={value} label={resolved} onClick={() => openSessionRef(value)} type="session" />
}

/** A `@session:` reference in assistant markdown (`#session/` links rewritten
 *  in `preprocessMarkdown`). Reads as an ordinary inline link — the agent wrote
 *  it mid-sentence — with the funnel icon leading the resolved title. */
export const SessionRefLink: FC<{
  label?: string
  value: string
}> = ({ label, value }) => {
  const resolved = useSessionLinkTitle(value, label)

  return (
    <a
      {...refAttrs('session', 'wrap-anywhere')}
      href="#"
      onClick={event => {
        event.preventDefault()
        event.stopPropagation()
        openSessionRef(value)
      }}
      title={value}
    >
      <DirectiveIcon type="session" />
      {resolved}
    </a>
  )
}

/** A skill referenced inside a sent message — the rendered twin of the
 *  composer's slash pill, so a picked skill stays a chip after send. */
const SlashChip: FC<{ kind: SlashChipKind; label: string; value: string }> = ({ kind, label, value }) => (
  <span {...refAttrs(kind)} data-slot="aui_slash-chip" title={value}>
    <DirectiveIcon type={kind} />
    {label}
  </span>
)

/** Inert by default; `onClick` promotes the chip to a real button (session
 *  refs, which open the session they name). */
const DirectiveChip: FC<{
  type: string
  label: string
  id: string
  onClick?: () => void
}> = ({ type, label, id, onClick }) => {
  const body = (
    <>
      <DirectiveIcon type={type} />
      {label}
    </>
  )

  const props = {
    ...refAttrs(type, cn('wrap-anywhere', onClick && 'cursor-pointer')),
    'data-directive-id': id,
    'data-slot': 'aui_directive-chip',
    title: id
  }

  return onClick ? (
    <button {...props} onClick={onClick} type="button">
      {body}
    </button>
  ) : (
    <span {...props}>{body}</span>
  )
}
