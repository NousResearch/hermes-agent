import type { ComponentProps, ReactNode } from 'react'
import { useEffect, useMemo, useState } from 'react'

import { ArrowUpRight } from '@/lib/icons'
import { IS_MAC } from '@/lib/keybinds/combo'

import { resolveBrandIcon } from './brand-icon'
import { cn } from './utils'

const titleCache = new Map<string, string>()
const titleInflight = new Map<string, Promise<string>>()
const titleSubs = new Map<string, Set<(value: string) => void>>()

const URL_RE =
  /(?:https?:\/\/|www\.)[^\s<>"'`]+[^\s<>"'`.,;:!?)]|[a-z0-9](?:[a-z0-9-]*\.)+[a-z]{2,}(?:\/[^\s<>"'`.,;:!?)]*)?/gi

// Explicit-scheme / www. URLs only — no bare-domain matching. Used where the
// surrounding text is full of filename-shaped tokens (e.g. `agent.log`,
// `errors.log` in a /debug report) that the bare-domain branch of URL_RE would
// otherwise mistake for domains and linkify.
const EXPLICIT_URL_RE = /(?:https?:\/\/|www\.)[^\s<>"'`]+[^\s<>"'`.,;:!?)]/gi

const DOMAIN_RE = /^(?:www\.)?[a-z0-9](?:[a-z0-9-]*\.)+[a-z]{2,}(?::\d+)?(?:[/?#][^\s]*)?$/i
const SKIP_PROTO_RE = /^(?:file|data|mailto|javascript|blob|chrome|about|hermes):/i
const LOCAL_HOST_RE = /^(?:localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\])(?::\d+)?$/i

const ERROR_TITLE_RE =
  /\b(?:access denied|attention required|captcha|error|forbidden|just a moment|not found|request blocked|too many requests)\b/i

export function normalizeExternalUrl(value: string): string {
  const trimmed = value.trim()

  if (!trimmed || /^https?:\/\//i.test(trimmed)) {
    return trimmed
  }

  return DOMAIN_RE.test(trimmed) ? `https://${trimmed}` : trimmed
}

function parseUrl(value: string): null | URL {
  try {
    return new URL(normalizeExternalUrl(value))
  } catch {
    return null
  }
}

function titleCacheKey(value: string): string {
  const url = parseUrl(value)

  if (!url) {
    return normalizeExternalUrl(value)
  }

  const host = url.hostname.replace(/^www\./i, '').toLowerCase()
  const pathname = url.pathname === '/' ? '/' : url.pathname.replace(/\/+$/, '') || '/'

  return `${host}${pathname}${url.search || ''}`
}

export function shortHostLabel(value: string): string {
  return parseUrl(value)?.hostname.replace(/^www\./, '') ?? value
}

export function hostPathLabel(value: string): string {
  const url = parseUrl(value)

  if (!url) {
    return value
  }

  const host = url.hostname.replace(/^www\./, '')
  const path = url.pathname && url.pathname !== '/' ? url.pathname.replace(/\/$/, '') : ''

  return `${host}${path}`
}

function cleanSlug(segment: string): string {
  try {
    return decodeURIComponent(segment)
      .replace(/\.a\d+\..*$/i, '')
      .replace(/\.(?:html?|php|aspx?)$/i, '')
      .replace(/(?:[-_.](?:[a-z]{1,3}\d{2,}|i\d{2,}))+$/i, '')
      .replace(/[_-]+/g, ' ')
      .replace(/\s+/g, ' ')
      .trim()
  } catch {
    return ''
  }
}

export function urlSlugTitleLabel(value: string): string {
  const url = parseUrl(value)

  for (const segment of url?.pathname.split('/').filter(Boolean).reverse() ?? []) {
    const cleaned = cleanSlug(segment)

    if (!cleaned || !/[a-z]/i.test(cleaned)) {
      continue
    }

    if (/^(?:[a-z]{1,3}\d+|\d+)$/i.test(cleaned.replace(/\s+/g, ''))) {
      continue
    }

    const titled = cleaned.replace(/\b[a-z]/g, c => c.toUpperCase())

    if (titled.length >= 4) {
      return titled
    }
  }

  return hostPathLabel(value)
}

export function isTitleFetchable(value: string): boolean {
  if (!value || SKIP_PROTO_RE.test(value)) {
    return false
  }

  const url = parseUrl(value)

  return Boolean(url && /^https?:$/.test(url.protocol) && !LOCAL_HOST_RE.test(url.host))
}

export function fetchLinkTitle(url: string): Promise<string> {
  const normalizedUrl = normalizeExternalUrl(url)
  const key = titleCacheKey(normalizedUrl)

  if (!isTitleFetchable(normalizedUrl)) {
    return Promise.resolve('')
  }

  if (titleCache.has(key)) {
    return Promise.resolve(titleCache.get(key) ?? '')
  }

  const pending = titleInflight.get(key)

  if (pending) {
    return pending
  }

  const bridge = typeof window === 'undefined' ? undefined : window.hermesDesktop?.fetchLinkTitle

  if (!bridge) {
    titleCache.set(key, '')

    return Promise.resolve('')
  }

  const promise = bridge(normalizedUrl)
    .then(value => (value || '').replace(/\s+/g, ' ').trim())
    .then(clean => (clean && !ERROR_TITLE_RE.test(clean) ? clean : ''))
    .catch(() => '')
    .then(safe => {
      titleCache.set(key, safe)
      titleInflight.delete(key)
      titleSubs.get(key)?.forEach(sub => sub(safe))

      return safe
    })

  titleInflight.set(key, promise)

  return promise
}

export function useLinkTitle(url?: null | string): string {
  const normalizedUrl = useMemo(() => (url ? normalizeExternalUrl(url) : ''), [url])
  const key = useMemo(() => (normalizedUrl ? titleCacheKey(normalizedUrl) : ''), [normalizedUrl])
  const [title, setTitle] = useState(() => (key ? (titleCache.get(key) ?? '') : ''))

  useEffect(() => {
    setTitle(key ? (titleCache.get(key) ?? '') : '')

    if (!key || !isTitleFetchable(normalizedUrl)) {
      return
    }

    const subs = titleSubs.get(key) ?? new Set<(value: string) => void>()

    subs.add(setTitle)
    titleSubs.set(key, subs)
    void fetchLinkTitle(normalizedUrl)

    return () => {
      subs.delete(setTitle)

      if (!subs.size) {
        titleSubs.delete(key)
      }
    }
  }, [key, normalizedUrl])

  return title
}

export function openExternalLink(href: string): void {
  if (href) {
    void window.hermesDesktop?.openExternal?.(href)
  }
}

/**
 * True when a click asked for the SYSTEM browser — ⌘ on macOS, Ctrl elsewhere,
 * the modifier every app uses for "open this somewhere else". Middle-click
 * counts too: it is the other half of the same convention.
 */
export function wantsNativeBrowser(event: Pick<MouseEvent, 'button' | 'ctrlKey' | 'metaKey'>): boolean {
  return event.button === 1 || (IS_MAC ? event.metaKey : event.ctrlKey)
}

/**
 * The HUD is a chrome-free bar with no in-app browser. A preview tile there
 * either no-ops or tries to paint a webview into the transparent overlay —
 * the OAuth-in-the-HUD case. Always hand off to the OS browser.
 */
export function hudForcesNativeLinks(
  search = typeof window === 'undefined' ? '' : window.location.search
): boolean {
  try {
    return new URLSearchParams(search).get('win') === 'hud'
  } catch {
    return false
  }
}

/**
 * Where a link the user clicked should open.
 *
 * A web page opens in the in-app browser — that pane exists so reading a doc
 * doesn't cost a context switch out of Hermes, and it is the surface the agent
 * can see. ⌘/Ctrl-click (or middle-click) escapes to the real browser, which is
 * where you go for anything needing your logged-in session or a password.
 *
 * Everything that ISN'T a web page — `mailto:`, `file:`, a custom scheme — has
 * no business in the webview and always hands off to the OS. The HUD has no
 * browser pane, so it always takes the OS path.
 */
export function openLink(href: string, options: { native?: boolean } = {}): void {
  const target = normalizeExternalUrl(href)

  if (!target) {
    return
  }

  if (
    options.native ||
    hudForcesNativeLinks() ||
    !/^https?:$/i.test(parseUrl(target)?.protocol ?? '')
  ) {
    openExternalLink(target)

    return
  }

  // Lazy: this module is a leaf every surface imports, and the preview store
  // pulls the layout/session graph behind it. A static edge would make one
  // link helper drag that whole tree into anything that renders a link. The
  // tab lands a microtask later, which is invisible.
  void import('@/store/preview').then(({ openPreview }) =>
    openPreview({ kind: 'url', label: hostPathLabel(target), source: target, url: target }, 'explicit-link')
  )
}

interface ExternalLinkProps extends Omit<ComponentProps<'a'>, 'href' | 'target'> {
  href: string
  children?: ReactNode
  /** Skip the in-app pane. For links whose whole point is the session you are
   *  signed into over there — a cloud console, an account page. */
  native?: boolean
  showExternalIcon?: boolean
}

export function ExternalLinkIcon({ className }: { className?: string }) {
  return <ArrowUpRight aria-hidden className={cn('ml-1 inline size-[0.78em] align-[-0.08em] opacity-70', className)} />
}

// Brand mark for a known host, sized in `em` so it tracks the surrounding text
// at any font size. It paints in `currentColor` rather than the brand hex —
// several brand colors (GitHub's near-black, Unity's white) vanish against one
// theme or the other.
//
// `title=""` is load-bearing: Simple Icons always renders a <title> defaulting
// to the brand name, which lands in the anchor's textContent and accessible
// name — a PR link would read "GitHub#123".
export function LinkBrandIcon({ className, href }: { className?: string; href: string }) {
  const Icon = resolveBrandIcon(shortHostLabel(href))

  return Icon ? (
    <Icon aria-hidden className={cn('mr-1 inline size-[0.85em] align-[-0.12em] opacity-80', className)} title="" />
  ) : null
}

export function ExternalLink({
  children,
  className,
  href,
  native = false,
  onClick,
  showExternalIcon = false,
  ...rest
}: ExternalLinkProps) {
  const target = normalizeExternalUrl(href)

  // No menu wiring here: the app context-menu coordinator resolves a
  // right-click on any `a[href]` to the link menu (open in-app / open
  // external / copy URL / copy resolved URL).
  return (
    <a
      className={cn('ref', className)}
      href={target}
      // Middle-click never fires `click`; it's the other half of the
      // open-elsewhere convention, so it has to be caught on its own.
      onAuxClick={event => {
        if (event.button !== 1) {
          return
        }

        event.preventDefault()
        event.stopPropagation()
        openExternalLink(target)
      }}
      onClick={event => {
        event.stopPropagation()
        onClick?.(event)

        if (event.defaultPrevented) {
          return
        }

        event.preventDefault()
        openLink(target, { native: native || wantsNativeBrowser(event.nativeEvent) })
      }}
      rel="noopener noreferrer"
      target="_blank"
      {...rest}
    >
      {children ?? urlSlugTitleLabel(target)}
      {showExternalIcon && <ExternalLinkIcon />}
    </a>
  )
}

interface PrettyLinkProps extends Omit<ComponentProps<'a'>, 'href' | 'target'> {
  href: string
  label?: string
  fallbackLabel?: string
}

// Title resolution is a fallback, not an override. Both props carry authored
// text — chat markdown passes `fallbackLabel` — so either one skips the fetch.
export function PrettyLink({ className, fallbackLabel, href, label, ...rest }: PrettyLinkProps) {
  const target = useMemo(() => normalizeExternalUrl(href), [href])
  const authoredLabel = label?.trim() || fallbackLabel?.trim()
  const fetched = useLinkTitle(authoredLabel ? null : target)
  const display = authoredLabel || fetched || urlSlugTitleLabel(target)

  return (
    <ExternalLink className={cn('wrap-break-word', className)} href={target} title={target} {...rest}>
      <LinkBrandIcon href={target} />
      {display}
    </ExternalLink>
  )
}

interface LinkifiedTextProps {
  className?: string
  text: string
  pretty?: boolean
  explicitOnly?: boolean
}

export function LinkifiedText({ className, explicitOnly = false, pretty = true, text }: LinkifiedTextProps) {
  const nodes: ReactNode[] = []
  let cursor = 0

  for (const match of text.matchAll(explicitOnly ? EXPLICIT_URL_RE : URL_RE)) {
    const raw = match[0]
    const url = normalizeExternalUrl(raw)
    const index = match.index ?? 0

    if (index > cursor) {
      nodes.push(text.slice(cursor, index))
    }

    nodes.push(
      pretty ? (
        <PrettyLink href={url} key={`${url}-${index}`} />
      ) : (
        <ExternalLink href={url} key={`${url}-${index}`}>
          {raw}
        </ExternalLink>
      )
    )

    cursor = index + raw.length
  }

  if (cursor < text.length) {
    nodes.push(text.slice(cursor))
  }

  return <span className={className}>{nodes.length ? nodes : text}</span>
}

const MD_LINK_RE = /\[([^\]]+)]\((https?:\/\/[^\s)]+)\)/g

/**
 * Inline `[label](url)` and nothing else.
 *
 * For short authored strings — a catalog entry's setup steps — where the
 * label carries the meaning ("enable the Docs API") and the URL is a console
 * page whose own title is useless or, behind a login wall, actively wrong.
 * `LinkifiedText` can't serve this: it finds bare URLs and guesses a label.
 * Full markdown is the other extreme, a block renderer inside a card row.
 *
 * These open in the real browser. The destination is a console the user is
 * already signed into there, and the work is a form to fill in and a secret to
 * copy back — none of which the in-app pane is for.
 */
export function MarkdownLinkText({ className, text }: { className?: string; text: string }) {
  const nodes: ReactNode[] = []
  let cursor = 0

  for (const match of text.matchAll(MD_LINK_RE)) {
    const [raw, label, href] = match
    const index = match.index ?? 0

    if (index > cursor) {
      nodes.push(text.slice(cursor, index))
    }

    nodes.push(
      <ExternalLink href={href} key={`${href}-${index}`} native title={href}>
        {label}
      </ExternalLink>
    )

    cursor = index + raw.length
  }

  if (cursor < text.length) {
    nodes.push(text.slice(cursor))
  }

  return <span className={className}>{nodes.length ? nodes : text}</span>
}

export function __resetLinkTitleCache(): void {
  titleCache.clear()
  titleInflight.clear()
  titleSubs.clear()
}
