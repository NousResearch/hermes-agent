import type { ComponentProps, ReactNode } from 'react'
import { useEffect, useMemo, useState } from 'react'

import { ArrowUpRight } from '@/lib/icons'
import { openWebLink } from '@/store/browser'

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
  /\b(?:access denied|attention required|captcha|error|forbidden|just a moment|request blocked|too many requests)\b/i

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

interface ExternalLinkProps extends Omit<ComponentProps<'a'>, 'href' | 'target'> {
  href: string
  children?: ReactNode
  showExternalIcon?: boolean
  // When true, a plain click opens the URL in the in-app browser panel (fast,
  // on-the-fly preview); ⌘/Ctrl-click or middle-click still routes to the system
  // browser. Used for chat-transcript links. Default false = always system.
  openInApp?: boolean
}

export function ExternalLinkIcon({ className }: { className?: string }) {
  return <ArrowUpRight aria-hidden className={cn('ml-1 inline size-[0.78em] align-[-0.08em] opacity-70', className)} />
}

export function ExternalLink({
  children,
  className,
  href,
  onClick,
  openInApp = false,
  showExternalIcon = true,
  ...rest
}: ExternalLinkProps) {
  const target = normalizeExternalUrl(href)

  return (
    <a
      className={cn('font-semibold text-foreground underline underline-offset-4 decoration-current/20', className)}
      href={target}
      onAuxClick={
        openInApp
          ? event => {
              if (event.button === 1) {
                event.preventDefault()
                event.stopPropagation()
                openWebLink(target, { button: 1 })
              }
            }
          : undefined
      }
      onClick={event => {
        event.stopPropagation()
        onClick?.(event)

        if (event.defaultPrevented) {
          return
        }

        event.preventDefault()

        if (openInApp) {
          openWebLink(target, { metaKey: event.metaKey, ctrlKey: event.ctrlKey })
        } else {
          openExternalLink(target)
        }
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
  // See ExternalLink.openInApp.
  openInApp?: boolean
}

export function PrettyLink({ className, fallbackLabel, href, label, openInApp, ...rest }: PrettyLinkProps) {
  const target = useMemo(() => normalizeExternalUrl(href), [href])
  const fetched = useLinkTitle(label ? null : target)
  const display = fetched || label?.trim() || fallbackLabel?.trim() || urlSlugTitleLabel(target)

  return (
    <ExternalLink className={cn('wrap-break-word', className)} href={target} openInApp={openInApp} title={target} {...rest}>
      <span className="font-medium">{display}</span>
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

export function __resetLinkTitleCache(): void {
  titleCache.clear()
  titleInflight.clear()
  titleSubs.clear()
}
