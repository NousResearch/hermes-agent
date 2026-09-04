'use client'

import { type ComponentProps, useState } from 'react'

import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n/context'
import { hostPathLabel } from '@/lib/external-link'
import { cn } from '@/lib/utils'

/**
 * Link preview metadata as resolved by the desktop main process
 * (electron/link-preview.ts via the `fetchLinkPreview` bridge).
 */
export interface LinkPreviewMeta {
  url: string
  title: string
  description: string
  imageUrl: string
  /** Thumbnail as a main-process-validated data URL ('' when unprovable). */
  image: string
  fetchedAt: number
}

export type LinkPreviewFailureReason = 'private-url' | 'error'

type LoadState =
  | { phase: 'idle' }
  | { phase: 'loading' }
  | { phase: 'loaded'; meta: LinkPreviewMeta }
  | { phase: 'failed'; reason: LinkPreviewFailureReason }

/**
 * Click-to-expand link preview (D7).
 *
 * Collapsed, this is a compact chip labeled with the host — nothing has been
 * fetched, and the chip is purely a local affordance. Clicking it performs
 * the ONE user-initiated main-process fetch and expands into the preview
 * card (thumbnail, title, description). Loading is one-shot and sticky for
 * the life of the mounted transcript: re-renders never re-fetch, and every
 * terminal state (loaded / failed) renders honest content — no silent legs.
 */
export function LinkPreviewChipCard({ href, className, ...props }: ComponentProps<'span'> & { href: string }) {
  const { t } = useI18n()
  const [state, setState] = useState<LoadState>({ phase: 'idle' })

  const load = () => {
    if (state.phase === 'loading' || state.phase === 'loaded') {
      return
    }

    setState({ phase: 'loading' })

    const bridge = window.hermesDesktop?.fetchLinkPreview

    if (!bridge) {
      // No desktop bridge (tests, future web target): nothing to fetch with.
      setState({ phase: 'failed', reason: 'error' })

      return
    }

    bridge(href)
      .then(result => {
        if (result.ok) {
          setState({ phase: 'loaded', meta: result.meta })
        } else {
          setState({ phase: 'failed', reason: result.reason })
        }
      })
      .catch(() => setState({ phase: 'failed', reason: 'error' }))
  }

  if (state.phase === 'idle') {
    return (
      <Tip label={href}>
        <button
          className={cn(
            'inline-flex max-w-full items-center gap-1.5 rounded-full border border-(--ui-stroke-tertiary) bg-muted/35 px-2.5 py-1 text-xs text-muted-foreground transition-colors hover:border-(--ui-stroke-secondary) hover:text-foreground',
            className
          )}
          data-link-preview="chip"
          onClick={load}
          type="button"
        >
          <span className="truncate">{hostPathLabel(href)}</span>
          <span className="shrink-0 font-medium">{t.assistant.thread.linkPreviewLoad}</span>
        </button>
      </Tip>
    )
  }

  const header = state.phase === 'loaded' ? state.meta.title || hostPathLabel(href) : hostPathLabel(href)

  return (
    <span
      className={cn('my-3 block max-w-xl rounded-xl border border-(--ui-stroke-tertiary) bg-muted/35 p-3 text-sm', className)}
      data-link-preview="card"
      {...props}
    >
      <a className="ref wrap-anywhere font-medium text-foreground" href={href} rel="noopener noreferrer" target="_blank">
        {header}
      </a>
      {state.phase === 'loading' && (
        <span className="mt-2 block text-xs text-muted-foreground">{t.assistant.thread.linkPreviewLoading}</span>
      )}
      {state.phase === 'loaded' && <LinkPreviewBody meta={state.meta} />}
      {state.phase === 'failed' && (
        <span className="mt-2 block text-xs text-muted-foreground">
          {state.reason === 'private-url' ? t.assistant.thread.linkPreviewPrivate : t.assistant.thread.linkPreviewUnavailable}
        </span>
      )}
    </span>
  )
}

function LinkPreviewBody({ meta }: { meta: LinkPreviewMeta }) {
  const { t } = useI18n()

  return (
    <span className="mt-2 block">
      {/*
        The thumbnail is a data URL fetched and validated by the main process
        (same per-hop SSRF admission as the page itself). The renderer NEVER
        GETs meta.imageUrl — an <img src> to a private/redirecting address
        would bypass every guard the main process applies.
      */}
      {meta.image && (
        <img
          alt=""
          className="mb-2 max-h-48 rounded-lg border border-(--ui-stroke-tertiary) object-cover"
          loading="lazy"
          src={meta.image}
        />
      )}
      {meta.title && <span className="mb-0.5 block font-medium text-foreground">{meta.title}</span>}
      {meta.description && <span className="block text-xs leading-5 text-muted-foreground">{meta.description}</span>}
      {!meta.title && !meta.description && !meta.image && (
        <span className="block text-xs text-muted-foreground">{t.assistant.thread.linkPreviewUnavailable}</span>
      )}
    </span>
  )
}
