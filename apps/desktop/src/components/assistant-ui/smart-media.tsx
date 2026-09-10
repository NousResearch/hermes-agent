'use client'

import { type ComponentProps, type ReactNode, useCallback, useState } from 'react'

import { ZoomableImage } from '@/components/chat/zoomable-image'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'

export interface SmartMediaProps extends Omit<ComponentProps<'img'>, 'src' | 'alt' | 'className'> {
  /** Already-resolved playback/display URL (http(s)/data/blob/hermes-media). */
  src: string
  alt?: string
  className?: string
  containerClassName?: string
  /** Explicit kind — required when `src` is a data/blob URL that has no extension. */
  kind: 'image' | 'video'
  /** Human label for error/retry chrome. */
  name?: string
  /** Optional ZoomableImage slot for images. */
  slot?: string
  /** Optional external-open affordance shown after a hard failure (video). */
  onOpenExternal?: () => void
  openFailedNote?: ReactNode
}

function LoadingSpinner() {
  return (
    <div className="flex items-center justify-center py-4">
      <div className="h-6 w-6 animate-spin rounded-full border-2 border-border border-t-foreground" />
    </div>
  )
}

function ErrorFallback({
  name,
  onRetry,
  onOpenExternal,
  openFailedNote
}: {
  name: string
  onRetry: () => void
  onOpenExternal?: () => void
  openFailedNote?: ReactNode
}) {
  const { t } = useI18n()
  const copy = t.desktop

  return (
    <div className="my-2 flex flex-col gap-1 rounded-lg border border-(--ui-stroke-tertiary) bg-muted/35 p-3 text-sm text-muted-foreground">
      <div className="flex items-center gap-2">
        <span aria-hidden>⚠️</span>
        <span className="min-w-0 flex-1 truncate">{copy.loadFailed ? copy.loadFailed(name) : `Couldn't load ${name}`}</span>
        <button
          className="shrink-0 text-xs font-medium text-foreground underline underline-offset-2"
          onClick={onRetry}
          type="button"
        >
          {copy.resumeRetry || 'Retry'}
        </button>
        {onOpenExternal ? (
          <button
            className="shrink-0 text-xs font-medium text-foreground underline underline-offset-2"
            onClick={onOpenExternal}
            type="button"
          >
            Open
          </button>
        ) : null}
      </div>
      {openFailedNote}
    </div>
  )
}

/**
 * Shared presentation layer for already-resolved image/video sources.
 *
 * Source resolution (`#media:`, remote gateway, hermes-media streaming) stays in
 * `MediaAttachment` / `resolveMedia*`. This component only handles load/error/
 * retry chrome and image lightbox (via ZoomableImage).
 */
export function SmartMedia({
  src,
  alt,
  className,
  containerClassName,
  kind,
  name,
  slot,
  onOpenExternal,
  openFailedNote,
  style,
  ...imgProps
}: SmartMediaProps) {
  const label = name || alt || 'media'
  const [loaded, setLoaded] = useState(false)
  const [failed, setFailed] = useState(false)
  // Bump to force <img>/<video> remount on retry so the browser re-fetches.
  const [retryKey, setRetryKey] = useState(0)

  const handleLoad = useCallback(() => {
    setLoaded(true)
    setFailed(false)
  }, [])

  const handleError = useCallback(() => {
    setFailed(true)
    setLoaded(false)
  }, [])

  const handleRetry = useCallback(() => {
    setFailed(false)
    setLoaded(false)
    setRetryKey(key => key + 1)
  }, [])

  if (kind === 'image') {
    if (failed) {
      return (
        <span className={cn('block', containerClassName)}>
          <ErrorFallback
            name={label}
            onOpenExternal={onOpenExternal}
            onRetry={handleRetry}
            openFailedNote={openFailedNote}
          />
        </span>
      )
    }

    const pendingStyle =
      !loaded && style && typeof style === 'object' && !Array.isArray(style)
        ? { ...style, opacity: 0 }
        : !loaded
          ? { opacity: 0 }
          : style

    return (
      <ZoomableImage
        key={retryKey}
        alt={alt}
        className={cn(!loaded && 'animate-pulse bg-muted/35', className)}
        containerClassName={containerClassName}
        slot={slot}
        src={src}
        style={pendingStyle}
        {...imgProps}
        // Keep load/error ownership after imgProps so callers cannot drop retry chrome.
        onError={handleError}
        onLoad={handleLoad}
      />
    )
  }

  return (
    <span
      className={cn(
        'my-3 block max-w-2xl rounded-xl border border-(--ui-stroke-tertiary) bg-muted/35 p-3',
        containerClassName
      )}
    >
      <span className="mb-2 block truncate text-xs font-medium text-muted-foreground">{label}</span>
      {!loaded && !failed ? <LoadingSpinner /> : null}
      {failed ? (
        <ErrorFallback
          name={label}
          onOpenExternal={onOpenExternal}
          onRetry={handleRetry}
          openFailedNote={openFailedNote}
        />
      ) : (
        <video
          key={retryKey}
          className={cn('block max-h-112 w-full rounded-lg bg-black', !loaded && 'hidden')}
          controls
          onError={handleError}
          onLoadedData={handleLoad}
          preload="metadata"
          src={src}
        />
      )}
    </span>
  )
}
