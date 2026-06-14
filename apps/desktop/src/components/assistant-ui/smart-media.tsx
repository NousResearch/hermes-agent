'use client'

import { type ComponentProps, useState, useCallback } from 'react'

import { Dialog, DialogContent } from '@/components/ui/dialog'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import { mediaKind, mediaName, mediaStreamUrl } from '@/lib/media'

export interface SmartMediaProps {
  src: string
  alt?: string
  className?: string
  containerClassName?: string
}

function LoadingSpinner() {
  return (
    <div className="flex items-center justify-center py-4">
      <div className="h-6 w-6 animate-spin rounded-full border-2 border-border border-t-foreground" />
    </div>
  )
}

function ErrorFallback({ name, onRetry }: { name: string; onRetry: () => void }) {
  const { t } = useI18n()
  const copy = t.desktop
  return (
    <div className="my-2 flex items-center gap-2 rounded-lg border border-border bg-muted/35 p-3 text-sm text-muted-foreground">
      <span>⚠️</span>
      <span className="flex-1 truncate">{name}</span>
      <button
        className="text-xs text-foreground underline underline-offset-2"
        onClick={onRetry}
        type="button"
      >
        {copy.retry || 'Retry'}
      </button>
    </div>
  )
}

export function SmartMedia({ src, alt, className, containerClassName }: SmartMediaProps) {
  const kind = mediaKind(src)
  const name = mediaName(src)
  const [loaded, setLoaded] = useState(false)
  const [failed, setFailed] = useState(false)
  const [lightboxOpen, setLightboxOpen] = useState(false)

  const isVideo = kind === 'video'
  const isImage = kind === 'image'

  const videoSrc = isVideo && !src.startsWith('http') && !src.startsWith('data:')
    ? mediaStreamUrl(src)
    : src

  const handleLoad = useCallback(() => setLoaded(true), [])
  const handleError = useCallback(() => setFailed(true), [])

  if (isImage) {
    return (
      <>
        <span className={cn('block', containerClassName)}>
          <img
            alt={alt || name}
            className={cn(
              'block h-auto w-auto max-h-(--image-preview-height) max-w-[min(100%,var(--image-preview-max-width))] rounded-lg object-contain shadow-[0_0.0625rem_0.125rem_color-mix(in_srgb,#000_4%,transparent),0_0.625rem_1.5rem_color-mix(in_srgb,#000_5%,transparent)]',
              !loaded && 'animate-pulse bg-muted/35',
              className
            )}
            onLoad={handleLoad}
            onError={handleError}
            src={src}
            style={!loaded ? { opacity: 0 } : undefined}
            onClick={() => setLightboxOpen(true)}
          />
        </span>
        <Dialog open={lightboxOpen} onOpenChange={setLightboxOpen}>
          <DialogContent className="max-w-[90vw] max-h-[90vh] bg-transparent p-0 shadow-none">
            <img
              alt={alt || name}
              className="max-h-[85vh] max-w-[90vw] rounded-lg object-contain"
              src={src}
            />
          </DialogContent>
        </Dialog>
      </>
    )
  }

  if (isVideo) {
    return (
      <span className={cn('my-3 block max-w-2xl rounded-xl border border-border bg-muted/35 p-3', containerClassName)}>
        {!loaded && !failed && <LoadingSpinner />}
        {failed ? (
          <ErrorFallback name={name} onRetry={() => { setFailed(false); setLoaded(false) }} />
        ) : (
          <video
            className="block max-h-112 w-full rounded-lg bg-black"
            controls
            onError={handleError}
            preload="metadata"
            src={videoSrc}
            onLoadedData={handleLoad}
          />
        )}
      </span>
    )
  }

  return (
    <a
      className="font-semibold text-foreground underline underline-offset-4 decoration-current/20"
      href={src}
      rel="noopener noreferrer"
      target="_blank"
    >
      {name}
    </a>
  )
}
