'use client'

import { useStore } from '@nanostores/react'
import { type ComponentProps } from 'react'

import { Dialog, DialogContent } from '@/components/ui/dialog'
import { useImageDownload } from '@/hooks/use-image-download'
import { useI18n } from '@/i18n'
import { Download } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { $transcriptLightbox, closeTranscriptLightbox, openTranscriptLightbox } from '@/store/transcript-lightbox'

export interface ZoomableImageProps extends ComponentProps<'img'> {
  containerClassName?: string
  slot?: string
  /** Full-resolution source for the lightbox and download. When set, the inline
   * `<img>` keeps its bounded `src` (avoiding heavy inline paints) while the
   * zoom view and Save action use the original — so a click-to-zoom shows real
   * detail instead of an upscaled thumbnail. Defaults to `src`. */
  zoomSrc?: string
}

export interface ImageActionCopy {
  downloadImage: string
  savingImage: string
}

export function ZoomableImage({
  className,
  containerClassName,
  src,
  zoomSrc,
  alt,
  slot,
  ...props
}: ZoomableImageProps) {
  const { t } = useI18n()
  const copy = t.desktop
  // The lightbox and Save action prefer the full-resolution source; the inline
  // thumbnail (`src`) stays the cheap paint.
  const fullSrc = zoomSrc || src || ''
  const { download, saving } = useImageDownload(fullSrc)
  // The open flag lives in a store keyed by source identity, not local state:
  // transcript rows (and the markdown leaves inside them) remount routinely
  // while a turn streams — the render-budget slice can drop the row, and the
  // streaming re-parse re-mounts AST leaves — which used to close a
  // user-opened lightbox with no gesture (#123018). A remounted row reads the
  // same store and re-presents its own open dialog.
  const openSrc = useStore($transcriptLightbox)
  const lightboxOpen = openSrc !== null && openSrc === fullSrc
  const canOpen = Boolean(src)

  return (
    <>
      <span
        className={cn('group/image relative inline-block max-w-full align-top', containerClassName)}
        data-slot={slot ?? 'aui_zoomable-image'}
      >
        <button
          aria-label={canOpen ? copy.openImage : undefined}
          className="contents"
          disabled={!canOpen}
          onClick={() => canOpen && openTranscriptLightbox(fullSrc)}
          type="button"
        >
          <img alt={alt ?? ''} className={className} src={src} {...props} />
        </button>
        {src && (
          <ImageActionButton className="group-hover/image:opacity-100" copy={copy} onClick={download} saving={saving} />
        )}
      </span>
      {src && (
        <ImageLightbox
          alt={alt}
          copy={copy}
          onClick={download}
          onOpenChange={open => (open ? openTranscriptLightbox(fullSrc) : closeTranscriptLightbox(fullSrc))}
          open={lightboxOpen}
          saving={saving}
          src={fullSrc}
        />
      )}
    </>
  )
}

export function ImageLightbox({
  alt,
  copy,
  onClick,
  onOpenChange,
  open,
  saving,
  src
}: {
  alt?: string
  copy: ImageActionCopy
  onClick: () => void
  onOpenChange: (open: boolean) => void
  open: boolean
  saving: boolean
  src: string
}) {
  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent
        bodyClassName="block overflow-visible p-0"
        className="w-auto max-h-[calc(100vh-12rem)] max-w-[calc(100vw-12rem)] border-0 bg-transparent shadow-none"
        overlayClassName="bg-black/60"
        showCloseButton={false}
      >
        <div className="group/lightbox relative inline-block">
          <img
            alt={alt ?? ''}
            className="block max-h-[calc(100vh-12rem)] max-w-[calc(100vw-12rem)] cursor-zoom-out select-auto rounded-lg object-contain shadow-2xl"
            onClick={() => onOpenChange(false)}
            src={src}
          />
          <ImageActionButton
            className="group-hover/lightbox:opacity-100"
            copy={copy}
            onClick={onClick}
            saving={saving}
          />
        </div>
      </DialogContent>
    </Dialog>
  )
}

export function ImageActionButton({
  className,
  copy,
  onClick,
  saving
}: {
  className?: string
  copy: ImageActionCopy
  onClick: () => void
  saving: boolean
}) {
  return (
    <button
      aria-label={saving ? copy.savingImage : copy.downloadImage}
      className={cn(
        'absolute right-2 top-2 grid size-8 place-items-center rounded-full border border-border/70 bg-background/80 text-muted-foreground opacity-0 shadow-sm backdrop-blur transition-opacity hover:bg-accent hover:text-foreground focus-visible:opacity-100 disabled:opacity-50',
        className
      )}
      disabled={saving}
      onClick={event => {
        event.stopPropagation()
        void onClick()
      }}
      type="button"
    >
      <Download className={cn('size-4', saving && 'animate-pulse')} />
    </button>
  )
}
