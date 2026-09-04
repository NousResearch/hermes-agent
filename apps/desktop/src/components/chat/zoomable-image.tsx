'use client'

import { type ComponentProps, useState } from 'react'

import { RevealInFolderTrigger } from '@/components/chat/reveal-in-folder'
import { Dialog, DialogContent } from '@/components/ui/dialog'
import { useImageDownload } from '@/hooks/use-image-download'
import { useI18n } from '@/i18n'
import { Download } from '@/lib/icons'
import { cn } from '@/lib/utils'

export interface ZoomableImageProps extends ComponentProps<'img'> {
  containerClassName?: string
  slot?: string
  /** Local file path this image was rendered from. When set (and the backend
   *  is local), right-clicking the image offers Reveal-in-file-manager +
   *  Copy Path — the transcript's "where is that file?" door. */
  revealPath?: string
}

export interface ImageActionCopy {
  downloadImage: string
  savingImage: string
}

export function ZoomableImage({
  className,
  containerClassName,
  src,
  alt,
  slot,
  revealPath,
  ...props
}: ZoomableImageProps) {
  const { t } = useI18n()
  const copy = t.desktop
  const { download, saving } = useImageDownload(src)
  const [lightboxOpen, setLightboxOpen] = useState(false)
  const canOpen = Boolean(src)

  const image = (
    <span
      className={cn('group/image relative inline-block max-w-full align-top', containerClassName)}
      data-slot={slot ?? 'aui_zoomable-image'}
    >
      <button
        className="contents"
        disabled={!canOpen}
        onClick={() => canOpen && setLightboxOpen(true)}
        title={canOpen ? copy.openImage : undefined}
        type="button"
      >
        <img alt={alt ?? ''} className={className} src={src} {...props} />
      </button>
      {src && (
        <ImageActionButton className="group-hover/image:opacity-100" copy={copy} onClick={download} saving={saving} />
      )}
    </span>
  )

  const lightbox = src && (
    <ImageLightbox
      alt={alt}
      copy={copy}
      onClick={download}
      onOpenChange={setLightboxOpen}
      open={lightboxOpen}
      saving={saving}
      src={src}
    />
  )

  if (!revealPath) {
    return (
      <>
        {image}
        {lightbox}
      </>
    )
  }

  // `asChild` merges into a single element, so only the visual span goes under
  // the trigger; the lightbox (a portaled dialog) stays a sibling.
  return (
    <>
      <RevealInFolderTrigger path={revealPath}>{image}</RevealInFolderTrigger>
      {lightbox}
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
      title={saving ? copy.savingImage : copy.downloadImage}
      type="button"
    >
      <Download className={cn('size-4', saving && 'animate-pulse')} />
    </button>
  )
}
