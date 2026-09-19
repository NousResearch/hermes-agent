'use client'

import { type ComponentProps, type ReactNode, useState } from 'react'

import { Dialog, DialogContent } from '@/components/ui/dialog'
import { useImageDownload } from '@/hooks/use-image-download'
import { useImageZoom } from '@/hooks/use-image-zoom'
import { useI18n } from '@/i18n'
import { Download, Maximize, ZoomIn, ZoomOut } from '@/lib/icons'
import { isVectorSource } from '@/lib/image-zoom'
import { cn } from '@/lib/utils'

export interface ZoomableImageProps extends ComponentProps<'img'> {
  containerClassName?: string
  slot?: string
}

export interface ImageActionCopy {
  downloadImage: string
  resetZoom: string
  savingImage: string
  zoomIn: string
  zoomOut: string
}

export function ZoomableImage({ className, containerClassName, src, alt, slot, ...props }: ZoomableImageProps) {
  const { t } = useI18n()
  const copy = t.desktop
  const { download, saving } = useImageDownload(src)
  const [lightboxOpen, setLightboxOpen] = useState(false)
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
          onClick={() => canOpen && setLightboxOpen(true)}
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
          onOpenChange={setLightboxOpen}
          open={lightboxOpen}
          saving={saving}
          src={src}
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
  const zoom = useImageZoom(open, isVectorSource(src), true)

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent
        bodyClassName="block overflow-visible p-0"
        className="w-auto max-h-[calc(100vh-12rem)] max-w-[calc(100vw-12rem)] border-0 bg-transparent shadow-none"
        showCloseButton={false}
      >
        {/* eslint-disable-next-line jsx-a11y/no-static-element-interactions -- wheel zoom needs the whole surface; keyboard zoom is bound in the hook */}
        <div className="group/lightbox relative inline-block overflow-hidden rounded-lg" onWheel={zoom.onWheel}>
          <img
            alt={alt ?? ''}
            className={cn(
              'block max-h-[calc(100vh-12rem)] max-w-[calc(100vw-12rem)] select-auto rounded-lg object-contain shadow-2xl',
              zoom.isZoomed ? 'cursor-grab active:cursor-grabbing' : 'cursor-zoom-out'
            )}
            draggable={false}
            onClick={() => {
              // A drag that ends over the image must not read as a click-to-close.
              if (!zoom.isPanning() && !zoom.isZoomed) onOpenChange(false)
            }}
            onDoubleClick={zoom.onDoubleClick}
            onPointerCancel={zoom.endDrag}
            onPointerDown={zoom.onPointerDown}
            onPointerMove={zoom.onPointerMove}
            onPointerUp={zoom.endDrag}
            ref={zoom.imageRef}
            src={src}
            style={{ transformOrigin: 'center center', willChange: 'transform' }}
          />
          <ImageActionButton
            className="group-hover/lightbox:opacity-100"
            copy={copy}
            onClick={onClick}
            saving={saving}
          />
          <ImageZoomControls className="group-hover/lightbox:opacity-100" copy={copy} zoom={zoom} />
        </div>
      </DialogContent>
    </Dialog>
  )
}

function ImageZoomControls({
  className,
  copy,
  zoom
}: {
  className?: string
  copy: ImageActionCopy
  zoom: ReturnType<typeof useImageZoom>
}) {
  return (
    <div
      className={cn(
        'absolute bottom-2 left-1/2 flex -translate-x-1/2 items-center gap-1 rounded-full border border-border/70 bg-background/80 p-1 opacity-0 shadow-sm backdrop-blur transition-opacity focus-within:opacity-100',
        className
      )}
    >
      <ZoomControlButton
        disabled={!zoom.canZoomOut}
        icon={<ZoomOut className="size-4" />}
        label={copy.zoomOut}
        onClick={zoom.zoomOut}
      />
      <span className="min-w-12 text-center text-xs tabular-nums text-muted-foreground">
        {Math.round(zoom.scale * 100)}%
      </span>
      <ZoomControlButton
        disabled={!zoom.canZoomIn}
        icon={<ZoomIn className="size-4" />}
        label={copy.zoomIn}
        onClick={zoom.zoomIn}
      />
      <ZoomControlButton
        disabled={!zoom.isZoomed}
        icon={<Maximize className="size-4" />}
        label={copy.resetZoom}
        onClick={zoom.reset}
      />
    </div>
  )
}

function ZoomControlButton({
  disabled,
  icon,
  label,
  onClick
}: {
  disabled: boolean
  icon: ReactNode
  label: string
  onClick: () => void
}) {
  return (
    <button
      aria-label={label}
      className="grid size-8 place-items-center rounded-full text-muted-foreground transition-colors hover:bg-accent hover:text-foreground disabled:opacity-40 disabled:hover:bg-transparent"
      disabled={disabled}
      onClick={event => {
        event.stopPropagation()
        onClick()
      }}
      title={label}
      type="button"
    >
      {icon}
    </button>
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
