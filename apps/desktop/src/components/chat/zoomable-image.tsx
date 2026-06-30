'use client'

import { type ComponentProps, useState } from 'react'

import { Dialog, DialogContent } from '@/components/ui/dialog'
import { useImageCopy } from '@/hooks/use-image-copy'
import { useImageDownload } from '@/hooks/use-image-download'
import { useI18n } from '@/i18n'
import { Copy, Download, type IconComponent } from '@/lib/icons'
import { cn } from '@/lib/utils'

export interface ZoomableImageProps extends ComponentProps<'img'> {
  containerClassName?: string
  slot?: string
}

export interface ImageActionCopy {
  copyingImage: string
  copyImage: string
  downloadImage: string
  savingImage: string
}

export function ZoomableImage({ className, containerClassName, src, alt, slot, ...props }: ZoomableImageProps) {
  const { t } = useI18n()
  const copy = t.desktop
  const { copyImage, copying } = useImageCopy(src)
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
          className="contents"
          disabled={!canOpen}
          onClick={() => canOpen && setLightboxOpen(true)}
          title={canOpen ? copy.openImage : undefined}
          type="button"
        >
          <img alt={alt ?? ''} className={className} src={src} {...props} />
        </button>
        {src && (
          <ImageActionButtons
            className="group-hover/image:opacity-100"
            copy={copy}
            copying={copying}
            onCopy={copyImage}
            onDownload={download}
            saving={saving}
          />
        )}
      </span>
      {src && (
        <ImageLightbox
          alt={alt}
          copy={copy}
          copying={copying}
          onCopy={copyImage}
          onDownload={download}
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
  copying,
  onCopy,
  onDownload,
  onOpenChange,
  open,
  saving,
  src
}: {
  alt?: string
  copy: ImageActionCopy
  copying: boolean
  onCopy: () => void
  onDownload: () => void
  onOpenChange: (open: boolean) => void
  open: boolean
  saving: boolean
  src: string
}) {
  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent
        className="block w-auto max-h-[calc(100vh-12rem)] max-w-[calc(100vw-12rem)] overflow-visible border-0 bg-transparent p-0 shadow-none"
        showCloseButton={false}
      >
        <div className="group/lightbox relative inline-block">
          <img
            alt={alt ?? ''}
            className="block max-h-[calc(100vh-12rem)] max-w-[calc(100vw-12rem)] cursor-zoom-out select-auto rounded-lg object-contain shadow-2xl"
            onClick={() => onOpenChange(false)}
            src={src}
          />
          <ImageActionButtons
            className="group-hover/lightbox:opacity-100"
            copy={copy}
            copying={copying}
            onCopy={onCopy}
            onDownload={onDownload}
            saving={saving}
          />
        </div>
      </DialogContent>
    </Dialog>
  )
}

export function ImageActionButtons({
  className,
  copying,
  copy,
  onCopy,
  onDownload,
  saving
}: {
  className?: string
  copying: boolean
  copy: ImageActionCopy
  onCopy: () => void
  onDownload: () => void
  saving: boolean
}) {
  return (
    <span
      className={cn(
        'absolute right-2 top-2 flex gap-1 opacity-0 transition-opacity focus-within:opacity-100',
        className
      )}
    >
      <ImageActionButton
        busy={copying}
        icon={Copy}
        label={copy.copyImage}
        loadingLabel={copy.copyingImage}
        onClick={onCopy}
      />
      <ImageActionButton
        busy={saving}
        icon={Download}
        label={copy.downloadImage}
        loadingLabel={copy.savingImage}
        onClick={onDownload}
      />
    </span>
  )
}

export function ImageActionButton({
  busy,
  icon: Icon,
  label,
  loadingLabel,
  onClick,
  className
}: {
  busy: boolean
  className?: string
  icon: IconComponent
  label: string
  loadingLabel: string
  onClick: () => void
}) {
  return (
    <button
      aria-label={busy ? loadingLabel : label}
      className={cn(
        'grid size-8 place-items-center rounded-full border border-border/70 bg-background/80 text-muted-foreground shadow-sm backdrop-blur hover:bg-accent hover:text-foreground disabled:opacity-50',
        className
      )}
      disabled={busy}
      onClick={event => {
        event.stopPropagation()
        void onClick()
      }}
      title={busy ? loadingLabel : label}
      type="button"
    >
      <Icon className={cn('size-4', busy && 'animate-pulse')} />
    </button>
  )
}
