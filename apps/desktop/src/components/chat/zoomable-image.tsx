'use client'

import {
  type ComponentProps,
  createContext,
  type ReactNode,
  type RefObject,
  useContext,
  useRef,
  useState
} from 'react'

import { Dialog, DialogContent } from '@/components/ui/dialog'
import { Tip } from '@/components/ui/tooltip'
import { useImageDownload } from '@/hooks/use-image-download'
import { useI18n } from '@/i18n'
import { ChevronLeft, ChevronRight, Download } from '@/lib/icons'
import { cn } from '@/lib/utils'

export interface ZoomableImageProps extends ComponentProps<'img'> {
  containerClassName?: string
  slot?: string
}

export interface ImageActionCopy {
  downloadImage: string
  nextImage: string
  previousImage: string
  savingImage: string
}

const ImageGalleryContext = createContext<RefObject<HTMLDivElement | null> | null>(null)

export function ImageGallery({ children }: { children: ReactNode }) {
  const rootRef = useRef<HTMLDivElement>(null)

  return (
    <ImageGalleryContext.Provider value={rootRef}>
      <div className="contents" data-slot="image-gallery" ref={rootRef}>
        {children}
      </div>
    </ImageGalleryContext.Provider>
  )
}

interface GalleryImage {
  alt: string
  src: string
}

export function ZoomableImage({ className, containerClassName, src, alt, slot, ...props }: ZoomableImageProps) {
  const { t } = useI18n()
  const copy = t.desktop
  const { download, saving } = useImageDownload(src)
  const galleryRoot = useContext(ImageGalleryContext)
  const imageRef = useRef<HTMLImageElement>(null)
  const [galleryImages, setGalleryImages] = useState<GalleryImage[]>([])
  const [activeIndex, setActiveIndex] = useState(0)
  const [lightboxOpen, setLightboxOpen] = useState(false)
  const canOpen = Boolean(src)
  const activeImage = galleryImages[activeIndex] ?? { alt: alt ?? '', src: String(src ?? '') }
  const { download: downloadActive, saving: savingActive } = useImageDownload(activeImage.src)

  const openLightbox = () => {
    if (!canOpen) {
      return
    }

    const imageElements = galleryRoot?.current
      ? Array.from(galleryRoot.current.querySelectorAll<HTMLImageElement>('img[data-gallery-image]'))
      : []

    const images = imageElements
      .map(image => ({ alt: image.alt, src: image.currentSrc || image.src }))
      .filter(image => Boolean(image.src))

    const index = imageRef.current ? imageElements.indexOf(imageRef.current) : -1

    setGalleryImages(images.length > 0 ? images : [{ alt: alt ?? '', src: String(src) }])
    setActiveIndex(index >= 0 ? index : 0)
    setLightboxOpen(true)
  }

  return (
    <>
      <span
        className={cn('group/image relative inline-block max-w-full align-top', containerClassName)}
        data-slot={slot ?? 'aui_zoomable-image'}
      >
        <Tip label={canOpen ? copy.openImage : undefined}>
          <button className="contents" disabled={!canOpen} onClick={openLightbox} type="button">
            <img alt={alt ?? ''} className={className} data-gallery-image ref={imageRef} src={src} {...props} />
          </button>
        </Tip>
        {src && (
          <ImageActionButton className="group-hover/image:opacity-100" copy={copy} onClick={download} saving={saving} />
        )}
      </span>
      {src && (
        <ImageLightbox
          alt={activeImage.alt}
          copy={copy}
          currentIndex={activeIndex}
          imageCount={galleryImages.length}
          onClick={downloadActive}
          onNext={() => setActiveIndex(index => Math.min(index + 1, galleryImages.length - 1))}
          onOpenChange={setLightboxOpen}
          onPrevious={() => setActiveIndex(index => Math.max(index - 1, 0))}
          open={lightboxOpen}
          saving={savingActive}
          src={activeImage.src}
        />
      )}
    </>
  )
}

export function ImageLightbox({
  alt,
  copy,
  currentIndex = 0,
  imageCount = 1,
  onClick,
  onNext,
  onOpenChange,
  onPrevious,
  open,
  saving,
  src
}: {
  alt?: string
  copy: ImageActionCopy
  currentIndex?: number
  imageCount?: number
  onClick: () => void
  onNext?: () => void
  onOpenChange: (open: boolean) => void
  onPrevious?: () => void
  open: boolean
  saving: boolean
  src: string
}) {
  const hasPrevious = currentIndex > 0
  const hasNext = currentIndex < imageCount - 1
  const hasGallery = imageCount > 1

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent
        bodyClassName="block overflow-visible p-0"
        className="w-auto max-h-[calc(100vh-12rem)] max-w-[calc(100vw-12rem)] border-0 bg-transparent shadow-none"
        onKeyDown={event => {
          if (event.key === 'ArrowLeft' && hasPrevious) {
            event.preventDefault()
            onPrevious?.()
          } else if (event.key === 'ArrowRight' && hasNext) {
            event.preventDefault()
            onNext?.()
          }
        }}
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
          {hasGallery && (
            <>
              <ImageNavigationButton
                ariaLabel={copy.previousImage}
                className="left-3"
                disabled={!hasPrevious}
                onClick={onPrevious}
              >
                <ChevronLeft className="size-5" />
              </ImageNavigationButton>
              <ImageNavigationButton
                ariaLabel={copy.nextImage}
                className="right-3"
                disabled={!hasNext}
                onClick={onNext}
              >
                <ChevronRight className="size-5" />
              </ImageNavigationButton>
              <span className="absolute bottom-3 left-1/2 -translate-x-1/2 rounded-full bg-black/65 px-2.5 py-1 text-xs font-medium text-white shadow-sm backdrop-blur">
                {currentIndex + 1} / {imageCount}
              </span>
            </>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}

function ImageNavigationButton({
  ariaLabel,
  children,
  className,
  disabled,
  onClick
}: {
  ariaLabel: string
  children: ReactNode
  className: string
  disabled: boolean
  onClick?: () => void
}) {
  return (
    <Tip label={ariaLabel}>
      <button
        aria-label={ariaLabel}
        className={cn(
          'absolute top-1/2 grid size-10 -translate-y-1/2 place-items-center rounded-full border border-white/25 bg-black/55 text-white shadow-lg backdrop-blur transition hover:bg-black/75 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white disabled:cursor-default disabled:opacity-25',
          className
        )}
        disabled={disabled}
        onClick={event => {
          event.stopPropagation()
          onClick?.()
        }}
        type="button"
      >
        {children}
      </button>
    </Tip>
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
  const label = saving ? copy.savingImage : copy.downloadImage

  return (
    <Tip label={label}>
      <button
        aria-label={label}
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
    </Tip>
  )
}
