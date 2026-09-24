'use client'

import { type FC, useState } from 'react'

import { DiffusionCanvas } from '@/components/chat/image-generation-placeholder'
import { ImageActionButton, ImageLightbox } from '@/components/chat/zoomable-image'
import { useImageDownload } from '@/hooks/use-image-download'
import { useMediaImage } from '@/hooks/use-media-image'
import { useI18n } from '@/i18n'
import { generatedImageDimensionsFromResult, generatedImageFromResult } from '@/lib/generated-images'
import { mediaExternalUrl, mediaName } from '@/lib/media'
import { cn } from '@/lib/utils'

// A hint is only a placeholder shape, not a promise about the delivered image.
const ASPECT_HINTS: Record<string, number> = {
  landscape: 16 / 9,
  square: 1,
  portrait: 9 / 16
}

function hintedRatio(aspectRatio?: string): number {
  return (
    ASPECT_HINTS[
      String(aspectRatio ?? '')
        .toLowerCase()
        .trim()
    ] ?? ASPECT_HINTS.landscape
  )
}

export const GeneratedImage: FC<{ aspectRatio?: string; result?: unknown }> = ({ aspectRatio, result }) => {
  const { t } = useI18n()
  const copy = t.desktop
  const image = result === undefined ? null : generatedImageFromResult(result)
  const pending = result === undefined

  const media = useMediaImage(image ?? '', hintedRatio(aspectRatio), generatedImageDimensionsFromResult(result), {
    preservePendingFrame: true
  })

  const { src, loaded, failed } = media
  const [canvasGone, setCanvasGone] = useState<string | null>(null)
  const [lightboxOpen, setLightboxOpen] = useState<string | null>(null)
  const { download, saving } = useImageDownload(src)

  // Completed but no usable image (generation failed): the agent's prose carries
  // the explanation, so render nothing here.
  if (!pending && !image) {
    return null
  }

  // Nothing will fill the frame, so collapse to the link line.
  if (failed && image) {
    return (
      <a
        className="mt-2 ref inline-block wrap-anywhere"
        href="#"
        onClick={event => {
          event.preventDefault()
          void window.hermesDesktop?.openExternal(mediaExternalUrl(image))
        }}
      >
        {copy.openImage}: {mediaName(image)}
      </a>
    )
  }

  return (
    <>
      <span
        aria-label={pending ? t.assistant.tool.renderingImage : undefined}
        aria-live={pending ? 'polite' : undefined}
        className="group/image relative block max-w-full overflow-hidden rounded-2xl"
        data-slot="aui_generated-image"
        role={pending ? 'status' : undefined}
        style={media.frameStyle}
      >
        {canvasGone !== media.key && (
          <div
            className={cn('absolute inset-0 transition-opacity duration-500 ease-out', loaded && 'opacity-0')}
            onTransitionEnd={() => loaded && setCanvasGone(media.key)}
          >
            <DiffusionCanvas />
          </div>
        )}
        {src && (
          <button
            aria-label={copy.openImage}
            className="absolute inset-0 block size-full cursor-zoom-in"
            onClick={() => setLightboxOpen(media.key)}
            type="button"
          >
            <img
              alt="Generated image"
              className={cn(
                'absolute inset-0 size-full object-contain opacity-0 transition-opacity duration-500 ease-out',
                loaded && 'opacity-100'
              )}
              draggable={false}
              onError={media.onError}
              onLoad={event => media.onLoad(event.currentTarget)}
              src={src}
            />
          </button>
        )}
        {loaded && src && (
          <ImageActionButton className="group-hover/image:opacity-100" copy={copy} onClick={download} saving={saving} />
        )}
      </span>
      {src && (
        <ImageLightbox
          alt="Generated image"
          copy={copy}
          onClick={download}
          onOpenChange={open => setLightboxOpen(open ? media.key : null)}
          open={lightboxOpen === media.key}
          saving={saving}
          src={src}
        />
      )}
    </>
  )
}
