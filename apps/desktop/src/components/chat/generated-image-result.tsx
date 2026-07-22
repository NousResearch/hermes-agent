'use client'

import { type FC, useEffect, useState } from 'react'

import { DiffusionCanvas } from '@/components/chat/image-generation-placeholder'
import { ImageActionButton, ImageLightbox } from '@/components/chat/zoomable-image'
import { useImageDownload } from '@/hooks/use-image-download'
import { useOpenMediaFile } from '@/hooks/use-open-media-file'
import { useI18n } from '@/i18n'
import { generatedImageFromResult } from '@/lib/generated-images'
import { filePathFromMediaPath, gatewayMediaDataUrl, isRemoteGateway, mediaExternalUrl, mediaName } from '@/lib/media'
import { cn } from '@/lib/utils'

// Aspect hint from the tool args sizes the frame *before* the image loads, so
// the placeholder and the resolved image occupy the same box — no layout shift.
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

function isInlineSrc(path: string): boolean {
  return /^(?:https?|data):/i.test(path)
}

async function resolveImageSrc(path: string): Promise<string> {
  if (isInlineSrc(path)) {
    return path
  }

  if (window.hermesDesktop && isRemoteGateway()) {
    return gatewayMediaDataUrl(path)
  }

  if (!window.hermesDesktop?.readFileDataUrl) {
    return mediaExternalUrl(path)
  }

  return window.hermesDesktop.readFileDataUrl(filePathFromMediaPath(path))
}

export const GeneratedImage: FC<{ aspectRatio?: string; result?: unknown }> = ({ aspectRatio, result }) => {
  const { t } = useI18n()
  const copy = t.desktop
  const resolvedImage = result === undefined ? null : generatedImageFromResult(result)
  const image = resolvedImage?.trim() || null
  const fullFile = image && !/^data:/i.test(image) ? image : undefined
  const downloadName = fullFile ? mediaName(fullFile) : undefined
  const pending = result === undefined

  const [ratio, setRatio] = useState(() => hintedRatio(aspectRatio))
  const [src, setSrc] = useState(() => (image && isInlineSrc(image) ? image : ''))
  const [loaded, setLoaded] = useState(false)
  const [canvasGone, setCanvasGone] = useState(false)
  const [failed, setFailed] = useState(false)
  const [lightboxOpen, setLightboxOpen] = useState(false)
  const { download, saving } = useImageDownload(src, downloadName)
  const { downloadsRemoteFile, open: openFullFile, openFailed } = useOpenMediaFile(fullFile)

  useEffect(() => setRatio(hintedRatio(aspectRatio)), [aspectRatio])

  // Resolve the deliverable path (local read / gateway proxy / remote URL). The
  // <img> stays mounted under the placeholder and only fades in once it decodes,
  // so the frame keeps its hinted size and never jumps.
  useEffect(() => {
    let cancelled = false
    setFailed(false)
    setLoaded(false)
    setCanvasGone(false)
    setSrc(image && isInlineSrc(image) ? image : '')

    if (!image || isInlineSrc(image)) {
      return
    }

    void resolveImageSrc(image)
      .then(resolved => !cancelled && setSrc(resolved))
      .catch(() => !cancelled && setFailed(true))

    return () => {
      cancelled = true
    }
  }, [image])

  // Completed but no usable image (generation failed): the agent's prose carries
  // the explanation, so render nothing here.
  if (!pending && !image) {
    return null
  }

  if (failed && image) {
    return (
      <>
        <button
          className="mt-2 inline-block bg-transparent font-semibold text-foreground underline underline-offset-4 decoration-current/20 wrap-anywhere"
          onClick={openFullFile}
          type="button"
        >
          {downloadsRemoteFile ? 'Download' : copy.openImage}: {mediaName(image)}
        </button>
        {openFailed && (
          <span className="mt-1 block text-xs text-muted-foreground">
            Couldn&apos;t open or download {mediaName(image)}.
          </span>
        )}
      </>
    )
  }

  return (
    <>
      <span
        aria-label={pending ? t.assistant.tool.renderingImage : undefined}
        aria-live={pending ? 'polite' : undefined}
        className="group/image relative block max-w-full overflow-hidden rounded-2xl transition-[width,height] duration-300 ease-out"
        data-slot="aui_generated-image"
        role={pending ? 'status' : undefined}
        style={{
          aspectRatio: ratio,
          // Width is capped so the derived height (width / ratio) never exceeds
          // --image-preview-height; the box then matches the image exactly with
          // no letterboxing.
          width: `min(calc(var(--image-preview-height) * ${ratio}), var(--image-preview-max-width), 100%)`
        }}
      >
        {!canvasGone && (
          <div
            className={cn('absolute inset-0 transition-opacity duration-500 ease-out', loaded && 'opacity-0')}
            onTransitionEnd={() => loaded && setCanvasGone(true)}
          >
            <DiffusionCanvas />
          </div>
        )}
        {src && (
          <button
            className="absolute inset-0 block size-full cursor-zoom-in"
            onClick={() => setLightboxOpen(true)}
            title={copy.openImage}
            type="button"
          >
            <img
              alt="Generated image"
              className={cn(
                'absolute inset-0 size-full object-contain opacity-0 transition-opacity duration-500 ease-out',
                loaded && 'opacity-100'
              )}
              draggable={false}
              onError={() => setFailed(true)}
              onLoad={event => {
                const { naturalHeight, naturalWidth } = event.currentTarget

                if (naturalWidth && naturalHeight) {
                  setRatio(naturalWidth / naturalHeight)
                }

                setLoaded(true)
              }}
              src={src}
            />
          </button>
        )}
        {loaded && src && (
          <ImageActionButton className="group-hover/image:opacity-100" copy={copy} onClick={download} saving={saving} />
        )}
      </span>
      {fullFile && (
        <>
          <button
            className="mt-1 block bg-transparent text-xs font-medium text-muted-foreground underline underline-offset-4 decoration-current/20 hover:text-foreground"
            onClick={openFullFile}
            type="button"
          >
            {downloadsRemoteFile ? 'Download' : 'Open'} full file
          </button>
          {openFailed && (
            <span className="mt-1 block text-xs text-muted-foreground">
              Couldn&apos;t open or download {mediaName(fullFile)}.
            </span>
          )}
        </>
      )}
      {src && (
        <ImageLightbox
          alt="Generated image"
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
