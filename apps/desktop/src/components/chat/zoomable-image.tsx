'use client'

import { type ComponentProps, useState } from 'react'

import { Dialog, DialogContent } from '@/components/ui/dialog'
import { Download } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { notify, notifyError } from '@/store/notifications'

function imageFilename(src?: string): string {
  if (!src) {
    return 'image'
  }

  try {
    const { pathname } = new URL(src, window.location.href)

    return pathname.split('/').filter(Boolean).pop() || 'image'
  } catch {
    return src.split(/[\\/]/).filter(Boolean).pop() || 'image'
  }
}

function isMissingIpcHandler(error: unknown): boolean {
  const message = error instanceof Error ? error.message : typeof error === 'string' ? error : ''

  return message.includes("No handler registered for 'hermes:saveImageFromUrl'")
}

async function startBrowserDownload(src: string) {
  const response = await fetch(src)

  if (!response.ok) {
    throw new Error(`Could not fetch image: ${response.status}`)
  }

  const blobUrl = URL.createObjectURL(await response.blob())
  const link = document.createElement('a')
  link.href = blobUrl
  link.download = imageFilename(src)
  link.rel = 'noopener noreferrer'
  document.body.appendChild(link)
  link.click()
  link.remove()
  window.setTimeout(() => URL.revokeObjectURL(blobUrl), 30_000)
}

export interface ZoomableImageProps extends ComponentProps<'img'> {
  containerClassName?: string
  slot?: string
}

export function ZoomableImage({ className, containerClassName, src, alt, slot, ...props }: ZoomableImageProps) {
  const [saving, setSaving] = useState(false)
  const [lightboxOpen, setLightboxOpen] = useState(false)
  const canOpen = Boolean(src)

  async function handleDownload() {
    if (!src || saving) {
      return
    }

    setSaving(true)

    try {
      if (window.hermesDesktop?.saveImageFromUrl) {
        const saved = await window.hermesDesktop.saveImageFromUrl(src)

        if (saved) {
          notify({ kind: 'success', title: 'Image saved', message: imageFilename(src) })
        }

        return
      }

      await startBrowserDownload(src)
    } catch (error) {
      if (isMissingIpcHandler(error)) {
        try {
          await startBrowserDownload(src)
          notify({
            kind: 'info',
            title: 'Download started',
            message: 'Restart Hermes Desktop to use Save Image.'
          })
        } catch (fallbackError) {
          notifyError(fallbackError, 'Restart Hermes Desktop to save images')
        }

        return
      }

      notifyError(error, 'Image download failed')
    } finally {
      setSaving(false)
    }
  }

  const lightbox = src ? (
    <Dialog onOpenChange={setLightboxOpen} open={lightboxOpen}>
      <DialogContent
        className="block w-auto max-h-[calc(100vh-12rem)] max-w-[calc(100vw-12rem)] overflow-visible border-0 bg-transparent p-0 shadow-none"
        showCloseButton={false}
      >
        <div className="group/lightbox relative inline-block">
          <img
            alt={alt ?? ''}
            className="block max-h-[calc(100vh-12rem)] max-w-[calc(100vw-12rem)] cursor-zoom-out select-auto rounded-lg object-contain shadow-2xl"
            onClick={() => setLightboxOpen(false)}
            src={src}
          />
          <ImageActionButton onClick={handleDownload} saving={saving} variant="lightbox" />
        </div>
      </DialogContent>
    </Dialog>
  ) : null

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
          title={canOpen ? 'Open image' : undefined}
          type="button"
        >
          <img alt={alt ?? ''} className={className} src={src} {...props} />
        </button>
        {src && <ImageActionButton onClick={handleDownload} saving={saving} variant="inline" />}
      </span>
      {lightbox}
    </>
  )
}

function ImageActionButton({
  onClick,
  saving,
  variant
}: {
  onClick: () => void
  saving: boolean
  variant: 'inline' | 'lightbox'
}) {
  return (
    <button
      aria-label={saving ? 'Saving image' : 'Download image'}
      className={cn(
        'absolute right-2 top-2 grid size-8 place-items-center rounded-full border border-border/70 bg-background/80 text-muted-foreground opacity-0 shadow-sm backdrop-blur transition-opacity hover:bg-accent hover:text-foreground focus-visible:opacity-100 disabled:opacity-50',
        variant === 'inline' ? 'group-hover/image:opacity-100' : 'group-hover/lightbox:opacity-100'
      )}
      disabled={saving}
      onClick={event => {
        event.stopPropagation()
        void onClick()
      }}
      title={saving ? 'Saving image' : 'Download image'}
      type="button"
    >
      <Download className={cn('size-4', saving && 'animate-pulse')} />
    </button>
  )
}
