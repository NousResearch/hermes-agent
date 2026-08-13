import { useStore } from '@nanostores/react'
import { useState } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import { ImageLightbox } from '@/components/chat/zoomable-image'
import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import { useImageDownload } from '@/hooks/use-image-download'
import { useI18n } from '@/i18n'
import { readDesktopFileDataUrlLocalFirst } from '@/lib/desktop-fs'
import { AlertCircle, FileText, FolderOpen, ImageIcon, Link, Loader2, MessageCode, Terminal } from '@/lib/icons'
import { normalizeOrLocalPreviewTarget } from '@/lib/local-preview'
import { cn } from '@/lib/utils'
import type { ComposerAttachment } from '@/store/composer'
import { notifyError } from '@/store/notifications'
import { openPreview } from '@/store/preview'

export function AttachmentList({
  attachments,
  onRemove
}: {
  attachments: ComposerAttachment[]
  onRemove?: (id: string) => void
}) {
  return (
    <div className="flex max-w-full flex-wrap gap-1.5 px-1 pt-1" data-slot="composer-attachments">
      {attachments.filter(Boolean).map(attachment => (
        <AttachmentPill attachment={attachment} key={attachment.id} onRemove={onRemove} />
      ))}
    </div>
  )
}

function AttachmentPill({ attachment, onRemove }: { attachment: ComposerAttachment; onRemove?: (id: string) => void }) {
  const { t } = useI18n()
  const c = t.composer

  const Icon = {
    file: FileText,
    folder: FolderOpen,
    image: ImageIcon,
    review: MessageCode,
    terminal: Terminal,
    url: Link
  }[attachment.kind]

  // The tile's cwd when this pill lives in a tile composer, not the primary's:
  // a relative attachment path has to resolve against its own session's root.
  const cwd = useStore(useSessionView().$cwd)
  const isUploading = attachment.uploadState === 'uploading'
  const hasUploadError = attachment.uploadState === 'error'

  // A review card's detail is its resolved-comment JSON, not a previewable
  // path — clicking it should do nothing rather than toast a bogus failure.
  const canPreview =
    attachment.kind !== 'folder' && attachment.kind !== 'terminal' && attachment.kind !== 'review' && !isUploading

  const detail =
    attachment.kind !== 'review' && attachment.detail && attachment.detail !== attachment.label
      ? attachment.detail
      : undefined

  // Keep full image bytes out of composer state. New chips read their path only
  // when clicked; previewUrl remains a compatibility fallback for older drafts.
  const [loadedImageSrc, setLoadedImageSrc] = useState<string>()
  const lightboxSrc = attachment.kind === 'image' && !isUploading ? attachment.previewUrl || loadedImageSrc : undefined
  const [lightboxOpen, setLightboxOpen] = useState(false)
  const { download, saving } = useImageDownload(lightboxSrc)

  async function openAttachment() {
    if (!canPreview) {
      return
    }

    if (attachment.kind === 'image') {
      try {
        const source = lightboxSrc || (attachment.path ? await readDesktopFileDataUrlLocalFirst(attachment.path) : '')

        if (!source) {
          throw new Error(c.couldNotPreview(attachment.label))
        }

        setLoadedImageSrc(source)
        setLightboxOpen(true)
      } catch (error) {
        notifyError(error, c.previewUnavailable)
      }

      return
    }

    const rawTarget =
      attachment.path ||
      attachment.detail ||
      attachment.refText?.replace(/^@(file|image|url):/, '') ||
      attachment.label ||
      ''

    const target = rawTarget.replace(/^`|`$/g, '')

    if (!target) {
      return
    }

    try {
      const preview = await normalizeOrLocalPreviewTarget(target, cwd || undefined)

      if (!preview) {
        throw new Error(c.couldNotPreview(attachment.label))
      }

      openPreview(preview, 'manual')
    } catch (error) {
      notifyError(error, c.previewUnavailable)
    }
  }

  return (
    <>
      <Tip label={attachment.path || attachment.detail || attachment.label}>
        <div className="group/attachment relative min-w-0 shrink-0">
          <button
            aria-busy={isUploading || undefined}
            aria-label={canPreview ? c.previewLabel(attachment.label) : attachment.label}
            className={cn(
              'flex max-w-56 items-center gap-2 rounded-2xl border bg-background/50 px-2 py-1.5 text-left shadow-[inset_0_1px_0_rgba(255,255,255,0.18)] transition-colors disabled:cursor-default',
              hasUploadError
                ? 'border-destructive/45 hover:border-destructive/60'
                : 'border-border/60 hover:border-primary/35 hover:bg-accent/45'
            )}
            disabled={!canPreview}
            onClick={() => void openAttachment()}
            type="button"
          >
            <span className="relative grid size-8 shrink-0 place-items-center overflow-hidden rounded-lg border border-border/55 bg-muted/35 text-muted-foreground">
              {(attachment.thumbnailUrl || attachment.previewUrl) && attachment.kind === 'image' ? (
                <img
                  alt={attachment.label}
                  className="size-full object-cover"
                  draggable={false}
                  src={attachment.thumbnailUrl ?? attachment.previewUrl}
                />
              ) : (
                <Icon className="size-3.5" />
              )}
              {isUploading && (
                <span className="absolute inset-0 grid place-items-center bg-background/60 backdrop-blur-[1px]">
                  <Loader2 className="size-3.5 animate-spin text-foreground/75" />
                </span>
              )}
              {hasUploadError && (
                <span className="absolute inset-0 grid place-items-center bg-destructive/15">
                  <AlertCircle className="size-3.5 text-destructive" />
                </span>
              )}
            </span>
            <span className="min-w-0">
              <span className="block truncate text-[0.72rem] font-medium leading-4 text-foreground/90">
                {attachment.label}
              </span>
              {detail && (
                <span
                  className={cn(
                    'block truncate text-[0.62rem] leading-3.5',
                    hasUploadError ? 'text-destructive/80' : 'text-muted-foreground/65'
                  )}
                >
                  {detail}
                </span>
              )}
            </span>
          </button>
          {onRemove && (
            <button
              aria-label={c.removeAttachment(attachment.label)}
              className="absolute -right-1 -top-1 grid size-3.5 place-items-center rounded-full border border-border/70 bg-background text-muted-foreground opacity-0 shadow-xs transition hover:bg-accent hover:text-foreground group-hover/attachment:opacity-100 focus-visible:opacity-100"
              onClick={() => onRemove(attachment.id)}
              type="button"
            >
              <Codicon name="close" size="0.625rem" />
            </button>
          )}
        </div>
      </Tip>
      {lightboxSrc && (
        <ImageLightbox
          alt={attachment.label}
          copy={t.desktop}
          onClick={download}
          onOpenChange={open => {
            setLightboxOpen(open)

            if (!open && !attachment.previewUrl) {
              setLoadedImageSrc(undefined)
            }
          }}
          open={lightboxOpen}
          saving={saving}
          src={lightboxSrc}
        />
      )}
    </>
  )
}
