import { useStore } from '@nanostores/react'
import { type KeyboardEvent, useMemo, useState } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import { ImageActionButton, ImageLightbox } from '@/components/chat/zoomable-image'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { useImageDownload } from '@/hooks/use-image-download'
import { useI18n } from '@/i18n'
import { mediaName } from '@/lib/media'
import { TOOL_IMAGE_PAGE_SIZE } from '@/lib/tool-images'
import { openPreview } from '@/store/preview'

import { type ToolImageContext, type ToolImageState, useToolImagePage } from './image-loader'

interface ToolImagePreviewsProps {
  active: boolean
  sources: string[]
  toolCallId: string
}

export function ToolImagePreviews({ active, sources, toolCallId }: ToolImagePreviewsProps) {
  const { $storedId, $runtimeId } = useSessionView()
  const sessionId = useStore($storedId)
  const runtimeId = useStore($runtimeId)

  return (
    <ToolImageGallery
      active={active}
      context={{ sessionId, runtimeId }}
      key={`${runtimeId}:${sessionId}:${toolCallId}`}
      sources={sources}
      toolCallId={toolCallId}
    />
  )
}

function ToolImageGallery({
  active,
  sources,
  toolCallId,
  context
}: ToolImagePreviewsProps & { context: ToolImageContext }) {
  const { t } = useI18n()
  const [selection, setSelection] = useState(sources[0])
  const [lightboxOpen, setLightboxOpen] = useState(false)
  const index = Math.max(0, sources.indexOf(selection))
  const source = sources[index]
  const multiple = sources.length > 1
  const pageStart = Math.floor(index / TOOL_IMAGE_PAGE_SIZE) * TOOL_IMAGE_PAGE_SIZE
  const pageSources = useMemo(() => sources.slice(pageStart, pageStart + TOOL_IMAGE_PAGE_SIZE), [pageStart, sources])
  const { images, retry, fail } = useToolImagePage(pageSources, active, context)
  const image = images.get(source)
  const src = image?.status === 'ready' ? image.src || '' : ''
  const { download, saving } = useImageDownload(src || undefined)
  const position = t.desktop.imagePosition(index + 1, sources.length)
  const name = /^data:/i.test(source) ? t.assistant.tool.outputAlt : mediaName(source)
  const label = multiple ? `${position} · ${name}` : name
  const select = (next: number) => setSelection(sources[Math.max(0, Math.min(next, sources.length - 1))])

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) {
      return
    }

    const target = event.target as HTMLElement

    if (target.closest('input, textarea, [contenteditable="true"]')) {
      return
    }

    const moves: Record<string, number> = {
      ArrowLeft: index - 1,
      ArrowRight: index + 1,
      Home: 0,
      End: sources.length - 1
    }

    if (!(event.key in moves) || !multiple) {
      return
    }

    event.preventDefault()
    event.stopPropagation()
    select(moves[event.key])
  }

  const controls = multiple ? (
    <div className="flex items-center justify-between gap-3 py-2" data-slot="image-gallery-navigation">
      <Button
        aria-label={t.desktop.previousImage}
        disabled={index === 0}
        onClick={() => select(index - 1)}
        size="icon-xs"
        variant="ghost"
      >
        <Codicon name="chevron-left" />
      </Button>
      <span aria-live="polite" className="text-xs tabular-nums text-muted-foreground">
        {position}
      </span>
      <Button
        aria-label={t.desktop.nextImage}
        disabled={index === sources.length - 1}
        onClick={() => select(index + 1)}
        size="icon-xs"
        variant="ghost"
      >
        <Codicon name="chevron-right" />
      </Button>
    </div>
  ) : undefined

  const placeholder = <ImageLoadState image={image} retry={() => retry(source)} />

  return (
    <div
      aria-label={t.desktop.imageGallery}
      className={active ? 'grid gap-2 p-2' : 'hidden'}
      data-slot="tool-image-previews"
      hidden={!active}
      onKeyDown={onKeyDown}
      role="region"
      tabIndex={0}
    >
      <figure className="m-0 min-w-0 max-w-xl" data-slot="tool-image-preview">
        <div className="group/image relative inline-block max-w-full align-top">
          {src ? (
            <button
              aria-label={t.desktop.openImage}
              className="block max-w-full cursor-zoom-in"
              onClick={() => setLightboxOpen(true)}
              type="button"
            >
              <img
                alt={label}
                className="max-h-80 w-auto max-w-full rounded-md object-contain"
                decoding="async"
                onError={() => fail(source)}
                src={src}
              />
            </button>
          ) : (
            placeholder
          )}
          {src && (
            <ImageActionButton
              className="group-hover/image:opacity-100"
              copy={t.desktop}
              onClick={download}
              saving={saving}
            />
          )}
        </div>
        {controls}
        <figcaption className="mt-1 flex min-w-0 items-center gap-3 text-xs text-muted-foreground">
          <span className="truncate">{name}</span>
          {src && (
            <Button
              onClick={() =>
                openPreview(
                  {
                    kind: 'file',
                    label,
                    source: `tool-image:${context.runtimeId}:${context.sessionId}:${toolCallId}:${index}`,
                    url: `tool-image:${context.runtimeId}:${context.sessionId}:${toolCallId}:${index}`,
                    dataUrl: src,
                    previewKind: 'image',
                    transient: true
                  },
                  'tool-result'
                )
              }
              size="xs"
              variant="text"
            >
              {t.preview.openPreview}
            </Button>
          )}
        </figcaption>
      </figure>
      {multiple && (
        <div className="grid max-w-xl gap-1">
          <span className="text-xs text-muted-foreground">
            {t.desktop.thumbnailRange(pageStart + 1, pageStart + pageSources.length, sources.length)}
          </span>
          <div className="flex gap-2" data-slot="image-gallery-thumbnails">
            {pageSources.map((item, offset) => {
              const itemIndex = pageStart + offset
              const thumbnail = images.get(item)

              return (
                <button
                  aria-label={`${t.desktop.openImage} ${itemIndex + 1}/${sources.length}`}
                  aria-pressed={index === itemIndex}
                  className="relative grid h-16 min-w-0 flex-1 cursor-pointer place-items-center overflow-hidden rounded-md border border-border bg-muted/30 focus-visible:outline-2 focus-visible:outline-ring aria-pressed:border-primary aria-pressed:ring-1 aria-pressed:ring-primary"
                  key={item}
                  onClick={() => select(itemIndex)}
                  type="button"
                >
                  {thumbnail?.status === 'ready' ? (
                    <img alt="" className="h-full w-full object-contain" decoding="async" src={thumbnail.src} />
                  ) : (
                    <Codicon name={thumbnail?.status === 'error' ? 'warning' : 'file-media'} />
                  )}
                  <span className="absolute right-1 bottom-1 rounded bg-background/90 px-1 text-[0.625rem] tabular-nums text-foreground">
                    {itemIndex + 1}
                  </span>
                </button>
              )
            })}
          </div>
        </div>
      )}
      <ImageLightbox
        alt={label}
        copy={t.desktop}
        navigation={controls ? <div className="rounded-b-md bg-background px-3">{controls}</div> : undefined}
        onClick={download}
        onKeyDown={onKeyDown}
        onOpenChange={setLightboxOpen}
        open={active && lightboxOpen}
        placeholder={<div className="min-w-80 rounded-md bg-background p-4">{placeholder}</div>}
        saving={saving}
        src={src}
      />
    </div>
  )
}

function ImageLoadState({ image, retry }: { image?: ToolImageState; retry: () => void }) {
  const { t } = useI18n()
  const failed = image?.status === 'error'

  return (
    <div className="flex min-h-24 items-center gap-2 text-xs text-muted-foreground" role="status">
      <Codicon name="file-media" />
      {failed ? t.preview.unavailable : t.preview.loading}
      {failed && (
        <Button onClick={retry} size="xs" variant="text">
          {t.common.retry}
        </Button>
      )}
    </div>
  )
}
