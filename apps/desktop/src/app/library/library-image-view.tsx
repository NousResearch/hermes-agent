import { Dialog as DialogPrimitive } from 'radix-ui'
import { type KeyboardEvent, type RefObject, useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { CopyButton } from '@/components/ui/copy-button'
import { PathText } from '@/components/ui/path-text'
import { PixelIcon } from '@/components/ui/pixel-icon'
import { Tip } from '@/components/ui/tooltip'
import { fmtDateTime } from '@/lib/time'

import type { ArtifactRecord } from '../artifacts/artifact-utils'
import type { IndexedChat } from '../artifacts/item-index-snapshot'

import { chatOfItem } from './library-actions'
import { formatItemTime, ItemThumb, itemTypeTag, useItemImageSrc } from './library-collections'
import { useLibraryCopy } from './library-copy'

/** Neighbours either side of the open image in the filmstrip. */
const FILMSTRIP_REACH = 5

const ZOOM_STEPS = [0.5, 0.75, 1, 1.5, 2, 3]

export interface ImageViewProps {
  items: readonly ArtifactRecord[]
  index: number
  chats: Record<string, IndexedChat>
  onIndex: (index: number) => void
  onOpenChat: (item: ArtifactRecord) => void
  onDownload: (item: ArtifactRecord) => void
}

/** Left/right step through the images; the caller owns what Escape does. */
function stepKeys(event: KeyboardEvent, { index, items, onIndex }: ImageViewProps) {
  const rtl = getComputedStyle(event.currentTarget).direction === 'rtl'
  const step = { ArrowLeft: rtl ? 1 : -1, ArrowRight: rtl ? -1 : 1 }[event.key]

  if (!step || (event.target as HTMLElement).closest('input, textarea')) {
    return
  }

  const next = index + step

  if (next >= 0 && next < items.length) {
    event.preventDefault()
    onIndex(next)
  }
}

function StepButtons({ copy, props }: { copy: ReturnType<typeof useLibraryCopy>; props: ImageViewProps }) {
  return (
    <>
      <Tip label={copy.previousImage}>
        <Button
          aria-label={copy.previousImage}
          disabled={props.index <= 0}
          onClick={() => props.onIndex(props.index - 1)}
          size="icon"
          variant="secondary"
        >
          <PixelIcon name="chevron-left" />
        </Button>
      </Tip>
      <Tip label={copy.nextImage}>
        <Button
          aria-label={copy.nextImage}
          disabled={props.index >= props.items.length - 1}
          onClick={() => props.onIndex(props.index + 1)}
          size="icon"
          variant="secondary"
        >
          <PixelIcon name="chevron-right" />
        </Button>
      </Tip>
    </>
  )
}

interface ImagePreviewProps extends ImageViewProps {
  /** The image button, which the full-window view gives focus back to. */
  enlargeRef: RefObject<HTMLButtonElement | null>
  backLabel: string
  onBack: () => void
  onEnlarge: () => void
}

/** 1170:8636 "Image preview": the image on the stage with zoom, its chat / created / path, the
 *  actions, and a filmstrip of its neighbours. Escape goes back to the list. */
export function ImagePreview(props: ImagePreviewProps) {
  const { backLabel, chats, enlargeRef, index, items, onBack, onDownload, onEnlarge, onIndex, onOpenChat } = props
  const copy = useLibraryCopy()
  const item = items[index]!
  const chat = chatOfItem(chats, item)
  const { failed, src } = useItemImageSrc(item, chat)
  const [zoom, setZoom] = useState<null | number>(null)
  const rootRef = useRef<HTMLDivElement>(null)

  useEffect(() => setZoom(null), [item.id])
  useEffect(() => rootRef.current?.focus({ preventScroll: true }), [])

  const zoomBy = (direction: 1 | -1) => {
    const current = zoom ?? 1
    const next = direction > 0 ? ZOOM_STEPS.find(step => step > current) : ZOOM_STEPS.findLast(step => step < current)

    setZoom(next ?? current)
  }

  const first = Math.max(0, index - FILMSTRIP_REACH)
  const strip = items.slice(first, index + FILMSTRIP_REACH + 1)

  return (
    <div
      className="library-preview"
      onKeyDown={event => {
        if (event.key === 'Escape' && !event.defaultPrevented) {
          event.preventDefault()
          onBack()
        } else {
          stepKeys(event, props)
        }
      }}
      ref={rootRef}
      tabIndex={-1}
    >
      <div className="library-preview-head">
        <Button className="library-back" onClick={onBack} size="sm" variant="secondary">
          <PixelIcon name="chevron-left" />
          {backLabel}
        </Button>
        <div className="library-preview-title">
          <h1 className="library-preview-name">{item.label}</h1>
          <p>{copy.position(index + 1, items.length)}</p>
        </div>
        <div className="library-preview-steps">
          <StepButtons copy={copy} props={props} />
        </div>
      </div>

      <div className="library-preview-stage">
        {failed ? (
          <p className="library-preview-missing">{copy.imageUnavailable}</p>
        ) : (
          <button
            aria-label={copy.enlarge}
            className="library-preview-image"
            onClick={onEnlarge}
            ref={enlargeRef}
            type="button"
          >
            {src && (
              <img
                alt={item.label}
                src={src}
                style={zoom === null ? undefined : { width: `${zoom * 100}%`, maxWidth: 'none', maxHeight: 'none' }}
              />
            )}
          </button>
        )}
        <div className="library-zoom">
          <Button aria-label={copy.zoomOut} onClick={() => zoomBy(-1)} size="icon-sm" variant="ghost">
            <PixelIcon name="zoom-out" />
          </Button>
          <Button aria-pressed={zoom === null} onClick={() => setZoom(null)} size="sm" variant="ghost">
            {zoom === null ? copy.fit : `${Math.round(zoom * 100)}%`}
          </Button>
          <Button aria-label={copy.zoomIn} onClick={() => zoomBy(1)} size="icon-sm" variant="ghost">
            <PixelIcon name="zoom-in" />
          </Button>
        </div>
      </div>

      <div className="library-preview-meta">
        <dl>
          <div>
            <dt>{copy.meta.chat}</dt>
            <dd>{chat?.title ?? item.sessionTitle}</dd>
          </div>
          <div>
            <dt>{copy.meta.created}</dt>
            <dd>{fmtDateTime.format(new Date(item.timestamp))}</dd>
          </div>
          <div className="library-preview-path">
            <dt>{copy.meta.path}</dt>
            <dd>{item.value.startsWith('data:') ? itemTypeTag(item) : <PathText path={item.value} />}</dd>
          </div>
        </dl>
        <div className="library-preview-actions">
          <CopyButton
            appearance="icon"
            buttonSize="icon"
            label={copy.copyPath}
            text={item.value}
            title={copy.copyPath}
          />
          <Tip label={copy.download}>
            <Button aria-label={copy.download} onClick={() => onDownload(item)} size="icon" variant="ghost">
              <PixelIcon name="download" />
            </Button>
          </Tip>
          <Button onClick={() => onOpenChat(item)}>{copy.openChat}</Button>
        </div>
      </div>

      <div aria-label={copy.thumbnails} className="library-filmstrip" role="list">
        {strip.map((entry, offset) => (
          <button
            aria-current={first + offset === index ? 'true' : undefined}
            aria-label={entry.label}
            key={entry.id}
            onClick={() => onIndex(first + offset)}
            role="listitem"
            type="button"
          >
            <ItemThumb chat={chatOfItem(chats, entry)} className="library-filmstrip-thumb" item={entry} />
          </button>
        ))}
      </div>
    </div>
  )
}

interface ImageLightboxProps extends ImageViewProps {
  onClose: () => void
  /** Where focus goes when it closes (the control that opened it). */
  returnFocus: RefObject<HTMLElement | null>
}

/** 1192:8764 "Image open": the image over the whole window with its name and source, Open chat,
 *  Download, previous / next and "1 of 161". A modal dialog: focus stays inside, Escape closes, and
 *  focus returns to what opened it. */
export function ImageLightbox(props: ImageLightboxProps) {
  const { chats, index, items, onClose, onDownload, onOpenChat, returnFocus } = props
  const copy = useLibraryCopy()
  const item = items[index]!
  const chat = chatOfItem(chats, item)
  const { failed, src } = useItemImageSrc(item, chat)

  return (
    <DialogPrimitive.Root onOpenChange={open => !open && onClose()} open>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="library-lightbox-scrim" />
        <DialogPrimitive.Content
          aria-describedby={undefined}
          className="library-lightbox"
          onCloseAutoFocus={event => {
            event.preventDefault()
            returnFocus.current?.focus()
          }}
          onKeyDown={event => stepKeys(event, props)}
          onOpenAutoFocus={event => {
            // The window itself takes focus, not its first button, so arrows step at once.
            event.preventDefault()
            ;(event.currentTarget as HTMLElement).focus()
          }}
          tabIndex={-1}
        >
          <div className="library-lightbox-head">
            <div className="library-lightbox-title">
              <DialogPrimitive.Title>{item.label}</DialogPrimitive.Title>
              <p>
                {copy.imageSource(
                  itemTypeTag(item).toUpperCase(),
                  chat?.title ?? item.sessionTitle,
                  formatItemTime(item.timestamp)
                )}
              </p>
            </div>
            <Button onClick={() => onOpenChat(item)} size="sm">
              <PixelIcon name="message" />
              {copy.openChat}
            </Button>
            <Button onClick={() => onDownload(item)} size="sm" variant="secondary">
              <PixelIcon name="download" />
              {copy.download}
            </Button>
            <DialogPrimitive.Close asChild>
              <Button aria-label={copy.close} size="icon" variant="ghost">
                <PixelIcon name="close" />
              </Button>
            </DialogPrimitive.Close>
          </div>
          <div className="library-lightbox-body">
            <span className="library-lightbox-step">
              <Button
                aria-label={copy.previousImage}
                disabled={index <= 0}
                onClick={() => props.onIndex(index - 1)}
                size="icon"
                variant="secondary"
              >
                <PixelIcon name="chevron-left" />
              </Button>
            </span>
            {failed ? (
              <p className="library-preview-missing">{copy.imageUnavailable}</p>
            ) : (
              src && <img alt={item.label} src={src} />
            )}
            <span className="library-lightbox-step">
              <Button
                aria-label={copy.nextImage}
                disabled={index >= items.length - 1}
                onClick={() => props.onIndex(index + 1)}
                size="icon"
                variant="secondary"
              >
                <PixelIcon name="chevron-right" />
              </Button>
            </span>
          </div>
          <p aria-live="polite" className="library-lightbox-position">
            {copy.position(index + 1, items.length)}
          </p>
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  )
}
