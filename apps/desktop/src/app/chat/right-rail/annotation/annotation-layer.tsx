import { useCallback, useEffect, useRef, useState } from 'react'

import { requestComposerFocus, requestComposerInsert } from '@/app/chat/composer/focus'
import { useI18n } from '@/i18n'
import { attachmentId } from '@/lib/chat-runtime'
import { addComposerAttachment } from '@/store/composer'
import { notify, notifyError } from '@/store/notifications'

import { AnnotationPopover } from './annotation-popover'
import { AnnotationToolbar, type AnnotationListEntry } from './annotation-toolbar'
import type { PickedElement, PickedRegion } from './element-picker'

function isPickedElement(target: PickedElement | PickedRegion): target is PickedElement {
  return 'selector' in target
}

function listSummary(item: AnnotationItem): string {
  if (isPickedElement(item.target)) {
    const text = item.target.text ? ` "${item.target.text}"` : ''
    return `<${item.target.tagName.toLowerCase()}>${item.target.id ? ` #${item.target.id}` : ''}${text}`
  }

  return `${Math.round(item.target.rect.width)}×${Math.round(item.target.rect.height)}px 区域`
}
import { dataUrlToBytes } from './image-annotate'
import {
  buildAddBadgeCall,
  buildFlashCall,
  buildRemoveBadgeCall,
  buildSessionProbeSource,
  buildSetPickingCall,
  buildTeardownCall,
  parseSessionEvent
} from './element-picker-session'
import { formatAnnotationSessionMessage } from './message-format'
import { captureRegionDataUrl } from './screenshot'

interface AnnotatableWebview {
  addEventListener?: (type: string, listener: (event: Event) => void) => void
  capturePage?: (rect?: { height: number; width: number; x: number; y: number }) => Promise<{ toDataURL: () => string }>
  executeJavaScript?: (code: string) => Promise<unknown>
  removeEventListener?: (type: string, listener: (event: Event) => void) => void
}

interface AnnotationLayerProps {
  /** The live preview webview element (null while preview is a local file). */
  webview: AnnotatableWebview | null
  /** Called when annotation mode ends for any reason (finish, cancel, error). */
  onExit: () => void
}

export interface AnnotationItem {
  /** Badge anchor — the actual click point (element) or region corner. */
  anchorX: number
  anchorY: number
  comment: string
  id: string
  kind: 'element' | 'region'
  number: number
  screenshot?: string
  target: PickedElement | PickedRegion
}

interface PendingPick {
  anchorX: number
  anchorY: number
  kind: 'element' | 'region'
  screenshot?: string
  target: PickedElement | PickedRegion
}

/** Keep badge anchors inside the viewport and clear of the top banner. */
function clampAnchor(x: number, y: number): { x: number; y: number } {
  return {
    x: Math.max(24, Math.min(x, window.innerWidth - 24)),
    y: Math.max(40, Math.min(y, window.innerHeight - 24))
  }
}

let idCounter = 0
function nextId(): string {
  idCounter += 1
  return `ann-${Date.now()}-${idCounter}`
}

/**
 * Drives one annotation *session*: injects the persistent probe, collects any
 * number of element/region annotations (each gets a numbered badge pinned on
 * the page), and finally assembles one composer message when the user hits
 * "完成". Badges are clickable to remove a single annotation.
 */
export function AnnotationLayer({ onExit, webview }: AnnotationLayerProps) {
  const { t } = useI18n()
  const copy = t.preview.web.annotation
  const [items, setItems] = useState<AnnotationItem[]>([])
  const [pending, setPending] = useState<PendingPick | null>(null)
  const itemsRef = useRef<AnnotationItem[]>([])
  const pendingRef = useRef<PendingPick | null>(null)
  const exitedRef = useRef(false)

  itemsRef.current = items
  pendingRef.current = pending

  const exit = useCallback(() => {
    if (exitedRef.current) {
      return
    }
    exitedRef.current = true
    if (webview?.executeJavaScript) {
      void webview.executeJavaScript(buildTeardownCall()).catch(() => undefined)
    }
    onExit()
  }, [onExit, webview])

  const submitCollected = useCallback(async () => {
    const collected = itemsRef.current
    if (collected.length === 0) {
      return false
    }

    // Screenshots go through the composer-images pipeline as real image
    // attachments (chips under the composer) — never inline base64, which
    // floods the plain-text composer with unreadable noise.
    let attachedCount = 0
    for (const item of collected) {
      if (!item.screenshot) {
        continue
      }

      try {
        const bytes = await dataUrlToBytes(item.screenshot)
        const savedPath = bytes ? await window.hermesDesktop?.saveImageBuffer(bytes, '.png') : ''

        if (savedPath) {
          addComposerAttachment({
            detail: savedPath,
            id: attachmentId('image', savedPath),
            kind: 'image',
            label: `标注-${item.number}.png`,
            path: savedPath,
            previewUrl: item.screenshot
          })
          attachedCount += 1
        }
      } catch {
        // A failed screenshot must not block the text report.
      }
    }

    const message = formatAnnotationSessionMessage(collected)
    const attachmentNote = attachedCount > 0 ? `\n\n📎 ${copy.screenshotsAttached(attachedCount)}` : ''

    requestComposerInsert(`${message}${attachmentNote}`, {
      mode: 'block',
      target: 'main'
    })
    requestComposerFocus('main')
    return true
  }, [copy])

  // Inject the probe + subscribe to its console channel.
  useEffect(() => {
    if (!webview || typeof webview.executeJavaScript !== 'function') {
      notifyError(new Error('webview unavailable'), copy.pickerFailed)
      onExit()
      return
    }

    let disposed = false

    const onConsoleMessage = (raw: Event) => {
      const message = (raw as Event & { message?: string }).message || ''
      const event = parseSessionEvent(message)
      if (!event || disposed) {
        return
      }

      if (event.type === 'cancel-request') {
        exit()
        return
      }

      if (event.type === 'iframe-blocked') {
        notify({ message: copy.iframeBlocked, kind: 'warning' })
        void webview.executeJavaScript!(buildSetPickingCall(true)).catch(() => undefined)
        return
      }

      if (event.type === 'badge-click') {
        removeItemRef.current(event.id)
        return
      }

      // pick event — the probe already paused picking and hid its highlight,
      // so the page is clean for the screenshot.
      void (async () => {
        const page = await measureViewport(webview)
        const screenshot = (await captureRegionDataUrl(webview, event.target.rect, page)) ?? undefined

        if (disposed) {
          return
        }

        const anchor = clampAnchor(event.clickX, event.clickY)
        setPending({ anchorX: anchor.x, anchorY: anchor.y, kind: event.kind, screenshot, target: event.target })
      })()
    }

    const onNavigate = () => {
      // Full navigation wipes the probe — submit collected work, then exit.
      if (disposed) {
        return
      }
      void submitCollected().then(submitted => {
        notify({ message: submitted ? copy.navigatedAwaySubmitted : copy.navigatedAway, kind: 'warning' })
      })
      exit()
    }

    webview.addEventListener?.('console-message', onConsoleMessage)
    webview.addEventListener?.('did-navigate', onNavigate)
    void webview.executeJavaScript(buildSessionProbeSource(copy.banner)).catch(error => {
      if (!disposed) {
        notifyError(error, copy.pickerFailed)
        onExit()
      }
    })

    return () => {
      disposed = true
      webview.removeEventListener?.('console-message', onConsoleMessage)
      webview.removeEventListener?.('did-navigate', onNavigate)
      // Best-effort probe cleanup on unmount (mode exit also tears down).
      void webview.executeJavaScript?.(buildTeardownCall()).catch(() => undefined)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [webview])

  const removeItemRef = useRef<(itemId: string) => void>(() => undefined)
  const locateItemRef = useRef<(item: AnnotationItem) => void>(() => undefined)

  const resumePicking = useCallback(() => {
    void webview?.executeJavaScript?.(buildSetPickingCall(true)).catch(() => undefined)
  }, [webview])


  const pinBadge = useCallback(
    (item: AnnotationItem) => {
      void webview?.executeJavaScript?.(
        buildAddBadgeCall(item.id, item.number, item.anchorX, item.anchorY)
      ).catch(() => undefined)
    },
    [webview]
  )

  const handleAdd = useCallback(
    (draft: { comment: string; kind: 'element' | 'region'; screenshotDataUrl?: string; target: PickedElement | PickedRegion }) => {
      const anchor = pendingRef.current
      const item: AnnotationItem = {
        anchorX: anchor?.anchorX ?? draft.target.rect.x,
        anchorY: anchor?.anchorY ?? draft.target.rect.y,
        comment: draft.comment,
        id: nextId(),
        kind: draft.kind,
        number: itemsRef.current.length + 1,
        screenshot: draft.screenshotDataUrl,
        target: draft.target
      }

      setItems(prev => [...prev, item])
      pinBadge(item)

      setPending(null)
      resumePicking()
    },
    [pinBadge, resumePicking]
  )

  /** Remove one annotation and re-pin survivors with fresh numbers. */
  const removeItem = useCallback(
    (itemId: string) => {
      const survivors = itemsRef.current
        .filter(item => item.id !== itemId)
        .map((item, index) => ({ ...item, number: index + 1 }))

      setItems(survivors)
      void webview?.executeJavaScript?.(buildRemoveBadgeCall(itemId)).catch(() => undefined)
      for (const item of survivors) {
        pinBadge(item)
      }
    },
    [pinBadge, webview]
  )

  /** Scroll to an annotation's target and flash it. */
  const locateItem = useCallback(
    (item: AnnotationItem) => {
      void webview?.executeJavaScript?.(buildFlashCall(item.target.rect)).catch(() => undefined)
    },
    [webview]
  )

  removeItemRef.current = removeItem
  locateItemRef.current = locateItem

  const handleDiscard = useCallback(() => {
    setPending(null)
    resumePicking()
  }, [resumePicking])

  const handleFinish = useCallback(() => {
    void submitCollected().then(submitted => {
      if (!submitted) {
        return
      }
      notify({ message: copy.sentToComposer, kind: 'success' })
      exit()
    })
  }, [copy.sentToComposer, exit, submitCollected])

  return (
    <>
      <AnnotationToolbar
        copy={{
          cancel: copy.cancelSession,
          finish: copy.finishSession,
          locate: copy.locate,
          remove: copy.remove,
          title: count => copy.sessionTitle(count)
        }}
        items={items.map(
          (item): AnnotationListEntry => ({
            commentPreview: item.comment.trim().slice(0, 40),
            id: item.id,
            number: item.number,
            summary: listSummary(item)
          })
        )}
        onCancel={exit}
        onFinish={handleFinish}
        onLocate={id => {
          const item = itemsRef.current.find(entry => entry.id === id)
          if (item) {
            locateItem(item)
          }
        }}
        onRemove={removeItem}
      />

      {pending && (
        <AnnotationPopover
          kind={pending.kind}
          number={items.length + 1}
          onAdd={handleAdd}
          onDiscard={handleDiscard}
          screenshotDataUrl={pending.screenshot}
          target={pending.target}
        />
      )}
    </>
  )
}

/** Viewport size is needed to clamp the capture rect to page bounds. */
async function measureViewport(webview: AnnotatableWebview): Promise<{ pageHeight: number; pageWidth: number }> {
  try {
    const size = (await webview.executeJavaScript?.(
      '({ pageWidth: window.innerWidth, pageHeight: window.innerHeight })'
    )) as { pageHeight: number; pageWidth: number } | undefined

    if (size && size.pageWidth > 0 && size.pageHeight > 0) {
      return size
    }
  } catch {
    // fall through to default
  }

  return { pageHeight: 1080, pageWidth: 1920 }
}
