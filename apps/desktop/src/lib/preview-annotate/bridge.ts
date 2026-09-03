import { requestComposerAttachImages, requestComposerFocus, requestComposerInsert } from '@/app/chat/composer/focus'
import { notifyError } from '@/store/notifications'

import { annotateFlushPrompt, type ComposerReadyAnnotation, dataUrlToFile, packageAnnotateStack } from './pack'
import type { AnnotatePin } from './stack'

/**
 * Cross-window annotate-flush relay (popped-out Browser → primary window).
 *
 * The pop-out (`?win=browser`) renders the preview pane with NO composer, so
 * the same-window composer bus (`hermes:composer-*` CustomEvents) has no
 * subscriber there and a flush would vanish. The pop-out instead packages its
 * pin stack and hands the envelope to the main process, which forwards it to
 * the primary window — the one renderer that runs this module's receiver and
 * owns a composer.
 */
export interface AnnotateFlushEnvelope {
  id: string
  items: ComposerReadyAnnotation[]
  pageUrl?: string
}

export interface AnnotateFlushPostResult {
  delivered: boolean
}

function makeEnvelopeId(): string {
  try {
    return crypto.randomUUID()
  } catch {
    return `flush-${Date.now().toString(36)}-${Math.floor(Math.random() * 0xffffffff).toString(36)}`
  }
}

type AnnotateFlushBridge = {
  onAnnotateFlushed?: (callback: (envelope: AnnotateFlushEnvelope) => void) => () => void
  postAnnotateFlush?: (envelope: AnnotateFlushEnvelope) => Promise<{ error?: string; ok: boolean }>
}

function bridge(): AnnotateFlushBridge | null {
  if (typeof window === 'undefined') {
    return null
  }

  const candidate = (window as unknown as { hermesDesktop?: AnnotateFlushBridge }).hermesDesktop

  if (!candidate || typeof candidate.postAnnotateFlush !== 'function') {
    return null
  }

  return candidate
}

/**
 * Package `pins` and post them to the primary window's composer. Resolves
 * `{ delivered: true }` only when the main process accepted the envelope;
 * resolves `{ delivered: false }` when this renderer has no flush bridge
 * (older shell) and rejects when the main process reports a failure (e.g.
 * the primary window is gone) — the caller keeps its pins in both cases.
 */
export async function postPopoutAnnotateFlush(
  pins: readonly AnnotatePin[],
  pageUrl?: string
): Promise<AnnotateFlushPostResult> {
  const candidate = bridge()

  if (!candidate?.postAnnotateFlush) {
    return { delivered: false }
  }

  const envelope: AnnotateFlushEnvelope = {
    id: makeEnvelopeId(),
    items: packageAnnotateStack(pins),
    pageUrl
  }

  const result = await candidate.postAnnotateFlush(envelope)

  return { delivered: result?.ok === true }
}

function isEnvelope(value: unknown): value is AnnotateFlushEnvelope {
  if (!value || typeof value !== 'object') {
    return false
  }

  const envelope = value as { id?: unknown; items?: unknown }

  return typeof envelope.id === 'string' && Array.isArray(envelope.items)
}

/**
 * Subscribe the primary window to pop-out flushes. Incoming envelopes are
 * attached to the local composer exactly like a docked-pane flush (numbered
 * crops + packed prompt, never auto-sent). Install once per renderer, in
 * composer-bearing windows only — the pop-out itself must never install this,
 * or its own flush would loop back onto a composer that isn't there.
 */
export function installAnnotateFlushReceiver(): () => void {
  if (typeof window === 'undefined') {
    return () => undefined
  }

  const candidate = (window as unknown as { hermesDesktop?: AnnotateFlushBridge }).hermesDesktop

  if (!candidate || typeof candidate.onAnnotateFlushed !== 'function') {
    return () => undefined
  }

  return candidate.onAnnotateFlushed(envelope => {
    try {
      if (!isEnvelope(envelope) || envelope.items.length === 0) {
        return
      }

      const files = envelope.items
        .filter(item => typeof item.imageDataUrl === 'string' && item.imageDataUrl.length > 0)
        .map(item => dataUrlToFile(item.imageDataUrl, `Comment_${item.number}.png`))

      if (files.length > 0) {
        requestComposerAttachImages(files)
      }

      requestComposerInsert(annotateFlushPrompt(envelope.items, envelope.pageUrl), { mode: 'block' })
      requestComposerFocus()
    } catch (error) {
      notifyError(error, 'Could not attach browser comments')
    }
  })
}
