import { queryAllVisible, queryVisible } from '@/components/pane-shell/pane-visibility'
import { $workspaceOwnerKey } from '@/components/pane-shell/workspace-scope'

import { annotateFlushPrompt, packageAnnotateStack } from './pack'
import type { AnnotatePin } from './stack'

/**
 * Cross-window handoff for Browser Comment Mode.
 *
 * A popped-out Browser is a separate Electron renderer. The ordinary composer
 * bus is intentionally window-local, so an annotation saved in that renderer
 * cannot reach the chat/group composer that opened the Browser. This bridge
 * pins the destination at pop-out time and relays payload/ack messages through
 * Electron IPC when available, with BroadcastChannel as a browser/dev fallback. The destination is exact and fail-closed:
 * a stale/missing owner never falls through to whatever composer happens to be
 * active later.
 */

export const PREVIEW_ANNOTATE_HANDOFF_CHANNEL = 'hermes.desktop.preview-annotate-handoff.v1'
export const PREVIEW_ANNOTATE_WINDOW_ID_KEY = 'hermes.desktop.previewAnnotate.windowId'

const DESTINATION_PREFIX = 'hermes.desktop.previewAnnotate.destination.v1:'
const DESTINATION_MAX_AGE_MS = 24 * 60 * 60 * 1000

export interface PreviewAnnotateComposerDestination {
  kind: 'composer'
  surfaceId: string
  target: string
  windowId: string
}

export interface PreviewAnnotateGroupDestination {
  composerKey: string
  group: string
  kind: 'group'
  windowId: string
}

export type PreviewAnnotateDestination = PreviewAnnotateComposerDestination | PreviewAnnotateGroupDestination

interface StoredDestination {
  at: number
  destination: PreviewAnnotateDestination
  version: 1
}

export interface PreviewAnnotateHandoffImage {
  dataUrl: string
  name: string
  number: number
}

export interface PreviewAnnotateHandoffRequest {
  count: number
  destination: PreviewAnnotateDestination
  images: PreviewAnnotateHandoffImage[]
  prompt: string
  requestId: string
  tabId: string
  type: 'preview-annotate-handoff'
}

export interface PreviewAnnotateHandoffAck {
  error?: string
  ok: boolean
  requestId: string
  type: 'preview-annotate-handoff-ack'
}

export interface PreviewAnnotateHandoffResult {
  error?: string
  ok: boolean
}

const randomId = () =>
  globalThis.crypto?.randomUUID?.() ?? `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`

/** One id per renderer window. sessionStorage is deliberately window-scoped. */
export function previewAnnotateWindowId(): string {
  if (typeof window === 'undefined') {
    return 'server'
  }

  try {
    const existing = window.sessionStorage.getItem(PREVIEW_ANNOTATE_WINDOW_ID_KEY)?.trim()

    if (existing) {
      return existing
    }

    const created = randomId()
    window.sessionStorage.setItem(PREVIEW_ANNOTATE_WINDOW_ID_KEY, created)

    return created
  } catch {
    // sessionStorage can be disabled. Keep the id stable for this module load.
    return fallbackWindowId
  }
}

const fallbackWindowId = randomId()

/**
 * Capture the exact surface that should receive Browser comments.
 * Group rooms opt in with data attributes because their New Thread composer is
 * not part of the ordinary chat composer bus.
 */
function rectDistance(a: DOMRect, b: DOMRect): number {
  const dx = Math.max(a.left - b.right, b.left - a.right, 0)
  const dy = Math.max(a.top - b.bottom, b.top - a.bottom, 0)

  if (dx || dy) {
    return Math.hypot(dx, dy)
  }

  const ax = (a.left + a.right) / 2
  const ay = (a.top + a.bottom) / 2
  const bx = (b.left + b.right) / 2
  const by = (b.top + b.bottom) / 2

  return Math.hypot(ax - bx, ay - by)
}

function nearestGroupToAnchor(groups: HTMLElement[], anchor: Element | null): HTMLElement | undefined {
  if (!anchor || groups.length < 2 || typeof anchor.getBoundingClientRect !== 'function') {
    return undefined
  }

  const anchorRect = anchor.getBoundingClientRect()

  if (!anchorRect.width && !anchorRect.height) {
    return undefined
  }

  return groups
    .map(group => ({ distance: rectDistance(group.getBoundingClientRect(), anchorRect), group }))
    .sort((a, b) => a.distance - b.distance)[0]?.group
}

export function capturePreviewAnnotateDestination(anchor: Element | null = null): PreviewAnnotateDestination | null {
  if (typeof document === 'undefined') {
    return null
  }

  const visibleGroups = queryAllVisible<HTMLElement>('[data-preview-annotate-destination="group"]')
  const workspaceOwnerKey = $workspaceOwnerKey.get()

  // The Browser lives in its own right-rail zone, while multiple group rooms
  // can be visible side by side. The global workspace owner can therefore
  // still point at the *other* room when the user clicks the Browser's Pop out
  // glyph. Prefer the room physically nearest that glyph; this binds the
  // Browser to the pane it is actually sitting beside. Fall back to the
  // workspace owner for keyboard/context-menu opens where no useful anchor is
  // available, and fail closed rather than picking the first visible room.
  const group =
    nearestGroupToAnchor(visibleGroups, anchor) ??
    (workspaceOwnerKey
      ? visibleGroups.find(item => item.dataset.previewAnnotateOwnerKey === workspaceOwnerKey)
      : undefined) ??
    (visibleGroups.length === 1 ? visibleGroups[0] : undefined)

  const groupName = group?.dataset.previewAnnotateGroup?.trim()
  const composerKey = group?.dataset.previewAnnotateComposerKey?.trim()

  if (groupName && composerKey) {
    return {
      composerKey,
      group: groupName,
      kind: 'group',
      windowId: previewAnnotateWindowId()
    }
  }

  const composer = queryVisible<HTMLElement>('[data-composer-target]')
  const target = composer?.dataset.composerTarget?.trim()
  const surfaceId = composer?.dataset.composerSurfaceId?.trim()

  if (target && surfaceId) {
    return {
      kind: 'composer',
      surfaceId,
      target,
      windowId: previewAnnotateWindowId()
    }
  }

  return null
}

const destinationKey = (tabId: string) => `${DESTINATION_PREFIX}${tabId}`

export function rememberPreviewAnnotateDestination(tabId: string, destination: PreviewAnnotateDestination | null) {
  if (typeof window === 'undefined' || !tabId) {
    return
  }

  try {
    if (!destination) {
      window.localStorage.removeItem(destinationKey(tabId))

      return
    }

    const record: StoredDestination = { at: Date.now(), destination, version: 1 }
    window.localStorage.setItem(destinationKey(tabId), JSON.stringify(record))
  } catch {
    // The pop-out still opens; Add comments will fail closed with guidance.
  }
}

export function clearPreviewAnnotateDestination(tabId: string) {
  if (typeof window === 'undefined' || !tabId) {
    return
  }

  try {
    window.localStorage.removeItem(destinationKey(tabId))
  } catch {
    /* storage unavailable */
  }
}

function isDestination(value: unknown): value is PreviewAnnotateDestination {
  if (!value || typeof value !== 'object') {
    return false
  }

  const row = value as Record<string, unknown>

  if (row.kind === 'group') {
    return (
      typeof row.windowId === 'string' &&
      typeof row.group === 'string' &&
      typeof row.composerKey === 'string' &&
      Boolean(row.windowId.trim() && row.group.trim() && row.composerKey.trim())
    )
  }

  return (
    row.kind === 'composer' &&
    typeof row.windowId === 'string' &&
    typeof row.target === 'string' &&
    typeof row.surfaceId === 'string' &&
    Boolean(row.windowId.trim() && row.target.trim() && row.surfaceId.trim())
  )
}

export function readPreviewAnnotateDestination(tabId: string): PreviewAnnotateDestination | null {
  if (typeof window === 'undefined' || !tabId) {
    return null
  }

  try {
    const raw = window.localStorage.getItem(destinationKey(tabId))

    if (!raw) {
      return null
    }

    const record = JSON.parse(raw) as Partial<StoredDestination>

    if (
      record.version !== 1 ||
      typeof record.at !== 'number' ||
      Date.now() - record.at > DESTINATION_MAX_AGE_MS ||
      !isDestination(record.destination)
    ) {
      clearPreviewAnnotateDestination(tabId)

      return null
    }

    return record.destination
  } catch {
    return null
  }
}

const isAck = (value: unknown): value is PreviewAnnotateHandoffAck => {
  if (!value || typeof value !== 'object') {
    return false
  }

  const row = value as Record<string, unknown>

  return row.type === 'preview-annotate-handoff-ack' && typeof row.requestId === 'string' && typeof row.ok === 'boolean'
}

export function isPreviewAnnotateHandoffRequest(value: unknown): value is PreviewAnnotateHandoffRequest {
  if (!value || typeof value !== 'object') {
    return false
  }

  const row = value as Record<string, unknown>

  return (
    row.type === 'preview-annotate-handoff' &&
    typeof row.requestId === 'string' &&
    typeof row.tabId === 'string' &&
    typeof row.prompt === 'string' &&
    typeof row.count === 'number' &&
    Array.isArray(row.images) &&
    isDestination(row.destination)
  )
}

/** Register one exact destination receiver in this renderer window. */
export function subscribePreviewAnnotateHandoff(
  handler: (
    request: PreviewAnnotateHandoffRequest
  ) => PreviewAnnotateHandoffResult | null | Promise<PreviewAnnotateHandoffResult | null>
): () => void {
  const desktopBridge = typeof window !== 'undefined' ? window.hermesDesktop?.previewAnnotate : undefined

  if (desktopBridge?.onMessage && desktopBridge.send) {
    return desktopBridge.onMessage(payload => {
      if (!isPreviewAnnotateHandoffRequest(payload)) {
        return
      }

      const request = payload

      if (request.destination.windowId !== previewAnnotateWindowId()) {
        return
      }

      void Promise.resolve(handler(request))
        .then(result => {
          if (!result) {
            return
          }

          const ack: PreviewAnnotateHandoffAck = {
            ...(result.error ? { error: result.error } : {}),
            ok: result.ok,
            requestId: request.requestId,
            type: 'preview-annotate-handoff-ack'
          }

          desktopBridge.send(ack)
        })
        .catch(error => {
          desktopBridge.send({
            error: error instanceof Error ? error.message : String(error),
            ok: false,
            requestId: request.requestId,
            type: 'preview-annotate-handoff-ack'
          } satisfies PreviewAnnotateHandoffAck)
        })
    })
  }

  if (typeof BroadcastChannel === 'undefined') {
    return () => undefined
  }

  const channel = new BroadcastChannel(PREVIEW_ANNOTATE_HANDOFF_CHANNEL)

  const listener = (event: MessageEvent<unknown>) => {
    if (!isPreviewAnnotateHandoffRequest(event.data)) {
      return
    }

    const request = event.data

    if (request.destination.windowId !== previewAnnotateWindowId()) {
      return
    }

    void Promise.resolve(handler(request))
      .then(result => {
        if (!result) {
          return
        }

        const ack: PreviewAnnotateHandoffAck = {
          ...(result.error ? { error: result.error } : {}),
          ok: result.ok,
          requestId: request.requestId,
          type: 'preview-annotate-handoff-ack'
        }

        channel.postMessage(ack)
      })
      .catch(error => {
        const ack: PreviewAnnotateHandoffAck = {
          error: error instanceof Error ? error.message : String(error),
          ok: false,
          requestId: request.requestId,
          type: 'preview-annotate-handoff-ack'
        }

        channel.postMessage(ack)
      })
  }

  channel.addEventListener('message', listener)

  return () => {
    channel.removeEventListener('message', listener)
    channel.close()
  }
}

/** Send one saved comment batch back to the renderer that owned the pop-out. */
export async function handoffPreviewAnnotateStack(
  tabId: string,
  pins: readonly AnnotatePin[],
  pageUrl?: string
): Promise<PreviewAnnotateHandoffResult> {
  if (!pins.length) {
    return { ok: true }
  }

  const destination = readPreviewAnnotateDestination(tabId)

  if (!destination) {
    return {
      error:
        'The Browser pop-out has no pinned chat destination. Pop it back in, open it again from the target chat, and retry.',
      ok: false
    }
  }

  if (typeof BroadcastChannel === 'undefined') {
    return { error: 'Cross-window comment handoff is unavailable in this renderer.', ok: false }
  }

  const items = packageAnnotateStack(pins)
  const requestId = randomId()

  const request: PreviewAnnotateHandoffRequest = {
    count: items.length,
    destination,
    images: items
      .filter(item => Boolean(item.imageDataUrl))
      .map(item => ({ dataUrl: item.imageDataUrl, name: `Comment_${item.number}.png`, number: item.number })),
    prompt: annotateFlushPrompt(items, pageUrl),
    requestId,
    tabId,
    type: 'preview-annotate-handoff'
  }

  const desktopBridge = typeof window !== 'undefined' ? window.hermesDesktop?.previewAnnotate : undefined

  if (desktopBridge?.onMessage && desktopBridge.send) {
    return new Promise(resolve => {
      let settled = false

      const finish = (result: PreviewAnnotateHandoffResult) => {
        if (settled) {
          return
        }

        settled = true
        window.clearTimeout(timer)
        stop()
        resolve(result)
      }

      const stop = desktopBridge.onMessage(payload => {
        if (!isAck(payload) || payload.requestId !== requestId) {
          return
        }

        finish({ ...(payload.error ? { error: payload.error } : {}), ok: payload.ok })
      })

      const timer = window.setTimeout(
        () =>
          finish({
            error: 'The original chat did not accept the comments. Keep the annotations and reopen the target chat.',
            ok: false
          }),
        4000
      )

      try {
        desktopBridge.send(request)
      } catch (error) {
        finish({ error: error instanceof Error ? error.message : String(error), ok: false })
      }
    })
  }

  return new Promise(resolve => {
    const channel = new BroadcastChannel(PREVIEW_ANNOTATE_HANDOFF_CHANNEL)
    let settled = false

    const finish = (result: PreviewAnnotateHandoffResult) => {
      if (settled) {
        return
      }

      settled = true
      window.clearTimeout(timer)
      channel.close()
      resolve(result)
    }

    channel.addEventListener('message', event => {
      if (!isAck(event.data) || event.data.requestId !== requestId) {
        return
      }

      finish({ ...(event.data.error ? { error: event.data.error } : {}), ok: event.data.ok })
    })

    const timer = window.setTimeout(
      () =>
        finish({
          error: 'The original chat did not accept the comments. Keep the annotations and reopen the target chat.',
          ok: false
        }),
      4000
    )

    try {
      channel.postMessage(request)
    } catch (error) {
      finish({ error: error instanceof Error ? error.message : String(error), ok: false })
    }
  })
}
