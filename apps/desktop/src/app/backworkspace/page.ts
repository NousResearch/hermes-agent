import type { BackworkspaceOpenResult, BackworkspaceSaveResult } from '@hermes/shared'
import { atom } from 'nanostores'

import { isMissingRpcMethod } from '@/lib/gateway-rpc'
import { requestGatewayForAgent } from '@/store/gateway'

export type BackworkspacePageStatus = 'error' | 'loading' | 'ready' | 'unsupported'

/** The backend that owns a page. A null connection keeps the legacy profile
 *  resolver (a window live on a source the registry cannot name); it must not
 *  be turned into the `local` registry source. */
export interface BackworkspaceRoute {
  connectionId: null | string
  profile: string
}

export interface BackworkspacePageState {
  /** `connection:profile` — the backend that owns this page. */
  key: string
  route: BackworkspaceRoute
  status: BackworkspacePageStatus
  content: string
  saveFailed: boolean
}

const SAVE_DEBOUNCE_MS = 400

/** The page shown on the back of this window: a cache of the owning backend's file. */
export const $backworkspacePage = atom<BackworkspacePageState | null>(null)

// Page id per owner. A save reads it when it RUNS, not when it is queued, so
// every save behind the first one reuses the id the first one minted instead
// of each starting another page.
const pageIds = new Map<string, string>()
// Last text handed to the save queue per owner; an unchanged page is not re-sent.
const queuedContent = new Map<string, string>()
// Text whose latest save failed, per owner. It outlives a switch to another
// profile, so coming back shows (and retries) it instead of the older file.
const unsavedContent = new Map<string, string>()
let saveTimer: ReturnType<typeof setTimeout> | undefined
let saveQueue: Promise<void> = Promise.resolve()
let loadGeneration = 0

export function backworkspaceOwnerKey(route: BackworkspaceRoute): string {
  return `${route.connectionId ?? ''}:${route.profile}`
}

function request<T>(route: BackworkspaceRoute, method: string, params: Record<string, unknown>): Promise<T> {
  return requestGatewayForAgent<T>(route.connectionId, route.profile, method, params, undefined, undefined, {
    spawnPriority: 'foreground'
  })
}

function patchPage(key: string, next: Partial<BackworkspacePageState>) {
  const current = $backworkspacePage.get()

  if (current?.key === key) {
    $backworkspacePage.set({ ...current, ...next })
  }
}

/**
 * Show `route`'s latest page. The same owner reopening keeps the text already
 * in memory — it is newer than anything the backend could return while a save
 * is still in flight.
 */
export async function loadBackworkspacePage(route: BackworkspaceRoute): Promise<void> {
  const key = backworkspaceOwnerKey(route)
  const current = $backworkspacePage.get()

  if (current?.key === key && current.status !== 'error') {
    return
  }

  const generation = ++loadGeneration
  // Queue the outgoing owner's text before this page replaces it in memory.
  const pendingSaves = flushBackworkspacePage()

  $backworkspacePage.set({ content: '', key, route, saveFailed: false, status: 'loading' })

  try {
    // Read only after every save queued so far landed, so a quick A → B → A
    // switch cannot read A's file from before its last save.
    await pendingSaves
    const { page } = await request<BackworkspaceOpenResult>(route, 'backworkspace.open', {})

    // A later load owns the page now; this answer is from the past.
    if (generation !== loadGeneration) {
      return
    }

    if (page) {
      pageIds.set(key, page.id)
    }

    const unsaved = unsavedContent.get(key)
    const content = unsaved ?? page?.content ?? ''

    if (unsaved === undefined) {
      queuedContent.set(key, content)
    } else {
      queuedContent.delete(key)
    }

    patchPage(key, { content, saveFailed: unsaved !== undefined, status: 'ready' })

    if (unsaved !== undefined) {
      void flushBackworkspacePage()
    }
  } catch (error) {
    if (generation === loadGeneration) {
      patchPage(key, { status: isMissingRpcMethod(error) ? 'unsupported' : 'error' })
    }
  }
}

async function savePage(key: string, route: BackworkspaceRoute, content: string): Promise<void> {
  try {
    const { id } = await request<BackworkspaceSaveResult>(route, 'backworkspace.save', {
      content,
      id: pageIds.get(key)
    })

    // Saves run in queue order, so a success supersedes any earlier failure.
    pageIds.set(key, id)
    unsavedContent.delete(key)
    patchPage(key, { saveFailed: false })
  } catch {
    // Forget what was queued so the next edit or flush sends this text again.
    queuedContent.delete(key)
    unsavedContent.set(key, content)
    patchPage(key, { saveFailed: true })
  }
}

/** Send the current text now (flip back, owner switch, window close). Resolves once every save queued so far settled. */
export function flushBackworkspacePage(): Promise<void> {
  clearTimeout(saveTimer)
  saveTimer = undefined

  const current = $backworkspacePage.get()

  if (current?.status === 'ready' && queuedContent.get(current.key) !== current.content) {
    const { content, key, route } = current

    queuedContent.set(key, content)
    saveQueue = saveQueue.then(() => savePage(key, route, content))
  }

  return saveQueue
}

export function editBackworkspacePage(content: string) {
  const current = $backworkspacePage.get()

  if (current?.status !== 'ready') {
    return
  }

  $backworkspacePage.set({ ...current, content })
  clearTimeout(saveTimer)
  saveTimer = setTimeout(() => void flushBackworkspacePage(), SAVE_DEBOUNCE_MS)
}
