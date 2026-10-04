import { backendScopeKey } from '@hermes/shared'
import { atom } from 'nanostores'

import { capturePreviewAnnotateDestination } from '@/lib/preview-annotate/handoff'
import { $rightRailActiveTabId, selectRightRailTab } from '@/store/layout'
import { noteExplicitPreviewOpen } from '@/store/preview-explicit'

import {
  type BrowserConversation,
  type BrowserProfileRetirement,
  type BrowserRequestTarget,
  browserTabAllowsRequest,
  type BrowserWorkspace,
  type BrowserWorkspaceCommand,
  sameBrowserConversation,
  sameBrowserOwner
} from '../../electron/browser-workspace-types'

import { $previewTabs, applyBrowserWorkspace, previewTabIdsVisibleTo } from './preview'
import { windowBrowserWorkspaceId } from './windows'

export const $browserWorkspaces = atom<Record<string, BrowserWorkspace>>({})

export function receiveBrowserWorkspace(state: BrowserWorkspace) {
  const previous = $browserWorkspaces.get()[state.id]

  if (previous && previous.revision >= state.revision) {
    return
  }

  const ownId = windowBrowserWorkspaceId()

  if (ownId && ownId !== state.id) {
    return
  }

  $browserWorkspaces.set({ ...$browserWorkspaces.get(), [state.id]: state })
  applyBrowserWorkspace(state, previous)

  // Selection intent is an explicit detached gesture; page reports are not. Keep
  // the opener's layout-only follow from replacing it with a docked sibling.
  // A background conversation must never change the foreground's selection.
  // Renewing this intent does not change the target epoch of in-flight tools.
  const selected = state.tabs.find(tab => tab.id === state.activeTabId)

  if (
    !ownId && previous && !state.closed && selected &&
    state.selectionIntentVersion !== previous.selectionIntentVersion &&
    sameBrowserConversation(state.owner.conversation, capturePreviewAnnotateDestination()?.conversation)
  ) {
    noteExplicitPreviewOpen(selected.id)
    selectRightRailTab(selected.id)
  }

  if (!ownId) {window.hermesDesktop?.browserWorkspace?.acknowledge?.(state.id, state.revision)}
}

let installed = false

/** Await main's terminal snapshots before deleting/moving persisted buckets. */
export function retireBrowserProfile(change: BrowserProfileRetirement): Promise<void> | undefined {
  return window.hermesDesktop?.browserWorkspace?.retireProfile?.(change).then(states => {
    for (const state of states) {receiveBrowserWorkspace(state)}
  })
}

/** Subscribe before fetching: a command can land while cold hydration is in flight. */
export function installBrowserWorkspaceSync(): () => void {
  const api = window.hermesDesktop?.browserWorkspace

  if (!api || installed) {
    return () => {}
  }

  installed = true
  let alive = true
  const stop = api.onChanged(receiveBrowserWorkspace)
  void api.snapshots().then(states => {
    if (alive) {
      for (const state of states) {
        receiveBrowserWorkspace(state)
      }
    }
  })

  return () => {
    alive = false
    installed = false
    stop()
  }
}

export async function commandBrowserWorkspace(command: BrowserWorkspaceCommand): Promise<BrowserWorkspace | null> {
  const id = windowBrowserWorkspaceId()

  if (!id) {
    return null
  }

  const state = await window.hermesDesktop?.browserWorkspace?.command(id, command)

  if (state) {
    receiveBrowserWorkspace(state)
  }

  return state ?? null
}

/** Capture selection once, in the authorized originating chat renderer. A
 * missing/ambiguous owner is not permission to use the first browser window. */
export function selectedPopoutTarget(
  conversation: BrowserConversation | null = capturePreviewAnnotateDestination()?.conversation ?? null,
  authorizedTabIds?: readonly string[]
): BrowserRequestTarget | null {
  if (!conversation) {
    return null
  }

  const allowed = new Set(authorizedTabIds ?? previewTabIdsVisibleTo({
    profile: backendScopeKey(conversation.connectionId, conversation.profile),
    runtimeId: conversation.kind === 'session' ? conversation.id : null,
    sessionId: conversation.kind === 'session' ? conversation.id : null
  }))

  const candidates = Object.values($browserWorkspaces.get()).filter(
    state =>
      !state.closed &&
      state.activeTabId &&
      allowed.has(state.activeTabId) &&
      browserTabAllowsRequest(state.owner, state.tabs.find(tab => tab.id === state.activeTabId)!, conversation)
  )

  const activeId = $rightRailActiveTabId.get()
  const explicit = candidates.find(state => state.tabs.some(tab => tab.id === activeId))

  // A selected local preview wins over an unrelated detached window. Only an
  // absent/stale local selection may use the sole unambiguous owned workspace.
  if (!explicit && sameBrowserConversation(conversation, capturePreviewAnnotateDestination()?.conversation) && $previewTabs.get().some(tab => tab.id === activeId)) {
    return null
  }

  const selected = explicit ?? (candidates.length === 1 ? candidates[0] : null)

  return selected?.activeTabId
    ? {
        windowId: selected.id,
        tabId: selected.activeTabId,
        selectionVersion: selected.selectionVersion,
        owner: selected.owner
      }
    : null
}

export function selectedDetachedBrowser(): boolean {
  return Object.values($browserWorkspaces.get()).some(state => !state.closed && state.tabs.some(tab => tab.id === $rightRailActiveTabId.get()))
}

export function validPopoutTarget(target: BrowserRequestTarget): boolean {
  if (!target || typeof target !== 'object' || !target.owner || typeof target.owner !== 'object') {
    return false
  }

  const state = $browserWorkspaces.get()[target.windowId]

  return (
    target.windowId === windowBrowserWorkspaceId() &&
    Boolean(
      state &&
      !state.closed &&
      state.selectionVersion === target.selectionVersion &&
      state.activeTabId === target.tabId &&
      sameBrowserOwner(state.owner, target.owner) &&
      state.tabs.some(tab => tab.id === target.tabId)
    )
  )
}
