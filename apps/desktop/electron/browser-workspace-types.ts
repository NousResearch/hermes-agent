export interface BrowserConversation {
  kind: 'session' | 'group'
  id: string
  connectionId: string
  profile: string
}

export type BrowserWorkspaceDestination = (
  | { kind: 'composer'; surfaceId: string; target: string; windowId: string }
  | { kind: 'group'; composerKey: string; group: string; windowId: string }
) & { conversation?: BrowserConversation }

export interface BrowserWorkspaceTab {
  id: `url:${string}`
  sessionId?: string
  /** Explicit on every new transfer; absence is never a legacy migration. */
  pinned?: boolean
  /** Memory-only ownership before the stored session id binds. */
  pendingRuntimeId?: string
  target: { kind: 'url'; label: string; source: string; url: string }
}

export interface BrowserWorkspaceOpen {
  tab: BrowserWorkspaceTab
  scope: string
  destination: BrowserWorkspaceDestination | null
}

export interface BrowserWorkspaceOwner {
  conversation?: BrowserConversation
  registryScoped?: boolean
  connectionId: string | null
  profile: string
  scope: string
  destination: BrowserWorkspaceDestination | null
}

export interface BrowserWorkspace {
  id: string
  revision: number
  owner: BrowserWorkspaceOwner
  tabs: BrowserWorkspaceTab[]
  activeTabId: string | null
  /** Target epoch: active-tab or membership changes retire in-flight requests. */
  selectionVersion: number
  /** Explicit intent, including reselecting the same tab to reclaim the opener. */
  selectionIntentVersion: number
  /** Includes retired IDs so a renderer recovering after reload removes them. */
  removed: string[]
  docked: BrowserWorkspaceTab[]
  closed: boolean
  /** Terminal profile transition: never replay a transfer into the old scope. */
  profileRetirement?: { replacementScope?: string }
}

export type BrowserWorkspaceCommand =
  | { kind: 'new' }
  | { kind: 'select' | 'close' | 'dock'; tabId: string }
  | { kind: 'page'; tabId: string; url: string; title: string }

export interface BrowserRequestTarget {
  windowId: string
  tabId: string
  /** Captured target epoch, not the count of same-tab interaction notices. */
  selectionVersion: number
  owner: BrowserWorkspaceOwner
}

export interface BrowserProfileRetirement {
  connectionId: string
  profile: string
  replacementProfile?: string
}

export interface BrowserTabOwnership {
  id: string
  sessionId?: string
  pinned: boolean
  pendingRuntimeId?: string
}

export interface BrowserOwnershipUpdate {
  tabs: BrowserTabOwnership[]
  rotation?: { previousId: string; nextId: string }
}

export interface BrowserWorkspaceApi {
  updateOwnership?: (update: BrowserOwnershipUpdate) => void
  retireProfile?: (change: BrowserProfileRetirement) => Promise<BrowserWorkspace[]>
  retireSession?: (sessionId: string) => void
  acknowledge?: (windowId: string, revision: number) => void
  snapshots: () => Promise<BrowserWorkspace[]>
  command: (windowId: string, command: BrowserWorkspaceCommand) => Promise<BrowserWorkspace | null>
  onChanged: (callback: (workspace: BrowserWorkspace) => void) => () => void
  setShortcuts: (bindings: Record<string, string[]>) => void
  onShortcut: (callback: (action: string) => void) => () => void
}

export function sameBrowserOwner(a: BrowserWorkspaceOwner, b: BrowserWorkspaceOwner): boolean {
  return (
    sameBrowserConversation(a.conversation, b.conversation) &&
    Boolean(a.registryScoped) === Boolean(b.registryScoped) &&
    a.connectionId === b.connectionId &&
    a.profile === b.profile &&
    a.scope === b.scope &&
    JSON.stringify(a.destination) === JSON.stringify(b.destination)
  )
}

export function isBrowserConversation(value: unknown): value is BrowserConversation {
  if (!value || typeof value !== 'object') {return false}
  const row = value as BrowserConversation

  return (row.kind === 'session' || row.kind === 'group') &&
    [row.id, row.connectionId, row.profile].every(item => typeof item === 'string' && Boolean(item.trim()))
}

export function sameBrowserConversation(a: BrowserConversation | undefined, b: BrowserConversation | undefined): boolean {
  return Boolean(a && b && a.kind === b.kind && a.id === b.id && a.connectionId === b.connectionId && a.profile === b.profile)
}

/** A shared pin grants use, not a new annotation destination. */
export function browserTabAllowsRequest(owner: BrowserWorkspaceOwner, tab: BrowserWorkspaceTab, requester: BrowserConversation | undefined): boolean {
  return isBrowserConversation(requester) &&
    requester.connectionId === owner.connectionId && requester.profile === owner.profile &&
    (Boolean(tab.pinned) || (tab.sessionId !== undefined || tab.pendingRuntimeId !== undefined
      ? requester.kind === 'session' && requester.id === (tab.sessionId ?? tab.pendingRuntimeId)
      : sameBrowserConversation(owner.conversation, requester)))
}

export function isBrowserLocation(url: unknown): url is string {
  if (typeof url !== 'string' || url.length > 32_768) {
    return false
  }

  if (url === 'about:blank') {
    return true
  }

  try {
    return ['http:', 'https:'].includes(new URL(url).protocol)
  } catch {
    return false
  }
}
