import { randomUUID } from 'node:crypto'

import { backendScopeKey } from '../../shared/src/backend-scope'

import { BROWSER_REQUEST_HISTORY_LIMIT, BrowserRequestHistory } from './browser-request-history'
import {
  type BrowserConversation,
  type BrowserOwnershipUpdate,
  type BrowserProfileRetirement,
  type BrowserRequestTarget,
  browserTabAllowsRequest,
  type BrowserWorkspace,
  type BrowserWorkspaceCommand,
  type BrowserWorkspaceOpen,
  isBrowserConversation,
  isBrowserLocation,
  sameBrowserConversation,
  sameBrowserOwner
} from './browser-workspace-types'

interface Entry {
  state: BrowserWorkspace
  opener: number
  renderer?: number
}

/** The trusted chat renderer supplies its established conversation owner, not
 * its active socket. Validate the route against main's connection registry. */
export function browserWorkspaceRoute(request: BrowserWorkspaceOpen, registry: { connections: readonly { id: string }[] }) {
  const owner = request?.destination?.conversation

  return isBrowserConversation(owner) && registry.connections.some(connection => connection.id === owner.connectionId)
    ? { connectionId: owner.connectionId, profile: owner.profile, registryScoped: true }
    : null
}

/** Runtime authority, not session restore. A process restart restores persisted
 * tabs docked; no persisted hidden bit can strand a page without a window. */
export class BrowserWorkspaces {
  private entries = new Map<string, Entry>()
  private renderers = new Set<number>()
  private sessionAliases = new Map<number, Map<string, string>>()
  private comments = new Map<string, { opener: number; renderer: number; windowId: string; at: number }>()
  private receivedRequests = new BrowserRequestHistory()
  // Active cancellation routes survive target retirement and deadline expiry.
  // Bound admission rather than evicting handles for operations still running.
  private pending = new Map<
    string,
    { opener: number; renderer: number; target: BrowserRequestTarget; kind: string }
  >()

  constructor(private changed: (recipients: number[], state: BrowserWorkspace) => void) {}

  open(
    opener: number,
    request: BrowserWorkspaceOpen,
    route: { connectionId: string | null; profile?: string; registryScoped?: boolean }
  ): BrowserWorkspace {
    if (
      !request ||
      typeof request.scope !== 'string' ||
      !request.scope ||
      !request.tab?.id?.startsWith('url:') ||
      request.tab.target?.kind !== 'url' ||
      !isBrowserLocation(request.tab.target.url) ||
      typeof request.tab.target.label !== 'string' ||
      typeof request.tab.target.source !== 'string' ||
      (request.tab.pinned !== undefined && typeof request.tab.pinned !== 'boolean') ||
      (request.tab.sessionId !== undefined && typeof request.tab.sessionId !== 'string') ||
      (request.tab.pendingRuntimeId !== undefined && typeof request.tab.pendingRuntimeId !== 'string')
    ) {
      throw new Error('Invalid browser workspace')
    }

    const destination = request.destination
    const conversation = destination?.conversation

    if (!isBrowserConversation(conversation) || conversation.connectionId !== route.connectionId || conversation.profile !== route.profile) {
      throw new Error('Browser conversation route is unavailable')
    }

    if (
      destination !== null &&
      (!destination ||
        typeof destination.windowId !== 'string' ||
        !(
          (destination.kind === 'composer' &&
            conversation.kind === 'session' &&
            typeof destination.surfaceId === 'string' &&
            typeof destination.target === 'string') ||
          (destination.kind === 'group' &&
            conversation.kind === 'group' && conversation.id === destination.composerKey &&
            typeof destination.group === 'string' &&
            typeof destination.composerKey === 'string')
        ))
    ) {
      throw new Error('Invalid browser owner')
    }

    for (const entry of this.entries.values()) {
      if (!entry.state.closed && entry.state.tabs.some(tab => tab.id === request.tab.id)) {
        if (entry.opener !== opener || !sameBrowserConversation(entry.state.owner.conversation, conversation)) {
          throw new Error('Browser tab already owned by another window')
        }

        return structuredClone(entry.state)
      }
    }

    const state: BrowserWorkspace = {
      id: randomUUID(),
      revision: 0,
      owner: {
        conversation: structuredClone(conversation),
        connectionId: route.connectionId,
        profile: route.profile || 'default',
        registryScoped: Boolean(route.registryScoped),
        scope: request.scope,
        destination: structuredClone(destination)
      },
      tabs: [{ ...structuredClone(request.tab), pinned: Boolean(request.tab.pinned) }],
      activeTabId: request.tab.id,
      selectionVersion: 0,
      selectionIntentVersion: 0,
      removed: [],
      docked: [],
      closed: false
    }

    this.entries.set(state.id, { state, opener })

    return structuredClone(state)
  }

  attach(id: string, renderer: number) {
    const entry = this.entries.get(id)

    if (!entry || entry.state.closed || entry.renderer !== undefined) {
      throw new Error('Unknown browser workspace')
    }

    entry.renderer = renderer
    this.renderers.add(renderer)
    this.publish(entry)
  }

  snapshots(sender: number): BrowserWorkspace[] {
    return [...this.entries.values()]
      .filter(entry => entry.opener === sender || entry.renderer === sender)
      .map(entry => structuredClone(entry.state))
  }

  /** The opener acknowledges only after persisting the transfer. Without a
   * receipt, a renderer reload must replay missed close/dock transitions; with
   * one, replay must not resurrect a tab the user later deleted locally. */
  acknowledge(sender: number, id: string, revision: number): void {
    const entry = this.entries.get(id)

    if (!entry || entry.opener !== sender || entry.state.revision !== revision) {return}
    entry.state.docked = []
    entry.state.removed = []

    if (entry.state.closed) {this.entries.delete(id)}
  }

  ownerForRenderer(sender: number) {
    const entry = [...this.entries.values()].find(entry => entry.renderer === sender && !entry.state.closed)

    return entry ? structuredClone(entry.state.owner) : null
  }

  isRenderer(sender: number): boolean {
    return [...this.entries.values()].some(entry => entry.renderer === sender && !entry.state.closed)
  }

  /** A destroyed host cannot receive input or replay its old IPC identities. */
  forgetHost(sender: number): void {
    for (const [id, request] of this.pending) {
      if (request.opener === sender || request.renderer === sender) {this.pending.delete(id)}
    }

    for (const [id, request] of this.comments) {
      if (request.opener === sender || request.renderer === sender) {this.comments.delete(id)}
    }

    this.renderers.delete(sender)
  }

  private publish(entry: Entry) {
    entry.state.revision++
    this.changed(
      [entry.opener, ...(entry.renderer === undefined ? [] : [entry.renderer])],
      structuredClone(entry.state)
    )
  }

  command(sender: number, id: string, command: BrowserWorkspaceCommand): BrowserWorkspace | null {
    const entry = this.entries.get(id)

    if (!entry || entry.renderer !== sender || entry.state.closed || !command || typeof command !== 'object') {
      return null
    }

    const state = entry.state
    const previousActiveTabId = state.activeTabId

    if (command.kind === 'new') {
      const tab = {
        id: `url:browser-${randomUUID()}` as const,
        pinned: false,
        sessionId: state.owner.conversation?.kind === 'session' && !state.tabs.some(tab => !tab.pinned && tab.sessionId === undefined)
          ? state.owner.conversation.id : undefined,
        pendingRuntimeId: state.owner.conversation?.kind === 'session'
          ? state.tabs.find(tab => !tab.pinned && tab.sessionId === undefined)?.pendingRuntimeId : undefined,
        target: { kind: 'url' as const, label: 'Browser', source: 'about:blank', url: 'about:blank' }
      }

      state.tabs.push(tab)
      state.activeTabId = tab.id
    } else {
      if (!('tabId' in command)) {
        return null
      }

      const index = state.tabs.findIndex(tab => tab.id === command.tabId)
      const tab = state.tabs[index]

      if (!tab) {
        return null
      }

      switch (command.kind) {
        case 'select':
          state.activeTabId = tab.id

          break

        case 'page':
          if (!isBrowserLocation(command.url) || typeof command.title !== 'string') {
            return null
          }

          tab.target = { ...tab.target, url: command.url, label: command.title || tab.target.label }

          break

        case 'close':

        case 'dock':
          state.tabs.splice(index, 1)

          if (command.kind === 'dock') {
            state.docked.push(tab)
          } else {
            state.removed.push(tab.id)
          }

          if (state.activeTabId === tab.id) {
            state.activeTabId = state.tabs[Math.min(index, state.tabs.length - 1)]?.id ?? null
          }

          break

        default:
          return null
      }
    }

    if (command.kind !== 'page') {
      state.selectionIntentVersion++

      // Trusted guest input includes the tool's own native clicks/keys. A
      // same-tab renewal must reclaim opener selection without retiring the
      // action producing it. All membership changes and actual switches still
      // retire the captured epoch, including switching away and back.
      if (command.kind !== 'select' || state.activeTabId !== previousActiveTabId) {
        state.selectionVersion++
      }
    }

    this.publish(entry)

    return structuredClone(state)
  }

  /** Native close and failed-open rollback share the return-to-owner transition.
   * Explicit tab removal has already removed the tab from membership. */
  close(id: string) {
    const entry = this.entries.get(id)

    if (!entry || entry.state.closed) {
      return
    }

    entry.state.docked.push(...entry.state.tabs)
    entry.state.tabs = []
    entry.state.activeTabId = null
    entry.state.closed = true
    entry.state.selectionVersion++
    entry.state.selectionIntentVersion++
    this.publish(entry)

    // Keep request identities: cancellation must still reach the original
    // responder after close/acknowledgement removes its workspace entry.
  }

  /** The profile operation is global to one backend, not the active window.
   * Retire authority before publishing or closing native windows. Rename is a
   * terminal dock into the replacement scope, never a live owner retarget. */
  retireProfile(change: BrowserProfileRetirement): BrowserWorkspace[] {
    if (!change || typeof change.connectionId !== 'string' || !change.connectionId.trim() ||
      typeof change.profile !== 'string' || !change.profile.trim() ||
      (change.replacementProfile !== undefined && (typeof change.replacementProfile !== 'string' || !change.replacementProfile.trim()))) {
      throw new Error('Invalid browser profile retirement')
    }

    const entries = [...this.entries.values()].filter(entry =>
      !entry.state.profileRetirement && entry.state.owner.connectionId === change.connectionId && entry.state.owner.profile === change.profile)

    for (const entry of entries) {
      const state = entry.state
      const tabs = [...state.docked, ...state.tabs]
      state.profileRetirement = change.replacementProfile
        ? { replacementScope: backendScopeKey(change.connectionId, change.replacementProfile) }
        : {}
      state.docked = change.replacementProfile ? tabs : []
      state.removed = [...new Set([...state.removed, ...tabs.map(tab => tab.id)])]
      state.tabs = []
      state.activeTabId = null
      state.closed = true
      state.selectionVersion++
      state.selectionIntentVersion++
    }

    for (const entry of entries) {this.publish(entry)}

    return entries.map(entry => structuredClone(entry.state))
  }

  private sessionTip(sender: number, id: string): string {
    const aliases = this.sessionAliases.get(sender)

    for (let hops = 0; aliases?.has(id) && hops < aliases.size; hops++) {id = aliases.get(id)!}

    return id
  }

  /** Only the opener owns session binding/pinning. Guests own page and selection. */
  updateOwnership(sender: number, update: BrowserOwnershipUpdate): void {
    if (!update || !Array.isArray(update.tabs) || update.tabs.some(tab =>
      !tab || typeof tab.id !== 'string' || typeof tab.pinned !== 'boolean' ||
      (tab.sessionId !== undefined && typeof tab.sessionId !== 'string') ||
      (tab.pendingRuntimeId !== undefined && typeof tab.pendingRuntimeId !== 'string'))) {return}

    const rotation = update.rotation

    if (rotation && (!rotation.previousId || !rotation.nextId ||
      typeof rotation.previousId !== 'string' || typeof rotation.nextId !== 'string')) {return}

    const from = rotation && this.sessionTip(sender, rotation.previousId)
    const to = rotation && this.sessionTip(sender, rotation.nextId)
    const moved = Boolean(from && to && from !== to)
    const rows = new Map(update.tabs.map(tab => [tab.id, tab]))

    for (const entry of this.entries.values()) {
      if (entry.opener !== sender || entry.state.profileRetirement || entry.state.closed) {continue}
      const state = entry.state
      let changed = false

      for (const tab of [...state.tabs, ...state.docked]) {
        const received = rows.get(tab.id)
        const row = received && { ...received, sessionId: received.sessionId ? this.sessionTip(sender, received.sessionId) : undefined }

        if (row && (tab.sessionId !== row.sessionId || Boolean(tab.pinned) !== row.pinned || tab.pendingRuntimeId !== row.pendingRuntimeId)) {
          tab.sessionId = row.sessionId
          tab.pinned = row.pinned
          tab.pendingRuntimeId = row.pendingRuntimeId
          changed = true
        } else if (moved && tab.sessionId && this.sessionTip(sender, tab.sessionId) === from) {
          tab.sessionId = to!
          changed = true
        }
      }

      if (moved && state.owner.conversation?.kind === 'session' && this.sessionTip(sender, state.owner.conversation.id) === from) {
        state.owner.conversation = { ...state.owner.conversation, id: to! }

        if (state.owner.destination?.conversation) {
          state.owner.destination = { ...state.owner.destination, conversation: { ...state.owner.conversation } }
        }

        changed = true
      }

      if (changed) {
        state.selectionVersion++
        this.publish(entry)
      }
    }

    if (moved) {
      const aliases = this.sessionAliases.get(sender) ?? new Map<string, string>()
      aliases.set(from!, to!)
      this.sessionAliases.set(sender, aliases)
    }
  }

  retireSession(sender: number, sessionId: string): string[] {
    const tip = this.sessionTip(sender, sessionId)
    const ids: string[] = []

    for (const [id, entry] of this.entries) {
      if (entry.opener !== sender || entry.state.profileRetirement) {continue}
      const state = entry.state
      const ownsDestination = state.owner.conversation?.kind === 'session' && this.sessionTip(sender, state.owner.conversation.id) === tip

      const ownsTab = (tab: typeof state.tabs[number]) =>
        (tab.sessionId !== undefined && this.sessionTip(sender, tab.sessionId) === tip) || tab.pendingRuntimeId === sessionId ||
        (ownsDestination && tab.sessionId === undefined && tab.pendingRuntimeId === undefined)

      if (!ownsDestination && ![...state.tabs, ...state.docked].some(ownsTab)) {continue}

      const survivors = [...state.tabs, ...state.docked].filter(tab => {
        if (!ownsTab(tab)) {return true}

        if (tab.pinned) {
          tab.sessionId = undefined
          tab.pendingRuntimeId = undefined

          return true
        }

        state.removed.push(tab.id)

        return false
      })

      // Retire the exact annotation authority; surviving pins dock, never acquire
      // whichever session happens to be focused when this snapshot arrives.
      state.owner.destination = null
      state.owner.conversation = undefined
      state.docked = survivors
      state.tabs = []
      state.activeTabId = null
      state.closed = true
      state.selectionVersion++
      state.selectionIntentVersion++
      this.publish(entry)
      ids.push(id)
    }

    return ids
  }

  closeOwned(sender: number): string[] {
    const ids = [...this.entries]
      .filter(([, entry]) => entry.opener === sender && !entry.state.closed)
      .map(([id]) => id)

    for (const id of ids) {
      this.close(id)
    }

    return ids
  }

  /** Comment payloads reach only their exact originating renderer. */
  relayComment(sender: number, value: unknown): number | null | undefined {
    for (const [id, pending] of this.comments) {
      if (Date.now() - pending.at > 30_000) {this.comments.delete(id)}
    }

    if (!value || typeof value !== 'object') {return undefined}
    const packet = value as { type?: string; requestId?: string; tabId?: string; destination?: unknown }

    if (packet.type === 'preview-annotate-handoff-ack') {
      const pending = this.comments.get(packet.requestId || '')

      if (!pending) {return undefined}

      const entry = this.entries.get(pending.windowId)

      if (pending.opener !== sender || !entry || entry.state.closed) {return null}
      this.comments.delete(packet.requestId!)

      return pending.renderer
    }

    if (packet.type !== 'preview-annotate-handoff') {return undefined}
    const entry = [...this.entries.values()].find(item => item.renderer === sender)

    if (!entry) {return this.renderers.has(sender) ? null : undefined} // Legacy one-tab shell uses the old bridge.

    if (entry.state.closed || !entry.state.owner.destination || !packet.requestId || this.comments.has(packet.requestId) ||
      !entry.state.tabs.some(tab => tab.id === packet.tabId) ||
      JSON.stringify(packet.destination) !== JSON.stringify(entry.state.owner.destination)) {return null}

    this.comments.set(packet.requestId, { opener: entry.opener, renderer: sender, windowId: entry.state.id, at: Date.now() })

    return entry.opener
  }

  /** Resolve both directions in main, using the actual IPC sender, never a
   * renderer-supplied identity as authority. Opaque non-browser relay stays separate. */
  relay(sender: number, payload: unknown): number | null {
    if (!payload || typeof payload !== 'object') {
      return null
    }

    const packet = payload as {
      id?: unknown
      kind?: unknown
      target?: BrowserRequestTarget
      requester?: BrowserConversation
      payload?: unknown
      result?: unknown
      error?: unknown
      deadline?: unknown
    }

    const target = packet.target

    if (
      typeof packet.id !== 'string' ||
      !target ||
      typeof target.windowId !== 'string' ||
      typeof target.tabId !== 'string' ||
      !target.owner
    ) {
      return null
    }

    // Cancellation belongs to the admitted request, not to current membership.
    // A stale target is precisely when the original operation still needs Stop.
    if (packet.kind === 'cancel') {
      const pending = this.pending.get(packet.id)

      if (!pending || pending.opener !== sender ||
        pending.target.windowId !== target.windowId || pending.target.tabId !== target.tabId ||
        pending.target.selectionVersion !== target.selectionVersion || !sameBrowserOwner(pending.target.owner, target.owner)) {
        return null
      }

      this.pending.delete(packet.id)

      return pending.renderer
    }

    const entry = this.entries.get(target.windowId)

    if (
      !entry ||
      entry.state.closed ||
      !sameBrowserOwner(entry.state.owner, target.owner) ||
      entry.state.activeTabId !== target.tabId ||
      entry.state.selectionVersion !== target.selectionVersion ||
      !entry.state.tabs.some(tab => tab.id === target.tabId)
    ) {
      return null
    }

    if ('payload' in packet) {
      if (
        entry.opener !== sender ||
        !browserTabAllowsRequest(entry.state.owner, entry.state.tabs.find(tab => tab.id === target.tabId)!, packet.requester) ||
        entry.renderer === undefined ||
        (packet.kind !== 'act' && packet.kind !== 'read') ||
        this.pending.has(packet.id) ||
        this.pending.size >= BROWSER_REQUEST_HISTORY_LIMIT ||
        !this.receivedRequests.admit(packet.id, packet.kind, packet.deadline)
      ) {
        return null
      }

      this.pending.set(packet.id, {
        opener: sender,
        renderer: entry.renderer,
        target: structuredClone(target),
        kind: packet.kind
      })

      return entry.renderer
    }

    const pending = this.pending.get(packet.id)

    if (
      !pending ||
      (packet.kind !== pending.kind && packet.kind !== 'error') ||
      pending.renderer !== sender ||
      pending.target.windowId !== target.windowId ||
      pending.target.tabId !== target.tabId ||
      pending.target.selectionVersion !== target.selectionVersion ||
      !sameBrowserOwner(pending.target.owner, target.owner) ||
      (packet.kind !== 'act' && packet.kind !== 'read' && packet.kind !== 'error')
    ) {
      return null
    }

    this.pending.delete(packet.id)

    return pending.opener
  }
}
