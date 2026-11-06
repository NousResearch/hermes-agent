import { randomBytes } from 'node:crypto'

import { app, BrowserWindow, dialog, Menu, screen, session, WebContentsView } from 'electron'
import type { Session } from 'electron'

import type {
  PlayerAgentOperation,
  PlayerAgentRequest,
  PlayerAgentResponse,
  PlayerAgentScope,
  PlayerAwarenessLinkState,
  PlayerContextMetadata,
  PlayerContextSnapshot,
  PlayerControlAction,
  PlayerMediaJob,
  PlayerMediaJobCapabilities,
  PlayerMediaJobStartOptions,
  PlayerMediaTarget
} from '../../src/lib/player-context-contract'
import type { PlayerCommand, PlayerListener, PlayerSnapshot, PlayerSource } from '../../src/lib/player-contract'
import { getPlayerService } from '../../src/lib/player-services'
import { getNativePlayerCopy } from '../player-native-copy'

import {
  type CapturedPlayerContext,
  PlayerContextRegistry,
  playerControlGenerationIsCurrent,
  playerOwnerMatches,
  samePlayerScope,
  youtubeTargetFromUrl
} from './context-capabilities'
import { trackGuestBounds } from './float-guest-bounds'
import {
  type AwarenessLink,
  type FixtureOrigin,
  fixtureOrigin,
  initialSnapshot,
  isDocked,
  safePageText,
  type Slot,
  validSeconds
} from './manager-state'
import hyprland from './platform-hyprland.cjs'
import {
  clampPlayerBounds,
  isAllowedPlayerFrame,
  isAllowedPlayerMediaKeyHost,
  isAllowedPlayerNavigation,
  isAllowedPlayerPopup,
  safePlayerOrigin
} from './policy'
import { resolvePlayerRuntime } from './runtime'
import { resetServiceScopedState } from './service-scope'

const PARTITION = 'persist:hermes-player'
const FLOAT_TITLE = 'Hermes Player · Service Guest'

export interface PlayerManagerOptions {
  /** Tests may opt into one explicit localhost fixture. Never exposed to IPC. */
  fixtureUrl?: string
  allowFixture?: boolean
  runtimeReady?: () => Promise<{ state: 'ready' | 'unsupported'; detail?: string }>
  createView?: (options: ConstructorParameters<typeof WebContentsView>[0]) => WebContentsView
  createWindow?: (options: ConstructorParameters<typeof BrowserWindow>[0]) => BrowserWindow
  getSession?: (partition: string) => Session
  platform?: NodeJS.Platform
  mediaJobs?: PlayerMediaJobHost
}

/** Root-owned implementation is injected by main; no host filesystem path crosses this interface. */
export interface PlayerMediaJobHost {
  setup(owner: BrowserWindow): Promise<PlayerMediaJobCapabilities>
  start(
    owner: BrowserWindow,
    scope: PlayerAgentScope,
    target: PlayerMediaTarget,
    options: PlayerMediaJobStartOptions
  ): Promise<PlayerMediaJob>
  list(scope: PlayerAgentScope): Promise<PlayerMediaJob[]> | PlayerMediaJob[]
  get(scope: PlayerAgentScope, jobId: string): Promise<PlayerMediaJob | null> | PlayerMediaJob | null
  cancel(scope: PlayerAgentScope, jobId: string): Promise<boolean> | boolean
  reveal(owner: BrowserWindow, scope: PlayerAgentScope, jobId: string): Promise<boolean> | boolean
  capabilities(): PlayerMediaJobCapabilities
  disposeScope?(scope: PlayerAgentScope): void | Promise<void>
}

/** Owns one service guest and moves that same WebContentsView between native parents. */
export class PlayerManager {
  private readonly options: PlayerManagerOptions
  private readonly fixture?: FixtureOrigin
  private readonly listeners = new Map<number, Set<PlayerListener>>()
  private readonly permissionDecisions = new Map<string, boolean>()
  private slot: Slot | null = null
  private statusOwnerId: number | null = null
  private awarenessLink: AwarenessLink | null = null
  private readonly capturedContexts = new PlayerContextRegistry()
  private disposed = false
  private operation: Promise<unknown> = Promise.resolve()
  private snapshot = initialSnapshot()

  constructor(options: PlayerManagerOptions = {}) {
    this.options = options
    this.fixture = options.allowFixture ? fixtureOrigin(options.fixtureUrl) : undefined
  }

  subscribe(owner: BrowserWindow, listener: PlayerListener): () => void {
    let listeners = this.listeners.get(owner.id)

    if (!listeners) {
      listeners = new Set()
      this.listeners.set(owner.id, listeners)
    }

    listeners.add(listener)

    const initial =
      this.slot?.ownerId === owner.id || this.statusOwnerId === owner.id ? this.snapshot : initialSnapshot()

    listener(this.slot ? initial : { ...initial, awarenessLinked: this.awarenessLink?.ownerId === owner.id })

    return () => {
      listeners?.delete(listener)

      if (listeners?.size === 0) {
        this.listeners.delete(owner.id)
      }
    }
  }

  async linkAwareness(owner: BrowserWindow, scope: PlayerAgentScope): Promise<PlayerAwarenessLinkState> {
    const slot = this.requireSlot(owner)

    if (slot.ownerId !== owner.id || !slot.owner || slot.owner.isDestroyed()) {
      throw new Error('player-owner-unavailable')
    }

    const priorLink = this.awarenessLink

    if (!playerOwnerMatches(owner.id, priorLink?.ownerId)) {
      throw new Error('player-linked-to-another-window')
    }

    if (priorLink && priorLink.ownerId === owner.id && !samePlayerScope(priorLink.scope, scope)) {
      void this.options.mediaJobs?.disposeScope?.(priorLink.scope)
    }

    this.invalidateContexts()
    this.awarenessLink = { ownerId: owner.id, linkId: randomBytes(24).toString('base64url'), scope: { ...scope } }
    this.publish()

    return this.awarenessLinkState(owner)
  }

  async unlinkAwareness(owner: BrowserWindow): Promise<PlayerAwarenessLinkState> {
    if (this.awarenessLink?.ownerId !== owner.id) {
      return { linked: false, linkId: null, scope: null }
    }

    const priorScope = this.awarenessLink.scope
    this.awarenessLink = null
    this.invalidateContexts()
    await this.options.mediaJobs?.disposeScope?.(priorScope)
    this.publish()

    return this.awarenessLinkState(owner)
  }

  async setupMediaHelpers(owner: BrowserWindow): Promise<PlayerMediaJobCapabilities> {
    return this.options.mediaJobs?.setup(owner) ?? { available: false, reason: 'media-helpers-not-installed' }
  }

  async revealMediaJob(owner: BrowserWindow, scope: PlayerAgentScope, jobId: string): Promise<boolean> {
    const link = this.requireLink(owner, scope)

    if (!link || !this.options.mediaJobs) {
      return false
    }

    return this.options.mediaJobs.reveal(owner, scope, jobId)
  }

  async clearLocalData(owner: BrowserWindow): Promise<PlayerSnapshot> {
    if (!playerOwnerMatches(owner.id, this.slot?.ownerId, this.statusOwnerId, this.awarenessLink?.ownerId)) {
      throw new Error('player-owned-by-another-window')
    }

    if (this.slot?.ownerId === owner.id) {
      const slot = this.slot
      this.destroySlot(slot)
      this.slot = null
    }

    const linkedScope = this.awarenessLink?.ownerId === owner.id ? this.awarenessLink.scope : null

    if (this.awarenessLink?.ownerId === owner.id) {
      this.awarenessLink = null
    }

    this.invalidateContexts()

    if (linkedScope) {
      await this.options.mediaJobs?.disposeScope?.(linkedScope)
    }

    const ses = (this.options.getSession ?? (partition => session.fromPartition(partition, { cache: true })))(PARTITION)
    await ses.clearStorageData()
    await ses.clearCache()
    this.snapshot = initialSnapshot()
    this.statusOwnerId = owner.id
    this.publish()

    return this.snapshot
  }

  async agentRequest(owner: BrowserWindow, request: PlayerAgentRequest): Promise<PlayerAgentResponse> {
    const link = this.requireLink(owner, request.scope)

    if (!link) {
      return playerOperationError(
        request.operation,
        'player-awareness-not-linked',
        'Link this chat to the player first.'
      )
    }

    const operation = request.operation

    if (operation.type === 'jobs.list' || operation.type === 'job.status' || operation.type === 'job.cancel') {
      return this.handleJobRequest(link.scope, operation)
    }

    if (operation.type === 'context' || operation.type === 'control') {
      return this.handleContextRequest(owner, link, operation)
    }

    return this.startMediaJob(owner, request.scope, link, operation)
  }

  private async handleJobRequest(
    scope: PlayerAgentScope,
    operation: Extract<PlayerAgentOperation, { type: 'jobs.list' | 'job.status' | 'job.cancel' }>
  ): Promise<PlayerAgentResponse> {
    if (operation.type === 'jobs.list') {
      if (!this.options.mediaJobs) {
        return playerOperationError(operation, 'media-helpers-not-installed', 'Media helpers are unavailable.')
      }

      return { operation: operation.type, jobs: await this.options.mediaJobs.list(scope) }
    }

    if (operation.type === 'job.status') {
      return this.jobStatus(scope, operation)
    }

    return this.cancelJob(scope, operation)
  }

  private async jobStatus(
    scope: PlayerAgentScope,
    operation: Extract<PlayerAgentOperation, { type: 'job.status' }>
  ): Promise<PlayerAgentResponse> {
    const job = (await this.options.mediaJobs?.get(scope, operation.jobId)) ?? null

    return job
      ? { operation: operation.type, job }
      : playerOperationError(operation, 'media-job-not-found', 'That media job is unavailable.')
  }

  private async cancelJob(
    scope: PlayerAgentScope,
    operation: Extract<PlayerAgentOperation, { type: 'job.cancel' }>
  ): Promise<PlayerAgentResponse> {
    const cancelled = (await this.options.mediaJobs?.cancel(scope, operation.jobId)) ?? false
    const job = cancelled ? ((await this.options.mediaJobs?.get(scope, operation.jobId)) ?? null) : null

    return job
      ? { operation: operation.type, job }
      : playerOperationError(operation, 'media-job-not-found', 'That media job is unavailable.')
  }

  private async handleContextRequest(
    owner: BrowserWindow,
    link: AwarenessLink,
    operation: Extract<PlayerAgentOperation, { type: 'context' | 'control' }>
  ): Promise<PlayerAgentResponse> {
    const slot = this.readySlot(owner)

    if (!slot) {
      return playerOperationError(operation, 'player-not-ready', 'The player is not ready.')
    }

    if (operation.type === 'context') {
      const context = await this.captureContext(owner, link, slot)

      return context
        ? {
            operation: operation.type,
            context,
            capabilities: this.options.mediaJobs?.capabilities() ?? {
              available: false,
              reason: 'media-helpers-not-installed'
            }
          }
        : playerOperationError(operation, 'player-context-unavailable', 'Current player details are unavailable.')
    }

    const captured = this.getCapturedContext(owner, link, operation.contextToken)

    if (!captured) {
      return playerOperationError(operation, 'player-context-expired', 'Refresh player context and try again.')
    }

    return this.controlCapturedContext(slot, operation, captured)
  }

  private readySlot(owner: BrowserWindow): Slot | null {
    const slot = this.slot

    return slot && slot.ownerId === owner.id && slot.status === 'ready' && !slot.guest.webContents.isDestroyed()
      ? slot
      : null
  }

  private async controlCapturedContext(
    slot: Slot,
    operation: Extract<PlayerAgentOperation, { type: 'control' }>,
    captured: CapturedPlayerContext
  ): Promise<PlayerAgentResponse> {
    if (
      !playerControlGenerationIsCurrent(captured, operation.navigationGeneration) ||
      !playerControlGenerationIsCurrent(captured, slot.navigationGeneration)
    ) {
      return playerOperationError(
        operation,
        'player-context-stale',
        'The player page changed. Refresh context and try again.'
      )
    }

    const control = await this.applyAgentControl(slot, operation.action, operation.positionSeconds)
    this.publish()

    return { operation: operation.type, control }
  }

  private async startMediaJob(
    owner: BrowserWindow,
    scope: PlayerAgentScope,
    link: AwarenessLink,
    operation: Extract<PlayerAgentOperation, { type: 'transcript.start' | 'download.start' }>
  ): Promise<PlayerAgentResponse> {
    const slot = this.readySlot(owner)

    if (!slot) {
      return playerOperationError(operation, 'player-not-ready', 'The player is not ready.')
    }

    const captured = this.getCapturedContext(owner, link, operation.contextToken)

    if (!captured) {
      return playerOperationError(operation, 'player-context-expired', 'Refresh player context and try again.')
    }

    if (!playerControlGenerationIsCurrent(captured, slot.navigationGeneration)) {
      return playerOperationError(
        operation,
        'player-context-stale',
        'The player page changed. Refresh context and try again.'
      )
    }

    if (!captured.target || captured.target.adStatus === 'ad') {
      return playerOperationError(
        operation,
        'media-target-unsupported',
        'Media jobs require a verified YouTube video outside an ad.'
      )
    }

    const jobCapabilities = this.options.mediaJobs?.capabilities() ?? {
      available: false,
      reason: 'media-helpers-not-installed'
    }

    if (!jobCapabilities.available) {
      return playerOperationError(
        operation,
        jobCapabilities.reason ?? 'media-helpers-not-installed',
        'Media helpers are unavailable.'
      )
    }

    const options: PlayerMediaJobStartOptions =
      operation.type === 'transcript.start'
        ? { kind: 'transcript', language: operation.language, range: operation.range }
        : { kind: 'download', quality: operation.quality }

    if (!this.isCurrentLink(link, scope)) {
      return playerOperationError(operation, 'player-awareness-not-linked', 'Link this chat to the player first.')
    }

    const job = await this.options.mediaJobs?.start(owner, scope, captured.target, options)

    if (job && !this.isCurrentLink(link, scope)) {
      await this.options.mediaJobs?.cancel(scope, job.id)

      return playerOperationError(
        operation,
        'player-awareness-not-linked',
        'The player link changed before the media job started.'
      )
    }

    return job
      ? { operation: operation.type, job, capabilities: jobCapabilities }
      : playerOperationError(operation, 'media-helpers-not-installed', 'Media helpers are unavailable.')
  }

  private isCurrentLink(link: AwarenessLink, scope: PlayerAgentScope): boolean {
    return this.awarenessLink?.linkId === link.linkId && samePlayerScope(this.awarenessLink.scope, scope)
  }

  request(owner: BrowserWindow, command: PlayerCommand): Promise<PlayerSnapshot> {
    const run = async (): Promise<PlayerSnapshot> => {
      if (this.disposed) {
        throw new Error('player-manager-disposed')
      }

      if (this.slot && this.slot.ownerId !== owner.id) {
        throw new Error('player-owned-by-another-window')
      }

      if (!this.slot && this.statusOwnerId !== null && this.statusOwnerId !== owner.id) {
        this.snapshot = initialSnapshot()
        this.statusOwnerId = null
      }

      await this.apply(owner, command)
      this.publish()

      return this.snapshot
    }

    const result = this.operation.then(run, run)
    this.operation = result.catch(() => undefined)

    return result
  }

  /** Call during main-window teardown. A floating guest may survive owner close until app quit. */
  dispose(): void {
    this.disposed = true
    const scope = this.awarenessLink?.scope
    this.awarenessLink = null
    this.invalidateContexts()

    if (scope) {
      void this.options.mediaJobs?.disposeScope?.(scope)
    }

    const slot = this.slot

    if (slot) {
      this.destroySlot(slot)
    }

    this.slot = null
    this.statusOwnerId = null
    this.snapshot = initialSnapshot()
    this.publish()
  }

  ownerClosed(owner: BrowserWindow): void {
    if (this.awarenessLink?.ownerId === owner.id) {
      const scope = this.awarenessLink.scope
      this.awarenessLink = null
      this.invalidateContexts()
      void this.options.mediaJobs?.disposeScope?.(scope)
    }

    const slot = this.slot

    if (!slot) {
      if (this.statusOwnerId === owner.id) {
        this.statusOwnerId = null
        this.snapshot = initialSnapshot()
        this.publish()
      }

      return
    }

    if (slot.ownerId !== owner.id) {
      return
    }

    if (slot.presentation === 'docked') {
      this.destroySlot(slot)
      this.slot = null
    } else {
      slot.owner = null

      if (slot.fullscreenOrigin === 'docked') {
        slot.fullscreenOrigin = 'floating'
      }
    }

    this.publish()
  }

  private async apply(owner: BrowserWindow, command: PlayerCommand): Promise<void> {
    switch (command.type) {
      case 'open':
        await this.open(owner, command.source)

        return

      case 'status':
        return

      case 'bounds':
        this.applyBounds(owner, command)

        return

      case 'dock':
        await this.dock(this.requireSlot(owner))

        return

      case 'popout':
        await this.popout(this.requireSlot(owner))

        return

      case 'home':
        await this.navigateHome(owner)

        return

      case 'back':

      case 'forward':
        this.navigateHistory(owner, command.type)

        return

      case 'media-control':
        await this.applyMediaControl(owner, command)

        return

      case 'pin':
        await this.setPinPreference(owner, command.pinned)

        return

      case 'fullscreen':
        await this.fullscreen(this.requireSlot(owner), command.fullscreen)

        return

      case 'close':
        this.closePlayer(owner)

        return
      case 'release': {
        // Unloading is stronger than hiding a pane or closing a video: revoke
        // the chat lease and stop its local work, while retaining sign-in data.
        const scope = this.awarenessLink?.ownerId === owner.id ? this.awarenessLink.scope : null

        if (scope) {
          this.awarenessLink = null
          this.invalidateContexts()
          void this.options.mediaJobs?.disposeScope?.(scope)
        }

        this.closePlayer(owner)

        return
      }
    }
  }

  private applyBounds(owner: BrowserWindow, command: Extract<PlayerCommand, { type: 'bounds' }>): void {
    const slot = this.requireSlot(owner)

    if (!Number.isSafeInteger(command.revision) || command.revision <= slot.boundsRevision) {
      return
    }

    slot.boundsRevision = command.revision
    slot.visible = command.visible
    slot.hasBounds = true

    if (slot.presentation !== 'docked' || slot.fullscreen) {
      return
    }

    const [width, height] = owner.getContentSize()
    slot.bounds = clampPlayerBounds(command.bounds, { width, height })
    slot.guest.setBounds(slot.visible ? slot.bounds : { ...slot.bounds, width: 0, height: 0 })
  }

  private async navigateHome(owner: BrowserWindow): Promise<void> {
    const slot = this.requireSlot(owner)
    slot.error = null
    slot.status = 'loading'
    this.loadServicePage(slot, this.fixture ? this.options.fixtureUrl! : getPlayerService(slot.source).homeUrl)
  }

  private navigateHistory(owner: BrowserWindow, direction: 'back' | 'forward'): void {
    const contents = this.requireSlot(owner).guest.webContents

    if (direction === 'back' && contents.canGoBack()) {
      contents.goBack()
    }

    if (direction === 'forward' && contents.canGoForward()) {
      contents.goForward()
    }
  }

  private async applyMediaControl(
    owner: BrowserWindow,
    command: Extract<PlayerCommand, { type: 'media-control' }>
  ): Promise<void> {
    const result = await this.applyAgentControl(this.requireSlot(owner), command.action, command.positionSeconds)

    if (!result.applied) {
      throw new Error('player-media-control-unavailable')
    }
  }

  private async setPinPreference(owner: BrowserWindow, pinned: boolean): Promise<void> {
    const slot = this.requireSlot(owner)
    slot.pinned = pinned

    if (slot.fullscreen) {
      slot.pinStatus = 'requested'
    } else {
      await this.applyPin(slot)
    }
  }

  private closePlayer(owner: BrowserWindow): void {
    if (this.slot) {
      const slot = this.requireSlot(owner)
      this.destroySlot(slot)
      this.slot = null
      this.invalidateContexts()
    } else if (this.statusOwnerId === owner.id) {
      this.statusOwnerId = null
      this.snapshot = initialSnapshot()
    }
  }

  private async open(owner: BrowserWindow, source: PlayerSource): Promise<void> {
    if (this.slot) {
      return this.openExistingSlot(this.slot, source)
    }

    await this.openNewSlot(owner, source)
  }

  private async openExistingSlot(existing: Slot, source: PlayerSource): Promise<void> {
    if (existing.source === source) {
      if (existing.error === 'provider-navigation-blocked') {
        await this.reloadServiceHome(existing, source)
      }

      return
    }

    resetServiceScopedState(existing.authWindows, this.permissionDecisions)
    const runtime = await this.resolveSourceRuntime(source)

    if (this.disposed || this.slot !== existing) {
      return
    }

    if (!this.allowRuntime(existing.ownerId, source, runtime, existing)) {
      return
    }

    existing.source = source
    existing.runtime = runtime
    existing.status = 'loading'
    existing.error = null
    this.loadServicePage(existing, this.serviceHomeUrl(source))
  }

  private async reloadServiceHome(slot: Slot, source: PlayerSource): Promise<void> {
    slot.status = 'loading'
    slot.error = null
    this.loadServicePage(slot, this.serviceHomeUrl(source))
  }

  private async openNewSlot(owner: BrowserWindow, source: PlayerSource): Promise<void> {
    const runtime = await this.resolveSourceRuntime(source)

    if (!this.allowRuntime(owner.id, source, runtime)) {
      return
    }

    if (this.disposed) {
      return
    }

    const url = this.serviceHomeUrl(source)
    const ses = (this.options.getSession ?? (partition => session.fromPartition(partition, { cache: true })))(PARTITION)
    this.configureSession(ses, owner)

    const guest = (this.options.createView ?? (options => new WebContentsView(options)))({
      webPreferences: {
        partition: PARTITION,
        nodeIntegration: false,
        sandbox: true,
        contextIsolation: true,
        webSecurity: true,
        webviewTag: false,
        backgroundThrottling: false,
        // The manager owns fullscreen after reparenting; prevent Electron's automatic HTML-fullscreen host race.
        disableHtmlFullscreenWindowResize: true
      }
    })

    const slot: Slot = {
      owner,
      ownerId: owner.id,
      guest,
      source,
      presentation: 'docked',
      floatWindow: null,
      floatBoundsTracking: null,
      floatBounds: null,
      fullscreen: false,
      fullscreenOrigin: null,
      pinned: false,
      pinStatus: 'requested',
      pinMode: 'topmost',
      visible: false,
      hasBounds: false,
      bounds: { x: 0, y: 0, width: 0, height: 0 },
      boundsRevision: -1,
      status: 'loading',
      currentTime: null,
      duration: null,
      paused: null,
      muted: null,
      error: null,
      runtime,
      mediaPoll: null,
      mediaPollInFlight: false,
      navigationGeneration: 0,
      authWindows: new Set(),
      closeInProgress: false
    }

    this.slot = slot
    this.statusOwnerId = null
    owner.contentView.addChildView(guest)
    guest.setBounds(slot.bounds)
    this.installGuestHandlers(slot)
    owner.once('closed', () => {
      this.ownerClosed(owner)
    })
    owner.webContents.on('did-navigate', () => {
      if (this.slot !== slot || slot.presentation !== 'docked') {
        return
      }

      slot.boundsRevision = -1
      slot.hasBounds = false
      slot.visible = false
      slot.bounds = { x: 0, y: 0, width: 0, height: 0 }
      slot.guest.setBounds(slot.bounds)
    })
    this.publish()
    this.loadServicePage(slot, url)
  }

  /** Page load completion must not hold the manager's command queue. */
  private loadServicePage(slot: Slot, url: string): void {
    void slot.guest.webContents.loadURL(url).catch(() => undefined)
  }

  private async resolveSourceRuntime(source: PlayerSource): Promise<PlayerSnapshot['runtime']> {
    return this.fixture || !getPlayerService(source).requiresWidevine
      ? { state: 'not-required', detail: 'Widevine readiness is not required for this service or local fixtures' }
      : resolvePlayerRuntime(this.options.runtimeReady)
  }

  private allowRuntime(
    ownerId: number,
    source: PlayerSource,
    runtime: PlayerSnapshot['runtime'],
    slot?: Slot
  ): boolean {
    if (!getPlayerService(source).requiresWidevine || this.fixture || runtime.state === 'ready') {
      return true
    }

    const error =
      runtime.state === 'restart-required'
        ? 'protected-playback-runtime-restart-required'
        : 'protected-playback-runtime-unavailable'

    if (slot) {
      slot.status = 'error'
      slot.error = error
      slot.runtime = runtime
    } else {
      this.snapshot = { ...initialSnapshot(), source, status: 'error', runtime, error }
      this.statusOwnerId = ownerId
    }

    this.publish()

    return false
  }

  private serviceHomeUrl(source: PlayerSource): string {
    return this.fixture ? this.options.fixtureUrl! : getPlayerService(source).homeUrl
  }

  private awarenessLinkState(owner: BrowserWindow): PlayerAwarenessLinkState {
    const link = this.awarenessLink

    return link?.ownerId === owner.id
      ? { linked: true, linkId: link.linkId, scope: { ...link.scope } }
      : { linked: false, linkId: null, scope: null }
  }

  private requireLink(owner: BrowserWindow, scope: PlayerAgentScope): AwarenessLink | null {
    const link = this.awarenessLink

    return link && link.ownerId === owner.id && samePlayerScope(link.scope, scope) ? link : null
  }

  private invalidateContexts(): void {
    this.capturedContexts.clear()
  }

  private getCapturedContext(owner: BrowserWindow, link: AwarenessLink, token: string) {
    return this.capturedContexts.resolve(token, owner.id, link.linkId, link.scope)
  }

  private async captureContext(
    owner: BrowserWindow,
    link: AwarenessLink,
    slot: Slot
  ): Promise<PlayerContextSnapshot | null> {
    if (this.slot !== slot || slot.ownerId !== owner.id || slot.status !== 'ready') {
      return null
    }

    const wc = slot.guest.webContents
    const generation = slot.navigationGeneration
    const source = slot.source
    const pageUrl = wc.getURL()
    let raw: unknown = null

    try {
      raw = await wc.executeJavaScript(
        `(() => {
        const media = [...document.querySelectorAll('video,audio')].find(item => Number.isFinite(item.currentTime));
        const ad = !!document.querySelector('#movie_player.ad-showing, .ytp-ad-player-overlay, .video-ads.ytp-ad-module:not(:empty)');
        const channel = document.querySelector('ytd-channel-name a, #channel-name a, #owner a')?.textContent ?? null;
        return { title: document.title, channel, currentTime: media?.currentTime ?? null,
          duration: Number.isFinite(media?.duration) ? media.duration : null,
          paused: media ? media.paused : null, muted: media ? media.muted : null,
          adStatus: ad ? 'ad' : 'unknown' };
      })()`,
        false
      )
    } catch {
      /* page metadata remains unknown */
    }

    if (
      this.slot !== slot ||
      slot.navigationGeneration !== generation ||
      slot.source !== source ||
      this.awarenessLink?.linkId !== link.linkId
    ) {
      return null
    }

    const sample = raw && typeof raw === 'object' ? (raw as Record<string, unknown>) : {}
    const adStatus = sample.adStatus === 'ad' ? 'ad' : 'unknown'

    const target = youtubeTargetFromUrl(source, pageUrl, {
      title: safePageText(sample.title, 512),
      capturedTimeSeconds: validSeconds(sample.currentTime),
      adStatus
    })

    const now = Date.now()

    const metadata: PlayerContextMetadata = {
      identity: target ? 'youtube-watch' : 'unknown',
      videoId: target?.videoId ?? null,
      canonicalUrl: target?.canonicalUrl ?? null,
      title: safePageText(sample.title, 512),
      channel: safePageText(sample.channel, 256),
      positionSeconds: validSeconds(sample.currentTime),
      durationSeconds: validSeconds(sample.duration),
      paused: typeof sample.paused === 'boolean' ? sample.paused : null,
      muted: typeof sample.muted === 'boolean' ? sample.muted : null,
      adStatus
    }

    const issued = this.capturedContexts.issue(
      {
        ownerId: owner.id,
        linkId: link.linkId,
        scope: { ...link.scope },
        generation,
        target
      },
      now
    )

    return {
      source,
      safeOrigin: safePlayerOrigin(pageUrl, this.fixture, source),
      navigationGeneration: generation,
      contextToken: issued.token,
      capturedAt: new Date(now).toISOString(),
      expiresAt: new Date(issued.expiresAtMs).toISOString(),
      metadata
    }
  }

  private async applyAgentControl(slot: Slot, action: PlayerControlAction, positionSeconds?: number) {
    if (slot.status !== 'ready' || slot.guest.webContents.isDestroyed()) {
      return { applied: false, action, positionSeconds: slot.currentTime, paused: slot.paused, muted: slot.muted }
    }

    if (action === 'pin' || action === 'unpin') {
      return this.applyPinControl(slot, action)
    }

    if (action === 'fullscreen' || action === 'exit-fullscreen') {
      return this.applyFullscreenControl(slot, action)
    }

    return this.applyMediaElementControl(slot, action, positionSeconds)
  }

  private async applyPinControl(slot: Slot, action: 'pin' | 'unpin') {
    slot.pinned = action === 'pin'
    await this.applyPin(slot)

    return {
      applied: true,
      action,
      positionSeconds: slot.currentTime,
      paused: slot.paused,
      muted: slot.muted,
      pinned: slot.pinned
    }
  }

  private async applyFullscreenControl(slot: Slot, action: 'fullscreen' | 'exit-fullscreen') {
    await this.fullscreen(slot, action === 'fullscreen')

    return {
      applied: true,
      action,
      positionSeconds: slot.currentTime,
      paused: slot.paused,
      muted: slot.muted,
      fullscreen: slot.fullscreen
    }
  }

  private async applyMediaElementControl(slot: Slot, action: PlayerControlAction, positionSeconds?: number) {
    if (action === 'seek' && validSeconds(positionSeconds) === null) {
      return { applied: false, action, positionSeconds: slot.currentTime, paused: slot.paused, muted: slot.muted }
    }

    const wc = slot.guest.webContents
    const desiredPosition = action === 'seek' ? Math.max(0, Number(positionSeconds)) : null

    const applied = await wc.executeJavaScript(
      `(() => {
      const media = [...document.querySelectorAll('video,audio')].find(item => Number.isFinite(item.currentTime));
      if (!media) return false;
      ${action === 'play' ? 'return media.play().then(() => true, () => false);' : ''}
      ${action === 'pause' ? 'media.pause();' : ''}
      ${action === 'mute' ? 'media.muted = true;' : ''}
      ${action === 'unmute' ? 'media.muted = false;' : ''}
      ${action === 'seek' ? `media.currentTime = ${JSON.stringify(desiredPosition)};` : ''}
      return true;
    })()`,
      false
    )

    if (applied !== true) {
      return { applied: false, action, positionSeconds: slot.currentTime, paused: slot.paused, muted: slot.muted }
    }

    this.updateMediaElementState(slot, action, desiredPosition)

    return { applied: true, action, positionSeconds: slot.currentTime, paused: slot.paused, muted: slot.muted }
  }

  private updateMediaElementState(slot: Slot, action: PlayerControlAction, desiredPosition: number | null): void {
    if (action === 'play') {
      slot.paused = false
    }

    if (action === 'pause') {
      slot.paused = true
    }

    if (action === 'mute') {
      slot.muted = true
    }

    if (action === 'unmute') {
      slot.muted = false
    }

    if (action === 'seek') {
      slot.currentTime = desiredPosition
    }
  }

  private isHyprland(): boolean {
    const platform = this.options.platform ?? process.platform

    return platform === 'linux' && Boolean(process.env.HYPRLAND_INSTANCE_SIGNATURE && process.env.WAYLAND_DISPLAY)
  }

  private async applyPin(slot: Slot): Promise<void> {
    const window = slot.floatWindow

    if (!window || window.isDestroyed()) {
      slot.pinStatus = 'requested'

      return
    }

    if (this.isHyprland()) {
      slot.pinMode = 'all-workspaces'
      const result = await hyprland.setHyprlandPlayerPin({ enabled: slot.pinned, title: FLOAT_TITLE })

      if (result.status === 'applied' || result.status === 'observed') {
        slot.pinStatus = result.observation?.pinned === slot.pinned ? 'observed' : 'unknown'
      } else {
        slot.pinStatus = result.supported ? 'unknown' : 'unsupported'
      }

      return
    }

    const platform = this.options.platform ?? process.platform

    if (platform === 'linux' && process.env.WAYLAND_DISPLAY && process.env.XDG_SESSION_TYPE !== 'x11') {
      slot.pinMode = 'unsupported'
      slot.pinStatus = 'unsupported'

      return
    }

    slot.pinMode = 'topmost'
    window.setAlwaysOnTop(slot.pinned, platform === 'darwin' ? 'floating' : undefined)

    if (platform === 'darwin') {
      window.setVisibleOnAllWorkspaces(slot.pinned, {
        visibleOnFullScreen: slot.pinned,
        skipTransformProcessType: true
      })
    }

    slot.pinStatus = 'requested'
  }

  private async applyFloatPolicy(slot: Slot): Promise<void> {
    if (!this.isHyprland()) {
      await this.applyPin(slot)

      return
    }

    slot.pinMode = 'all-workspaces'
    const result = await hyprland.applyHyprlandPlayerPolicy({ enabled: slot.pinned, title: FLOAT_TITLE })

    if (result.status === 'applied' && result.observation?.pinned === slot.pinned) {
      slot.pinStatus = 'observed'
    } else {
      slot.pinStatus = result.supported ? 'unknown' : 'unsupported'
    }
  }

  private configureSession(ses: Session, owner: BrowserWindow): void {
    const approved = (permission: string, raw: string, source: PlayerSource): boolean => {
      let url: URL

      try {
        url = new URL(raw)
      } catch {
        return false
      }

      if (permission === 'geolocation') {
        return url.protocol === 'https:' && url.hostname === 'tv.youtube.com'
      }

      if (permission === 'mediaKeySystem') {
        return url.protocol === 'https:' && isAllowedPlayerMediaKeyHost(url.hostname, source)
      }

      return false
    }

    const decisionKey = (permission: string, raw: string): string | null => {
      try {
        return `${permission}:${new URL(raw).origin}`
      } catch {
        return null
      }
    }

    ses.setPermissionRequestHandler((contents, permission, callback, details) => {
      const url = details?.requestingUrl ?? ''
      const player = this.slot
      const source = player?.source ?? 'youtube'

      if (permission === 'fullscreen' && isAllowedPlayerNavigation(url, this.fixture, source)) {
        callback(true)

        return
      }

      const key = decisionKey(permission, url)

      if (
        !key ||
        !approved(permission, url, source) ||
        !player ||
        player.ownerId !== owner.id ||
        contents.id !== player.guest.webContents.id ||
        (permission === 'geolocation' && player.source !== 'tv')
      ) {
        callback(false)

        return
      }

      const prior = this.permissionDecisions.get(key)

      if (prior !== undefined) {
        callback(prior)

        return
      }

      void dialog
        .showMessageBox(owner, {
          type: 'question',
          buttons: [
            getNativePlayerCopy(app.getLocale(), source).allowSession,
            getNativePlayerCopy(app.getLocale(), source).deny
          ],
          defaultId: 1,
          cancelId: 1,
          title:
            permission === 'geolocation'
              ? getNativePlayerCopy(app.getLocale(), source).locationTitle
              : getNativePlayerCopy(app.getLocale(), source).protectedTitle,
          message:
            permission === 'geolocation'
              ? getNativePlayerCopy(app.getLocale(), source).locationMessage
              : getNativePlayerCopy(app.getLocale(), source).protectedMessage
        })
        .then(
          result => {
            const allow =
              result.response === 0 &&
              this.slot?.ownerId === owner.id &&
              this.slot.guest.webContents.id === contents.id &&
              this.slot.source === source

            this.permissionDecisions.set(key, allow)
            callback(allow)
          },
          () => {
            this.permissionDecisions.set(key, false)
            callback(false)
          }
        )
    })
    ses.setPermissionCheckHandler((contents, permission, origin) => {
      const player = this.slot
      const source = player?.source ?? 'youtube'

      if (permission === 'fullscreen') {
        return isAllowedPlayerNavigation(origin, this.fixture, source)
      }

      if (!approved(permission, origin, source)) {
        return false
      }

      const key = decisionKey(permission, origin)

      return Boolean(
        key &&
        this.permissionDecisions.get(key) &&
        player &&
        player.ownerId === owner.id &&
        player.guest.webContents.id === contents.id &&
        (permission !== 'geolocation' || player.source === 'tv')
      )
    })
  }

  private markProviderNavigationBlocked(slot: Slot): void {
    if (this.slot !== slot) {
      return
    }

    slot.status = 'error'
    slot.error = 'provider-navigation-blocked'
    this.publish()
  }

  private installGuestHandlers(slot: Slot): void {
    const wc = slot.guest.webContents
    wc.setWindowOpenHandler(({ url }) => {
      if (!isAllowedPlayerPopup(url, slot.source)) {
        // This request belongs to an auxiliary window. Denying it leaves the
        // committed service document intact, so it must not revoke readiness.
        return { action: 'deny' }
      }

      const popup = (this.options.createWindow ?? (options => new BrowserWindow(options)))({
        parent: slot.floatWindow && !slot.floatWindow.isDestroyed() ? slot.floatWindow : (slot.owner ?? undefined),
        width: 720,
        height: 760,
        show: true,
        title: getNativePlayerCopy(app.getLocale(), slot.source).signIn,
        webPreferences: {
          partition: PARTITION,
          nodeIntegration: false,
          sandbox: true,
          contextIsolation: true,
          webSecurity: true
        }
      })

      slot.authWindows.add(popup)
      popup.webContents.setWindowOpenHandler(() => {
        return { action: 'deny' }
      })

      const guard = (event: Electron.Event, target: string) => {
        if (!isAllowedPlayerNavigation(target, this.fixture, slot.source)) {
          event.preventDefault()
        }
      }

      popup.webContents.on('will-navigate', guard)
      popup.webContents.on('will-redirect', guard)

      const popupOn = popup.webContents as unknown as {
        on: (event: string, listener: (...args: any[]) => void) => void
      }

      popupOn.on(
        'will-frame-navigate',
        (
          event: Electron.Event & { url?: string; isMainFrame?: boolean },
          details?: { url?: string; isMainFrame?: boolean }
        ) => {
          const target = details?.url ?? event.url ?? ''
          const mainFrame = details?.isMainFrame ?? event.isMainFrame ?? false

          if (
            mainFrame
              ? !isAllowedPlayerNavigation(target, this.fixture, slot.source)
              : !isAllowedPlayerFrame(target, this.fixture, slot.source)
          ) {
            event.preventDefault()
          }
        }
      )
      popup.once('closed', () => slot.authWindows.delete(popup))
      void popup.loadURL(url)

      return { action: 'deny' }
    })

    const guardNavigation = (event: Electron.Event, url: string) => {
      if (!isAllowedPlayerNavigation(url, this.fixture, slot.source)) {
        event.preventDefault()
        this.markProviderNavigationBlocked(slot)
      }
    }

    wc.on('will-navigate', guardNavigation)
    wc.on('will-redirect', guardNavigation)
    const guestOn = wc as unknown as { on: (event: string, listener: (...args: any[]) => void) => void }
    guestOn.on(
      'will-frame-navigate',
      (
        event: Electron.Event & { url?: string; isMainFrame?: boolean },
        details?: { url?: string; isMainFrame?: boolean }
      ) => {
        const target = details?.url ?? event.url ?? ''
        const mainFrame = details?.isMainFrame ?? event.isMainFrame ?? false

        if (
          mainFrame
            ? !isAllowedPlayerNavigation(target, this.fixture, slot.source)
            : !isAllowedPlayerFrame(target, this.fixture, slot.source)
        ) {
          event.preventDefault()

          if (mainFrame) {
            this.markProviderNavigationBlocked(slot)
          }
        }
      }
    )
    wc.on('did-start-navigation', (_event, url, inPlace, isMainFrame) => {
      if (isMainFrame) {
        slot.navigationGeneration += 1
        slot.mediaPollInFlight = false

        // Same-document routes (YouTube's video picker, history/hash changes)
        // never emit did-finish-load. Keep document readiness and any existing
        // error, while invalidating controls and samples from the previous URL.
        if (!inPlace) {
          slot.status = 'loading'
          slot.error = null
        }

        slot.currentTime = null
        slot.duration = null
        slot.paused = null
        slot.muted = null
        this.publish()
      }
    })
    wc.on('did-finish-load', () => {
      if (slot.error === 'provider-navigation-blocked') {
        slot.status = 'error'
      } else {
        slot.status = 'ready'
        slot.error = null
      }

      if (slot.mediaPoll) {
        clearInterval(slot.mediaPoll)
      }

      slot.mediaPoll = setInterval(() => {
        if (
          wc.isDestroyed() ||
          this.slot !== slot ||
          slot.mediaPollInFlight ||
          !safePlayerOrigin(wc.getURL(), this.fixture, slot.source)
        ) {
          return
        }

        const generation = slot.navigationGeneration
        const source = slot.source
        slot.mediaPollInFlight = true
        void wc
          .executeJavaScript(
            `(() => { const media = [...document.querySelectorAll('video,audio')].find(item => Number.isFinite(item.currentTime)); if (!media) return null; return { currentTime: media.currentTime, duration: Number.isFinite(media.duration) ? media.duration : null, paused: media.paused, muted: media.muted }; })()`,
            false
          )
          .then(value => {
            if (
              !value ||
              typeof value !== 'object' ||
              this.slot !== slot ||
              generation !== slot.navigationGeneration ||
              source !== slot.source
            ) {
              return
            }

            const sample = value as { currentTime?: unknown; duration?: unknown; paused?: unknown; muted?: unknown }
            slot.currentTime =
              typeof sample.currentTime === 'number' && Number.isFinite(sample.currentTime) && sample.currentTime >= 0
                ? sample.currentTime
                : null
            slot.duration =
              typeof sample.duration === 'number' && Number.isFinite(sample.duration) && sample.duration >= 0
                ? sample.duration
                : null
            slot.paused = typeof sample.paused === 'boolean' ? sample.paused : null
            slot.muted = typeof sample.muted === 'boolean' ? sample.muted : null
            this.publish()
          })
          .catch(() => undefined)
          .finally(() => {
            slot.mediaPollInFlight = false
          })
      }, 1000)
      this.publish()
    })
    wc.on('did-fail-load', (_event, code, _description, _url, isMainFrame) => {
      // A newer user navigation commonly cancels the prior request with ERR_ABORTED.
      if (!isMainFrame || Number(code) === -3) {
        return
      }

      slot.status = 'error'
      slot.error = `navigation-failed-${Number(code)}`
      this.publish()
    })
    wc.on('render-process-gone', () => {
      slot.status = 'error'
      slot.error = 'guest-crashed'
      this.publish()
    })
    wc.on('media-started-playing', () => {
      slot.paused = false
      this.publish()
    })
    wc.on('media-paused', () => {
      slot.paused = true
      this.publish()
    })
    wc.on('page-title-updated', event => event.preventDefault())
    wc.on('will-prevent-unload', event => event.preventDefault())
    wc.on('before-input-event', (event, input) => {
      if (input.type === 'keyDown' && input.key === 'Escape' && slot.fullscreen) {
        event.preventDefault()
        void this.enqueueInternal(slot, () => this.fullscreen(slot, false)).catch(() => undefined)
      } else if (
        input.type === 'keyDown' &&
        input.key.toLowerCase() === 'p' &&
        input.control &&
        input.shift &&
        this.isHyprland()
      ) {
        event.preventDefault()
        slot.pinned = false
        slot.pinStatus = 'requested'
        void this.enqueueInternal(slot, async () => {
          if (slot.fullscreen) {
            await this.fullscreen(slot, false)
          }

          await this.applyPin(slot)
        }).catch(() => undefined)
      }
    })
    wc.on('enter-html-full-screen', () => {
      void this.enqueueInternal(slot, () => this.fullscreen(slot, true)).catch(() => undefined)
    })
    wc.on('leave-html-full-screen', () => {
      void this.enqueueInternal(slot, () => this.fullscreen(slot, false)).catch(() => undefined)
    })
  }

  private async popout(slot: Slot): Promise<void> {
    if (slot.presentation === 'floating' && slot.floatWindow && !slot.floatWindow.isDestroyed()) {
      return
    }

    if (!slot.owner || slot.owner.isDestroyed()) {
      throw new Error('player-dock-owner-unavailable')
    }

    const floatWindow = (this.options.createWindow ?? (options => new BrowserWindow(options)))({
      width: 960,
      height: 540,
      minWidth: 320,
      minHeight: 180,
      show: true,
      title: FLOAT_TITLE,
      backgroundColor: '#101317',
      webPreferences: { nodeIntegration: false, sandbox: true, contextIsolation: true, webSecurity: true }
    })

    slot.floatWindow = floatWindow
    floatWindow.setMenu(
      Menu.buildFromTemplate([
        {
          label: getNativePlayerCopy(app.getLocale(), slot.source).player,
          submenu: [
            {
              label: getNativePlayerCopy(app.getLocale(), slot.source).dock,
              enabled: Boolean(slot.owner && !slot.owner.isDestroyed()),
              click: () => void this.enqueueInternal(slot, () => this.dock(slot))
            },
            {
              label: getNativePlayerCopy(app.getLocale(), slot.source).togglePin,
              click: () =>
                void this.enqueueInternal(slot, async () => {
                  slot.pinned = !slot.pinned
                  await this.applyPin(slot)
                })
            },
            {
              label: slot.fullscreen
                ? getNativePlayerCopy(app.getLocale(), slot.source).exitFullscreen
                : getNativePlayerCopy(app.getLocale(), slot.source).fullscreen,
              click: () => void this.enqueueInternal(slot, () => this.fullscreen(slot, !slot.fullscreen))
            },
            { type: 'separator' },
            {
              label: getNativePlayerCopy(app.getLocale(), slot.source).close,
              click: () =>
                void this.enqueueInternal(slot, () => {
                  this.destroySlot(slot)
                  this.slot = null
                })
            }
          ]
        }
      ])
    )
    slot.presentation = 'floating'

    try {
      slot.owner.contentView.removeChildView(slot.guest)
      floatWindow.contentView.addChildView(slot.guest)
      const boundsTracking = trackGuestBounds(floatWindow.contentView, slot.guest)
      slot.floatBoundsTracking = boundsTracking
      floatWindow.on('resize', boundsTracking.resize)
      await this.applyFloatPolicy(slot)
      boundsTracking.resize()
    } catch (error) {
      this.stopFloatBoundsTracking(slot)

      try {
        floatWindow.contentView.removeChildView(slot.guest)
      } catch {
        /* view may not have moved */
      }

      if (!slot.owner.isDestroyed()) {
        slot.owner.contentView.addChildView(slot.guest)
        slot.guest.setBounds(slot.hasBounds && slot.visible ? slot.bounds : { x: 0, y: 0, width: 0, height: 0 })
      }

      slot.presentation = 'docked'
      slot.floatWindow = null
      slot.closeInProgress = true
      floatWindow.destroy()
      slot.closeInProgress = false
      throw error
    }

    floatWindow.on('enter-full-screen', () => {
      if (this.isHyprland()) {
        return
      }

      slot.fullscreen = true
      this.publish()
    })
    floatWindow.on('leave-full-screen', () => {
      if (this.isHyprland()) {
        return
      }

      void this.enqueueInternal(slot, async () => {
        if (slot.presentation !== 'floating' || !slot.floatWindow || slot.floatWindow.isDestroyed()) {
          return
        }

        slot.fullscreen = false
        await this.restoreNativeFloatAfterFullscreen(slot)

        if (slot.fullscreenOrigin === 'docked' && slot.owner && !slot.owner.isDestroyed()) {
          await this.dock(slot)
        } else {
          slot.fullscreenOrigin = null
        }
      })
    })
    floatWindow.on('close', event => {
      if (slot.closeInProgress || this.disposed) {
        return
      }

      event.preventDefault()
      void this.enqueueInternal(slot, async () => {
        if (slot.owner && !slot.owner.isDestroyed()) {
          await this.dock(slot)
        } else {
          this.destroySlot(slot)
          this.slot = null
        }
      })
    })
    floatWindow.on('closed', () => {
      this.stopFloatBoundsTracking(slot)

      if (this.slot !== slot || slot.closeInProgress) {
        return
      }

      slot.floatWindow = null

      if (slot.owner && !slot.owner.isDestroyed()) {
        void this.enqueueInternal(slot, () => this.dock(slot))
      } else {
        this.destroySlot(slot)
        this.slot = null
      }
    })
    this.publish()
  }

  private async dock(slot: Slot): Promise<void> {
    if (slot.presentation === 'docked') {
      return
    }

    if (!slot.owner || slot.owner.isDestroyed()) {
      throw new Error('player-dock-owner-unavailable')
    }

    if (slot.fullscreen && this.isHyprland()) {
      await this.fullscreen(slot, false)

      if (isDocked(slot)) {
        return
      }
    }

    const floating = slot.floatWindow

    if (floating && !floating.isDestroyed() && floating.isFullScreen()) {
      slot.fullscreenOrigin = 'docked'
      await this.transitionNativeFullscreen(floating, false)
    }

    slot.fullscreen = false
    this.stopFloatBoundsTracking(slot)

    if (floating && !floating.isDestroyed()) {
      floating.contentView.removeChildView(slot.guest)
    }

    slot.owner.contentView.addChildView(slot.guest)
    slot.guest.setBounds(slot.hasBounds && slot.visible ? slot.bounds : { x: 0, y: 0, width: 0, height: 0 })
    slot.presentation = 'docked'
    slot.fullscreenOrigin = null

    if (floating && !floating.isDestroyed()) {
      slot.closeInProgress = true
      floating.destroy()
      slot.closeInProgress = false
      slot.floatWindow = null
    }

    this.publish()
  }

  private async fullscreen(slot: Slot, fullscreen: boolean): Promise<void> {
    if (fullscreen === slot.fullscreen) {
      return
    }

    if (fullscreen) {
      return this.enterFullscreen(slot)
    }

    return this.exitFullscreen(slot)
  }

  private async enterFullscreen(slot: Slot): Promise<void> {
    slot.fullscreenOrigin = slot.presentation

    if (slot.presentation === 'docked') {
      await this.popout(slot)
    }

    const floating = slot.floatWindow

    if (!floating || floating.isDestroyed()) {
      throw new Error('player-window-unavailable')
    }

    if (this.isHyprland()) {
      await this.setHyprlandFullscreen(slot, true)

      return
    }

    slot.floatBounds = floating.getBounds()
    await this.transitionNativeFullscreen(floating, true)
    slot.fullscreen = floating.isFullScreen()
  }

  private async exitFullscreen(slot: Slot): Promise<void> {
    const floating = slot.floatWindow

    if (!floating || floating.isDestroyed()) {
      slot.fullscreen = false

      return
    }

    if (this.isHyprland()) {
      await this.setHyprlandFullscreen(slot, false)
      await this.restoreFullscreenOrigin(slot)

      return
    }

    await this.transitionNativeFullscreen(floating, false)
    slot.fullscreen = false
    await this.restoreNativeFloatAfterFullscreen(slot)
    await this.restoreFullscreenOrigin(slot)
  }

  private async setHyprlandFullscreen(slot: Slot, enabled: boolean): Promise<void> {
    const result = await hyprland.setHyprlandPlayerFullscreen({
      enabled,
      title: FLOAT_TITLE,
      desiredPinned: () => slot.pinned,
      timeoutMs: 10_000
    })

    const expectedState = enabled ? 2 : 0

    if (
      !result.supported ||
      result.observation?.fullscreen !== expectedState ||
      result.observation?.fullscreenClient !== expectedState
    ) {
      slot.pinStatus = slot.pinned ? (enabled ? 'requested' : 'unknown') : slot.pinStatus
      throw new Error(`hyprland-fullscreen-${String(result.status || 'unobserved')}`)
    }

    slot.fullscreen = enabled
    slot.pinMode = 'all-workspaces'
    slot.pinStatus = enabled
      ? slot.pinned
        ? 'requested'
        : result.observation?.pinned === false
          ? 'observed'
          : 'unknown'
      : slot.pinned && result.observation?.pinned
        ? 'observed'
        : 'requested'
  }

  private async restoreFullscreenOrigin(slot: Slot): Promise<void> {
    if (slot.fullscreenOrigin === 'docked' && slot.owner && !slot.owner.isDestroyed()) {
      await this.dock(slot)
    } else {
      slot.fullscreenOrigin = null
    }
  }

  private transitionNativeFullscreen(window: BrowserWindow, desired: boolean): Promise<void> {
    if (window.isDestroyed()) {
      return Promise.reject(new Error('fullscreen-window-destroyed'))
    }

    if (window.isFullScreen() === desired) {
      return Promise.resolve()
    }

    const emitter = window as unknown as {
      on: (event: string, listener: () => void) => void
      removeListener: (event: string, listener: () => void) => void
    }

    return new Promise((resolve, reject) => {
      const event = desired ? 'enter-full-screen' : 'leave-full-screen'
      let eventSeen = false
      let poll: ReturnType<typeof setTimeout> | undefined
      const timeout = setTimeout(() => finish(new Error('fullscreen-transition-timeout')), 10_000)

      const handler = () => {
        eventSeen = true
        check()
      }

      const closed = () => finish(new Error('fullscreen-window-destroyed'))

      const check = () => {
        if (!eventSeen) {
          return
        }

        if (window.isDestroyed()) {
          return finish(new Error('fullscreen-window-destroyed'))
        }

        if (window.isFullScreen() === desired) {
          return finish()
        }

        poll = setTimeout(check, 25)
      }

      const finish = (error?: Error) => {
        cleanup()

        if (error) {
          reject(error)
        } else {
          resolve()
        }
      }

      const cleanup = () => {
        clearTimeout(timeout)

        if (poll) {
          clearTimeout(poll)
        }

        emitter.removeListener(event, handler)
        window.removeListener('closed', closed)
      }

      emitter.on(event, handler)
      window.once('closed', closed)

      try {
        window.setFullScreen(desired)
      } catch (error) {
        finish(error instanceof Error ? error : new Error('fullscreen-transition-failed'))
      }
    })
  }

  private clampWindowBounds(bounds: Electron.Rectangle): Electron.Rectangle {
    const display = screen.getDisplayMatching(bounds)
    const area = display.workArea
    const width = Math.min(area.width, Math.max(320, bounds.width))
    const height = Math.min(area.height, Math.max(180, bounds.height))

    return {
      x: Math.min(Math.max(bounds.x, area.x), area.x + area.width - width),
      y: Math.min(Math.max(bounds.y, area.y), area.y + area.height - height),
      width,
      height
    }
  }

  private async restoreNativeFloatAfterFullscreen(slot: Slot): Promise<void> {
    const floating = slot.floatWindow

    if (!floating || floating.isDestroyed()) {
      return
    }

    if (slot.floatBounds) {
      floating.setBounds(this.clampWindowBounds(slot.floatBounds))
      slot.floatBounds = null
      slot.floatBoundsTracking?.resize()
    }

    await this.applyPin(slot)
  }

  private stopFloatBoundsTracking(slot: Slot): void {
    const tracking = slot.floatBoundsTracking

    if (!tracking) {
      return
    }

    tracking.dispose()

    if (slot.floatWindow && !slot.floatWindow.isDestroyed()) {
      slot.floatWindow.off('resize', tracking.resize)
    }

    slot.floatBoundsTracking = null
  }

  private requireSlot(owner: BrowserWindow): Slot {
    if (!this.slot || this.slot.ownerId !== owner.id) {
      throw new Error('player-unavailable')
    }

    return this.slot
  }

  private destroySlot(slot: Slot): void {
    slot.closeInProgress = true
    this.stopFloatBoundsTracking(slot)

    if (slot.mediaPoll) {
      clearInterval(slot.mediaPoll)
    }

    resetServiceScopedState(slot.authWindows, this.permissionDecisions)

    if (slot.presentation === 'floating' && slot.floatWindow && !slot.floatWindow.isDestroyed()) {
      try {
        slot.floatWindow.contentView.removeChildView(slot.guest)
      } catch {
        /* view may already be detached */
      }
    }

    if (slot.presentation === 'docked' && slot.owner && !slot.owner.isDestroyed()) {
      try {
        slot.owner.contentView.removeChildView(slot.guest)
      } catch {
        /* view may already be detached */
      }
    }

    if (!slot.guest.webContents.isDestroyed()) {
      slot.guest.webContents.close()
    }

    if (slot.floatWindow && !slot.floatWindow.isDestroyed()) {
      slot.floatWindow.destroy()
    }

    slot.closeInProgress = false
    this.snapshot = initialSnapshot()
  }

  private enqueueInternal(slot: Slot, work: () => Promise<void> | void): Promise<void> {
    const run = async () => {
      if (this.disposed || this.slot !== slot) {
        return
      }

      await work()
      this.publish()
    }

    const result = this.operation.then(run, run)
    this.operation = result.catch(() => undefined)

    return result
  }

  private publish(): void {
    const slot = this.slot

    if (!slot) {
      if (this.snapshot.status !== 'error') {
        this.snapshot = initialSnapshot()
      }
    } else {
      const url = slot.guest.webContents.getURL()
      const observedOrigin = safePlayerOrigin(url, this.fixture, slot.source)
      this.snapshot = {
        status: slot.status,
        source: slot.source,
        presentation: slot.presentation,
        pinned: slot.pinned,
        pinStatus: slot.pinStatus,
        pinMode: slot.pinMode,
        fullscreen: slot.fullscreen,
        currentTime: slot.currentTime,
        duration: slot.duration,
        paused: slot.paused,
        muted: slot.muted,
        canGoBack: !slot.guest.webContents.isDestroyed() && slot.guest.webContents.canGoBack(),
        canGoForward: !slot.guest.webContents.isDestroyed() && slot.guest.webContents.canGoForward(),
        awarenessLinked: this.awarenessLink?.ownerId === slot.ownerId,
        safeOrigin: observedOrigin,
        guestId: slot.guest.webContents.id,
        runtime: slot.runtime,
        error: slot.error
      }
    }

    for (const [ownerId, listeners] of this.listeners) {
      const ownsState = this.slot?.ownerId === ownerId || this.statusOwnerId === ownerId
      const base = ownsState ? this.snapshot : initialSnapshot()
      const snapshot = this.slot ? base : { ...base, awarenessLinked: this.awarenessLink?.ownerId === ownerId }

      for (const listener of listeners) {
        listener(snapshot)
      }
    }
  }
}

function playerOperationError(operation: PlayerAgentOperation, code: string, message: string): PlayerAgentResponse {
  return { operation: operation.type, error: { code, message } }
}
