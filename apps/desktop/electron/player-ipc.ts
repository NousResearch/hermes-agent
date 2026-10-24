import type { BrowserWindow, IpcMain, IpcMainInvokeEvent } from 'electron'

import type {
  PlayerAgentRequest,
  PlayerAgentResponse,
  PlayerAgentScope,
  PlayerAwarenessLinkState,
  PlayerMediaJobCapabilities
} from '../src/lib/player-context-contract'
import type { PlayerCommand, PlayerListener, PlayerSnapshot } from '../src/lib/player-contract'
import { isPlayerSource } from '../src/lib/player-services'

interface PlayerHost {
  request(owner: BrowserWindow, command: PlayerCommand): Promise<PlayerSnapshot>
  linkAwareness(owner: BrowserWindow, scope: PlayerAgentScope): Promise<PlayerAwarenessLinkState>
  unlinkAwareness(owner: BrowserWindow): Promise<PlayerAwarenessLinkState>
  agentRequest(owner: BrowserWindow, request: PlayerAgentRequest): Promise<PlayerAgentResponse>
  setupMediaHelpers(owner: BrowserWindow): Promise<PlayerMediaJobCapabilities>
  revealMediaJob(owner: BrowserWindow, scope: PlayerAgentScope, jobId: string): Promise<boolean>
  clearLocalData(owner: BrowserWindow): Promise<PlayerSnapshot>
  subscribe(owner: BrowserWindow, listener: PlayerListener): () => void
  ownerClosed(owner: BrowserWindow): void
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object' && !Array.isArray(value)
}

function validScope(value: unknown): value is PlayerAgentScope {
  if (!isRecord(value) || Object.keys(value).length !== 3) {
    return false
  }

  return ['connectionId', 'profileId', 'sessionId'].every(
    key => typeof value[key] === 'string' && (value[key] as string).length > 0 && (value[key] as string).length <= 256
  )
}

export function parsePlayerAgentRequest(value: unknown): PlayerAgentRequest {
  if (!validAgentRequest(value)) {
    throw new Error('Invalid player agent request')
  }

  return value
}

function validAgentRequest(value: unknown): value is PlayerAgentRequest {
  if (
    !isRecord(value) ||
    !only(value, ['scope', 'operation']) ||
    !validScope(value.scope) ||
    !isRecord(value.operation)
  ) {
    return false
  }

  return isValidAgentOperation(value.operation)
}

function isValidAgentOperation(operation: Record<string, unknown>): boolean {
  return (
    typeof operation.type === 'string' &&
    Object.hasOwn(agentOperationValidators, operation.type) &&
    agentOperationValidators[operation.type](operation)
  )
}

const agentOperationValidators: Record<string, (operation: Record<string, unknown>) => boolean> = {
  context: operation => Object.keys(operation).length === 1,
  control: validControlOperation,
  'transcript.start': validTranscriptOperation,
  'download.start': validDownloadOperation,
  'jobs.list': operation => Object.keys(operation).length === 1,
  'job.status': validJobOperation,
  'job.cancel': validJobOperation
}

function validControlOperation(operation: Record<string, unknown>): boolean {
  return (
    only(operation, ['type', 'contextToken', 'navigationGeneration', 'action', 'positionSeconds']) &&
    typeof operation.contextToken === 'string' &&
    operation.contextToken.length >= 24 &&
    operation.contextToken.length <= 256 &&
    Number.isSafeInteger(operation.navigationGeneration) &&
    Number(operation.navigationGeneration) >= 0 &&
    ['play', 'pause', 'mute', 'unmute', 'seek', 'pin', 'unpin', 'fullscreen', 'exit-fullscreen'].includes(
      String(operation.action)
    ) &&
    (operation.action !== 'seek' || typeof operation.positionSeconds === 'number') &&
    (operation.action === 'seek' || operation.positionSeconds === undefined) &&
    (operation.positionSeconds === undefined ||
      (typeof operation.positionSeconds === 'number' &&
        Number.isFinite(operation.positionSeconds) &&
        operation.positionSeconds >= 0))
  )
}

function validTranscriptOperation(operation: Record<string, unknown>): boolean {
  return (
    only(operation, ['type', 'contextToken', 'language', 'range']) &&
    typeof operation.contextToken === 'string' &&
    operation.contextToken.length >= 24 &&
    (operation.language === undefined ||
      (typeof operation.language === 'string' && /^[A-Za-z0-9-]{1,32}$/.test(operation.language))) &&
    (operation.range === undefined || validRange(operation.range))
  )
}

function validDownloadOperation(operation: Record<string, unknown>): boolean {
  return (
    only(operation, ['type', 'contextToken', 'quality']) &&
    typeof operation.contextToken === 'string' &&
    operation.contextToken.length >= 24 &&
    (operation.quality === undefined || ['best', '720p', '1080p', 'audio'].includes(String(operation.quality)))
  )
}

function validJobOperation(operation: Record<string, unknown>): boolean {
  return (
    only(operation, ['type', 'jobId']) &&
    typeof operation.jobId === 'string' &&
    /^[A-Za-z0-9_-]{1,128}$/.test(operation.jobId)
  )
}

function validRange(value: unknown): boolean {
  if (!isRecord(value) || !only(value, ['startSeconds', 'endSeconds']) || Object.keys(value).length === 0) {
    return false
  }

  if (
    !['startSeconds', 'endSeconds'].every(
      key =>
        value[key] === undefined ||
        (typeof value[key] === 'number' &&
          Number.isFinite(value[key]) &&
          Number(value[key]) >= 0 &&
          Number(value[key]) <= 86_400)
    )
  ) {
    return false
  }

  return (
    value.startSeconds === undefined ||
    value.endSeconds === undefined ||
    Number(value.endSeconds) >= Number(value.startSeconds)
  )
}

const noArguments = (value: Record<string, unknown>) => Object.keys(value).length === 1
const only = (value: Record<string, unknown>, keys: string[]) => Object.keys(value).every(key => keys.includes(key))

const parsers: Record<string, (value: Record<string, unknown>) => boolean> = {
  status: noArguments,
  dock: noArguments,
  popout: noArguments,
  close: noArguments,
  home: noArguments,
  back: noArguments,
  forward: noArguments,
  'media-control': value =>
    only(value, ['type', 'action', 'positionSeconds']) &&
    ['play', 'pause', 'mute', 'unmute', 'seek'].includes(String(value.action)) &&
    (value.action !== 'seek' || typeof value.positionSeconds === 'number') &&
    (value.action === 'seek' || value.positionSeconds === undefined) &&
    (value.positionSeconds === undefined ||
      (typeof value.positionSeconds === 'number' &&
        Number.isFinite(value.positionSeconds) &&
        value.positionSeconds >= 0)),
  open: value => only(value, ['type', 'source']) && isPlayerSource(value.source),
  pin: value => only(value, ['type', 'pinned']) && typeof value.pinned === 'boolean',
  fullscreen: value => only(value, ['type', 'fullscreen']) && typeof value.fullscreen === 'boolean',
  bounds: value => {
    if (!only(value, ['type', 'bounds', 'visible', 'revision']) || typeof value.visible !== 'boolean') {
      return false
    }

    if (!Number.isSafeInteger(value.revision) || Number(value.revision) < 0) {
      return false
    }

    const bounds = value.bounds

    if (!bounds || typeof bounds !== 'object' || Array.isArray(bounds)) {
      return false
    }

    const rectangle = bounds as Record<string, unknown>

    return (
      Object.keys(rectangle).length === 4 &&
      ['x', 'y', 'width', 'height'].every(
        key =>
          typeof rectangle[key] === 'number' &&
          Number.isFinite(rectangle[key]) &&
          Math.abs(Number(rectangle[key])) <= 100000
      ) &&
      Number(rectangle.width) >= 0 &&
      Number(rectangle.height) >= 0
    )
  }
}

export function parsePlayerCommand(value: unknown): PlayerCommand {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error('Invalid player request')
  }

  const command = value as Record<string, unknown>

  if (typeof command.type !== 'string' || !Object.hasOwn(parsers, command.type) || !parsers[command.type](command)) {
    throw new Error('Invalid player request')
  }

  return command as PlayerCommand
}

function trustedPage(actual: string, expected: string): boolean {
  try {
    const page = new URL(actual)
    const target = new URL(expected)

    return page.protocol === target.protocol && page.host === target.host && page.pathname === target.pathname
  } catch {
    return false
  }
}

/** Only explicitly registered chat windows can own the player; guests never join this map. */
export function registerPlayerIpc(ipc: IpcMain, host: PlayerHost) {
  const owners = new Map<number, { window: BrowserWindow; page: string; unsubscribe: () => void }>()

  const authenticate = (event: IpcMainInvokeEvent) => {
    const entry = owners.get(event.sender.id)

    if (
      !entry ||
      entry.window.isDestroyed() ||
      entry.window.webContents !== event.sender ||
      !event.senderFrame ||
      event.senderFrame !== event.sender.mainFrame ||
      !trustedPage(event.senderFrame.url, entry.page)
    ) {
      throw new Error('Player request is unavailable from this surface')
    }

    return entry.window
  }

  ipc.handle('hermes:player:request', (event, value) => {
    const owner = authenticate(event)
    const command = parsePlayerCommand(value)

    if (command.type === 'bounds') {
      // DOM rectangles are CSS pixels; contentView bounds are device-independent pixels.
      const zoom = event.sender.getZoomFactor()

      if (!Number.isFinite(zoom) || zoom <= 0) {
        throw new Error('Invalid player layout scale')
      }

      const { x, y, width, height } = command.bounds

      return host.request(owner, {
        ...command,
        bounds: { x: x * zoom, y: y * zoom, width: width * zoom, height: height * zoom }
      })
    }

    return host.request(owner, command)
  })
  ipc.handle('hermes:player:link-awareness', (event, value) => {
    const owner = authenticate(event)

    if (!validScope(value)) {
      throw new Error('Invalid player awareness scope')
    }

    return host.linkAwareness(owner, value)
  })
  ipc.handle('hermes:player:unlink-awareness', event => host.unlinkAwareness(authenticate(event)))
  ipc.handle('hermes:player:agent-request', (event, value) => {
    const owner = authenticate(event)

    return host.agentRequest(owner, parsePlayerAgentRequest(value))
  })
  ipc.handle('hermes:player:setup-media-helpers', event => host.setupMediaHelpers(authenticate(event)))
  ipc.handle('hermes:player:reveal-media-job', (event, value) => {
    const owner = authenticate(event)

    if (
      !isRecord(value) ||
      !only(value, ['scope', 'jobId']) ||
      !validScope(value.scope) ||
      typeof value.jobId !== 'string' ||
      !/^[A-Za-z0-9_-]{1,128}$/.test(value.jobId)
    ) {
      throw new Error('Invalid media job request')
    }

    return host.revealMediaJob(owner, value.scope, value.jobId)
  })
  ipc.handle('hermes:player:clear-local-data', event => host.clearLocalData(authenticate(event)))

  return {
    registerOwner(window: BrowserWindow, page: string) {
      const id = window.webContents.id

      if (owners.has(id)) {
        return
      }

      const unsubscribe = host.subscribe(window, snapshot => {
        if (!window.isDestroyed() && !window.webContents.isDestroyed()) {
          window.webContents.send('hermes:player:state', snapshot)
        }
      })

      owners.set(id, { window, page, unsubscribe })
      window.once('closed', () => {
        unsubscribe()
        owners.delete(id)
        host.ownerClosed(window)
      })
    },
    dispose() {
      for (const entry of owners.values()) {
        entry.unsubscribe()
      }

      owners.clear()
      ipc.removeHandler('hermes:player:request')
      ipc.removeHandler('hermes:player:link-awareness')
      ipc.removeHandler('hermes:player:unlink-awareness')
      ipc.removeHandler('hermes:player:agent-request')
      ipc.removeHandler('hermes:player:setup-media-helpers')
      ipc.removeHandler('hermes:player:reveal-media-job')
      ipc.removeHandler('hermes:player:clear-local-data')
    }
  }
}
