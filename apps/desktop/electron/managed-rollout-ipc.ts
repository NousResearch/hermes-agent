/**
 * Narrow main-process IPC boundary for managed-rollout reads and commands.
 * Electron registration is intentionally left to T12.3; this module receives
 * an already-validated sender predicate so it can be unit-tested without a
 * BrowserWindow or a general-purpose bridge.
 */

import {
  validateRolloutCommand,
  type PromotionPolicy,
  type RolloutAction
} from '../src/lib/managed-rollout-contract'

const MAX_REQUEST_BYTES = 256 * 1024
const MAX_SNAPSHOT_BYTES = 8 * 1024 * 1024
const MAX_PAGE_SIZE = 50
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/

export type ManagedRolloutIpcMethod =
  | 'capabilities'
  | 'inventory'
  | 'resolveTarget'
  | 'preflight'
  | 'start'
  | 'activeRevision'
  | 'get'
  | 'command'
  | 'history'
  | 'events'

export interface ManagedRolloutIpcCapabilities {
  protocol: 1
  available: boolean
  reason: string | null
  maxConcurrency: number
  maxInstallations: number
}

export interface ManagedRolloutIpcCommand {
  id: string
  revision: number
  requestId: string
  kind: RolloutAction
  installId?: string
  action?: RolloutAction
  expectedRevision?: number
  reason?: string | null
  promotionPolicy?: PromotionPolicy | null
}

export interface ManagedRolloutIpcAdapter {
  capabilities: () => Promise<ManagedRolloutIpcCapabilities>
  inventory?: () => Promise<unknown>
  resolveTarget?: (request: { connectionIds: string[]; inventoryRevision: string; retryOf: string | null }) => Promise<unknown>
  preflight?: (draft: unknown) => Promise<unknown>
  start?: (request: { token: string; requestId: string }) => Promise<unknown>
  activeRevision: () => Promise<number | null>
  get: (id: string) => Promise<unknown | null>
  command: (command: ManagedRolloutIpcCommand) => Promise<unknown>
  history: (page: { cursor?: string; limit: number }) => Promise<unknown>
  events: (page: { id: string; cursor?: string; limit: number }) => Promise<unknown>
}

export interface ManagedRolloutIpcContext {
  sender: unknown
}

export type ManagedRolloutIpcResult =
  | { ok: true; value: unknown }
  | { ok: false; code: 'forbidden' | 'invalid-request' | 'snapshot-too-large' | 'unavailable'; message: string }

function byteLength(value: unknown): number {
  if (value === undefined) return 0

  try {
    const encoded = JSON.stringify(value)
    return encoded === undefined ? 0 : Buffer.byteLength(encoded, 'utf8')
  } catch {
    return Number.POSITIVE_INFINITY
  }
}

function isObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function exactKeys(value: Record<string, unknown>, keys: readonly string[]): boolean {
  const actual = Object.keys(value).sort()
  return actual.length === keys.length && actual.every((key, index) => key === [...keys].sort()[index])
}

const INSTALL_ID_RE = /^[0-9a-f]{32}$/

function identifier(value: unknown): string | null {
  const normalized = typeof value === 'string' ? value.trim().toLowerCase() : ''
  return UUID_RE.test(normalized) ? normalized : null
}

function installationIdentifier(value: unknown): string | null {
  return typeof value === 'string' && INSTALL_ID_RE.test(value) ? value : null
}

function page(value: unknown, withId: boolean): { id?: string; cursor?: string; limit: number } | null {
  if (!isObject(value)) return null
  const allowed = new Set(withId ? ['id', 'cursor', 'limit'] : ['cursor', 'limit'])
  if (!Object.keys(value).every(key => allowed.has(key)) || !Object.hasOwn(value, 'limit') || (withId && !Object.hasOwn(value, 'id'))) {
    return null
  }
  const limit = value.limit
  const cursor = value.cursor
  const id = withId ? identifier(value.id) : undefined
  if (!Number.isSafeInteger(limit) || (limit as number) < 1 || (limit as number) > MAX_PAGE_SIZE) return null
  if (cursor !== undefined && (typeof cursor !== 'string' || cursor.length > 1024)) return null
  if (withId && !id) return null
  return { ...(id ? { id } : {}), ...(typeof cursor === 'string' ? { cursor } : {}), limit: limit as number }
}

function command(value: unknown): ManagedRolloutIpcCommand | null {
  if (!isObject(value)) return null

  try {
    const parsed = validateRolloutCommand(value)
    if (!identifier(parsed.id) || !identifier(parsed.requestId)) return null
    const installId = parsed.installId === null ? undefined : installationIdentifier(parsed.installId)
    if (parsed.installId !== null && !installId) return null
    return {
      id: parsed.id,
      revision: parsed.expectedRevision,
      requestId: parsed.requestId,
      kind: parsed.action,
      ...(installId ? { installId } : {}),
      action: parsed.action,
      expectedRevision: parsed.expectedRevision,
      reason: parsed.reason,
      promotionPolicy: parsed.promotionPolicy
    }
  } catch {
    return null
  }
}

function exactString(value: unknown, maxLength: number): value is string {
  return typeof value === 'string' && value.length > 0 && value.length <= maxLength && !/[\x00-\x1f\x7f]/.test(value)
}

function resolveTargetRequest(value: unknown): { connectionIds: string[]; inventoryRevision: string; retryOf: string | null } | null {
  if (!isObject(value) || !exactKeys(value, ['connectionIds', 'inventoryRevision', 'retryOf'])) return null
  if (!Array.isArray(value.connectionIds) || value.connectionIds.length === 0 || value.connectionIds.length > 500) return null
  const connectionIds = value.connectionIds.map(identifier)
  if (connectionIds.some(id => id === null) || new Set(connectionIds).size !== connectionIds.length) return null
  if (!exactString(value.inventoryRevision, 256)) return null
  if (value.retryOf !== null && !identifier(value.retryOf)) return null
  return { connectionIds: connectionIds as string[], inventoryRevision: value.inventoryRevision, retryOf: value.retryOf as string | null }
}

function preflightDraft(value: unknown): unknown | null {
  if (!isObject(value) || !exactKeys(value, ['draft']) || !isObject(value.draft)) return null
  return value.draft
}

function startRequest(value: unknown): { token: string; requestId: string } | null {
  if (!isObject(value) || !exactKeys(value, ['token', 'requestId'])) return null
  const requestId = identifier(value.requestId)
  if (!requestId || !exactString(value.token, 4096)) return null
  return { token: value.token, requestId }
}

function rejected(code: Extract<ManagedRolloutIpcResult, { ok: false }>['code'], message: string): ManagedRolloutIpcResult {
  return { ok: false, code, message }
}

export function createManagedRolloutIpcHandler(
  adapter: ManagedRolloutIpcAdapter,
  isTrustedSender: (sender: unknown) => boolean
) {
  return async (context: ManagedRolloutIpcContext, method: unknown, payload: unknown): Promise<ManagedRolloutIpcResult> => {
    if (!isTrustedSender(context.sender)) return rejected('forbidden', 'Managed rollout IPC requires a trusted sender.')
    if (byteLength(payload) > MAX_REQUEST_BYTES) return rejected('invalid-request', 'Managed rollout IPC request exceeds 256 KiB.')
    if (typeof method !== 'string' || ![
      'capabilities', 'inventory', 'resolveTarget', 'preflight', 'start',
      'activeRevision', 'get', 'command', 'history', 'events'
    ].includes(method)) {
      return rejected('invalid-request', 'Managed rollout IPC method is not allowed.')
    }

    try {
      if (method === 'capabilities' || method === 'inventory' || method === 'activeRevision') {
        if (payload !== undefined) return rejected('invalid-request', 'This managed rollout IPC method does not accept a payload.')
        if (method === 'inventory') {
          if (!adapter.inventory) throw new Error('Managed rollout service is unavailable.')
          return { ok: true, value: await adapter.inventory() }
        }
        return { ok: true, value: method === 'capabilities' ? await adapter.capabilities() : await adapter.activeRevision() }
      }

      if (method === 'resolveTarget') {
        const parsed = resolveTargetRequest(payload)
        if (!parsed) return rejected('invalid-request', 'Managed rollout target-resolution request is invalid.')
        if (!adapter.resolveTarget) throw new Error('Managed rollout service is unavailable.')
        return { ok: true, value: await adapter.resolveTarget(parsed) }
      }

      if (method === 'preflight') {
        const draft = preflightDraft(payload)
        if (draft === null) return rejected('invalid-request', 'Managed rollout preflight request is invalid.')
        if (!adapter.preflight) throw new Error('Managed rollout service is unavailable.')
        return { ok: true, value: await adapter.preflight(draft) }
      }

      if (method === 'start') {
        const parsed = startRequest(payload)
        if (!parsed) return rejected('invalid-request', 'Managed rollout start request is invalid.')
        if (!adapter.start) throw new Error('Managed rollout service is unavailable.')
        return { ok: true, value: await adapter.start(parsed) }
      }

      if (method === 'get') {
        if (!isObject(payload) || !exactKeys(payload, ['id'])) return rejected('invalid-request', 'Managed rollout snapshot request is invalid.')
        const id = identifier(payload.id)
        if (!id) return rejected('invalid-request', 'Managed rollout identifier is invalid.')
        const snapshot = await adapter.get(id)
        if (byteLength(snapshot) > MAX_SNAPSHOT_BYTES) return rejected('snapshot-too-large', 'Managed rollout snapshot exceeds 8 MiB.')
        return { ok: true, value: snapshot }
      }

      if (method === 'command') {
        const parsed = command(payload)
        if (!parsed) return rejected('invalid-request', 'Managed rollout command is invalid.')
        return { ok: true, value: await adapter.command(parsed) }
      }

      const parsed = page(payload, method === 'events')
      if (!parsed) return rejected('invalid-request', 'Managed rollout page request is invalid.')
      return {
        ok: true,
        value: method === 'events' ? await adapter.events(parsed as { id: string; cursor?: string; limit: number }) : await adapter.history(parsed)
      }
    } catch (error) {
      return rejected('unavailable', error instanceof Error ? error.message : 'Managed rollout service is unavailable.')
    }
  }
}

export { MAX_PAGE_SIZE, MAX_REQUEST_BYTES, MAX_SNAPSHOT_BYTES }
