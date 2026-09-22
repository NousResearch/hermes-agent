/**
 * Narrow main-process IPC boundary for managed-rollout reads and commands.
 * Electron registration is intentionally left to T12.3; this module receives
 * an already-validated sender predicate so it can be unit-tested without a
 * BrowserWindow or a general-purpose bridge.
 */

const MAX_REQUEST_BYTES = 256 * 1024
const MAX_SNAPSHOT_BYTES = 8 * 1024 * 1024
const MAX_PAGE_SIZE = 50
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/

export type ManagedRolloutIpcMethod = 'capabilities' | 'activeRevision' | 'get' | 'command' | 'history' | 'events'

export interface ManagedRolloutIpcCapabilities {
  protocol: 1
  available: boolean
  reason?: string
  maxConcurrency: number
  maxInstallations: number
}

export interface ManagedRolloutIpcCommand {
  id: string
  revision: number
  requestId: string
  kind: 'pause' | 'resume' | 'stop' | 'promote' | 'exclude'
  installId?: string
}

export interface ManagedRolloutIpcAdapter {
  capabilities: () => Promise<ManagedRolloutIpcCapabilities>
  activeRevision: () => Promise<number | null>
  get: (id: string) => Promise<unknown | null>
  command: (command: ManagedRolloutIpcCommand) => Promise<{ ok: boolean; id: string; revision: number; code?: string }>
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

function identifier(value: unknown): string | null {
  const normalized = typeof value === 'string' ? value.trim().toLowerCase() : ''
  return UUID_RE.test(normalized) ? normalized : null
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
  const base = ['id', 'revision', 'requestId', 'kind']
  const kind = value.kind
  const requiresInstall = kind === 'exclude'
  if (!exactKeys(value, requiresInstall ? [...base, 'installId'] : base)) return null
  const id = identifier(value.id)
  const requestId = identifier(value.requestId)
  if (!id || !requestId || !Number.isSafeInteger(value.revision) || (value.revision as number) < 0) return null
  if (!['pause', 'resume', 'stop', 'promote', 'exclude'].includes(String(kind))) return null
  const installId = requiresInstall ? identifier(value.installId) : undefined
  if (requiresInstall && !installId) return null
  return { id, revision: value.revision as number, requestId, kind: kind as ManagedRolloutIpcCommand['kind'], ...(installId ? { installId } : {}) }
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
    if (typeof method !== 'string' || !['capabilities', 'activeRevision', 'get', 'command', 'history', 'events'].includes(method)) {
      return rejected('invalid-request', 'Managed rollout IPC method is not allowed.')
    }

    try {
      if (method === 'capabilities' || method === 'activeRevision') {
        if (payload !== undefined) return rejected('invalid-request', 'This managed rollout IPC method does not accept a payload.')
        return { ok: true, value: method === 'capabilities' ? await adapter.capabilities() : await adapter.activeRevision() }
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
