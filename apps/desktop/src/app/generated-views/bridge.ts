import { getStatus, getUsageAnalytics } from '@/hermes'
import { readJson, writeJson } from '@/lib/storage'

import type { GeneratedViewBinding, GeneratedViewCapability } from './manifest'
import type { GeneratedViewDocument } from './store'

export const GENERATED_VIEW_BRIDGE_VERSION = 1
export const GENERATED_VIEW_STATE_MAX_BYTES = 64 * 1024
const GENERATED_VIEW_DATA_MAX_BYTES = 256 * 1024
const REQUEST_TIMEOUT_MS = 10_000
const STATE_KEY = 'hermes.desktop.generatedViewState.v1'

type BridgeErrorCode =
  | 'binding_denied'
  | 'capability_denied'
  | 'payload_too_large'
  | 'resolve_failed'
  | 'timeout'

type OutboundMessage =
  | { v: 1; type: 'hermes:data'; requestId: string; bindingId: string; data: unknown }
  | { v: 1; type: 'hermes:theme'; requestId: string; tokens: Record<string, string> }
  | { v: 1; type: 'hermes:state'; requestId: string; state: unknown; version: number }
  | { v: 1; type: 'hermes:error'; requestId: string; code: BridgeErrorCode; message: string }

interface StateRecord {
  state: unknown
  version: number
}

interface StateStore {
  version: 1
  views: Record<string, StateRecord>
}

export interface GeneratedViewBridge {
  dispose: () => void
  handleMessage: (data: unknown, post: (message: OutboundMessage) => void) => void
}

interface BridgeOptions {
  capabilities: readonly GeneratedViewCapability[]
  bindings: readonly GeneratedViewBinding[]
  connectionKey: string
  id: string
  resolveBinding?: (binding: GeneratedViewBinding) => Promise<unknown>
  resolveTheme?: () => Record<string, string>
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value))
}

function serializedBytes(value: unknown): number | null {
  try {
    const json = JSON.stringify(value)

    return json === undefined ? null : new TextEncoder().encode(json).byteLength
  } catch {
    return null
  }
}

function readStateStore(): StateStore {
  const parsed = readJson<Partial<StateStore>>(STATE_KEY)

  if (!parsed || parsed.version !== 1 || !parsed.views || typeof parsed.views !== 'object') {
    return { version: 1, views: {} }
  }

  return { version: 1, views: parsed.views as Record<string, StateRecord> }
}

function stateIdentity(connectionKey: string, id: string): string {
  return JSON.stringify([connectionKey, id])
}

function readViewState(connectionKey: string, id: string): StateRecord {
  const record = readStateStore().views[stateIdentity(connectionKey, id)]

  return record && Number.isSafeInteger(record.version) && record.version >= 0
    ? record
    : { state: null, version: 0 }
}

function writeViewState(connectionKey: string, id: string, state: unknown): StateRecord {
  const size = serializedBytes(state)

  if (size === null) {
    throw new Error('state must be JSON-serializable')
  }

  if (size > GENERATED_VIEW_STATE_MAX_BYTES) {
    throw new Error(`state exceeds ${GENERATED_VIEW_STATE_MAX_BYTES} bytes`)
  }

  const store = readStateStore()
  const key = stateIdentity(connectionKey, id)
  const next = { state, version: (store.views[key]?.version ?? 0) + 1 }
  store.views[key] = next
  writeJson(STATE_KEY, store)

  return next
}

const THEME_PROPERTIES = [
  '--theme-background-seed',
  '--theme-foreground',
  '--theme-primary',
  '--theme-secondary',
  '--dt-border',
  '--dt-muted',
  '--dt-font-sans',
  '--dt-font-mono'
] as const

export function readGeneratedViewTheme(): Record<string, string> {
  if (typeof document === 'undefined' || typeof getComputedStyle !== 'function') {
    return {}
  }

  const styles = getComputedStyle(document.documentElement)

  return Object.fromEntries(
    THEME_PROPERTIES.flatMap(property => {
      const value = styles.getPropertyValue(property).trim()

      return value ? [[property, value]] : []
    })
  )
}

export async function resolveGeneratedViewBinding(binding: GeneratedViewBinding): Promise<unknown> {
  if (binding === 'hermes:status') {
    const status = await getStatus()

    return {
      activeSessions: status.active_sessions,
      gateway: {
        running: status.gateway_running,
        state: status.gateway_state,
        platforms: Object.fromEntries(
          Object.entries(status.gateway_platforms).map(([id, platform]) => [id, { state: platform.state }])
        )
      },
      releaseDate: status.release_date,
      version: status.version
    }
  }

  const analytics = await getUsageAnalytics(30)

  return {
    periodDays: analytics.period_days,
    totals: analytics.totals,
    daily: analytics.daily,
    byModel: analytics.by_model,
    skills: analytics.skills.summary,
    tools: analytics.tools ?? []
  }
}

function requestId(data: Record<string, unknown>): string | null {
  return typeof data.requestId === 'string' && data.requestId.length > 0 && data.requestId.length <= 128
    ? data.requestId
    : null
}

export function createGeneratedViewBridge(options: BridgeOptions): GeneratedViewBridge {
  const capabilities = new Set(options.capabilities)
  const bindings = new Set(options.bindings)
  const resolveBinding = options.resolveBinding ?? resolveGeneratedViewBinding
  const resolveTheme = options.resolveTheme ?? readGeneratedViewTheme
  const timers = new Set<ReturnType<typeof setTimeout>>()
  let disposed = false

  const error = (post: (message: OutboundMessage) => void, id: string, code: BridgeErrorCode, message: string) => {
    if (!disposed) {
      post({ v: 1, type: 'hermes:error', requestId: id, code, message })
    }
  }

  const resolveData = async (
    post: (message: OutboundMessage) => void,
    id: string,
    binding: GeneratedViewBinding
  ) => {
    let settled = false

    const timer = setTimeout(() => {
      if (!settled && !disposed) {
        settled = true
        timers.delete(timer)
        error(post, id, 'timeout', 'binding resolution timed out')
      }
    }, REQUEST_TIMEOUT_MS)

    timers.add(timer)

    try {
      const data = await resolveBinding(binding)

      if (settled || disposed) {
        return
      }

      settled = true
      clearTimeout(timer)
      timers.delete(timer)
      const bytes = serializedBytes(data)

      if (bytes === null || bytes > GENERATED_VIEW_DATA_MAX_BYTES) {
        error(post, id, 'payload_too_large', 'binding result exceeds the generated-view data limit')

        return
      }

      post({ v: 1, type: 'hermes:data', requestId: id, bindingId: binding, data })
    } catch (cause) {
      if (!settled && !disposed) {
        settled = true
        clearTimeout(timer)
        timers.delete(timer)
        error(post, id, 'resolve_failed', cause instanceof Error ? cause.message : String(cause))
      }
    }
  }

  return {
    handleMessage(data, post) {
      if (disposed || !isRecord(data) || data.v !== GENERATED_VIEW_BRIDGE_VERSION || typeof data.type !== 'string') {
        return
      }

      const id = requestId(data)

      if (!id) {
        return
      }

      if (data.type === 'hermes:getTheme') {
        if (!capabilities.has('theme:read')) {
          error(post, id, 'capability_denied', 'view lacks the theme:read capability')

          return
        }

        post({ v: 1, type: 'hermes:theme', requestId: id, tokens: resolveTheme() })

        return
      }

      if (data.type === 'hermes:getState') {
        if (!capabilities.has('state:persist')) {
          error(post, id, 'capability_denied', 'view lacks the state:persist capability')

          return
        }

        const record = readViewState(options.connectionKey, options.id)
        post({ v: 1, type: 'hermes:state', requestId: id, state: record.state, version: record.version })

        return
      }

      if (data.type === 'hermes:setState') {
        if (!capabilities.has('state:persist')) {
          error(post, id, 'capability_denied', 'view lacks the state:persist capability')

          return
        }

        if (!Object.hasOwn(data, 'state')) {
          return
        }

        try {
          const record = writeViewState(options.connectionKey, options.id, data.state)
          post({ v: 1, type: 'hermes:state', requestId: id, state: record.state, version: record.version })
        } catch (cause) {
          error(post, id, 'payload_too_large', cause instanceof Error ? cause.message : String(cause))
        }

        return
      }

      if (data.type === 'hermes:getData') {
        const binding = typeof data.bindingId === 'string' ? data.bindingId : null

        if (!binding || !bindings.has(binding as GeneratedViewBinding)) {
          error(post, id, 'binding_denied', `binding is not declared: ${String(binding ?? '')}`)

          return
        }

        void resolveData(post, id, binding as GeneratedViewBinding)
      }
      // Unknown message types are intentionally dropped without a response.
    },
    dispose() {
      disposed = true
      timers.forEach(clearTimeout)
      timers.clear()
    }
  }
}

export function bridgeForGeneratedView(view: GeneratedViewDocument): GeneratedViewBridge {
  return createGeneratedViewBridge({
    capabilities: view.manifest.capabilities,
    bindings: view.manifest.bindings,
    connectionKey: view.connectionKey,
    id: view.manifest.id
  })
}
