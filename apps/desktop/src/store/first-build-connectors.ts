import type { ToolCallMessagePart } from '@assistant-ui/react'
import { map } from 'nanostores'

import { endFirstBuildConnect, isFirstBuildSession } from '@/app/contrib/handoff-receipt'
import {
  connectionRows,
  connectorAuthorizationUrl,
  type ConnectorRow,
  connectorText,
  recordOf
} from '@/lib/connector-tools'
import { buildConnectionStartMessage, canStartWithConnections } from '@/lib/first-build-start'
import { readKey, writeKey } from '@/lib/storage'
import type { ConnectorFlowDeps, ConnectorFlowRow } from '@/store/connector-flow'

export type FirstBuildConnectorPart = Pick<ToolCallMessagePart, 'toolCallId' | 'toolName' | 'args' | 'result'>

export interface FirstBuildConnectorRow extends ConnectorFlowRow {
  connectUrl?: string
}

export interface FirstBuildConnectorState {
  toolCallId: string
  rows: FirstBuildConnectorRow[]
  started: boolean
  /** The "[setup] links opened" note, held until the session is idle. A submit
   *  while the model's turn runs is rejected by the gateway, and the model
   *  usually calls wait in that same turn, so the note is a fallback cue. */
  pendingNote?: string
}

export const $firstBuildConnections = map<Record<string, FirstBuildConnectorState>>({})

interface OpenLinksDeps {
  open?: (url: string) => Promise<void>
}

export async function openFirstBuildLinks(storedId: string, part: FirstBuildConnectorPart, deps: OpenLinksDeps) {
  if (
    !isFirstBuildSession(storedId) ||
    part.toolName !== 'manage_connections' ||
    !['connect', 'reconnect'].includes(String(recordOf(part.args).action))
  ) {
    return
  }

  const output = recordOf(part.result)

  if (!Array.isArray(output.results)) {
    return
  }

  const entries = output.results.map(recordOf)
  const previous = $firstBuildConnections.get()[storedId]

  const rows = connectionRows(part.args, part.result).map((seed): FirstBuildConnectorRow => {
    const existing = previous?.rows.find(row => row.connector === seed.connector)
    const entry = entries.find(row => row.connector === seed.connector)
    const connectUrl = entry?.status === 'initiated' ? connectorAuthorizationUrl(entry.connect_url) : null

    return {
      ...seed,
      ...existing,
      phase:
        entry?.status === 'active'
          ? 'connected'
          : connectUrl && previous?.toolCallId !== part.toolCallId
            ? 'waiting'
            : (existing?.phase ?? 'idle'),
      connectUrl: connectUrl ?? undefined
    }
  })

  $firstBuildConnections.setKey(storedId, {
    toolCallId: part.toolCallId,
    rows,
    started: previous?.started ?? readKey(`hermes.onboarding.started.v1.${storedId}`) === '1'
  })

  const links = entries.flatMap(entry => {
    const url = entry.status === 'initiated' ? connectorAuthorizationUrl(entry.connect_url) : null
    const connector = connectorText(entry.connector)

    return url && connector !== undefined ? [{ connector, url }] : []
  })

  const key = `hermes.onboarding.links-opened.v1.${part.toolCallId}`

  if (!deps.open || !links.length || readKey(key) === '1') {
    return
  }

  // Claim before opening so concurrent renders and relaunches cannot open the batch twice.
  writeKey(key, '1')
  const open = deps.open
  const outcomes = await Promise.allSettled(links.map(link => open(link.url)))
  const opened = links.filter((_link, index) => outcomes[index].status === 'fulfilled')

  const state = $firstBuildConnections.get()[storedId]

  if (opened.length && state?.toolCallId === part.toolCallId) {
    $firstBuildConnections.setKey(storedId, {
      ...state,
      pendingNote: `[setup] links opened for ${opened.map(link => link.connector).join(', ')}`
    })
  }
}

/** Deliver the held note once the session is idle, if the connect that minted
 *  the links is still the newest connector part. A newer part means the model
 *  already moved on (it called wait itself), and the note would only confuse it. */
export function flushFirstBuildNote(
  storedId: string,
  newestToolCallId: string | undefined,
  busy: boolean,
  submit: (text: string) => boolean
): void {
  const state = $firstBuildConnections.get()[storedId]

  if (!state?.pendingNote || busy) {
    return
  }

  if (!isFirstBuildSession(storedId) || newestToolCallId !== state.toolCallId || submit(state.pendingNote)) {
    $firstBuildConnections.setKey(storedId, { ...state, pendingNote: undefined })
  }
}

export function watchFirstBuildRows(
  storedId: string,
  runtimeId: string,
  part: FirstBuildConnectorPart,
  request: ConnectorFlowDeps['request']
) {
  const action = recordOf(part.args).action

  if (
    !isFirstBuildSession(storedId) ||
    part.toolName !== 'manage_connections' ||
    (action !== 'wait' && !(action === 'connect' && canStartWithConnections({ ...part, type: 'tool-call' })))
  ) {
    return
  }

  const output = recordOf(part.result)
  const polling = action === 'connect' || part.result === undefined || output.status === 'pending'

  if (action === 'wait' && ['connected', 'timeout', 'interrupted'].includes(String(output.status))) {
    endFirstBuildConnect(storedId)
  }

  const connected = new Set(
    (Array.isArray(output.connectors) ? output.connectors : []).flatMap(item => {
      const entry = recordOf(item)
      const slug = connectorText(item) ?? connectorText(entry.connector)

      return slug !== undefined && entry.connected !== false ? [slug] : []
    })
  )
  const pending = new Set(Array.isArray(output.pending) ? output.pending : [])
  const previous = $firstBuildConnections.get()[storedId]

  const rows = connectionRows(part.args, part.result).map((seed): FirstBuildConnectorRow => {
    const existing = previous?.rows.find(row => row.connector === seed.connector)
    let phase = existing?.phase ?? 'waiting'

    if (output.status !== 'interrupted') {
      if (connected.has(seed.connector)) {
        phase = 'connected'
      } else if (output.status === 'timeout' && pending.has(seed.connector)) {
        phase = 'timeout'
      } else if (polling && phase !== 'connected') {
        phase = 'waiting'
      }
    }

    return { ...seed, ...existing, phase }
  })

  $firstBuildConnections.setKey(storedId, {
    toolCallId: part.toolCallId,
    rows,
    started: previous?.started ?? readKey(`hermes.onboarding.started.v1.${storedId}`) === '1'
  })

  if (!polling) {
    return
  }

  let cancelled = false
  let timer: ReturnType<typeof setTimeout> | undefined
  const current = () => !cancelled && $firstBuildConnections.get()[storedId]?.toolCallId === part.toolCallId

  const poll = async () => {
    if (!current()) {
      return
    }

    try {
      const result = await request<{ available: boolean; connectors: ConnectorRow[] }>('connectors.list', {
        session_id: runtimeId
      })

      if (!current()) {
        return
      }

      const state = $firstBuildConnections.get()[storedId]

      const rows = state.rows.map((row): FirstBuildConnectorRow => {
        const live = result.connectors.find(item => item.connector === row.connector)

        if (!result.available || !live || live.enabled === false) {
          return { ...row, enabled: false, phase: 'error', error: 'unavailable' }
        }

        return { ...row, ...live, phase: live.connected ? 'connected' : 'waiting', error: undefined }
      })

      $firstBuildConnections.setKey(storedId, { ...state, rows })
    } catch {
      if (!current()) {
        return
      }

      const state = $firstBuildConnections.get()[storedId]
      $firstBuildConnections.setKey(storedId, {
        ...state,
        rows: state.rows.map(row => (row.phase === 'connected' ? row : { ...row, phase: 'error', error: 'status' }))
      })
    }

    if (current()) {
      timer = setTimeout(() => void poll(), 2000)
    }
  }

  void poll()

  return () => {
    cancelled = true
    clearTimeout(timer)
  }
}

/** A true submit result means the composer owns delivery through send, steer or queue. */
export function startFirstBuild(storedId: string, submit: (text: string) => boolean): void {
  const state = $firstBuildConnections.get()[storedId]
  const key = `hermes.onboarding.started.v1.${storedId}`

  if (!isFirstBuildSession(storedId) || !state || state.started || readKey(key) === '1') {
    return
  }

  if (submit(buildConnectionStartMessage(state.rows))) {
    writeKey(key, '1')
    endFirstBuildConnect(storedId)
    $firstBuildConnections.setKey(storedId, { ...state, started: true })
  }
}
