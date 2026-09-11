import type { ToolCallMessagePart } from '@assistant-ui/react'
import { map } from 'nanostores'

import { isFirstBuildSession } from '@/app/contrib/handoff-receipt'
import { connectionRows, connectorAuthorizationUrl, recordOf } from '@/lib/connector-tools'
import { readKey, writeKey } from '@/lib/storage'
import type { ConnectorFlowRow } from '@/store/connector-flow'

export type FirstBuildConnectorPart = Pick<ToolCallMessagePart, 'toolCallId' | 'toolName' | 'args' | 'result'>

export interface FirstBuildConnectorRow extends ConnectorFlowRow {
  connectUrl?: string
}

export interface FirstBuildConnectorState {
  toolCallId: string
  rows: FirstBuildConnectorRow[]
}

export const $firstBuildConnections = map<Record<string, FirstBuildConnectorState>>({})

interface OpenLinksDeps {
  open?: (url: string) => Promise<void>
  submit: (text: string) => void
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
  $firstBuildConnections.setKey(storedId, { toolCallId: part.toolCallId, rows })

  const links = entries.flatMap(entry => {
    const url = entry.status === 'initiated' ? connectorAuthorizationUrl(entry.connect_url) : null

    return url && typeof entry.connector === 'string' ? [{ connector: entry.connector, url }] : []
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

  if (opened.length) {
    deps.submit(`[setup] links opened for ${opened.map(link => link.connector).join(', ')}`)
  }
}
