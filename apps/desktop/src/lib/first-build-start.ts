import type { ChatMessagePart } from '@/lib/chat-messages'
import { connectorAuthorizationUrl, connectorTitle, recordOf } from '@/lib/connector-tools'
import type { ConnectorFlowRow } from '@/store/connector-flow'

function naturalJoin(names: string[]): string {
  return names.length < 2 ? names.join('') : `${names.slice(0, -1).join(', ')} and ${names.at(-1)}`
}

export function buildConnectionStartMessage(rows: readonly ConnectorFlowRow[]): string {
  const connected = rows.filter(row => row.phase === 'connected').map(row => connectorTitle(row.connector))
  const skipped = rows.filter(row => row.phase !== 'connected').map(row => connectorTitle(row.connector))

  return (
    (connected.length ? `Start with ${naturalJoin(connected)} connected.` : 'Start without connections.') +
    (skipped.length ? ` I skipped ${naturalJoin(skipped)}.` : '')
  )
}

export function canStartWithConnections(part: ChatMessagePart): boolean {
  if (part.type !== 'tool-call' || part.toolName !== 'manage_connections') {
    return false
  }

  const action = recordOf(part.args).action
  const output = recordOf(part.result)

  if (action === 'wait') {
    return part.result === undefined || output.status === 'pending'
  }

  return (
    action === 'connect' &&
    Array.isArray(output.results) &&
    output.results.some(item => {
      const entry = recordOf(item)

      return entry.status === 'initiated' && connectorAuthorizationUrl(entry.connect_url) !== null
    })
  )
}
