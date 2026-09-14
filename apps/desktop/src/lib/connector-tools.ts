import { isRecord } from '@assistant-ui/core/internal'
import type { ToolCallMessagePart } from '@assistant-ui/react'

import type { ChatMessage } from '@/lib/chat-messages'

export function latestConnectorPart(messages: ChatMessage[]) {
  return messages
    .flatMap(message => message.parts)
    .filter(part => {
      if (part.type !== 'tool-call') {
        return false
      }

      if (part.toolName === 'manage_connections') {
        const input = recordOf(part.args)

        return (
          (input.action ?? 'status') !== 'status' || (Array.isArray(input.connectors) && input.connectors.length > 0)
        )
      }

      return connectorCalls(part.toolName, part.args).length > 0
    })
    .at(-1)
}

/** Connector names and statuses from the tool payload, for display only. No field here grants access. */
export interface ConnectorRow {
  connector: string
  connected?: boolean
  enabled?: boolean
  connectionStatus?: string | null
  name?: string
  description?: string
}

export function connectorText(value: ToolCallMessagePart['result']): string | undefined {
  return typeof value === 'string' ? value : undefined
}

export const recordOf = (value: ToolCallMessagePart['result']): ToolCallMessagePart['args'] => {
  const text = connectorText(value)

  if (text !== undefined) {
    try {
      return recordOf(JSON.parse(text))
    } catch {
      return {}
    }
  }

  // SAFETY: tool payloads arrive as JSON-RPC or stored JSON; the object guard excludes arrays and primitives.
  return isRecord(value) ? (value as ToolCallMessagePart['args']) : {}
}

interface ConnectorTitles {
  [slug: string]: string
}

const TITLES: ConnectorTitles = {
  gmail: 'Gmail',
  googlecalendar: 'Google Calendar',
  googledrive: 'Google Drive',
  googledocs: 'Google Docs',
  slack: 'Slack',
  github: 'GitHub',
  notion: 'Notion',
  linear: 'Linear',
  jira: 'Jira',
  todoist: 'Todoist',
  figma: 'Figma',
  discord: 'Discord',
  stripe_mcp: 'Stripe',
  outlook: 'Outlook'
}

export function connectorTitle(slug: string): string {
  return TITLES[slug] ?? slug.replace(/[_-]+/g, ' ').replace(/\b\w/g, letter => letter.toUpperCase())
}

export function connectorToolName(name: string): { connector: string; action: string } | null {
  const match = /^connectors__([a-z0-9_-]+)__(.+)$/i.exec(name)

  return match ? { connector: match[1], action: match[2].replace(/_/g, ' ').toLowerCase() } : null
}

interface ConnectorCall {
  name: string
  arguments: ToolCallMessagePart['result']
}

export function connectorCalls(name: string, args: ToolCallMessagePart['result']): ConnectorCall[] {
  if (connectorToolName(name)) {
    return [{ name, arguments: args }]
  }

  if (name !== 'tool_call') {
    return []
  }

  const source = recordOf(args)
  const calls = Array.isArray(source.calls) ? source.calls : [source]

  return calls.flatMap(item => {
    const call = recordOf(item)

    const callName = connectorText(call.name)

    return callName !== undefined && connectorToolName(callName) ? [{ name: callName, arguments: call.arguments }] : []
  })
}

export function connectionRows(
  args: ToolCallMessagePart['result'],
  result: ToolCallMessagePart['result']
): ConnectorRow[] {
  const input = recordOf(args)
  const output = recordOf(result)
  const rows = new Map<string, ConnectorRow>()

  const add = (item: ToolCallMessagePart['result']) => {
    const slug = connectorText(item)

    if (slug !== undefined) {
      if (/^[a-z0-9_-]+$/i.test(slug)) {
        rows.set(slug, rows.get(slug) ?? { connector: slug })
      }

      return
    }

    const row = recordOf(item)
    const connector = connectorText(row.connector)

    if (connector === undefined || !/^[a-z0-9_-]+$/i.test(connector)) {
      return
    }

    const merged: ConnectorRow = { ...rows.get(connector), connector }

    if (row.connected === true || row.connected === false) {
      merged.connected = row.connected
    }

    if (row.enabled === true || row.enabled === false) {
      merged.enabled = row.enabled
    }

    for (const key of ['connectionStatus', 'name', 'description'] as const) {
      const text = connectorText(row[key])

      if (text !== undefined) {
        merged[key] = text
      }
    }

    rows.set(connector, merged)
  }

  if (Array.isArray(input.connectors)) {
    input.connectors.forEach(add)
  } else if (connectorText(input.connectors) !== undefined) {
    add(input.connectors)
  }

  for (const key of ['connectors', 'results', 'pending']) {
    if (Array.isArray(output[key])) {
      output[key].forEach(add)
    }
  }

  return [...rows.values()]
}

/** The connect URL carries an authorization token, so only https with no embedded credentials is returned. */
export function connectorAuthorizationUrl(value: ToolCallMessagePart['result']): string | null {
  const text = connectorText(value)

  if (text === undefined) {
    return null
  }

  try {
    const url = new URL(text)

    return url.protocol === 'https:' && !url.username && !url.password ? text : null
  } catch {
    return null
  }
}
