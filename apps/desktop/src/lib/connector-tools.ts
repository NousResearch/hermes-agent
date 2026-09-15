import { isRecord } from '@assistant-ui/core/internal'
import type { ToolCallMessagePart } from '@assistant-ui/react'

export interface McpTarget {
  name: string
  action: 'authorize' | 'enable' | 'install'
}

const MCP_ACTIONS: readonly McpTarget['action'][] = ['install', 'enable', 'authorize']

/** Reads args so live and settled rows classify alike. */
export function mcpTargets(toolName: string, args: ToolCallMessagePart['result']): McpTarget[] {
  if (toolName !== 'manage_connections') {
    return []
  }

  const input = recordOf(args)
  const action = MCP_ACTIONS.find(a => a === input.action) ?? 'install'

  if (!Array.isArray(input.connectors)) {
    return []
  }

  return input.connectors.flatMap(entry => {
    const name = isRecord(entry) && entry.mcp === true ? connectorText(entry.name)?.trim() : undefined

    return name ? [{ action, name: name.toLowerCase() }] : []
  })
}

/** The gateway's six-state account status; `pending` covers the vendor's INITIALIZING and INITIATED. */
export type ConnectionStatus = 'active' | 'expired' | 'failed' | 'inactive' | 'pending' | 'revoked'

/** One `GET /v1/connectors` item as the gateway sends it. Display-only; these fields never grant access. */
export interface ConnectorRow {
  connected: boolean
  connectionStatus?: ConnectionStatus
  connector: string
  disabledTools?: string[]
  enabled: boolean
  statusReason?: string
}

/** The vendor's public logo for a toolkit, keyed by its slug (the gateway slug is the vendor slug; checked
 *  for every lead-order pick). Served as an SVG with no CORS header, so it is only ever an `<img src>`. */
export const connectorIconUrl = (slug: string): string => `https://logos.composio.dev/api/${slug}`


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

  // SAFETY: isRecord excludes arrays and primitives from the JSON payload.
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

/** Authorization URLs may carry tokens; reject non-HTTPS or embedded credentials. */
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
