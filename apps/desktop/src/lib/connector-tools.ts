/** Connector names/results as presentation data, never authorization. */
export interface ConnectorRow {
  connector: string
  connected?: boolean
  enabled?: boolean
  connectionStatus?: string | null
  name?: string
  description?: string
}

export const recordOf = (value: unknown): Record<string, unknown> => {
  if (typeof value === 'string') {
    try {
      return recordOf(JSON.parse(value))
    } catch {
      return {}
    }
  }

  return value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : {}
}

const TITLES: Record<string, string> = {
  gmail: 'Gmail',
  googlecalendar: 'Google Calendar',
  googledrive: 'Google Drive',
  slack: 'Slack',
  github: 'GitHub',
  notion: 'Notion',
  linear: 'Linear',
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

export function connectorCalls(name: string, args: unknown): { name: string; arguments: unknown }[] {
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

    return typeof call.name === 'string' && connectorToolName(call.name)
      ? [{ name: call.name, arguments: call.arguments }]
      : []
  })
}

export function connectionRows(args: unknown, result: unknown): ConnectorRow[] {
  const input = recordOf(args)
  const output = recordOf(result)
  const rows = new Map<string, ConnectorRow>()

  const add = (item: unknown) => {
    if (typeof item === 'string') {
      if (/^[a-z0-9_-]+$/i.test(item)) {
        rows.set(item, rows.get(item) ?? { connector: item })
      }

      return
    }

    const row = recordOf(item)

    if (typeof row.connector !== 'string' || !/^[a-z0-9_-]+$/i.test(row.connector)) {
      return
    }

    rows.set(row.connector, {
      ...rows.get(row.connector),
      connector: row.connector,
      ...(typeof row.connected === 'boolean' ? { connected: row.connected } : {}),
      ...(typeof row.enabled === 'boolean' ? { enabled: row.enabled } : {}),
      ...(typeof row.connectionStatus === 'string' ? { connectionStatus: row.connectionStatus } : {}),
      ...(typeof row.name === 'string' ? { name: row.name } : {}),
      ...(typeof row.description === 'string' ? { description: row.description } : {})
    })
  }

  if (Array.isArray(input.connectors)) {
    input.connectors.forEach(add)
  } else if (typeof input.connectors === 'string') {
    add(input.connectors)
  }

  for (const key of ['connectors', 'results', 'pending']) {
    if (Array.isArray(output[key])) {
      output[key].forEach(add)
    }
  }

  return [...rows.values()]
}

/** Token-bearing auth links are opened only by a deliberate user action. */
export function connectorAuthorizationUrl(value: unknown): string | null {
  if (typeof value !== 'string') {
    return null
  }

  try {
    const url = new URL(value)

    return url.protocol === 'https:' && !url.username && !url.password ? value : null
  } catch {
    return null
  }
}
