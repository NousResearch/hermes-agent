// The `mcp.json` document: how it is written, parsed, and mapped back to the
// text the editor shows.
//
// Extracted from `mcp-tab.tsx` so the Connectors page can open the SAME editor
// for a server on this Mac without a second copy of the format. The editor
// always speaks the ecosystem's mcp.json document format — names are the JSON
// keys, transport is inferred from `command` vs `url` — so any README's "add
// this to your mcp.json" snippet pastes verbatim. Storage stays the config.yaml
// `mcp_servers` map (CLI/TUI untouched).

import { isServerShape, type McpServers, normalizeEntry } from '@/lib/mcp-servers'

export const STARTER_ENTRY = { command: 'npx', args: ['-y', '@modelcontextprotocol/server-filesystem', '/path/to/dir'] }

export const pretty = (value: unknown) => JSON.stringify(value, null, 2)

export const wrapDoc = (entries: McpServers) => pretty({ mcpServers: entries })

/** Accepts `{"mcpServers": {...}}` (ecosystem), a bare name→config map, or throws. */
export function parseServersDoc(raw: string): McpServers {
  const parsed = JSON.parse(raw) as unknown

  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error('Expected a JSON object')
  }

  const doc = parsed as Record<string, unknown>

  if (isServerShape(doc)) {
    throw new Error('Wrap the server in {"mcpServers": {"name": …}} so it has a name')
  }

  const wrapper = doc.mcpServers ?? doc.mcp_servers

  const map =
    wrapper && typeof wrapper === 'object' && !Array.isArray(wrapper) ? (wrapper as McpServers) : (doc as McpServers)

  return Object.fromEntries(Object.entries(map).map(([name, entry]) => [name, normalizeEntry(entry)]))
}

// The runtime gate is `enabled: false` — the same flag `hermes mcp` and the
// agent's MCP loader read.
export const serverEnabled = (server: Record<string, unknown>) => server.enabled !== false

/** `enabled: false` written or removed. Absent means on, so on deletes the key. */
export function withEnabled(server: Record<string, unknown>, enabled: boolean): Record<string, unknown> {
  const next = { ...server }

  if (enabled) {
    delete next.enabled
  } else {
    next.enabled = false
  }

  return next
}

/** A key nothing in `taken` already uses: `name`, `name-2`, `name-3`, … */
export function uniqueServerKey(taken: McpServers, name: string): string {
  let key = name

  for (let i = 2; key in taken; i++) {
    key = `${name}-${i}`
  }

  return key
}

// ---------------------------------------------------------------------------
// Cursor → server-block mapping. A tolerant character walker (not JSON.parse —
// it must work mid-edit) that finds each server's key+object range inside the
// mcpServers container, so the editor cursor selects a server and the block
// can be highlighted.
// ---------------------------------------------------------------------------

export interface ServerBlock {
  from: number
  name: string
  to: number
}

export function scanServerBlocks(text: string): ServerBlock[] {
  const skipString = (index: number): number => {
    let i = index + 1

    while (i < text.length) {
      if (text[i] === '\\') {
        i += 2
      } else if (text[i] === '"') {
        return i + 1
      } else {
        i++
      }
    }

    return i
  }

  // Container: the object after "mcpServers"/"mcp_servers", else the doc root.
  let start = -1
  const wrapper = /"mcpServers"|"mcp_servers"/.exec(text)

  if (wrapper) {
    let i = wrapper.index + wrapper[0].length

    while (i < text.length && text[i] !== '{') {
      i++
    }

    start = i
  } else {
    start = text.indexOf('{')
  }

  if (start < 0 || text[start] !== '{') {
    return []
  }

  const blocks: ServerBlock[] = []
  let i = start + 1

  while (i < text.length) {
    const ch = text[i]

    if (ch === '}') {
      break
    }

    if (ch !== '"') {
      i++

      continue
    }

    const keyStart = i
    const keyEnd = skipString(i)
    const name = text.slice(keyStart + 1, keyEnd - 1)
    i = keyEnd

    while (i < text.length && text[i] !== ':') {
      i++
    }

    i++

    while (i < text.length && /\s/.test(text[i])) {
      i++
    }

    if (text[i] === '{') {
      let depth = 0
      let j = i

      while (j < text.length) {
        const c = text[j]

        if (c === '"') {
          j = skipString(j)

          continue
        }

        if (c === '{') {
          depth++
        } else if (c === '}') {
          depth--

          if (depth === 0) {
            j++

            break
          }
        }

        j++
      }

      blocks.push({ from: keyStart, name, to: j })
      i = j
    } else {
      // Non-object value — skip to the next sibling.
      while (i < text.length && text[i] !== ',' && text[i] !== '}') {
        if (text[i] === '"') {
          i = skipString(i)

          continue
        }

        i++
      }
    }
  }

  return blocks
}
