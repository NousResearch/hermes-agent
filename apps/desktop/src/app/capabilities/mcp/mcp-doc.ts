// oxlint-disable-next-line anti-slop/no-shape-in-symbol-names -- `isServerShape` is main's exported name, shared with mcp-tab.tsx; it is bound to a domain name here instead of renaming the export.
import { type McpServerEntry, type McpServers, isServerShape as namesAServer, normalizeEntry } from '@/lib/mcp-servers'

export const STARTER_ENTRY = { command: 'npx', args: ['-y', '@modelcontextprotocol/server-filesystem', '/path/to/dir'] }

export const wrapDoc = (entries: McpServers) => JSON.stringify({ mcpServers: entries }, null, 2)

export function parseServersDoc(raw: string): McpServers {
  const parsed: unknown = JSON.parse(raw)

  if (!(parsed instanceof Object) || Array.isArray(parsed)) {
    throw new Error('Expected a JSON object')
  }

  // SAFETY: an object and not an array, checked on the line above; every property read below is optional.
  const doc = parsed as McpServerEntry

  if (namesAServer(doc)) {
    throw new Error('Wrap the server in {"mcpServers": {"name": …}} so it has a name')
  }

  const wrapper = doc.mcpServers ?? doc.mcp_servers

  // SAFETY: an object and not an array; `normalizeEntry` runs over every value below, so a non-object entry cannot reach a reader.
  const map = (wrapper instanceof Object && !Array.isArray(wrapper) ? wrapper : doc) as McpServers

  return Object.fromEntries(Object.entries(map).map(([name, entry]) => [name, normalizeEntry(entry)]))
}

export const serverEnabled = (server: McpServerEntry) => server.enabled !== false

export function withEnabled(server: McpServerEntry, enabled: boolean): McpServerEntry {
  const next = { ...server }

  if (enabled) {
    delete next.enabled
  } else {
    next.enabled = false
  }

  return next
}

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

function skipString(text: string, index: number): number {
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
function containerStart(text: string): number {
  const wrapper = /"mcpServers"|"mcp_servers"/.exec(text)

  if (!wrapper) {
    return text.indexOf('{')
  }

  let i = wrapper.index + wrapper[0].length

  while (i < text.length && text[i] !== '{') {
    i++
  }

  return i
}

/** The index just past the `{…}` that starts at `index`, balancing braces and skipping strings. */
function objectEnd(text: string, index: number): number {
  let depth = 0
  let i = index

  while (i < text.length) {
    const c = text[i]

    if (c === '"') {
      i = skipString(text, i)

      continue
    }

    if (c === '{') {
      depth++
    } else if (c === '}') {
      depth--

      if (depth === 0) {
        return i + 1
      }
    }

    i++
  }

  return i
}

/** The index of the next sibling key after a non-object value. */
function siblingStart(text: string, index: number): number {
  let i = index

  while (i < text.length && text[i] !== ',' && text[i] !== '}') {
    if (text[i] === '"') {
      i = skipString(text, i)

      continue
    }

    i++
  }

  return i
}

/** The index of the value after the `:` that follows the key ending at `index`. */
function valueStart(text: string, index: number): number {
  let i = index

  while (i < text.length && text[i] !== ':') {
    i++
  }

  i++

  while (i < text.length && /\s/.test(text[i])) {
    i++
  }

  return i
}

export function scanServerBlocks(text: string): ServerBlock[] {
  const start = containerStart(text)

  if (start < 0 || text[start] !== '{') {
    return []
  }

  const blocks: ServerBlock[] = []
  let i = start + 1

  while (i < text.length && text[i] !== '}') {
    if (text[i] !== '"') {
      i++

      continue
    }

    const keyStart = i
    const keyEnd = skipString(text, i)
    const name = text.slice(keyStart + 1, keyEnd - 1)

    i = valueStart(text, keyEnd)

    if (text[i] === '{') {
      const to = objectEnd(text, i)

      blocks.push({ from: keyStart, name, to })
      i = to
    } else {
      i = siblingStart(text, i)
    }
  }

  return blocks
}
