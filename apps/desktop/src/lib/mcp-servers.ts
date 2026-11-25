// Shape helpers for the `mcp_servers` config map, shared by everything that
// reads or writes it: the MCP tab editor, the paste-anything importer, and the
// `hermes://mcp/install` deeplink dialog. These agree on what a server entry
// looks like, so they belong in one place — a config written by one path has to
// be readable by the others.

export type McpServers = Record<string, Record<string, unknown>>

export type McpServerEntry = McpServers[string]

export const isServerShape = (value: Record<string, unknown>) =>
  typeof value.command === 'string' || typeof value.url === 'string'

/** Cursor/Claude write `type`; Hermes reads `transport`. Normalizing on the way
 *  in makes pasted configs behave identically under the CLI/TUI loader. */
export function normalizeEntry(entry: Record<string, unknown>): Record<string, unknown> {
  if (typeof entry.type === 'string' && entry.transport === undefined) {
    const { type, ...rest } = entry

    return { ...rest, transport: type }
  }

  return entry
}

const ON_WORDS = new Set(['true', '1', 'yes', 'on'])
const OFF_WORDS = new Set(['false', '0', 'no', 'off'])

/** The backend's `_parse_boolish` minus its default: a bool, a number (0 is false) or a
 *  word (any case, trimmed) parses; `undefined` means unparseable (null, absent, a list,
 *  a mapping, junk) and the caller applies its default, as the backend's `default=` does. */
const parseBoolish = (value: unknown): boolean | undefined => {
  if (typeof value === 'boolean') {
    return value
  }

  if (typeof value === 'number') {
    return value !== 0
  }

  if (typeof value !== 'string') {
    return undefined
  }

  const word = value.trim().toLowerCase()

  return ON_WORDS.has(word) ? true : OFF_WORDS.has(word) ? false : undefined
}

/** Whether a server entry is on. Mirrors the backend's one reader
 *  (`tools/mcp_tool_common.py::mcp_server_enabled`): false/0 and the off words
 *  (any case, trimmed) are off; absent, `null`, `""` and junk are on. A truthy
 *  legacy `disabled` wins over `enabled`.
 *  `mcp-enabled-cases.json` pins both sides to the same table, so the MCP page
 *  never shows a server on that the runtime skips. */
export const serverEnabled = (entry: McpServerEntry) =>
  (parseBoolish(entry.enabled) ?? true) && !(parseBoolish(entry.disabled) ?? false)

/** A value a reader can reach into: an object, not `null`, an array or a scalar. */
const isEntry = (value: unknown): value is Record<string, unknown> =>
  !!value && typeof value === 'object' && !Array.isArray(value)

/** The `mcp_servers` map out of a config record, or `{}` when absent/malformed.
 *
 *  Entries that aren't objects are dropped rather than handed on. Every reader
 *  takes properties straight off the value — `enabled` for the runtime gate,
 *  `command`/`url` for the transport, `tools` for the filter — so one bad entry
 *  throws on the first property read and, with nothing between it and the pane,
 *  takes the whole Capabilities workspace down with it. A `name:` left without
 *  a value in `config.yaml` parses as `null` and does exactly that. The backend
 *  already refuses to *write* a non-object entry (`_replace_mcp_servers`), so
 *  this only has to survive a config edited by hand or by another tool. */
export function getServers(config: { mcp_servers?: unknown } | null): McpServers {
  const raw = config?.mcp_servers

  if (!isEntry(raw)) {
    return {}
  }

  return Object.fromEntries(
    Object.entries(raw).filter((entry): entry is [string, Record<string, unknown>] => isEntry(entry[1]))
  )
}
