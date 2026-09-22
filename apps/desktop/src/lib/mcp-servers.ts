// Shape helpers for the `mcp_servers` config map, shared by everything that
// reads or writes it: the MCP tab editor, the paste-anything importer, and the
// `hermes://mcp/install` deeplink dialog. These agree on what a server entry
// looks like, so they belong in one place — a config written by one path has to
// be readable by the others.

export type McpServers = Record<string, Record<string, unknown>>

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
