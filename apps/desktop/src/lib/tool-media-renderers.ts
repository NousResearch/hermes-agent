import type { ReactNode } from 'react'

import type { MediaKind } from '@/lib/media'

/**
 * TOOL MEDIA RENDERERS — a plugin draws the media ITS OWN tools produced.
 *
 * When a transcript shows a `MEDIA:` attachment, the host finds the tool call
 * in that turn whose result produced the path. If the plugin that owns that
 * tool registered a renderer for the tool's name, its component replaces the
 * built-in player; every other file (core voice replies, other plugins' and
 * core tools' output, user uploads, paths no tool in the turn produced) keeps
 * the built-in player.
 *
 * Ownership is provenance, never file type or name: the backend reports which
 * tools each installed package's own code registered (`plugins.manage list` →
 * `tools`), and a claim on any other tool is refused.
 */

export const TOOL_MEDIA_RENDERER_AREA = 'transcript.toolMedia'

/** The file the tool produced, as the transcript names it. */
export interface ToolMediaFile {
  /** Path on the BACKEND's machine, exactly as the `MEDIA:` line wrote it.
   *  Never open it directly (the backend may be remote): use `resolveUrl()`,
   *  or hand the path to your own backend route. */
  path: string
  name: string
  kind: MediaKind
  mime: string
  /** A URL this window can load, local or remote backend alike: a range
   *  streaming route for audio/video, a data URL for anything else (size-capped
   *  by the host's file reader). */
  resolveUrl: () => Promise<string>
}

/** Props handed to a tool media renderer's `render`. */
export interface ToolMediaRendererProps {
  file: ToolMediaFile
  /** The producing tool call: registry name, call id, arguments and result. */
  toolName: string
  toolCallId: string
  args: Readonly<Record<string, unknown>>
  result: unknown
  /** The chat whose transcript renders this (stored id, else runtime id), or null. */
  sessionId: null | string
  /** The built-in player for this file: return it for anything you don't draw. */
  fallback: ReactNode
}

/** Payload of a `transcript.toolMedia` contribution's `data`. */
export interface ToolMediaRendererContribution {
  /** Tool names this renderer draws. A trailing `*` matches by prefix
   *  (`mcp__meshy__*` for an MCP server's tools); either way only tools this
   *  plugin's package registered are served, and other claims are refused with
   *  a console warning. */
  tools: readonly string[]
  /** Mounted inside an error boundary whose fallback is the built-in player. */
  render: (props: ToolMediaRendererProps) => ReactNode
}

/** The exact tool names one package's own code registered (native tools and
 *  its MCP servers' live tools), as the backend reports them. */
export interface ToolOwnership {
  tools: ReadonlySet<string>
}

/** The tool-call fields a producer lookup reads (an assistant-ui part). */
export interface ToolCallLike {
  args?: unknown
  isError?: boolean
  result?: unknown
  toolCallId?: string
  toolName?: string
  type?: string
}

interface MessageLike {
  content?: readonly unknown[] | string
  id?: string
  role?: string
}

function payloadText(value: unknown): string {
  if (typeof value === 'string') {
    return value
  }

  try {
    return JSON.stringify(value) ?? ''
  } catch {
    return ''
  }
}

function pathNeedles(path: string): string[] {
  const bare = path.startsWith('file:') ? decodeURIComponentSafe(path.replace(/^file:\/\//, '')) : path

  // JSON-escaped forms too: a Windows path sits in a JSON result as `C:\\Users\\…`.
  return [...new Set([path, bare, JSON.stringify(path).slice(1, -1), JSON.stringify(bare).slice(1, -1)])].filter(
    Boolean
  )
}

function decodeURIComponentSafe(value: string): string {
  try {
    return decodeURIComponent(value)
  } catch {
    return value
  }
}

const escapeRegExp = (value: string) => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

/** Whether `text` names `needle` as a WHOLE path: `/out/a.glb` matches in
 *  `{"path":"/out/a.glb"}`, `saved to /out/a.glb.` and `MEDIA:/out/a.glb`, but
 *  never inside `/out/a.glb.bak` or `/x/out/a.glb`. */
export function namesWholePath(text: string, needle: string): boolean {
  return new RegExp(`(?<=^|[\\s"'\`(=\\[{,:])${escapeRegExp(needle)}(?=$|[\\s"'\`),;\\]}>]|\\.(?:$|\\s))`).test(text)
}

function toolCallsOf(message: MessageLike): ToolCallLike[] {
  return Array.isArray(message.content)
    ? (message.content as ToolCallLike[]).filter(
        part => part?.type === 'tool-call' && typeof part.toolName === 'string'
      )
    : []
}

/**
 * The tool call whose result produced `path`, from the turn that holds message
 * `messageId`: that message and the assistant messages back to the user's
 * prompt. The EARLIEST matching call wins, because a file is produced once and
 * later calls (`ls`, a read) only mention it. Failed calls produced nothing.
 */
export function findProducingToolCall(
  messages: readonly MessageLike[],
  messageId: null | string | undefined,
  path: string
): ToolCallLike | null {
  const index = messageId ? messages.findIndex(message => message.id === messageId) : -1

  if (index < 0) {
    return null
  }

  let start = index

  while (start > 0 && messages[start - 1]?.role !== 'user') {
    start -= 1
  }

  const needles = pathNeedles(path)

  for (const message of messages.slice(start, index + 1)) {
    for (const call of toolCallsOf(message)) {
      if (call.isError || call.result === undefined) {
        continue
      }

      const text = payloadText(call.result)

      if (needles.some(needle => namesWholePath(text, needle))) {
        return call
      }
    }
  }

  return null
}

/** The tool a call really ran. The deferred-tool bridge (`tool_call`) runs one
 *  local tool per call (batches are split before they run), with the tool's
 *  name and arguments in `args.calls[0]` and its own result as the result, so
 *  a plugin tool loaded on demand is still recognised. A multi-entry batch
 *  (connectors only) stays the bridge: no plugin owns it. */
export function effectiveToolCall(part: ToolCallLike): ToolCallLike {
  const calls =
    part.toolName === 'tool_call' && part.args && typeof part.args === 'object'
      ? (part.args as { calls?: unknown }).calls
      : undefined

  if (!Array.isArray(calls) || calls.length !== 1) {
    return part
  }

  const entry = calls[0] as { arguments?: unknown; name?: unknown } | null

  return typeof entry?.name === 'string' && entry.name
    ? { ...part, args: entry.arguments ?? {}, toolName: entry.name }
    : part
}

/** The package that owns `toolName`: the one package whose registered tools
 *  include it (the backend already drops a name two packages list). */
export function ownerOfTool(toolName: string, owners: ReadonlyMap<string, ToolOwnership>): null | string {
  for (const [pkg, owned] of owners) {
    if (owned.tools.has(toolName)) {
      return pkg
    }
  }

  return null
}

export function claimPatternMatches(pattern: string, toolName: string): boolean {
  return pattern.endsWith('*') ? pattern.length > 1 && toolName.startsWith(pattern.slice(0, -1)) : toolName === pattern
}

/** Whether `pattern` names a tool `owned` registered (a wildcard must match at
 *  least one). Only what the claim's own package registered is ever served. */
export function claimPatternOwned(pattern: string, owned: ToolOwnership | undefined): boolean {
  return owned ? [...owned.tools].some(tool => claimPatternMatches(pattern, tool)) : false
}

export interface ToolMediaClaim {
  id: string
  source?: string
  data?: unknown
}

export interface ResolveToolMediaOptions {
  owners: ReadonlyMap<string, ToolOwnership>
  /** The package a contribution's plugin belongs to (null: none, e.g. a standalone desktop plugin). */
  packageOf: (source: string | undefined) => null | string
  onRefused?: (claim: ToolMediaClaim, toolName: string) => void
}

/** The first claim on `toolName` whose plugin's package owns that tool. A
 *  matching claim from any other plugin is refused (and reported). */
export function resolveToolMediaClaim<C extends ToolMediaClaim>(
  claims: readonly C[],
  toolName: string,
  { onRefused, owners, packageOf }: ResolveToolMediaOptions
): C | null {
  const owner = ownerOfTool(toolName, owners)

  for (const claim of claims) {
    const data = claim.data as Partial<ToolMediaRendererContribution> | undefined

    if (typeof data?.render !== 'function' || !Array.isArray(data.tools)) {
      continue
    }

    if (!data.tools.some(pattern => typeof pattern === 'string' && claimPatternMatches(pattern, toolName))) {
      continue
    }

    const pkg = packageOf(claim.source)

    if (owner !== null && pkg === owner) {
      return claim
    }

    onRefused?.(claim, toolName)
  }

  return null
}

/** Every pattern in `claims` its plugin's package does not own — what the
 *  host warns about once the ownership list is known. */
export function refusedClaimPatterns<C extends ToolMediaClaim>(
  claims: readonly C[],
  owners: ReadonlyMap<string, ToolOwnership>,
  packageOf: (source: string | undefined) => null | string
): { claim: C; pattern: string }[] {
  return claims.flatMap(claim => {
    const tools = (claim.data as Partial<ToolMediaRendererContribution> | undefined)?.tools
    const pkg = packageOf(claim.source)
    const owned = pkg ? owners.get(pkg) : undefined

    return Array.isArray(tools)
      ? tools
          .filter(pattern => !claimPatternOwned(String(pattern), owned))
          .map(pattern => ({ claim, pattern: String(pattern) }))
      : []
  })
}
