import type { ReactNode } from 'react'

import type { MediaKind } from '@/lib/media'

/**
 * TOOL MEDIA RENDERERS — a plugin draws the media ITS OWN tools produced.
 *
 * When a transcript shows a `MEDIA:` attachment, the host finds the tool call
 * whose result first named the path (earlier in the transcript). If the plugin that owns that
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

// The texts a value carries, decoded: a JSON-string result is parsed and its
// string leaves read, so a path next to an escaped `\n`, or a non-ASCII path
// written as `\u00e9`, reads as the path itself. Bounded (depth, leaf count).
function textLeaves(value: unknown, out: string[] = [], depth = 0): string[] {
  if (out.length >= MAX_LEAVES || depth > 6) {
    return out
  }

  if (typeof value === 'string') {
    out.push(value)
    const head = value.trimStart()[0]

    if ((head === '{' || head === '[' || head === '"') && value.length <= MAX_PARSE_CHARS) {
      try {
        const parsed: unknown = JSON.parse(value)

        if (parsed !== value) {
          textLeaves(parsed, out, depth + 1)
        }
      } catch {
        // Plain text that happens to start with a bracket.
      }
    }
  } else if (value && typeof value === 'object') {
    for (const item of Array.isArray(value) ? value : Object.values(value)) {
      textLeaves(item, out, depth + 1)
    }
  }

  return out
}

const MAX_LEAVES = 2_000
const MAX_PARSE_CHARS = 2_000_000

const WINDOWS_PATH = /^(?:[a-z]:[\\/]|\\\\|\/\/)/i

/** One spelling for every form a path takes: `file://` URLs unwrapped,
 *  backslashes as `/`, and Windows paths (case-insensitive) lower-cased. */
function canonicalText(text: string, windows: boolean): string {
  const unwrapped = text.replace(/file:\/\/(?:\/(?=[a-z]:))?/gi, '')
  const slashed = windows ? unwrapped.replace(/\\/g, '/') : unwrapped

  return windows ? slashed.toLowerCase() : slashed
}

interface PathNeedle {
  needles: string[]
  windows: boolean
}

function pathNeedle(path: string): PathNeedle {
  const bare = canonicalText(path, false)
  const windows = WINDOWS_PATH.test(bare)
  const decoded = decodeURIComponentSafe(bare)

  return { needles: [...new Set([bare, decoded].map(p => canonicalText(p, windows)))].filter(Boolean), windows }
}

function decodeURIComponentSafe(value: string): string {
  try {
    return decodeURIComponent(value)
  } catch {
    return value
  }
}

const escapeRegExp = (value: string) => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

// A path continues through these; anything else (whitespace, quotes, `:`,
// brackets, a sentence's final `.`) ends it.
const PATH_CHAR = String.raw`[\p{L}\p{N}_\-~%+@/\\]`

/** Whether `text` names `needle` as a WHOLE path: `/out/a.glb` matches in
 *  `{"path":"/out/a.glb"}`, `saved to /out/a.glb.`, `/out/a.glb: 3 MB` and
 *  `MEDIA:/out/a.glb`, but never inside `/out/a.glb.bak` or `/x/out/a.glb`. */
export function namesWholePath(text: string, needle: string): boolean {
  if (!needle || !text.includes(needle)) {
    return false
  }

  return new RegExp(`(?<!${PATH_CHAR}|\\.)${escapeRegExp(needle)}(?!${PATH_CHAR}|\\.${PATH_CHAR})`, 'u').test(text)
}

function toolCallsOf(message: MessageLike): ToolCallLike[] {
  return Array.isArray(message.content)
    ? (message.content as ToolCallLike[]).filter(
        part => part?.type === 'tool-call' && typeof part.toolName === 'string'
      )
    : []
}

/** A call that failed produced nothing. A reloaded transcript marks every
 *  stored result as a success, so the payload's own verdict counts too. */
function callFailed(call: ToolCallLike): boolean {
  if (call.isError) {
    return true
  }

  let payload = call.result

  if (typeof payload === 'string' && payload.trimStart().startsWith('{') && payload.length <= MAX_PARSE_CHARS) {
    try {
      payload = JSON.parse(payload)
    } catch {
      return false
    }
  }

  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
    return false
  }

  const { error, success } = payload as { error?: unknown; success?: unknown }

  return success === false || (error !== undefined && error !== null && error !== false && error !== '')
}

// Decoded, canonical texts per part and spelling. Parts and messages are the
// thread store's own objects, stable across streaming deltas, so a long
// transcript is decoded once instead of on every update.
const leavesCache = new WeakMap<object, Map<boolean, string[]>>()

function canonicalLeaves(owner: object, value: unknown, windows: boolean): string[] {
  let byForm = leavesCache.get(owner)

  if (!byForm) {
    byForm = new Map()
    leavesCache.set(owner, byForm)
  }

  let leaves = byForm.get(windows)

  if (!leaves) {
    leaves = textLeaves(value).map(text => canonicalText(text, windows))
    byForm.set(windows, leaves)
  }

  return leaves
}

type Mention = 'user' | ToolCallLike | null

const mentionCache = new WeakMap<object, Map<string, Mention>>()

/** The first thing in `message` that names the path: the user's own text
 *  (an upload, a file they pointed at) or a successful tool call's result. */
function firstMention(message: MessageLike, target: PathNeedle): Mention {
  const key = `${target.windows ? 'w' : 'p'}\u0000${target.needles.join('\u0000')}`
  let cached = mentionCache.get(message)

  if (cached?.has(key)) {
    return cached.get(key) ?? null
  }

  const names = (leaves: string[]) => leaves.some(text => target.needles.some(needle => namesWholePath(text, needle)))
  let found: Mention = null

  if (message.role === 'user') {
    found = names(canonicalLeaves(message, message.content, target.windows)) ? 'user' : null
  } else {
    found =
      toolCallsOf(message).find(
        call =>
          call.result !== undefined &&
          !callFailed(call) &&
          names(canonicalLeaves(call as object, call.result, target.windows))
      ) ?? null
  }

  if (!cached) {
    cached = new Map()
    mentionCache.set(message, cached)
  }

  cached.set(key, found)

  return found
}

/**
 * The tool call whose result produced `path`, read from the transcript up to
 * message `messageId`. The EARLIEST mention wins, because a file is produced
 * once and later calls (`ls`, a read, a plugin taking it as input) only name
 * it. A path the user named first (an upload) has no producer, and failed
 * calls produced nothing. Rewriting a file in place later never moves it to
 * another tool: ownership stays with the call that first produced it.
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

  const target = pathNeedle(path)

  for (let at = 0; at <= index; at += 1) {
    const message = messages[at]
    const mention = message ? firstMention(message, target) : null

    if (mention) {
      return mention === 'user' ? null : mention
    }
  }

  return null
}

function bridgeEntries(args: Record<string, unknown>): unknown[] | null {
  let calls: unknown = args.calls

  if (calls === undefined || calls === null) {
    calls = typeof args.name === 'string' && args.name.trim() ? [{ arguments: args.arguments, name: args.name }] : null
  }

  if (typeof calls === 'string') {
    try {
      calls = JSON.parse(calls)
    } catch {
      return null
    }
  }

  if (calls && typeof calls === 'object' && !Array.isArray(calls)) {
    calls = [calls]
  }

  return Array.isArray(calls) ? calls : null
}

function bridgeArguments(raw: unknown): Record<string, unknown> {
  if (typeof raw === 'string') {
    try {
      raw = raw.trim() ? JSON.parse(raw) : {}
    } catch {
      return {}
    }
  }

  return raw && typeof raw === 'object' && !Array.isArray(raw) ? (raw as Record<string, unknown>) : {}
}

/** The tool a call really ran. The deferred-tool bridge (`tool_call`) runs one
 *  local tool per call (batches are split before they run), with the tool's
 *  name and arguments in `args.calls[0]` (or the legacy `{name, arguments}`
 *  shape, a JSON-string `calls`, a single object) and its own result as the
 *  result, so a plugin tool loaded on demand is still recognised. Mirrors the
 *  backend's `normalize_tool_call_entries`. A multi-entry batch (connectors
 *  only) stays the bridge: no plugin owns it. */
export function effectiveToolCall(part: ToolCallLike): ToolCallLike {
  const entries =
    part.toolName === 'tool_call' && part.args && typeof part.args === 'object'
      ? bridgeEntries(part.args as Record<string, unknown>)
      : null

  if (!entries || entries.length !== 1) {
    return part
  }

  const entry = entries[0] as { arguments?: unknown; name?: unknown } | null
  const name = typeof entry?.name === 'string' ? entry.name.trim() : ''

  return name && name !== 'tool_call' ? { ...part, args: bridgeArguments(entry?.arguments), toolName: name } : part
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
