// Paste-anything MCP import: turns whatever a README tells the user to copy —
// an mcp.json snippet, a bare `npx`/`docker` command, a `claude mcp add` line,
// a plain server URL, or a Cursor install deeplink — into named mcp.json
// entries. Pure and side-effect free; the MCP tab merges the result into the
// editor draft. Returns null when nothing recognizable is in the text.

import { isServerShape, normalizeEntry } from '@/lib/mcp-servers'

export interface McpImportEntry {
  config: Record<string, unknown>
  name: string
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  !!value && typeof value === 'object' && !Array.isArray(value)

const URL_RE = /^https?:\/\/\S+$/i

// "mcp.linear.app" → "linear": drop generic front labels, keep the brand one.
function nameFromUrl(raw: string): string {
  try {
    const labels = new URL(raw).hostname.split('.').filter(Boolean)

    while (labels.length > 1 && ['api', 'mcp', 'www'].includes(labels[0])) {
      labels.shift()
    }

    return labels[0] || 'server'
  } catch {
    return 'server'
  }
}

// "@modelcontextprotocol/server-filesystem@latest" → "filesystem".
function cleanupName(base: string): string {
  const stripped = base
    .replace(/\.(cjs|js|mjs|py|ts)$/i, '')
    .replace(/^(mcp-server-|server-|mcp-)/, '')
    .replace(/(-mcp-server|-mcp|-server)$/, '')

  return stripped || base
}

function packageBasename(spec: string): string {
  // uvx-style version pins ("pkg==1.0"), then npm-style ("pkg@latest",
  // "@scope/pkg@1.2.3" — the scope's leading @ is at index 0 and survives).
  let bare = spec.split('==')[0]
  const at = bare.lastIndexOf('@')

  if (at > 0) {
    bare = bare.slice(0, at)
  }

  return cleanupName(bare.split('/').pop() ?? bare)
}

// Shell-ish tokenizer: whitespace-split with single/double quotes (and
// backslash escapes inside double quotes). Null on an unterminated quote.
function tokenize(line: string): null | string[] {
  const tokens: string[] = []
  let current = ''
  let sawQuote = false
  let quote: '"' | "'" | null = null

  for (let i = 0; i < line.length; i++) {
    const ch = line[i]

    if (quote) {
      if (ch === quote) {
        quote = null
      } else if (ch === '\\' && quote === '"' && i + 1 < line.length) {
        current += line[++i]
      } else {
        current += ch
      }
    } else if (ch === '"' || ch === "'") {
      quote = ch
      sawQuote = true
    } else if (/\s/.test(ch)) {
      if (current || sawQuote) {
        tokens.push(current)
        current = ''
        sawQuote = false
      }
    } else {
      current += ch
    }
  }

  if (quote) {
    return null
  }

  if (current || sawQuote) {
    tokens.push(current)
  }

  return tokens
}

const STDIO_COMMANDS = new Set(['bunx', 'docker', 'node', 'npx', 'uvx'])

// docker flags that consume the following token (so the image isn't mistaken
// for a flag value or vice versa).
const DOCKER_VALUE_FLAGS = new Set([
  '--entrypoint',
  '--env',
  '--label',
  '--mount',
  '--name',
  '--network',
  '--platform',
  '--publish',
  '--pull',
  '--user',
  '--volume',
  '--workdir',
  '-e',
  '-l',
  '-p',
  '-u',
  '-v',
  '-w'
])

function inferCommandName(command: string, args: string[]): string {
  if (command === 'docker') {
    let i = args[0] === 'run' ? 1 : 0

    for (; i < args.length; i++) {
      const arg = args[i]

      if (arg.startsWith('-')) {
        if (DOCKER_VALUE_FLAGS.has(arg)) {
          i++
        }

        continue
      }

      // Image ref: strip registry/repo path and the :tag.
      return cleanupName((arg.split('/').pop() ?? arg).split(':')[0])
    }

    return 'docker'
  }

  if (command === 'node') {
    const script = args.find(arg => !arg.startsWith('-'))

    return script ? cleanupName(script.split(/[/\\]/).pop() ?? script) : 'node'
  }

  // npx / bunx / uvx: the first positional is the package spec.
  for (let i = 0; i < args.length; i++) {
    const arg = args[i]

    if (arg.startsWith('-')) {
      if (arg === '--from' || arg === '--package' || arg === '-p') {
        i++
      }

      continue
    }

    return packageBasename(arg)
  }

  return command
}

function fromCommandLine(tokens: string[]): McpImportEntry | null {
  const command = tokens[0]

  if (!STDIO_COMMANDS.has(command)) {
    return null
  }

  return {
    config: { args: tokens.slice(1), command },
    name: inferCommandName(command, tokens.slice(1))
  }
}

// `claude mcp add NAME [--transport http|sse] [-e K=V]... [--] CMD ARGS...`
// and `claude mcp add NAME URL`. Everything after the first positional
// following NAME (or after `--`) is the command verbatim.
function fromClaudeAdd(tokens: string[]): McpImportEntry | null {
  let name: null | string = null
  let transport: null | string = null
  let rest: null | string[] = null
  const env: Record<string, string> = {}
  const headers: Record<string, string> = {}

  for (let i = 3; i < tokens.length; i++) {
    const token = tokens[i]

    if (token === '--') {
      rest = tokens.slice(i + 1)

      break
    }

    if (token === '--transport' || token === '-t') {
      transport = tokens[++i] ?? null

      continue
    }

    if (token === '--env' || token === '-e') {
      const pair = tokens[++i] ?? ''
      const eq = pair.indexOf('=')

      if (eq > 0) {
        env[pair.slice(0, eq)] = pair.slice(eq + 1)
      }

      continue
    }

    if (token === '--header' || token === '-H') {
      const pair = tokens[++i] ?? ''
      const colon = pair.indexOf(':')

      if (colon > 0) {
        headers[pair.slice(0, colon).trim()] = pair.slice(colon + 1).trim()
      }

      continue
    }

    if (token === '--scope' || token === '-s') {
      i++

      continue
    }

    if (token.startsWith('-')) {
      continue
    }

    if (!name) {
      name = token

      continue
    }

    // First non-flag token after the name starts the command (no `--` given).
    rest = tokens.slice(i)

    break
  }

  if (!name || !rest || rest.length === 0) {
    return null
  }

  if (rest.length === 1 && URL_RE.test(rest[0])) {
    const config: Record<string, unknown> = { url: rest[0] }

    if (transport) {
      config.transport = transport
    }

    if (Object.keys(headers).length > 0) {
      config.headers = headers
    }

    return { config, name }
  }

  const config: Record<string, unknown> = { args: rest.slice(1), command: rest[0] }

  if (Object.keys(env).length > 0) {
    // Placeholder values (YOUR_KEY, xxx, TOKEN_HERE) are kept verbatim — the
    // user replaces them in the JSON editor.
    config.env = env
  }

  return { config, name }
}

// cursor://anysphere.cursor-deeplink/mcp/install?name=X&config=<base64 JSON>
function fromCursorDeeplink(text: string): McpImportEntry[] | null {
  const match = /^cursor:\/\/anysphere\.cursor-deeplink\/mcp\/install\?(.+)$/i.exec(text)

  if (!match) {
    return null
  }

  const params = new URLSearchParams(match[1])
  const name = params.get('name')
  const encoded = params.get('config')

  if (!name || !encoded) {
    return null
  }

  try {
    const config = JSON.parse(atob(encoded)) as unknown

    if (!isRecord(config) || !isServerShape(config)) {
      return null
    }

    return [{ config: normalizeEntry(config), name }]
  } catch {
    return null
  }
}

function inferNameFromConfig(config: Record<string, unknown>): string {
  if (typeof config.url === 'string') {
    return nameFromUrl(config.url)
  }

  if (typeof config.command === 'string') {
    const args = Array.isArray(config.args) ? config.args.filter((a): a is string => typeof a === 'string') : []

    return inferCommandName(config.command, args)
  }

  return 'server'
}

// mcp.json snippets: `{"mcpServers": {...}}`, a bare name→config map, or a
// single unnamed server object (name inferred from its command/url).
function fromJson(text: string): McpImportEntry[] | null {
  let parsed: unknown

  try {
    parsed = JSON.parse(text)
  } catch {
    return null
  }

  if (!isRecord(parsed)) {
    return null
  }

  if (isServerShape(parsed)) {
    const config = normalizeEntry(parsed)

    return [{ config, name: inferNameFromConfig(config) }]
  }

  const wrapper = parsed.mcpServers ?? parsed.mcp_servers
  const map = isRecord(wrapper) ? wrapper : parsed
  const entries = Object.entries(map)

  if (entries.length === 0) {
    return null
  }

  const out: McpImportEntry[] = []

  for (const [name, entry] of entries) {
    if (!isRecord(entry) || !isServerShape(entry)) {
      return null
    }

    out.push({ config: normalizeEntry(entry), name })
  }

  return out
}

function parseLine(line: string): McpImportEntry[] | null {
  if (URL_RE.test(line)) {
    return [{ config: { url: line }, name: nameFromUrl(line) }]
  }

  const tokens = tokenize(line)

  if (!tokens || tokens.length === 0) {
    return null
  }

  if (tokens[0] === 'claude' && tokens[1] === 'mcp' && tokens[2] === 'add') {
    const entry = fromClaudeAdd(tokens)

    return entry ? [entry] : null
  }

  const entry = fromCommandLine(tokens)

  return entry ? [entry] : null
}

/**
 * Parse any pasted MCP server description into named mcp.json entries.
 * Returns null when the text doesn't contain a recognizable server config.
 */
export function parseMcpImport(text: string): McpImportEntry[] | null {
  const trimmed = text.trim()

  if (!trimmed) {
    return null
  }

  const deeplink = fromCursorDeeplink(trimmed)

  if (deeplink) {
    return deeplink
  }

  const json = fromJson(trimmed)

  if (json) {
    return json
  }

  // Command / URL territory: fold shell line continuations, then parse each
  // remaining line independently (a README block can list several commands).
  const folded = trimmed.replace(/\\\r?\n/g, ' ')
  const results: McpImportEntry[] = []

  for (const line of folded.split('\n')) {
    const clean = line.trim()

    if (!clean) {
      continue
    }

    const entries = parseLine(clean)

    if (entries) {
      results.push(...entries)
    }
  }

  return results.length > 0 ? results : null
}
