// Per-tool MCP gating. A server's optional `tools.include` (whitelist) /
// `tools.exclude` (denylist) decide which discovered tools the agent registers
// — `include` wins, no filter means all. Entries may be exact tool names or
// fnmatch-style globs (`*`, `?`, `[seq]`) — mirrors `_register_server_tools` /
// `matches_name_filter` in `tools/mcp_tool.py`, including glob support: a
// catalog manifest's `default_excluded: ["*instructions*"]` must gate the
// same tools here that the backend actually blocks, or this panel shows the
// wrong checkbox state and enabled-count for any server whose filter uses a
// glob (several curated catalog entries do).

export interface McpToolsFilter {
  exclude?: string[]
  include?: string[]
}

type ServerConfig = Record<string, unknown>

const GLOB_METACHARS = /[*?[]/

const asNames = (value: unknown): string[] | undefined =>
  Array.isArray(value) ? value.filter((v): v is string => typeof v === 'string') : undefined

const toolsObject = (server: ServerConfig | null | undefined): Record<string, unknown> => {
  const tools = server?.tools

  return tools && typeof tools === 'object' && !Array.isArray(tools) ? (tools as Record<string, unknown>) : {}
}

export function readToolsFilter(server: ServerConfig | null | undefined): McpToolsFilter {
  const tools = toolsObject(server)

  return { exclude: asNames(tools.exclude), include: asNames(tools.include) }
}

// Translate one fnmatch-style pattern into a case-sensitive RegExp, mirroring
// Python's fnmatch.translate: `*` → any run (incl. empty), `?` → any one
// char, `[seq]`/`[!seq]` → a (negated) character class, everything else
// literal. An unterminated `[` (no closing `]`) falls back to a literal `[`,
// same as CPython's translate.
function fnmatchToRegExp(pattern: string): RegExp {
  let source = ''
  let i = 0

  while (i < pattern.length) {
    const char = pattern[i]

    i += 1

    if (char === '*') {
      source += '.*'
    } else if (char === '?') {
      source += '.'
    } else if (char === '[') {
      let j = i

      if (pattern[j] === '!') {
        j += 1
      }

      if (pattern[j] === ']') {
        j += 1
      }

      while (j < pattern.length && pattern[j] !== ']') {
        j += 1
      }

      if (j >= pattern.length) {
        source += '\\['
      } else {
        let charClass = pattern.slice(i, j).replace(/\\/g, '\\\\')

        if (charClass.startsWith('!')) {
          charClass = `^${charClass.slice(1)}`
        }

        source += `[${charClass}]`
        i = j + 1
      }
    } else {
      source += char.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
    }
  }

  return new RegExp(`^${source}$`)
}

// True if `name` matches any entry in `patterns` — exact names match
// literally; entries containing a glob metacharacter match as a
// case-sensitive fnmatch pattern. Exact membership is checked first, same as
// the backend's matches_name_filter.
function matchesNameFilter(name: string, patterns: string[]): boolean {
  if (!patterns.length) {
    return false
  }

  if (patterns.includes(name)) {
    return true
  }

  return patterns.some(pattern => GLOB_METACHARS.test(pattern) && fnmatchToRegExp(pattern).test(name))
}

export function isToolEnabled(server: ServerConfig | null | undefined, name: string): boolean {
  const { exclude, include } = readToolsFilter(server)

  return include?.length ? matchesNameFilter(name, include) : !matchesNameFilter(name, exclude ?? [])
}

// Toggle one tool, preserving the config's mode (include if present, else an
// exclude denylist). Empty lists — and an emptied `tools` — are dropped.
export function toggleToolInServer(server: ServerConfig, name: string): ServerConfig {
  const { exclude, include } = readToolsFilter(server)
  const key = include?.length ? 'include' : 'exclude'
  const current = (key === 'include' ? include : exclude) ?? []
  const names = current.includes(name) ? current.filter(n => n !== name) : [...current, name]
  const tools = { ...toolsObject(server) }

  if (names.length) {
    tools[key] = names
  } else {
    delete tools[key]
  }

  const next = { ...server }

  if (Object.keys(tools).length) {
    next.tools = tools
  } else {
    delete next.tools
  }

  return next
}

export const countEnabledTools = (server: ServerConfig | null | undefined, names: string[]): number =>
  names.filter(name => isToolEnabled(server, name)).length
