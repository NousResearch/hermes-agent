import { translateNow } from '@/i18n'
import { summarizeShellCommand } from '@/lib/summarize-command'
import { firstStringField } from '@/lib/text'
import { extractToolErrorMessage } from '@/lib/tool-result-summary'

import { fileEditBasename, isFileEditTool, parseMaybeObject } from './fallback-model'
import { skillActivityTitle } from './skill-activity'

/**
 * The little a summary needs from a tool call, stated structurally so both
 * shapes of tool part satisfy it — the stored `ChatMessagePart` and the live
 * one assistant-ui hands to a renderer.
 */
export interface ToolCallLike {
  args?: unknown
  completedAt?: number
  isError?: boolean
  result?: unknown
  toolCallId?: string
  toolName: string
}

export function isToolCallPart<T extends { type: string }>(part: T): part is Extract<T, { type: 'tool-call' }> {
  return part.type === 'tool-call'
}

type RunCategory =
  | 'analyze'
  | 'ask'
  | 'browse'
  | 'delegate'
  | 'edit'
  | 'grep'
  | 'interact'
  | 'list'
  | 'mcp'
  | 'other'
  | 'page'
  | 'read'
  | 'recall'
  | 'run'
  | 'save'
  | 'todo'
  | 'web'

// MCP calls group per server, so each server is its own category key.
type CategoryKey = `mcp:${string}` | Exclude<RunCategory, 'mcp'>

// Clause order is fixed so the same run always reads the same way, whichever
// category happens to be live. MCP servers sit together, alphabetically.
const CATEGORY_ORDER: readonly RunCategory[] = [
  'edit',
  'read',
  'grep',
  'list',
  'web',
  'page',
  'browse',
  'interact',
  'analyze',
  'run',
  'recall',
  'save',
  'todo',
  'ask',
  'delegate',
  'mcp',
  'other'
]

/**
 * How a category reads. With a `noun` the clause names its lone target or
 * counts ("Read 3 files"), `prep` sitting between verb and object ("Searched
 * for 2 patterns"). With a fixed `object` there is nothing to count, so
 * repeats read as "twice" / "N times" ("Searched the web twice").
 */
interface CategoryCopy {
  noun?: [string, string]
  object?: string
  past: string
  prep?: string
  present: string
}

const CATEGORY_COPY: Record<Exclude<RunCategory, 'mcp'>, CategoryCopy> = {
  analyze: { noun: ['image', 'images'], past: 'Analyzed', present: 'Analyzing' },
  ask: { noun: ['question', 'questions'], past: 'Asked', present: 'Asking' },
  browse: { noun: ['page', 'pages'], past: 'Opened', present: 'Opening' },
  delegate: { noun: ['agent', 'agents'], past: 'Launched', present: 'Launching' },
  edit: { noun: ['file', 'files'], past: 'Edited', present: 'Editing' },
  grep: { noun: ['pattern', 'patterns'], past: 'Searched', prep: 'for', present: 'Searching' },
  interact: { noun: ['browser action', 'browser actions'], past: 'Performed', present: 'Performing' },
  list: { noun: ['directory', 'directories'], past: 'Listed', present: 'Listing' },
  other: { noun: ['tool', 'tools'], past: 'Used', present: 'Using' },
  page: { noun: ['page', 'pages'], past: 'Read', present: 'Reading' },
  read: { noun: ['file', 'files'], past: 'Read', present: 'Reading' },
  recall: { object: 'memory', past: 'Checked', present: 'Checking' },
  run: { noun: ['command', 'commands'], past: 'Ran', present: 'Running' },
  save: { noun: ['memory', 'memories'], past: 'Saved', present: 'Saving' },
  todo: { object: 'todos', past: 'Updated', present: 'Updating' },
  web: { object: 'the web', past: 'Searched', present: 'Searching' }
}

// Routed by name so a web search never counts as a read file. Browser tools
// other than navigation are interaction, not page loads: a screenshot or a
// click fetches nothing, so they must not be counted as pages.
const TOOL_CATEGORY: Record<string, RunCategory> = {
  browser_navigate: 'browse',
  clarify: 'ask',
  delegate_task: 'delegate',
  execute_code: 'run',
  list_files: 'list',
  memory: 'save',
  read_file: 'read',
  search_files: 'grep',
  session_search: 'recall',
  session_search_recall: 'recall',
  terminal: 'run',
  todo: 'todo',
  vision_analyze: 'analyze',
  web_extract: 'page',
  web_search: 'web'
}

// `mcp__<server>__<tool>`, as tools/mcp_tool_schema.py names them.
const MCP_TOOL_NAME = /^mcp__(.+?)__(.+)$/

// Memory reads as memory whichever surface it came through — a plugin
// (`mnemosyne_recall`) or an MCP server (`mcp__mnemosyne__store`). For an MCP
// call the server decides whether it is memory at all, by exact name: a
// substring would claim any server that merely mentions memory. Only the tool
// part then decides recall versus write.
const MEMORY_TOOL = /mnemosyne|memory/i
const MEMORY_MCP_SERVER = /^(?:mnemosyne|memory)$/i
const MEMORY_RECALL = /recall|search|read|open|get|list|stats/i

function memoryCategory(toolName: string): 'recall' | 'save' | undefined {
  const mcp = MCP_TOOL_NAME.exec(toolName)
  const isMemory = mcp ? MEMORY_MCP_SERVER.test(mcp[1]) : MEMORY_TOOL.test(toolName)

  if (!isMemory) {
    return undefined
  }

  return MEMORY_RECALL.test(mcp ? mcp[2] : toolName) ? 'recall' : 'save'
}

// Servers whose brand casing a title-case guess would get wrong.
const MCP_SERVER_NAME: Record<string, string> = {
  context7: 'Context7',
  github: 'GitHub',
  gitlab: 'GitLab',
  tinyfish: 'TinyFish'
}

function mcpServerName(server: string): string {
  const key = server.toLowerCase()

  return (
    MCP_SERVER_NAME[key] ??
    key
      .split(/[-_]+/)
      .filter(Boolean)
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ')
  )
}

// Name → category, first match wins. A table rather than a ladder: each row
// is one rule and the order is the precedence.
const CATEGORY_RULES: readonly ((toolName: string) => CategoryKey | undefined)[] = [
  name => (isFileEditTool(name) ? 'edit' : undefined),
  name => TOOL_CATEGORY[name] as CategoryKey | undefined,
  memoryCategory,
  name => {
    const server = MCP_TOOL_NAME.exec(name)?.[1]

    return server ? `mcp:${server}` : undefined
  },
  name => (name.startsWith('browser_') ? 'interact' : undefined)
]

function toolCategory(toolName: string): CategoryKey {
  for (const rule of CATEGORY_RULES) {
    const category = rule(toolName)

    if (category) {
      return category
    }
  }

  return 'other'
}

function isMcpCategory(category: CategoryKey): category is `mcp:${string}` {
  return category.startsWith('mcp:')
}

function categoryCopy(category: CategoryKey): CategoryCopy {
  if (isMcpCategory(category)) {
    return { object: mcpServerName(category.slice('mcp:'.length)), past: 'Called', present: 'Calling' }
  }

  return CATEGORY_COPY[category]
}

function categoryRank(category: CategoryKey): number {
  return CATEGORY_ORDER.indexOf(isMcpCategory(category) ? 'mcp' : category)
}

// Tools that act on a batch in one call, and the argument holding the batch.
// `clarify` and `delegate_task` batch too, but they render as cards and never
// reach a run summary (`isCardTool`); their copy above serves the drafting line.
const BATCH_FIELD: Record<string, string> = {
  web_extract: 'urls'
}

/**
 * How many things one call acted on. One call is one thing except for the
 * batching tools: `web_extract` takes up to five URLs — counting its calls
 * would report five fetched pages as one.
 */
function unitCount(tool: ToolCallLike): number {
  const field = BATCH_FIELD[tool.toolName]
  const batch = field ? parseMaybeObject(tool.args)[field] : undefined

  return Array.isArray(batch) && batch.length > 0 ? batch.length : 1
}

function isPending(tool: ToolCallLike): boolean {
  return tool.result === undefined && tool.completedAt === undefined
}

/**
 * How a tool reads while it is happening — "Editing", "Calling GitHub". Shared with
 * the status line that covers the gap before a tool starts, so the same run is
 * described in the same words from the moment the model drafts it.
 */
export function toolPresentVerb(toolName: string): string {
  if (toolName === 'skill_view') {
    return translateNow('assistant.tool.skillActivity.loading')
  }

  const copy = categoryCopy(toolCategory(toolName))

  return copy.object ? `${copy.present} ${copy.object}` : copy.present
}

/** The thing a tool acted on, as the header should name it. */
function toolTarget(tool: ToolCallLike): string {
  const args = parseMaybeObject(tool.args)

  if (toolCategory(tool.toolName) === 'run') {
    return summarizeShellCommand(firstStringField(args, ['command', 'code']))
  }

  const pattern = firstStringField(args, ['pattern'])

  if (pattern) {
    return pattern
  }

  const path = firstStringField(args, ['path', 'file', 'filepath'])

  if (path) {
    return fileEditBasename(path)
  }

  const urls = Array.isArray(args.urls) ? args.urls : []

  return firstStringField(args, ['query', 'url']) || (urls.length === 1 && typeof urls[0] === 'string' ? urls[0] : '')
}

function times(count: number): string {
  return count === 2 ? 'twice' : `${count} times`
}

/**
 * One clause per category. A category holding a single thing says what it was
 * ("Read wiring.tsx"); anything else counts ("read 3 files"). A category with
 * a fixed object has nothing to name, so it counts repeats instead ("searched
 * the web twice"). A settled command is the exception — "ran 5 commands" is
 * the useful reading, and a command line only earns its space while it's the
 * thing you're waiting on.
 */
function clause(category: CategoryKey, tools: ToolCallLike[], live: boolean): string {
  const copy = categoryCopy(category)
  const verb = live ? copy.present : copy.past
  const count = tools.reduce((sum, tool) => sum + unitCount(tool), 0)

  if (copy.object) {
    return count === 1 ? `${verb} ${copy.object}` : `${verb} ${copy.object} ${times(count)}`
  }

  const lead = copy.prep ? `${verb} ${copy.prep}` : verb
  const target = count === 1 && category !== 'interact' ? toolTarget(tools[0]) : ''

  if (target && (live || category !== 'run')) {
    return `${lead} ${target}`
  }

  const [one, many] = copy.noun ?? CATEGORY_COPY.other.noun!

  return `${lead} ${count} ${count === 1 ? one : many}`
}

function lowerFirst(text: string): string {
  return text.charAt(0).toLowerCase() + text.slice(1)
}

/**
 * Collapse a run of tool calls into the single grey line that stands in for it
 * — "Read 3 files, ran 5 commands". While the run is live, the category
 * holding its most recent call speaks in the present tense so the line reads as
 * work in progress rather than work already done.
 *
 * Whether the run is `live` is the caller's to say, not something readable off
 * the calls: a call can be left without a result by a turn that ended or an
 * agent that moved on, and a run like that has to read as finished rather than
 * narrate work that stopped happening.
 *
 * A run only ever holds ephemeral activity — file edits and other cards are
 * split out before this sees them (`splitRunItems`), so there is no aggregate
 * diff to report here; each edit carries its own +N/−M on its card.
 */
export function summarizeToolRun(tools: readonly ToolCallLike[], live: boolean): string {
  // Which clause narrates in the present tense: normally the outstanding call,
  // but sequential calls leave gaps where the run is still going and nothing is
  // pending. The most recent call covers those, and it's the one the ticker is
  // showing anyway.
  const narrating = live ? (tools.find(isPending) ?? tools.at(-1)) : undefined
  const liveCategory = narrating ? toolCategory(narrating.toolName) : null

  const byCategory = new Map<CategoryKey, ToolCallLike[]>()
  const skillClauses: string[] = []

  for (const tool of tools) {
    const skill = skillActivityTitle(tool, live)

    if (skill) {
      skillClauses.push(skill)

      continue
    }

    const category = toolCategory(tool.toolName)
    const group = byCategory.get(category)

    if (group) {
      group.push(tool)
    } else {
      byCategory.set(category, [tool])
    }
  }

  const clauses = [...byCategory.keys()]
    .sort((a, b) => categoryRank(a) - categoryRank(b) || a.localeCompare(b))
    .map(category => clause(category, byCategory.get(category)!, category === liveCategory))

  const failed = tools.filter(tool => {
    const result = parseMaybeObject(tool.result)

    // Explicit success beats stale envelope errors, as in individual rows.
    return (
      result.success !== true &&
      result.ok !== true &&
      Boolean(tool.isError || result.success === false || result.ok === false || extractToolErrorMessage(tool.result))
    )
  }).length

  if (failed) {
    clauses.push(translateNow('assistant.tool.failedCalls', failed))
  }

  return [...skillClauses, ...clauses].map((text, index) => (index === 0 ? text : lowerFirst(text))).join(', ')
}
