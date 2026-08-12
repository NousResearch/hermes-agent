import { summarizeShellCommand } from '@/lib/summarize-command'

import { fileEditBasename, firstStringField, isFileEditTool, parseMaybeObject } from './fallback-model'

/**
 * The little a summary needs from a tool call, stated structurally so both
 * shapes of tool part satisfy it — the stored `ChatMessagePart` and the live
 * one assistant-ui hands to a renderer.
 */
export interface ToolCallLike {
  args?: unknown
  result?: unknown
  toolCallId?: string
  toolName: string
}

export function isToolCallPart<T extends { type: string }>(part: T): part is Extract<T, { type: 'tool-call' }> {
  return part.type === 'tool-call'
}

type RunCategory = 'delegate' | 'edit' | 'explore' | 'other' | 'run'

// Clause order is fixed so the same run always reads the same way, whichever
// category happens to be live.
const CATEGORY_ORDER: readonly RunCategory[] = ['edit', 'explore', 'run', 'delegate', 'other']

const CATEGORY_COPY: Record<RunCategory, { noun: [string, string]; past: string; present: string }> = {
  delegate: { noun: ['task', 'tasks'], past: 'Delegated', present: 'Delegating' },
  edit: { noun: ['file', 'files'], past: 'Edited', present: 'Editing' },
  explore: { noun: ['file', 'files'], past: 'Explored', present: 'Exploring' },
  other: { noun: ['tool', 'tools'], past: 'Used', present: 'Using' },
  run: { noun: ['command', 'commands'], past: 'Ran', present: 'Running' }
}

const EXPLORE_TOOLS = new Set([
  'list_files',
  'read_file',
  'search_files',
  'session_search_recall',
  'vision_analyze',
  'web_extract',
  'web_search'
])

function toolCategory(toolName: string): RunCategory {
  if (isFileEditTool(toolName)) {
    return 'edit'
  }

  if (toolName === 'terminal' || toolName === 'execute_code') {
    return 'run'
  }

  if (toolName === 'delegate_task') {
    return 'delegate'
  }

  if (EXPLORE_TOOLS.has(toolName) || toolName.startsWith('browser_')) {
    return 'explore'
  }

  return 'other'
}

function isPending(tool: ToolCallLike): boolean {
  return tool.result === undefined
}

/**
 * How a tool reads while it is happening — "Editing", "Exploring". Shared with
 * the status line that covers the gap before a tool starts, so the same run is
 * described in the same words from the moment the model drafts it.
 */
export function toolPresentVerb(toolName: string): string {
  return CATEGORY_COPY[toolCategory(toolName)].present
}

/** The thing a tool acted on, as the header should name it. */
function toolTarget(tool: ToolCallLike): string {
  const args = parseMaybeObject(tool.args)

  if (toolCategory(tool.toolName) === 'run') {
    return summarizeShellCommand(firstStringField(args, ['command', 'code']))
  }

  const path = firstStringField(args, ['path', 'file', 'filepath'])

  return path ? fileEditBasename(path) : firstStringField(args, ['query', 'url'])
}

/**
 * One clause per category. A category holding a single thing says what it was
 * ("Edited wiring.tsx"); anything else counts ("explored 3 files"). A settled
 * command is the exception — "ran 5 commands" is the useful reading, and a
 * command line only earns its space while it's the thing you're waiting on.
 */
function clause(category: RunCategory, tools: ToolCallLike[], live: boolean): string {
  const copy = CATEGORY_COPY[category]
  const verb = live ? copy.present : copy.past
  const target = tools.length === 1 ? toolTarget(tools[0]) : ''

  if (target && (live || category !== 'run')) {
    return `${verb} ${target}`
  }

  return `${verb} ${tools.length} ${copy.noun[tools.length === 1 ? 0 : 1]}`
}

function lowerFirst(text: string): string {
  return text.charAt(0).toLowerCase() + text.slice(1)
}

/**
 * Collapse a run of tool calls into the single grey line that stands in for it
 * — "Explored 3 files, ran 5 commands". While the run is live, the category
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

  const byCategory = new Map<RunCategory, ToolCallLike[]>()

  for (const tool of tools) {
    const category = toolCategory(tool.toolName)
    const group = byCategory.get(category)

    if (group) {
      group.push(tool)
    } else {
      byCategory.set(category, [tool])
    }
  }

  const clauses = CATEGORY_ORDER.flatMap(category => {
    const group = byCategory.get(category)

    return group ? [clause(category, group, category === liveCategory)] : []
  })

  return clauses.map((text, index) => (index === 0 ? text : lowerFirst(text))).join(', ')
}
