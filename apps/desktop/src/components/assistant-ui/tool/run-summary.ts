import { summarizeShellCommand } from '@/lib/summarize-command'

import {
  countDiffLineStats,
  fileEditBasename,
  firstStringField,
  inlineDiffFromResult,
  isFileEditTool,
  parseMaybeObject
} from './fallback-model'

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

export interface RunSummary {
  added: number
  removed: number
  text: string
}

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

/** The thing a tool acted on, as the header should name it. */
function toolTarget(tool: ToolCallLike): string {
  const args = parseMaybeObject(tool.args)

  if (toolCategory(tool.toolName) === 'run') {
    return summarizeShellCommand(firstStringField(args, ['command', 'code']))
  }

  const path = firstStringField(args, ['path', 'file', 'filepath'])

  return path ? fileEditBasename(path) : firstStringField(args, ['query', 'url'])
}

function diffStats(tools: readonly ToolCallLike[]): { added: number; removed: number } {
  let added = 0
  let removed = 0

  for (const tool of tools) {
    if (!isFileEditTool(tool.toolName)) {
      continue
    }

    const stats = countDiffLineStats(inlineDiffFromResult(tool.result))

    added += stats.added
    removed += stats.removed
  }

  return { added, removed }
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
 * — "Edited wiring.tsx, explored 3 files, ran 5 commands". The category holding
 * the still-running tool speaks in the present tense so a live run reads as
 * work in progress rather than work already done.
 */
export function summarizeToolRun(tools: readonly ToolCallLike[]): RunSummary {
  const running = tools.find(isPending)
  const liveCategory = running ? toolCategory(running.toolName) : null

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

  return {
    ...diffStats(tools),
    text: clauses.map((text, index) => (index === 0 ? text : lowerFirst(text))).join(', ')
  }
}
