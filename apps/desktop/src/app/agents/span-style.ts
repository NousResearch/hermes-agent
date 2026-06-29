import { toolIconName } from '@/components/assistant-ui/tool-fallback-model'
import type { TraceSpanKind, TraceSpanNode } from '@/store/trace'

// Category → bar classes. Tailwind -500 family reads at even weight on dark;
// error tint overrides downstream.
const KIND_BAR: Record<TraceSpanKind, string> = {
  AGENT: 'bg-violet-500/70',
  CHAIN: 'bg-slate-500/70',
  LLM: 'bg-sky-500/70',
  TOOL: 'bg-emerald-500/70'
}

const TOOL_BAR: Record<string, string> = {
  delegate_task: 'bg-violet-500/70',
  patch: 'bg-amber-500/70',
  read_file: 'bg-cyan-500/70',
  search_files: 'bg-cyan-500/70',
  terminal: 'bg-zinc-400/70',
  web_extract: 'bg-teal-500/70',
  web_search: 'bg-teal-500/70',
  write_file: 'bg-amber-500/70'
}

/** Bar color for a span: error tint wins, then per-tool, then per-kind. */
export function barClass(node: TraceSpanNode): string {
  if (node.status === 'error') {
    return 'bg-red-500/80'
  }

  const tool = String(node.attributes['tool.name'] ?? '')

  if (node.kind === 'TOOL' && TOOL_BAR[tool]) {
    return TOOL_BAR[tool]
  }

  return KIND_BAR[node.kind]
}

/** Icon for a span, reusing the thread's tool icons so the trace matches what
 *  the conversation shows mid-thread. */
export function spanIconName(node: TraceSpanNode): string {
  if (node.kind === 'TOOL') {
    return toolIconName(String(node.attributes['tool.name'] ?? 'tools'))
  }

  // The root span is the user's turn (its label is the prompt) — mark it human.
  if (node.kind === 'AGENT' && node.parentId === null) {
    return 'account'
  }

  if (node.kind === 'AGENT' || node.kind === 'LLM') {
    return 'hubot'
  }

  return 'list-tree'
}
