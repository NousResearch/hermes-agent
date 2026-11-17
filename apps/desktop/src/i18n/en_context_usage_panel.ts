import type { ContextUsagePanelCopy } from './types_context_usage_panel'

export const enContextUsagePanel = {
  categories: {
    conversation: 'Conversation',
    mcp: 'MCP',
    memory: 'Memory',
    rules: 'Rules',
    skills: 'Skills',
    subagent_definitions: 'Subagent definitions',
    system_prompt: 'System prompt',
    tool_definitions: 'Tool definitions'
  },
  compact: {
    action: 'Compact conversation',
    busyHint: 'Finish or stop the current reply first',
    confirm: 'Compact',
    hint: 'Summarize older context to make room. Continue in this same conversation.',
    question: 'Compress this chat into a summary?',
    questionNamed: title => `Compress “${title}” into a summary?`,
    working: 'Compacting…'
  },
  empty: 'No context data yet',
  loading: 'Loading breakdown…',
  percentFull: percent => `${percent}% Full`,
  title: 'Context Usage',
  tokenSummary: (used, max) => `${used} / ${max} Tokens`,
  tabs: { inspector: 'Inspector', usage: 'Usage' },
  inspector: {
    changes: {
      model_changed: (before, after) => `Model changed: ${before} → ${after}`,
      prompt_changed: 'Prompt changed',
      provider_changed: (before, after) => `Provider changed: ${before} → ${after}`,
      tool_added: name => `Tool added: ${name}`,
      tool_removed: name => `Tool removed: ${name}`
    },
    error: 'Could not read what the agent saw',
    lineage: count => `Compression lineage · ${count} segments`,
    loading: 'Reading the stored snapshot…',
    mcpServers: 'MCP servers',
    model: 'Model',
    noChanges: 'No changes',
    noPrompt: 'No stored prompt yet: it is saved after the first turn.',
    none: 'None',
    promptChars: count => `${count} chars`,
    provider: 'Provider',
    segment: index => `Segment ${index}`,
    sourceLive: 'Live agent',
    sourceStored: 'Stored snapshot',
    systemPrompt: 'System prompt',
    tools: count => `Tools (${count})`
  }
} satisfies ContextUsagePanelCopy
