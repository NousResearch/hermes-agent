/** Statusbar ▸ context meter popover: the usage breakdown and the turn inspector. */
export interface ContextUsagePanelCopy {
  categories: {
    conversation: string
    mcp: string
    memory: string
    rules: string
    skills: string
    subagent_definitions: string
    system_prompt: string
    tool_definitions: string
  }
  compact: {
    action: string
    busyHint: string
    confirm: string
    hint: string
    question: string
    questionNamed: (title: string) => string
    working: string
  }
  empty: string
  loading: string
  percentFull: (percent: number) => string
  title: string
  tokenSummary: (used: string, max: string) => string
  tabs: { inspector: string; usage: string }
  /** The turn inspector: what the agent saw, read from the stored prompt (never a rebuild). */
  inspector: {
    changes: {
      model_changed: (before: string, after: string) => string
      prompt_changed: string
      provider_changed: (before: string, after: string) => string
      tool_added: (name: string) => string
      tool_removed: (name: string) => string
    }
    error: string
    lineage: (count: number) => string
    loading: string
    mcpServers: string
    model: string
    noChanges: string
    noPrompt: string
    none: string
    promptChars: (count: string) => string
    provider: string
    segment: (index: number) => string
    sourceLive: string
    sourceStored: string
    systemPrompt: string
    tools: (count: number) => string
  }
}
