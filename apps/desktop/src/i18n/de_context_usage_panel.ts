import type { ContextUsagePanelCopy } from './types_context_usage_panel'

export const deContextUsagePanel = {
  categories: {
    conversation: 'Konversation',
    mcp: 'MCP',
    memory: 'Memory',
    rules: 'Regeln',
    skills: 'Skills',
    subagent_definitions: 'Subagent-Definitionen',
    system_prompt: 'System-Prompt',
    tool_definitions: 'Tool-Definitionen'
  },
  compact: {
    action: 'Unterhaltung verdichten',
    busyHint: 'Beende oder stoppe zuerst die laufende Antwort',
    confirm: 'Verdichten',
    hint: 'Älteren Kontext zusammenfassen, um Platz zu schaffen. Dieses Gespräch wird fortgesetzt.',
    question: 'Diese Unterhaltung zu einer Zusammenfassung verdichten?',
    questionNamed: title => `„${title}“ zu einer Zusammenfassung verdichten?`,
    working: 'Verdichte …'
  },
  empty: 'Noch keine Kontext-Daten',
  loading: 'Aufschlüsselung wird geladen…',
  percentFull: percent => `${percent}% voll`,
  title: 'Kontext-Verbrauch',
  tokenSummary: (used, max) => `${used} / ${max} Tokens`,
  tabs: { inspector: 'Inspektor', usage: 'Verbrauch' },
  inspector: {
    changes: {
      model_changed: (before, after) => `Modell geändert: ${before} → ${after}`,
      prompt_changed: 'Prompt geändert',
      provider_changed: (before, after) => `Anbieter geändert: ${before} → ${after}`,
      tool_added: name => `Tool hinzugefügt: ${name}`,
      tool_removed: name => `Tool entfernt: ${name}`
    },
    error: 'Konnte nicht lesen, was der Agent gesehen hat',
    lineage: count => `Komprimierungsverlauf · ${count} Segmente`,
    loading: 'Gespeicherter Stand wird gelesen…',
    mcpServers: 'MCP-Server',
    model: 'Modell',
    noChanges: 'Keine Änderungen',
    noPrompt: 'Noch kein gespeicherter Prompt: er wird nach dem ersten Zug gespeichert.',
    none: 'Keine',
    promptChars: count => `${count} Zeichen`,
    provider: 'Anbieter',
    segment: index => `Segment ${index}`,
    sourceLive: 'Aktiver Agent',
    sourceStored: 'Gespeicherter Stand',
    systemPrompt: 'System-Prompt',
    tools: count => `Tools (${count})`
  }
} satisfies ContextUsagePanelCopy
