import type { ContextUsagePanelCopy } from './types_context_usage_panel'

export const frContextUsagePanel = {
  categories: {
    conversation: 'Conversation',
    mcp: 'MCP',
    memory: 'Mémoire',
    rules: 'Règles',
    skills: 'Skills',
    subagent_definitions: 'Définitions de sous-agents',
    system_prompt: 'Invite système',
    tool_definitions: "Définitions d'outils"
  },
  compact: {
    action: 'Compacter la conversation',
    busyHint: "Terminez ou arrêtez d'abord la réponse en cours",
    confirm: 'Compacter',
    hint: 'Résumez le contexte ancien pour libérer de la place. Continuez dans cette même conversation.',
    question: 'Compacter cette conversation en un résumé ?',
    questionNamed: title => `Compacter « ${title} » en un résumé ?`,
    working: 'Compaction…'
  },
  empty: 'Aucune donnée de contexte pour le moment',
  loading: 'Chargement du détail…',
  percentFull: percent => `${percent}% plein`,
  title: 'Utilisation du contexte',
  tokenSummary: (used, max) => `${used} / ${max} jetons`,
  tabs: { inspector: 'Inspecteur', usage: 'Utilisation' },
  inspector: {
    changes: {
      model_changed: (before, after) => `Modèle modifié: ${before} → ${after}`,
      prompt_changed: 'Invite modifiée',
      provider_changed: (before, after) => `Fournisseur modifié: ${before} → ${after}`,
      tool_added: name => `Outil ajouté: ${name}`,
      tool_removed: name => `Outil retiré: ${name}`
    },
    error: "Impossible de lire ce que l'agent a vu",
    lineage: count => `Lignée de compression · ${count} segments`,
    loading: "Lecture de l'instantané enregistré…",
    mcpServers: 'Serveurs MCP',
    model: 'Modèle',
    noChanges: 'Aucun changement',
    noPrompt: "Aucune invite enregistrée : elle l'est après le premier tour.",
    none: 'Aucun',
    promptChars: count => `${count} caractères`,
    provider: 'Fournisseur',
    segment: index => `Segment ${index}`,
    sourceLive: 'Agent actif',
    sourceStored: 'Instantané enregistré',
    systemPrompt: 'Invite système',
    tools: count => `Outils (${count})`
  }
} satisfies ContextUsagePanelCopy
