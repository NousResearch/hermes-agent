import type { ContextUsagePanelCopy } from './types_context_usage_panel'

export const esContextUsagePanel = {
  categories: {
    conversation: 'Conversación',
    mcp: 'MCP',
    memory: 'Memoria',
    rules: 'Reglas',
    skills: 'Skills',
    subagent_definitions: 'Definiciones de subagente',
    system_prompt: 'Prompt del sistema',
    tool_definitions: 'Definiciones de herramientas'
  },
  compact: {
    action: 'Compactar conversación',
    busyHint: 'Termina o detén primero la respuesta en curso',
    confirm: 'Compactar',
    hint: 'Resume el contexto anterior para hacer espacio. Continúa en esta misma conversación.',
    question: '¿Compactar esta conversación en un resumen?',
    questionNamed: title => `¿Compactar «${title}» en un resumen?`,
    working: 'Compactando…'
  },
  empty: 'Aún no hay datos de contexto',
  loading: 'Cargando desglose…',
  percentFull: percent => `${percent}% lleno`,
  title: 'Uso del contexto',
  tokenSummary: (used, max) => `${used} / ${max} tokens`,
  tabs: { inspector: 'Inspector', usage: 'Uso' },
  inspector: {
    changes: {
      model_changed: (before, after) => `Modelo cambiado: ${before} → ${after}`,
      prompt_changed: 'Prompt cambiado',
      provider_changed: (before, after) => `Proveedor cambiado: ${before} → ${after}`,
      tool_added: name => `Herramienta añadida: ${name}`,
      tool_removed: name => `Herramienta quitada: ${name}`
    },
    error: 'No se pudo leer lo que vio el agente',
    lineage: count => `Linaje de compresión · ${count} segmentos`,
    loading: 'Leyendo la instantánea guardada…',
    mcpServers: 'Servidores MCP',
    model: 'Modelo',
    noChanges: 'Sin cambios',
    noPrompt: 'Aún no hay prompt guardado: se guarda tras el primer turno.',
    none: 'Ninguno',
    promptChars: count => `${count} caracteres`,
    provider: 'Proveedor',
    segment: index => `Segmento ${index}`,
    sourceLive: 'Agente activo',
    sourceStored: 'Instantánea guardada',
    systemPrompt: 'Prompt del sistema',
    tools: count => `Herramientas (${count})`
  }
} satisfies ContextUsagePanelCopy
