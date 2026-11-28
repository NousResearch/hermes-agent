import type { AuxTaskCopyMap } from './types_aux_tasks'

export const esAuxTasks: AuxTaskCopyMap = {
  vision: {
    label: 'Visión',
    hint: 'Análisis de imágenes'
  },
  compression: {
    label: 'Compresión',
    hint: 'Compactación de contexto'
  },
  approval: {
    label: 'Aprobación',
    hint: 'Aprobación automática inteligente'
  },
  mcp: {
    label: 'MCP',
    hint: 'Enrutamiento de herramientas MCP'
  },
  title_generation: {
    label: 'Generación de títulos',
    hint: 'Títulos de sesión'
  },
  review: {
    label: 'Revisión',
    hint: 'subagente revisor de /review'
  },
  voice_chat: { label: 'Chat de voz', hint: 'Respuestas habladas del modo voz' },
  triage_specifier: {
    label: 'Especificador de triaje',
    hint: 'Detalle de especificaciones de Kanban'
  },
  kanban_decomposer: {
    label: 'Descomponedor de Kanban',
    hint: 'Descomposición de tareas'
  },
  profile_describer: {
    label: 'Descriptor de perfiles',
    hint: 'Descripciones automáticas de perfiles'
  },
  curator: {
    label: 'Curador',
    hint: 'Revisión de uso de skills'
  },
  memory_query_rewrite: { label: 'Reescritura de consultas de memoria', hint: 'Consultas de búsqueda en memoria' },
  tts_audio_tags: { label: 'Etiquetas de audio TTS', hint: 'Inserción de etiquetas para Gemini TTS' },
  group_router: { label: 'Enrutador de grupos', hint: 'Quién responde en grupos con enrutamiento automático' },
  goal_judge: { label: 'Juez de objetivos', hint: 'Redacción y evaluación de contratos /goal' },
  background_review: { label: 'Revisión en segundo plano', hint: 'Actualización de memoria y skills tras cada turno' },
  monitor: { label: 'Monitor', hint: 'Puntuación de correos importantes' },
  moa_reference: { label: 'Referencia MoA', hint: 'Modelos de referencia de Mixture of Agents' },
  moa_aggregator: { label: 'Agregador MoA', hint: 'Paso de agregación de Mixture of Agents' }
}
