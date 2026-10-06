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
  skills_hub: {
    label: 'Hub de skills',
    hint: 'Búsqueda de skills'
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
  }
}
