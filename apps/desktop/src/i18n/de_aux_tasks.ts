import type { AuxTaskCopyMap } from './types_aux_tasks'

export const deAuxTasks: AuxTaskCopyMap = {
  vision: {
    label: 'Sehen',
    hint: 'Bildanalyse'
  },
  compression: {
    label: 'Kompression',
    hint: 'Kontext-Verdichtung'
  },
  skills_hub: {
    label: 'Skills-Hub',
    hint: 'Skill-Suche'
  },
  approval: {
    label: 'Freigabe',
    hint: 'Intelligente Auto-Freigabe'
  },
  mcp: {
    label: 'MCP',
    hint: 'MCP-Tool-Routing'
  },
  title_generation: {
    label: 'Titel-Generierung',
    hint: 'Session-Titel'
  },
  review: {
    label: 'Review',
    hint: '/review Bewertungs-Subagent'
  },
  triage_specifier: {
    label: 'Triage-Spezifizierer',
    hint: 'Kanban-Spezifikation ausarbeiten'
  },
  kanban_decomposer: {
    label: 'Kanban-Zerleger',
    hint: 'Aufgaben zerlegen'
  },
  profile_describer: {
    label: 'Profil-Beschreiber',
    hint: 'Automatische Profilbeschreibungen'
  },
  curator: {
    label: 'Kurator',
    hint: 'Skill-Nutzungs-Review'
  }
}
