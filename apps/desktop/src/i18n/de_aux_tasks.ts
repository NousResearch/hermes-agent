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
  voice_chat: { label: 'Sprachchat', hint: 'Gesprochene Antworten im Sprachmodus' },
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
  },
  memory_query_rewrite: { label: 'Gedächtnis-Abfragen', hint: 'Suchanfragen an das Gedächtnis umschreiben' },
  tts_audio_tags: { label: 'TTS-Audio-Tags', hint: 'Tag-Einfügung für Gemini TTS' },
  group_router: { label: 'Gruppen-Router', hint: 'Wer in automatisch gerouteten Gruppen antwortet' },
  goal_judge: { label: 'Ziel-Prüfer', hint: '/goal-Verträge entwerfen und bewerten' },
  background_review: { label: 'Hintergrund-Review', hint: 'Gedächtnis- und Skill-Updates nach jeder Runde' },
  monitor: { label: 'Monitor', hint: 'Bewertung wichtiger E-Mails' },
  moa_reference: { label: 'MoA-Referenz', hint: 'Referenzmodelle für Mixture of Agents' },
  moa_aggregator: { label: 'MoA-Aggregator', hint: 'Aggregationsschritt für Mixture of Agents' }
}
