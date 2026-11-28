import type { AuxTaskCopyMap } from './types_aux_tasks'

export const frAuxTasks: AuxTaskCopyMap = {
  vision: {
    label: 'Vision',
    hint: "Analyse d'image"
  },
  compression: {
    label: 'Compression',
    hint: 'Compaction de contexte'
  },
  approval: {
    label: 'Approbation',
    hint: 'Auto-approbation intelligente'
  },
  mcp: {
    label: 'MCP',
    hint: "Routage d'outils MCP"
  },
  title_generation: {
    label: 'Génération de titre',
    hint: 'Titres de session'
  },
  review: {
    label: 'Révision',
    hint: 'Sous-agent de révision /review'
  },
  voice_chat: { label: 'Chat vocal', hint: 'Réponses parlées du mode vocal' },
  triage_specifier: {
    label: 'Précision du triage',
    hint: 'Détail des spécifications Kanban'
  },
  kanban_decomposer: {
    label: 'Décomposition Kanban',
    hint: 'Décomposition des tâches'
  },
  profile_describer: {
    label: 'Description de profil',
    hint: 'Descriptions automatiques des profils'
  },
  curator: {
    label: 'Curateur',
    hint: "Revue d'utilisation des skills"
  },
  memory_query_rewrite: { label: 'Réécriture des requêtes mémoire', hint: 'Requêtes de recherche en mémoire' },
  tts_audio_tags: { label: 'Balises audio TTS', hint: 'Insertion de balises pour Gemini TTS' },
  group_router: { label: 'Routeur de groupe', hint: 'Qui répond dans les groupes à routage automatique' },
  goal_judge: { label: 'Juge des objectifs', hint: 'Rédaction et évaluation des contrats /goal' },
  background_review: { label: 'Revue en arrière-plan', hint: 'Mise à jour de la mémoire et des skills après chaque tour' },
  monitor: { label: 'Surveillance', hint: 'Notation des e-mails importants' },
  moa_reference: { label: 'Référence MoA', hint: 'Modèles de référence Mixture of Agents' },
  moa_aggregator: { label: 'Agrégateur MoA', hint: "Étape d'agrégation Mixture of Agents" }
}
