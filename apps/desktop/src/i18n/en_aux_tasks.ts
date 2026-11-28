import type { AuxTaskCopyMap } from './types_aux_tasks'

export const enAuxTasks: AuxTaskCopyMap = {
  vision: { label: 'Vision', hint: 'Image analysis' },
  compression: { label: 'Compression', hint: 'Context compaction' },
  approval: { label: 'Approval', hint: 'Smart auto-approve' },
  mcp: { label: 'MCP', hint: 'MCP tool routing' },
  title_generation: { label: 'Title gen', hint: 'Session titles' },
  review: { label: 'Review', hint: '/review reviewer subagent' },
  voice_chat: { label: 'Voice chat', hint: 'Spoken voice-mode replies' },
  triage_specifier: { label: 'Triage specifier', hint: 'Kanban spec fleshing' },
  kanban_decomposer: { label: 'Kanban decomposer', hint: 'Task decomposition' },
  profile_describer: { label: 'Profile describer', hint: 'Auto profile descriptions' },
  curator: { label: 'Curator', hint: 'Skill-usage review' },
  memory_query_rewrite: { label: 'Memory query rewrite', hint: 'Memory retrieval queries' },
  tts_audio_tags: { label: 'TTS audio tags', hint: 'Gemini TTS tag insertion' },
  group_router: { label: 'Group router', hint: 'Who answers in auto-routed group chats' },
  goal_judge: { label: 'Goal judge', hint: '/goal contract drafting and judging' },
  background_review: { label: 'Background review', hint: 'Post-turn memory and skill updates' },
  monitor: { label: 'Monitor', hint: 'Important-mail scoring' },
  moa_reference: { label: 'MoA reference', hint: 'Mixture of Agents reference models' },
  moa_aggregator: { label: 'MoA aggregator', hint: 'Mixture of Agents aggregation step' }
}
