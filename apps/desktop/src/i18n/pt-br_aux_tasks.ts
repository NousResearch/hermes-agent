import type { AuxTaskCopyMap } from './types_aux_tasks'

export const ptBrAuxTasks: AuxTaskCopyMap = {
  vision: { label: 'Visão', hint: 'Análise de imagens' },
  compression: { label: 'Compressão', hint: 'Compactação do contexto' },
  approval: { label: 'Aprovação', hint: 'Aprovação automática inteligente' },
  mcp: { label: 'MCP', hint: 'Roteamento de ferramentas MCP' },
  title_generation: { label: 'Geração de títulos', hint: 'Títulos das sessões' },
  review: { label: 'Revisão', hint: 'Subagente revisor de /review' },
  voice_chat: { label: 'Conversa por voz', hint: 'Respostas faladas no modo de voz' },
  triage_specifier: { label: 'Detalhamento da triagem', hint: 'Detalhamento de especificações do Kanban' },
  kanban_decomposer: { label: 'Decomposição do Kanban', hint: 'Divisão de tarefas' },
  profile_describer: { label: 'Descrição de perfis', hint: 'Descrições automáticas dos perfis' },
  curator: { label: 'Curadoria', hint: 'Revisão do uso de habilidades' }
}
