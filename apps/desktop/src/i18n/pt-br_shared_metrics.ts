import type { Translations } from './types'

export const ptBrSharedMetrics: Translations['sharedMetrics'] = {
  consentTitle: 'Compartilhar estatísticas de uso?',
  dialogTitle: 'Estatísticas de uso',
  consentBody:
    'O Hermes pode contabilizar como você o usa: duração das sessões, quais modelos e ferramentas são executados e quando algo falha. Ele nunca registra suas mensagens, arquivos, caminhos ou textos de erro.',
  whatIsCollected: 'O que é contabilizado',
  collectedActivity: 'Sessões: duração, resultado, tipo de erro, tempo ativo por dia',
  collectedModels: 'Modelos: quais são usados, totais de tokens',
  collectedNames:
    'Recursos: ferramentas integradas, comandos, áreas do aplicativo e configurações usadas ou desativadas',
  collectedMilestones:
    'Configuração: etapas concluídas, conexões com provedores, quantidade de habilidades, plugins e tarefas',
  collectedReliability:
    'Saúde do aplicativo: falhas, velocidade de inicialização e resposta, atualizações, conexões de mensagens',
  collectedUsage:
    'Qualidade do agente: edições não realizadas, chamadas de ferramenta com falha, ciclos travados, custo por tarefa',
  collectedMachine:
    'Computador: sistema operacional, faixa de RAM, tipo de GPU, versão do Hermes, uso de modelos locais',
  sending:
    'Se você compartilhar, o Hermes enviará estatísticas à Nous uma vez por dia com um ID aleatório para este perfil. Exceto por um registro único de que o Hermes foi instalado, as estatísticas anteriores à sua autorização nunca são enviadas. Altere isso a qualquer momento nas Configurações.',
  readDocs: 'Ler todos os detalhes',
  share: 'Compartilhar com a Nous',
  off: 'Não, obrigado',
  saveFailed: 'Não foi possível salvar sua escolha',
  sendLabel: 'Compartilhar estatísticas de uso com a Nous',
  sendDesc:
    'Apenas contagens, enviadas uma vez por dia com um ID aleatório para este perfil. Nunca suas mensagens, arquivos, caminhos ou textos de erro.',
  unavailable: 'Atualize o backend do Hermes para alterar esta configuração.',
  stripBody: 'Apenas contagens. Nunca suas mensagens ou arquivos.',
  stripReaskBody: 'Perguntando mais uma vez: uma versão anterior podia salvar “Não, obrigado” antes de você ver isto.',
  stripChoices: { share: 'Compartilhar com a Nous', off: 'Não, obrigado' },
  stripDetails: 'Detalhes'
}
