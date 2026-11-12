import type { ClarifyCopy } from './types_clarify'

export const ptBrClarify: ClarifyCopy = {
  notReady: 'A solicitação de esclarecimento ainda não está pronta',
  gatewayDisconnected: 'O Hermes está desconectado no momento. Reconecte e envie novamente.',
  sendFailed: 'Não foi possível enviar a resposta de esclarecimento',
  loadingQuestion: 'Carregando pergunta…',
  other: 'Outra opção (digite sua resposta)',
  placeholder: 'Digite sua resposta…',
  skip: 'Pular',
  skipped: 'Pulado',
  noAnswer: 'Sem resposta',
  confirmAndContinueLabel: 'Confirmar e continuar',
  singleSelectHint: 'Escolha uma opção',
  multiSelectHint: 'Selecione todas as opções que se aplicam',
  oneQuestion: '1 pergunta',
  questionProgress: (answered, total) => `${answered} de ${total} respondidas`,
  notDelivered:
    'Esta pergunta não chegou ao aplicativo, então não é possível respondê-la aqui. Pressione Parar para encerrar o turno e responda na conversa.'
}
