import type { SessionCommandNotices } from './types_notices'

// Mensagens dos comandos /undo /retry /steer /queue no mesmo gateway.
export const ptBrSessionCommands: SessionCommandNotices = {
  undoUsage: 'uso: /undo [N]',
  undoNothing: 'nada para desfazer',
  undidOne: (count: string) => `${count} mensagem desfeita — a mensagem removida está na caixa de mensagem`,
  undidOther: (count: string) => `${count} mensagens desfeitas — a mensagem removida está na caixa de mensagem`,
  retryNothing: 'nada para tentar novamente',
  retryUnsupported: '/retry: a nova tentativa não oferece suporte a mídia ou partes de conteúdo desconhecidas',
  busy: (command: string) =>
    `sessão ocupada — o Hermes ainda está respondendo. Pare a resposta atual primeiro (botão Parar ou Ctrl+C em um terminal) e depois execute /${command}.`,
  busyCompressing: (command: string) =>
    `sessão ocupada — um /compress está em execução; execute /${command} quando terminar.`,
  steerUsage: 'uso: /steer <prompt>',
  steerQueued: (preview: string) =>
    `orientação na fila — será entregue após a próxima chamada de ferramenta: "${preview}"`,
  steerNoActiveTurn: (preview: string) => `nenhum turno ativo — na fila para o próximo: "${preview}"`,
  steerRejected: 'orientação rejeitada — nenhum turno ativo, adicionada à fila do próximo turno',
  queued: (preview: string) => `na fila: "${preview}"`,
  queueFailed: 'não foi possível adicionar a mensagem à fila',
  countOne: (count: string) => `${count} mensagem na fila`,
  countOther: (count: string) => `${count} mensagens na fila`,
  queueEditKeptTitle: 'Mensagem na fila não alterada',
  queueEditKept:
    'Ela começou a ser processada ou foi recusada antes de sua edição ser salva. Seu texto editado está na caixa de mensagem.',
  queueEditUnconfirmedTitle: 'Edição não confirmada',
  queueEditUnconfirmed:
    'O Hermes não conseguiu confirmar sua edição. Salve novamente para tentar a mesma edição; ela nunca é adicionada à fila duas vezes.'
}
