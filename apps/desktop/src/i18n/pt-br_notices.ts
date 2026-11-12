import { ptBrDesktopCommands } from './pt-br_desktop_commands'
import { ptBrSessionCommands } from './pt-br_session_commands'
import type { Translations } from './types'

// Avisos da interface principal, incorporados em pt-br.ts.
export const ptBrNotices = {
  remoteDisplayBanner: {
    message: reason =>
      `Renderização por software ativa — tela remota detectada (${reason}). A aceleração por GPU está desativada para evitar cintilação.`
  },
  butterbar: {
    goTo: (index, total) => `Mostrar aviso ${index} de ${total}`,
    legal: {
      before: 'O uso do Hermes Agent está sujeito aos nossos ',
      terms: 'Termos de Serviço',
      between: ' e à nossa ',
      privacy: 'Política de Privacidade',
      after: '.'
    }
  },
  promptNotices: {
    legacySendUnconfirmed:
      'Este servidor não conseguiu confirmar o envio anterior desta mensagem, então ela pode já ter sido executada. Confira a conversa antes de enviá-la novamente.',
    restoredDraftAttachmentsDropped:
      'Os anexos da sua mensagem não enviada não foram restaurados. Anexe-os novamente antes de enviá-la.',
    mcpReloadDeferred:
      'Esta conversa está ocupada: a alteração do MCP será aplicada quando o turno terminar e o MCP for recarregado novamente, ou quando você reabrir a conversa',
    mcpReloadUnavailable: servers => `Não foi possível conectar: ${servers}`,
    compactedTurnUnavailable:
      'Essa mensagem foi resumida quando a conversa foi compactada, então não pode mais ser restaurada ou editada.',
    staleTurnUnavailable: 'Essa mensagem não está mais no histórico desta conversa.',
    sessionCommands: ptBrSessionCommands,
    desktopCommands: ptBrDesktopCommands
  },
  keepAwakeOffer: {
    title: 'Seu computador entrou em suspensão enquanto o Hermes trabalhava',
    message:
      'O Hermes pode manter seu computador ativo enquanto uma tarefa é executada. A tela ainda será desligada e, na bateria, o computador ainda poderá entrar em suspensão.',
    action: 'Manter ativo enquanto trabalha'
  },
  connectorOffer: {
    heading: 'Mantenha seus aplicativos conectados',
    body: 'Entre com uma conta Nous gratuita para manter na sua conta os aplicativos que você conectou, como Gmail ou Slack.'
  }
} satisfies Pick<
  Translations,
  'remoteDisplayBanner' | 'butterbar' | 'promptNotices' | 'keepAwakeOffer' | 'connectorOffer'
>
