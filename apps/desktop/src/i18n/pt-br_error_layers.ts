import type { Translations } from './types'

type ThreadCopy = Translations['assistant']['thread']

export const ptBrErrorLayers: Pick<ThreadCopy, 'errorLayerBodies' | 'errorLayers'> = {
  errorLayers: {
    auth: 'Problema no login',
    billing: 'Sem créditos',
    disk: 'Disco cheio',
    endpoint: 'Não foi possível acessar seu servidor de modelos',
    gateway: 'Não foi possível enviar sua mensagem',
    generic: 'O Hermes não conseguiu concluir esta resposta',
    provider: 'O serviço de IA retornou um erro',
    runtime: 'O Hermes encontrou um problema',
    streaming: 'A resposta foi interrompida'
  },
  errorLayerBodies: {
    auth: 'O serviço de IA rejeitou seu login. Verifique as credenciais deste provedor e envie sua mensagem novamente.',
    billing:
      'Sua conta não tem mais créditos neste provedor. Adicione créditos ou troque de provedor e envie novamente.',
    disk: 'Seu disco está cheio, por isso o Hermes não pôde salvar esta conversa. Libere espaço e tente novamente.',
    endpoint:
      'O Hermes não consegue acessar seu servidor de modelos personalizado. Verifique se ele está em execução e envie sua mensagem novamente.',
    gateway:
      'O Hermes encontrou um problema interno ao iniciar esta resposta. Envie sua mensagem novamente; se o problema persistir, envie os diagnósticos.',
    generic:
      'Algo deu errado enquanto o Hermes respondia. Tente novamente ou copie os detalhes se o problema persistir.',
    provider:
      'O serviço de IA não conseguiu concluir esta solicitação. Tente novamente em instantes ou troque de provedor.',
    runtime:
      'O Hermes encontrou um problema interno ao iniciar esta resposta. Envie sua mensagem novamente; se o problema persistir, envie os diagnósticos.',
    streaming: 'A conexão caiu antes da conclusão da resposta. Tente novamente para reenviá-la.'
  }
}
