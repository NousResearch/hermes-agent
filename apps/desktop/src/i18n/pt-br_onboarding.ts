import type { Translations } from './types'

export const ptBrOnboarding: Translations['onboarding'] = {
  headerTitle: 'Vamos configurar o Hermes Agent',
  headerDesc: 'Conecte um provedor de modelos para começar a conversar. A maioria das opções leva apenas um clique.',
  preparingInstall:
    'O Hermes está finalizando a instalação. Na primeira execução, isso costuma levar menos de um minuto.',
  starting: 'Iniciando o Hermes…',
  setupSlowTitle: 'A configuração está demorando mais que o normal.',
  setupSlowBody: 'O Hermes ainda está iniciando em segundo plano.',
  continueWithoutSetup: 'Continuar sem configurar',
  lookingUpProviders: 'Buscando provedores...',
  collapse: 'Recolher',
  otherProviders: 'Outros provedores',
  haveApiKey: 'Tenho uma chave de API',
  chooseLater: 'Vou escolher um provedor depois',
  recommended: 'Recomendado',
  connected: 'Conectado',
  featuredPitch: 'Uma assinatura, mais de 300 modelos de ponta — a forma recomendada de usar o Hermes',
  fireworksPitch: 'API direta de modelos — modelos de ponta hospedados pela Fireworks',
  localModelsTitle: 'Executar modelos localmente',
  localModelsPitch: 'Não precisa de conta — baixe um modelo e execute neste computador',
  openRouterPitch: 'Uma chave, centenas de modelos — uma boa opção padrão',
  apiKeyOptions: {
    fireworks: {
      short: 'API direta de modelos',
      description: 'Acesso direto aos modelos hospedados pela Fireworks AI.'
    },
    openrouter: {
      short: 'uma chave, muitos modelos',
      description: 'Oferece centenas de modelos com uma única chave. Uma boa opção padrão para novas instalações.'
    },
    openai: { short: 'modelos da classe GPT', description: 'Acesso direto aos modelos da OpenAI.' },
    gemini: { short: 'modelos Gemini', description: 'Acesso direto aos modelos Google Gemini.' },
    xai: { short: 'modelos Grok', description: 'Acesso direto aos modelos Grok da xAI.' },
    local: {
      short: 'hospedagem própria',
      description:
        'Conecte o Hermes a um endpoint compatível com OpenAI, local ou de hospedagem própria (vLLM, llama.cpp, Ollama etc.).'
    }
  },
  backToSignIn: 'Voltar para entrar',
  getKey: 'Obter uma chave',
  replaceCurrent: 'Substituir valor atual',
  pasteApiKey: 'Colar chave de API',
  localApiKeyPlaceholder: 'Chave de API (opcional — apenas se o endpoint exigir)',
  localModelNamePlaceholder: 'Nome do modelo (ex.: command-a-plus-05-2026)',
  couldNotSave: 'Não foi possível salvar a credencial.',
  connecting: 'Conectando',
  update: 'Atualizar',
  flowSubtitles: {
    pkce: 'Abre seu navegador para entrar e depois continua aqui',
    device_code: 'Abre uma página de verificação no navegador — o Hermes se conecta automaticamente',
    external: 'Entre uma vez pelo terminal e depois volte para conversar'
  },
  startingSignIn: provider => `Iniciando a entrada em ${provider}...`,
  verifyingCode: provider => `Verificando seu código com ${provider}...`,
  connectedProvider: provider => `${provider} conectado`,
  connectedPicking: provider => `${provider} conectado. Escolhendo um modelo padrão...`,
  signInFailed: 'Não foi possível entrar. Tente novamente.',
  signInExpired:
    'A página de entrada expirou antes de você terminar. Tente novamente e conclua a etapa no navegador em poucos minutos ou use uma chave de API.',
  signInDidNotFinish: provider =>
    `A entrada em ${provider} não foi concluída. Verifique sua conexão com a internet e tente novamente ou escolha outro provedor.`,
  tryAgain: 'Tentar novamente',
  useApiKeyInstead: 'Usar uma chave de API',
  errorDetails: 'Detalhes',
  pickDifferentProvider: 'Escolher outro provedor',
  signInWith: provider => `Entrar com ${provider}`,
  openedBrowser: provider => `Abrimos ${provider} no seu navegador.`,
  authorizeThere: 'Autorize o Hermes lá.',
  copyAuthCode: 'Copie o código de autorização e cole abaixo.',
  pasteAuthCode: 'Colar código de autorização',
  reopenAuthPage: 'Reabrir página de autorização',
  waitingAuthorize: 'Aguardando sua autorização...',
  externalPending: provider =>
    `${provider} usa sua própria CLI para entrar. Execute este comando em um terminal e depois volte e escolha "Já entrei":`,
  signedIn: 'Já entrei',
  deviceCodeOpened: provider => `Abrimos ${provider} no seu navegador. Insira este código lá:`,
  reopenVerification: 'Reabrir página de verificação',
  copy: 'Copiar',
  defaultModel: 'Modelo padrão',
  freeTier: 'Plano gratuito',
  pro: 'Pro',
  free: 'Grátis',
  price: (input, output) => `${input} de entrada / ${output} de saída por Mtok`,
  change: 'Alterar',
  startChatting: 'Começar',
  docs: provider => `Documentação de ${provider}`
}
