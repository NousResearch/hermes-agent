import type { Translations } from './types'

type ThreadCopy = Translations['assistant']['thread']

export const ptBrErrorCodes: ThreadCopy['errorCodes'] = {
  auth: {
    title: provider => `${provider} rejeitou sua chave de API`,
    body: () => 'A chave salva é inválida ou foi revogada. Atualize-a e envie sua mensagem novamente.'
  },
  auth_permanent: {
    title: provider => `${provider} rejeitou sua chave de API`,
    body: () => 'A chave salva é inválida ou foi revogada. Atualize-a e envie sua mensagem novamente.'
  },
  billing: {
    title: provider => `Sem créditos em ${provider}`,
    body: provider => `Adicione créditos para continuar usando modelos pagos em ${provider}.`
  },
  rate_limit: {
    title: provider => `${provider} está ocupado no momento`,
    body: provider => `Você atingiu o limite de solicitações de ${provider}. Aguarde um pouco e tente novamente.`
  },
  upstream_rate_limit: {
    title: (_provider, model) => `${model || 'Este modelo'} está ocupado no momento`,
    body: (provider, model) =>
      `${model || 'Este modelo'} está com limite de solicitações na infraestrutura que atende ${provider}, não na sua conta. Tente novamente em instantes.`
  },
  overloaded: {
    title: provider => `${provider} está ocupado no momento`,
    body: provider =>
      `${provider} está sobrecarregado e não pôde aceitar esta solicitação. Tente novamente em instantes.`
  },
  server_error: {
    title: provider => `${provider} está com problemas`,
    body: provider => `${provider} retornou um erro de servidor. Tente novamente em instantes.`
  },
  timeout: {
    title: provider => `Não foi possível acessar ${provider}`,
    body: provider => `O Hermes não conseguiu acessar ${provider}. Verifique sua conexão e tente novamente.`
  },
  stream_drop: {
    title: 'A resposta foi interrompida',
    body: 'A conexão caiu antes da conclusão da resposta. Tente novamente para reenviá-la.'
  },
  no_reply: {
    title: 'A resposta não chegou',
    body: 'O Hermes concluiu este turno, mas nenhuma resposta chegou a esta janela. Tente novamente para reenviá-la.'
  },
  upstream_blocked: {
    title: provider => `${provider} bloqueou esta solicitação`,
    body: provider =>
      `Um firewall na frente de ${provider} recusou a solicitação antes que ela chegasse ao modelo. Troque de provedor ou defina um User-Agent personalizado para ele.`
  },
  ssl_cert_verification: {
    title: provider => `Não foi possível verificar a conexão com ${provider}`,
    body: provider =>
      `Um proxy ou um repositório de certificados desatualizado bloqueou a conexão segura com ${provider}. O guia de configuração mostra como resolver.`
  },
  context_overflow: {
    title: 'Esta conversa está longa demais',
    body: (_provider, model) =>
      `A conversa não cabe mais na janela de contexto ${model ? `de ${model}` : 'do modelo'}. Comprima-a para continuar.`
  },
  payload_too_large: {
    title: 'Esta mensagem é grande demais',
    body: provider =>
      `A solicitação, incluindo os anexos, excede o limite de tamanho de ${provider}. Comprima a conversa ou remova anexos.`
  },
  model_not_found: {
    title: (_provider, model) => `${model || 'Este modelo'} não está disponível`,
    body: (provider, model) => `${provider} não oferece ${model || 'este modelo'} na sua conta. Escolha outro modelo.`
  },
  provider_policy_blocked: {
    title: (provider, model) => `As configurações de ${provider} bloqueiam ${model || 'este modelo'}`,
    body: provider =>
      `Suas configurações de privacidade em ${provider} excluem todos os endpoints deste modelo. Altere-as em ${provider} ou escolha outro modelo.`
  },
  content_policy_blocked: {
    title: provider => `${provider} recusou esta mensagem`,
    body: provider => `O filtro de segurança de ${provider} recusou a mensagem. Edite-a e envie novamente.`
  },
  format_error: {
    title: provider => `${provider} rejeitou esta solicitação`,
    body: provider =>
      `${provider} não conseguiu aceitar a estrutura desta conversa. Troque de provedor ou inicie uma nova conversa se o problema persistir.`
  },
  truncated: {
    title: 'A resposta terminou antes da hora',
    body: (_provider, model) =>
      `${model || 'O modelo'} parou antes de concluir a resposta. Continue para retomar de onde parou.`
  },
  invalid_response: {
    title: 'Não foi possível ler a resposta',
    body: provider => `${provider} retornou uma resposta vazia ou inválida. Tente novamente em instantes.`
  },
  empty_response: {
    title: 'Não foi possível obter uma resposta',
    body: provider => `${provider} não retornou nada para esta mensagem. Tente novamente em instantes.`
  },
  loop_error: {
    title: 'O Hermes não conseguiu concluir esta etapa',
    body: 'Algo deu errado ao processar a resposta. Tente novamente; se acontecer de novo, envie os diagnósticos pela seção de detalhes.'
  },
  SESSION_OWNER_BUSY: {
    title: 'O Hermes está ocupado nesta conversa',
    body: 'Outro turno está em execução aqui e sua mensagem não pôde entrar na fila. Nada foi enviado. Tente novamente quando esse turno terminar.'
  },
  cron_session_read_only: {
    title: 'Esta conversa pertence a uma tarefa agendada',
    body: 'Ela executa apenas essa tarefa, então sua mensagem não foi executada. Inicie uma nova conversa para enviá-la.'
  },
  session_read_only: {
    title: 'Esta conversa é somente leitura',
    body: 'É um histórico salvo que o Hermes pode exibir, mas não continuar, então sua mensagem não foi executada. Inicie uma nova conversa para enviá-la.'
  },
  workspace_unavailable: {
    title: 'A pasta desta conversa não está mais disponível',
    body: 'A pasta de trabalho desta conversa foi movida ou excluída, então sua mensagem não foi executada. Recrie a pasta ou use Mover para projeto para atribuir outra pasta à conversa e tente novamente.'
  },
  launch_credentials_unavailable: {
    title: 'A chave de API desta conversa mudou',
    body: 'Esta conversa começou com uma chave de API que foi alterada ou removida, então sua mensagem não foi enviada. Uma conversa aberta nunca muda para outra chave por conta própria. Inicie uma nova sessão para usar sua chave atual.'
  },
  local_runtime_unavailable: {
    title: 'Não foi possível acessar o servidor de modelos local',
    body: 'O servidor de modelos local ainda não está atendendo: ele pode estar iniciando. Tente novamente em instantes ou mude para outro modelo.',
    bodyNoRetry:
      'O servidor de modelos local está desativado. Inicie-o pelo aplicativo Hermes Desktop ou com `hermes model` em um terminal, ou mude para outro modelo.'
  },
  SESSION_OWNER_UNCONFIRMED: {
    title: 'Sua mensagem pode já estar na fila',
    body: 'O gateway não confirmou sua mensagem a tempo. Se ela entrou na fila, será executada quando o turno anterior terminar ou for resolvido. Verifique a conversa antes de enviar novamente.'
  },
  agent_init_failed: {
    title: 'Não foi possível iniciar esta conversa',
    body: provider =>
      `O Hermes não conseguiu se conectar a ${provider} para esta conversa. Tente novamente; se a falha persistir, verifique o provedor em Configurações.`
  },
  SESSION_NOT_OWNED: {
    title: 'Conversa aberta em outro lugar',
    body: 'Esta conversa está aberta em outra janela ou terminal do Hermes. Inicie uma nova conversa aqui ou feche-a lá primeiro.'
  },
  disk_full: {
    title: 'Seu disco está cheio',
    body: 'O Hermes não conseguiu salvar esta conversa. Libere espaço e tente novamente.'
  },
  free_tier_disabled: {
    title: 'Entre para continuar conversando',
    body: 'Entre com uma conta Nous para continuar conversando. É grátis.'
  },
  free_tier_rate_limited: {
    title: 'Limite gratuito atingido',
    body: 'Ele será renovado em breve. Entre com uma conta Nous para ter um limite maior. É grátis.'
  },
  free_tier_at_capacity: {
    title: 'O plano gratuito está ocupado',
    body: 'Entre para pular a fila, é grátis, ou tente novamente daqui a pouco.'
  },
  free_tier_model_not_free: {
    title: 'Este modelo exige login',
    body: 'O Hermes usa o modelo gratuito por enquanto. Entre com uma conta Nous para acessar mais modelos. É grátis.'
  },
  free_tier_route: {
    title: 'Não foi possível acessar o modelo gratuito',
    body: 'Entre com uma conta Nous, é grátis, ou verifique a configuração NOUS_INFERENCE_BASE_URL.'
  },
  free_tier_outage: {
    title: 'O modelo gratuito está com problemas',
    body: 'Tente enviar sua mensagem novamente em um minuto.'
  },
  free_tier_refused: {
    title: 'Entre para continuar conversando',
    body: 'Entrar com uma conta Nous é grátis.'
  },
  usage_window: {
    title: 'O uso incluído no plano acabou',
    body: 'O uso incluído no seu plano acabou até a próxima renovação. Adicione créditos em /topup para continuar.'
  }
}
