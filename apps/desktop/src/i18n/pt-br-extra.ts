import type { TranslationOverrides } from './define-locale'

export const ptBrExtra: TranslationOverrides = {
  settings: {
    subpages: { safetyCheckpoints: 'Pontos de restauração' },
    nav: { gateway: 'Gateways', mcp: 'MCP' },
    plugins: { kinds: { runtime: 'ambiente de execução' } },
    vault: { kinds: { login: 'Credencial de acesso' }, originPlaceholder: 'https://github.com' },
    notifications: { testTitle: 'Hermes' },
    sections: { chat: 'Chat' },
    connections: {
      kindLocal: 'Neste computador',
      kindCloud: 'Hermes Cloud',
      kindSsh: 'SSH'
    },
    gateway: {
      cloudTitle: 'Hermes Cloud',
      cloudSignInTitle: 'Entrar no Hermes Cloud',
      sshHostTitle: 'Host',
      sshHostPickTitle: 'Escolher host'
    },
    model: { tasks: { mcp: { label: 'MCP' } } },
    localModels: { quickstartStageEngine: 'Mecanismo', browseDownloads: 'downloads' },
    providers: {
      localEndpoint: {
        title: 'Endpoint local ou personalizado',
        description:
          'Conecte o Hermes a qualquer endpoint compatível com OpenAI (Zyphra, vLLM, llama.cpp, Ollama etc.).'
      }
    },
    sessions: {
      archivedIntro:
        'Chats arquivados ficam ocultos na barra lateral, mas mantêm todas as mensagens. Use Alt/⌥+Shift-clique em um chat na barra lateral para arquivá-lo.'
    },
    toolsets: {
      postSetupInstalledHint: 'Instalado. Execute a configuração novamente apenas se algo não estiver funcionando.',
      terminalBackend: {
        needsSetupHint:
          'Este backend está selecionado, mas não foi totalmente configurado. Os comandos falharão até que a configuração seja concluída.',
        needsSetupConfirmDescriptionGeneric:
          'Este backend ainda não foi configurado. As sessões iniciadas depois desta alteração não terão ferramentas de terminal ou arquivos até que a configuração seja concluída.',
        needsSetupConfirmAction: 'Selecionar mesmo assim',
        unavailableTitle: 'Os comandos do terminal estão indisponíveis',
        openBackendSettings: 'Abrir configurações do terminal'
      }
    }
  },
  starmap: {
    shareHint:
      'Copie o código para compartilhar este mapa ou cole um código para carregá-lo. Ele inclui apenas o layout, não o texto das suas memórias ou skills.'
  },
  webhooks: {
    hint: 'As alterações nas assinaturas são aplicadas automaticamente quando o receptor está em execução. Assinaturas desativadas recusam eventos recebidos.',
    disabledBody:
      'Webhooks são uma plataforma própria do gateway. Ative-os aqui para receber eventos HTTP. Canais de chat só são necessários se uma assinatura enviar conteúdo ao Telegram, Discord, Slack ou outro canal.',
    deleteDescPrefix: 'Isso removerá permanentemente ',
    deleteDescSuffix: '. Esta ação não pode ser desfeita.',
    restartNeeded: 'Os webhooks estão ativados, mas o gateway precisa ser reiniciado para iniciar o receptor.',
    enabledRestarting: 'Webhooks ativados; reiniciando o gateway…',
    fieldSkills: 'Skills',
    fieldPrompt: 'Prompt',
    deliverOptions: { log: 'Log', telegram: 'Telegram', discord: 'Discord', slack: 'Slack' }
  },
  cron: {
    deliveryLabels: { telegram: 'Telegram', discord: 'Discord', slack: 'Slack' },
    emptyDescNew:
      'Agende um prompt para ser executado segundo uma expressão cron. O Hermes executará a tarefa e enviará os resultados ao destino escolhido.',
    emptyDescSearch: 'Tente buscar por outros termos.',
    resumeTitle: 'Retomar',
    pauseTitle: 'Pausar',
    deleteDescPrefix: 'Isso removerá permanentemente ',
    deleteDescSuffix: '. A tarefa deixará de ser executada imediatamente.',
    editDesc:
      'Atualize o agendamento, o prompt ou o destino de entrega. As alterações serão aplicadas na próxima execução.',
    createDesc:
      'Agende um prompt para ser executado automaticamente. Use a sintaxe cron ou uma frase como “a cada 15 minutos”.',
    promptLabel: 'Prompt',
    promptPlaceholder: 'Resuma minhas conversas não lidas no Slack e envie por e-mail os cinco pontos principais…',
    deliverNeedsHomeChannel: 'defina primeiro um canal principal',
    customScheduleLabel: 'Agendamento personalizado',
    customPlaceholder: '0 9 * * * ou dias úteis às 9h',
    customHint: 'Expressão cron ou frases como “a cada hora” ou “dias úteis às 9h”.',
    scriptOnlyEditHint: 'Tarefa somente com script (sem prompt de IA). ID da tarefa:',
    blueprints: {
      dialogDesc: 'Preencha os detalhes e configure o agendamento.',
      emptyDesc: 'Nenhum modelo de automação está disponível neste backend.'
    }
  },
  artifacts: {
    tabLinks: 'Links',
    itemsLink: 'links',
    colLocationLink: 'URL',
    kindLink: 'link'
  },
  handoffTour: {
    profileTitle: 'Sua primeira tarefa é executada no perfil padrão',
    profileText:
      'Esta barra alterna entre perfis. O perfil destacado é o padrão, onde fica a sessão da tarefa. O outro é o perfil de configuração, onde fica o chat de boas-vindas.',
    sessionsTitle: 'Cada perfil mantém suas próprias sessões',
    sessionsText:
      'Esta lista pertence ao perfil padrão. Uma nova sessão é iniciada no perfil selecionado. Troque de perfil na barra para ver a lista correspondente.',
    stayTitle: 'O Hermes está sempre a um clique de distância',
    stayText:
      'Troque para o perfil de configuração e abra Boas-vindas ao Hermes sempre que precisar de ajuda. O chat continuará lá.'
  },
  guidedGreeting: {
    line: 'Oi, que bom ter você aqui. Sou o Hermes. Vou levar dois minutos para deixar tudo do seu jeito e, depois, você pode me passar uma tarefa de verdade.\n\nAntes de começar, como devo chamar você?',
    nameSuggestion: name => `(Também posso chamar você de ${name}, se preferir.)`
  },
  onboarding: {
    signInExpired:
      'A página de login expirou antes de você concluir. Tente novamente e finalize a etapa no navegador em alguns minutos ou use uma chave de API.'
  },
  assistant: {
    thread: {
      errorLayerBodies: {
        gateway:
          'O Hermes encontrou um problema interno ao iniciar esta resposta. Envie sua mensagem novamente; se acontecer de novo, envie os diagnósticos.',
        runtime:
          'O Hermes encontrou um problema interno ao iniciar esta resposta. Envie sua mensagem novamente; se acontecer de novo, envie os diagnósticos.'
      },
      errorCodes: {
        server_error: { title: 'O serviço de IA teve um problema' },
        upstream_blocked: { title: 'Um firewall bloqueou a solicitação' },
        ssl_cert_verification: { title: 'Falha na conexão segura' },
        provider_policy_blocked: { title: 'Sua conta não permite usar este modelo' },
        format_error: { title: 'O serviço de IA recusou a solicitação' },
        invalid_response: { title: 'O serviço de IA enviou uma resposta ilegível' },
        empty_response: { title: 'O serviço de IA enviou uma resposta vazia' },
        loop_error: {
          title: 'O Hermes entrou em um ciclo',
          body: 'A resposta repetiu as mesmas etapas, então o Hermes parou. Tente novamente ou inicie um chat novo se isso acontecer outra vez.'
        },
        free_tier_disabled: {
          title: 'O uso do Hermes sem login está desativado no momento',
          body: 'Entre com uma conta Nous para continuar conversando. É grátis.'
        },
        free_tier_rate_limited: {
          title: 'Você atingiu o limite para conversar sem login',
          body: 'O limite será renovado em breve. Entre com uma conta Nous para ter um limite maior. É grátis.'
        },
        free_tier_at_capacity: {
          title: 'O chat sem login está com muita demanda agora',
          body: 'Entre para pular a fila, sem custo, ou tente novamente daqui a pouco.'
        },
        free_tier_model_not_free: {
          title: 'Este modelo não está disponível sem login',
          body: 'Por enquanto, o Hermes está usando o modelo gratuito. Entre com uma conta Nous para acessar mais modelos. É grátis.'
        },
        free_tier_route: {
          title: 'O Hermes não conseguiu acessar o modelo gratuito por esta rota',
          body: 'Entre com uma conta Nous, sem custo, ou confira a configuração NOUS_INFERENCE_BASE_URL.'
        },
        free_tier_outage: {
          title: 'O modelo gratuito está com dificuldades para responder',
          body: 'Tente enviar sua mensagem novamente daqui a um minuto.'
        },
        free_tier_refused: {
          title: 'O Hermes não conseguiu enviar isso sem login',
          body: 'Entrar com uma conta Nous é grátis.'
        }
      },
      errorDetails: 'Detalhes',
      errorGenericProvider: 'O serviço de IA',
      errorToastTitle: 'O Hermes não conseguiu concluir a resposta',
      errorRetry: 'Tentar novamente',
      errorRetryScheduledCancel: 'Cancelar',
      errorStartNewSession: 'Iniciar nova sessão',
      errorSwitchProvider: 'Trocar de provedor',
      errorChooseModel: 'Escolher modelo',
      errorCompressConversation: 'Compactar conversa',
      errorCompressFailed: 'Não foi possível compactar a conversa',
      errorOpenHermesFolder: 'Abrir pasta do Hermes',
      errorOpenHermesFolderFailed: 'Não foi possível abrir a pasta do Hermes',
      errorUpdateApiKey: 'Atualizar chave de API',
      errorSignInFreeTier: 'Entrar com uma conta Nous',
      errorOpenLogs: 'Abrir logs',
      errorOpenLogsFailed: 'Não foi possível abrir a pasta de logs',
      errorOpenDesktopLogs: 'Abrir logs do Desktop',
      errorCopyDiagnostics: 'Copiar detalhes do erro',
      errorSendDiagnostics: 'Enviar diagnósticos',
      reviewChanges: 'Revisar',
      readAloudFailed: 'Falha na leitura em voz alta',
      preparingAudio: 'Preparando áudio…',
      stopReading: 'Parar leitura',
      readAloud: 'Ler em voz alta',
      editMessage: 'Editar mensagem',
      expandMessage: 'Expandir mensagem',
      scrollToBottom: 'Rolar até o final',
      stop: 'Parar',
      restorePrevious: 'Restaurar checkpoint anterior',
      restoreCheckpoint: 'Restaurar checkpoint',
      restoreFromHere: 'Restaurar checkpoint e executar novamente a partir deste prompt',
      restoreTitle: 'Restaurar a partir deste checkpoint?',
      restoreBody:
        'Tudo o que vier depois deste prompt será removido da conversa, e o prompt será executado novamente a partir daqui.',
      restoreConfirm: 'Restaurar e executar novamente',
      restoreNext: 'Restaurar próximo checkpoint',
      goForward: 'Avançar',
      sendEdited: 'Enviar mensagem editada',
      attachingFile: 'Anexando…'
    },
    approval: {
      gatewayDisconnected:
        'O Hermes está offline. O comando continuará aguardando sua resposta até o limite de aprovação. Reconecte e envie-o novamente.',
      timedOutSystemLine:
        'A aprovação expirou; o comando não foi executado. Peça ao Hermes para tentar novamente ou aumente o tempo limite em Configurações → Segurança → Tempo limite de aprovação.'
    },
    clarify: {
      gatewayDisconnected: 'O Hermes está offline. Reconecte e envie a mensagem novamente.',
      lateAnswerTip: 'Escreva esta resposta como uma mensagem de continuação',
      lateAnswerHint:
        'Este prompt não está mais aguardando resposta. Escolha uma opção para preparar uma mensagem de continuação.'
    },
    catalogInstall: { commitLabel: 'Commit', subdirLabel: 'Pasta' },
    mcpSetup: { gatewayDisconnected: 'O Hermes está offline. Reconecte e envie a mensagem novamente.' },
    tool: {
      skillActivity: {
        loading: 'Carregando skill',
        loaded: 'Skill carregada',
        loadFailed: 'Falha ao carregar skill',
        readingResource: 'Lendo recurso da skill',
        readResource: 'Recurso da skill lido',
        resourceFailed: 'Falha ao ler recurso da skill',
        listing: 'Listando skills',
        listed: 'Skills listadas',
        listFailed: 'Falha ao listar skills',
        unavailable: 'Resultado da skill indisponível'
      },
      prefixes: { web: 'Web' }
    }
  },
  prompts: {
    gatewayDisconnected: 'O Hermes está offline. Reconecte e envie a mensagem novamente.',
    reconnect: 'Reconectar',
    sudoSendFailed: 'Não foi possível enviar a senha de administrador',
    secretSendFailed: 'Não foi possível enviar o segredo',
    sudoTitle: 'Senha de administrador',
    sudoDesc:
      'Confira o comando antes de informar sua senha sudo. A senha será enviada ao agente que vai executá-lo e ficará em cache durante esta sessão.',
    sudoCommandUnavailable: 'Este agente não informou o comando. Cancele se não puder verificá-lo na conversa.',
    sudoInstallDesc:
      'O Hermes precisa da sua senha sudo para instalar os pacotes Bot Screen (TigerVNC + Xfce) no host do gateway. Ela será enviada apenas para esse host.',
    sudoPlaceholder: 'Senha sudo',
    secretTitle: 'Credencial necessária',
    secretDesc: 'O Hermes precisa de uma credencial para continuar.',
    secretPlaceholder: 'Valor do segredo',
    vaultUnlockSendFailed: 'Não foi possível enviar a senha mestra',
    vaultUnlockPlaceholder: 'Senha mestra',
    vaultUnlockKeepLocked: 'Manter bloqueado',
    vaultUnlockConfirm: 'Desbloquear',
    vaultSaveSendFailed: 'Não foi possível salvar o login',
    vaultSaveIdentifierLabel: 'E-mail ou nome de usuário',
    vaultSaveIdentifierPlaceholder: 'voce@example.com',
    vaultSavePasswordPlaceholder: 'Senha',
    vaultSaveFootnote: 'Gerencie os logins salvos em Configurações → Senhas e logins.',
    vaultSaveDecline: 'Não salvar',
    vaultSaveConfirm: 'Salvar e entrar',
    vaultCodeSendFailed: 'Não foi possível enviar o código',
    vaultCodeLabel: 'Código',
    vaultCodeFootnote:
      'Dica: salve a chave do autenticador com este login em Configurações → Senhas e logins para que o Hermes preencha os códigos.',
    vaultCodeSkip: 'Pular',
    vaultCodeConfirm: 'Informar código'
  },
  shell: {
    modelOptions: { ultra: 'Ultra' },
    gatewayMenu: { gateway: 'Gateway' },
    approvalMode: { manual: 'Manual' },
    statusbar: {
      gateway: 'Gateway',
      gatewayTitle: 'Gateway',
      toggleCacheHitRate: 'Taxa de acerto do cache',
      toggleTerminal: 'Terminal',
      toggleTokensPerSecond: 'Tokens por segundo',
      cacheHitRateTitle:
        'Taxa de acerto do cache de prompt nesta sessão. Tokens em cache custam menos; quanto maior, menor o custo.',
      tokensPerSecondTitle: 'Tokens de saída por segundo, em média nas últimas 10 chamadas ao modelo',
      webhooks: 'Webhooks',
      systemResources: {
        ram: 'RAM',
        unifiedNote: 'Memória unificada: a GPU e o sistema compartilham este espaço.'
      },
      contextUsagePanel: { categories: { mcp: 'MCP', skills: 'Skills' } },
      yoloOn: 'YOLO ativado: aprovação automática de comandos perigosos. Shift-clique alterna globalmente.',
      yoloOff: 'YOLO desativado. Shift-clique alterna globalmente.',
      modelNone: 'nenhum',
      noModel: 'nenhum modelo',
      switchModel: 'Trocar modelo',
      openModelPicker: 'Abrir seletor de modelos',
      modelPinned: 'fixado por você; novos chats usarão este modelo em vez do padrão das Configurações'
    }
  },
  messaging: {
    sharedListenerUrl: 'Disponível no listener compartilhado do gateway em',
    noTokenNeeded: 'Esta plataforma não precisa de um token aqui. Use o guia de configuração acima e ative-a abaixo.',
    approvedHint: 'A pessoa será reconhecida automaticamente na próxima mensagem.',
    pairingLockedOut:
      'Houve muitas tentativas de aprovação. Esta plataforma foi bloqueada; tente novamente mais tarde.',
    restartFailedManual: 'O Hermes não conseguiu reiniciar para aplicar as configurações de mensagens.',
    restartFailedManualDetail:
      'Tente reiniciar novamente. Se ainda não funcionar, abra os logs e envie os diagnósticos.',
    telegramQr: {
      replaceWarning:
        'As credenciais do Telegram já estão configuradas. Uma nova configuração por QR code ou token de bot substituirá o bot atual quando você salvar.'
    },
    fieldCopy: {
      TELEGRAM_PROXY: { label: 'URL do proxy', help: 'Necessário apenas em redes que bloqueiam o Telegram.' },
      DISCORD_ALLOWED_USERS: {
        label: 'IDs de usuários permitidos no Discord',
        help: 'Recomendado. Separe os IDs de usuários do Discord por vírgulas.'
      },
      DISCORD_REPLY_TO_MODE: {
        label: 'Estilo das respostas',
        help: 'first (primeiro), all (todos) ou off (desativado).'
      },
      DISCORD_ALLOW_ALL_USERS: {
        label: 'Permitir todos os usuários do Discord',
        help: 'Apenas para desenvolvimento. Quando ativado, qualquer pessoa pode enviar mensagens diretas ao bot, sem lista de permissão.'
      },
      DISCORD_HOME_CHANNEL: {
        label: 'ID do canal principal',
        help: 'Canal em que o bot envia mensagens proativas, como resultados de tarefas agendadas e lembretes.'
      },
      DISCORD_HOME_CHANNEL_NAME: {
        label: 'Nome do canal principal',
        help: 'Nome exibido nos logs e no status do canal principal.'
      },
      BLUEBUBBLES_ALLOW_ALL_USERS: {
        label: 'Permitir todos os usuários do iMessage',
        help: 'Quando ativado, ignora a lista de permissão do BlueBubbles.'
      },
      MATTERMOST_ALLOW_ALL_USERS: { label: 'Permitir todos os usuários do Mattermost' },
      MATTERMOST_HOME_CHANNEL: { label: 'Canal principal' },
      QQ_ALLOW_ALL_USERS: { label: 'Permitir todos os usuários do QQ' },
      QQBOT_HOME_CHANNEL: {
        label: 'Canal principal do QQ',
        help: 'Canal ou grupo padrão para entrega de tarefas agendadas.'
      },
      QQBOT_HOME_CHANNEL_NAME: { label: 'Nome do canal principal do QQ' },
      SLACK_BOT_TOKEN: { help: 'Use o token do bot em OAuth e permissões, depois de instalar seu app do Slack.' },
      SLACK_APP_TOKEN: {
        label: 'Token do app do Slack',
        help: 'Use o token no nível do app, necessário para o Socket Mode.',
        placeholder: 'Cole o token do app do Slack'
      },
      SLACK_ALLOWED_USERS: {
        label: 'IDs de usuários permitidos no Slack',
        help: 'Recomendado. Separe os IDs de usuários do Slack por vírgulas.'
      },
      MATTERMOST_URL: { label: 'URL do servidor', placeholder: 'https://mattermost.example.com' },
      MATTERMOST_TOKEN: { label: 'Token do bot' },
      MATTERMOST_ALLOWED_USERS: {
        label: 'IDs de usuários permitidos',
        help: 'Recomendado. Separe os IDs de usuários do Mattermost por vírgulas.'
      },
      MATRIX_HOMESERVER: { label: 'URL do homeserver', placeholder: 'https://matrix.org' },
      MATRIX_ACCESS_TOKEN: { label: 'Token de acesso' },
      MATRIX_USER_ID: { label: 'ID do usuário do bot', placeholder: '@hermes:example.org' },
      MATRIX_ALLOWED_USERS: {
        label: 'IDs de usuários permitidos no Matrix',
        help: 'Recomendado. Separe por vírgulas os IDs no formato @usuario:servidor.'
      },
      SIGNAL_HTTP_URL: {
        label: 'URL da ponte Signal',
        placeholder: 'http://127.0.0.1:8080',
        help: 'URL de uma ponte signal-cli REST em execução.'
      },
      SIGNAL_ACCOUNT: { label: 'Número de telefone', help: 'Número registrado na ponte signal-cli.' },
      SIGNAL_ALLOWED_USERS: {
        label: 'Usuários permitidos no Signal',
        help: 'Recomendado. Separe os identificadores do Signal por vírgulas.'
      },
      WHATSAPP_ENABLED: {
        label: 'Ativar ponte do WhatsApp',
        help: 'Definido automaticamente pelo botão abaixo. Altere apenas se souber que é necessário.'
      },
      WHATSAPP_MODE: { label: 'Modo da ponte' },
      WHATSAPP_ALLOWED_USERS: {
        label: 'Usuários permitidos no WhatsApp',
        help: 'Recomendado. Separe por vírgulas os números de telefone ou IDs do WhatsApp.'
      }
    }
  },
  skills: {
    tabSkills: 'Skills',
    visionModelHint:
      'A visão usa a configuração do modelo auxiliar. O modelo com suporte a imagens é escolhido lá, e não em cada provedor.',
    provenance: { hub: 'Catálogo' },
    tabPlugins: 'Plugins',
    plugins: {
      halfDesktop: 'Desktop',
      kindDesktop: 'Desktop',
      installAgentHereNoOrigin:
        'A parte do agente não está instalada neste perfil, e este pacote foi copiado manualmente (sem entrada no catálogo ou repositório Git remoto). Não é possível instalá-lo daqui. Copie a pasta para o perfil ou reinstale pelo Git.',
      desktopHalfPending: 'copiando…',
      desktopHalfPendingTip:
        'Este pacote inclui uma parte para desktop que ainda não foi copiada para o app. Use Verificar novamente ou reinicie o app.',
      desktopHalfRemote: 'indisponível (backend remoto)',
      desktopHalfRemoteTip:
        'A parte para desktop deste pacote está no disco do backend remoto e este app não consegue acessá-la. Para usá-la aqui, escolha Instalar pelo Git, informe a URL do repositório e marque o destino Desktop. Isso copiará a parte para desktop para este computador.',
      legacyBackend:
        'Este backend é antigo e não oferece controles de plugins por chave. Atualize o Hermes para gerenciá-lo aqui.',
      serverStates: {
        connected: 'conectado',
        app_not_running: 'app não está em execução',
        endpoint_unavailable: 'endpoint indisponível',
        no_interactive_session: 'sem sessão interativa',
        version_too_old: 'versão antiga demais',
        missing_app: 'app ausente',
        unknown: 'status desconhecido'
      },
      catalogHint:
        'Use “Adicionar a este agente” em qualquer plugin. Itens revisados são instalados no perfil selecionado, no commit fixado. Plugins incluídos com parte para agente e desktop oferecem as duas versões.',
      deepLinkErrorTitle: 'Link de instalação do plugin recusado',
      deepLinkCatalogInvalidName: 'O nome do catálogo no link está ausente ou é inválido.',
      deepLinkCatalogUnavailable:
        'Não foi possível carregar o catálogo de plugins Hermes. Confira sua conexão e abra o link novamente.',
      settingsForm: { secretSet: '•••••••• (definido)' }
    },
    hub: {
      landingHint:
        'Pesquise no catálogo para encontrar skills instaláveis no índice oficial, no GitHub e na comunidade.',
      updateStarted: 'Atualizando skills instaladas…',
      actionFailed: 'Falha ao executar a ação da skill',
      pickerHint: 'Use “Adicionar a este agente” em qualquer skill. Ela será instalada e aparecerá na lista acima.'
    }
  },
  freeTier: {
    providerName: 'Nous',
    signInHeading: 'Entre com uma conta Nous para liberar mais modelos e ferramentas.',
    finishingBody: 'Aprovação concluída no navegador. Obtendo os tokens da sua conta.',
    completedBody: 'Sua conta agora tem acesso a inferência e ferramentas.',
    didNotComplete: 'O login não foi concluído',
    rejectedBody: 'Tudo bem; você continua no serviço gratuito da Nous. Entre quando quiser.',
    supersededBody: 'Um código de login mais recente substituiu este. Use o mais recente ou comece de novo.',
    timedOutHeading: 'Este link de login expirou',
    timedOutBody: 'Comece novamente quando quiser. Você continua no serviço gratuito da Nous.',
    retiredBody:
      'Sua sessão terminou antes do login ser concluído. O Hermes iniciará outra; depois, entre quando quiser.',
    errorBody: 'O login não foi concluído. Tente novamente quando quiser.',
    busyHeading: 'Quase lá',
    unreachableBody:
      'O Hermes não conseguiu acessar o serviço da Nous para concluir seu login. Confira sua conexão e tente novamente. Sua sessão continua disponível.',
    setupFailed: {
      gateClosed:
        'Esta versão do Hermes precisa de uma conta Nous para iniciar. Entre ou crie uma conta gratuita; leva apenas um minuto.',
      paused:
        'O uso do Hermes sem login está temporariamente pausado. O Hermes continuará verificando. Entrar é grátis e permite começar agora.',
      unreachable:
        'O Hermes não conseguiu acessar o serviço da Nous. Confira sua conexão e toque em Tentar novamente. Você também pode conectar outro provedor por enquanto.',
      serverError:
        'O serviço da Nous teve um problema. Toque em Tentar novamente daqui a pouco ou conecte outro provedor por enquanto.',
      powRequired:
        'O servidor Nous solicitou uma prova de trabalho, mas seu Agent ainda não oferece esse recurso. Entre ou crie uma conta Nous gratuita para continuar.',
      locked: 'Esta sessão precisa de login para continuar. Entre ou crie uma conta Nous gratuita.',
      generic:
        'O Hermes não conseguiu configurar o acesso gratuito sem login. Entrar é grátis; você também pode conectar outro provedor.',
      signInBelow: 'Entrar é grátis. Escolha Nous abaixo.',
      tryAgain: 'Tentar novamente',
      retrying: 'Tentando novamente…'
    }
  },
  updates: {
    discontinuedTitle: 'Esta versão do Hermes não recebe mais suporte',
    discontinuedBody:
      'Esta versão do Hermes não recebe mais suporte e pode parar de funcionar. Desinstale-a. Seus dados continuarão no disco.',
    channels: { stable: 'Estável', canary: 'Canary' },
    bundleSwapPending: 'Reinicie para concluir a atualização',
    bundleSwapPendingDesc:
      'O app atualizado já está instalado — o Hermes só precisa reiniciar para carregá-lo. Conversas e configurações não são afetadas.',
    bundleSwapPendingAction: 'Reiniciar o Hermes',
    stages: {
      idle: 'Preparando…',
      prepare: 'Preparando…',
      fetch: 'Baixando…',
      pull: 'Quase pronto…',
      pydeps: 'Finalizando…',
      update: 'Atualizando o Hermes…',
      rebuild: 'Reconstruindo o app para desktop…',
      restart: 'Reiniciando o Hermes…',
      done: 'Atualização concluída',
      manual: 'Atualize pelo terminal',
      guiSkew: 'Atualize o app para desktop',
      error: 'Atualização pausada'
    },
    checking: 'Buscando atualizações…',
    checkFailedTitle: 'Não foi possível buscar atualizações',
    tryAgain: 'Tentar novamente',
    notAvailableTitle: 'Atualização indisponível',
    unsupportedMessage: 'Esta versão do Hermes não pode ser atualizada pelo próprio app.',
    connectionRetry:
      'O Hermes não conseguiu acessar o servidor de atualizações. Confira sua conexão com a internet e tente novamente. Se você usa um Hermes remoto, confirme se ele está online.',
    gitUnusable: 'O Hermes não conseguiu executar o Git neste computador para buscar atualizações.',
    connectionSettings: 'Configurações de conexão',
    openDownloadPage: 'Abrir página de download',
    latestBody: 'Você está usando a versão mais recente.',
    latestBodyBackend: 'O backend está usando a versão mais recente.',
    allSetTitle: 'Tudo pronto',
    availableTitle: 'Nova atualização disponível',
    availableBody: 'Uma nova versão do Hermes está pronta para instalar.',
    availableTitleBackend: 'Atualização do backend disponível',
    availableBodyBackend: 'Há uma nova versão do backend Hermes conectado, pronta para instalar.',
    availableBodyNoChangelog:
      'Há uma nova versão, mas as notas de lançamento não estão disponíveis para este tipo de instalação.',
    availableBodyAppInstaller:
      'Uma nova versão do Hermes está pronta. O Hermes será fechado, o Windows concluirá a atualização e o app abrirá novamente sozinho.',
    updateNow: 'Atualizar agora',
    maybeLater: 'Talvez depois',
    moreChanges: count => `+ mais ${count} alteraç${count === 1 ? 'ão incluída' : 'ões incluídas'}.`,
    copyFullLog: 'Copiar todas as notas da versão',
    manualTitle: 'Atualize pelo terminal',
    manualUnavailableTitle: 'Não é possível atualizar por aqui',
    manualBody:
      'Você instalou o Hermes pela linha de comando, então as atualizações também são feitas por lá. Cole este comando no terminal:',
    manualBodyBackend: 'O backend do Hermes é gerenciado fora deste app. Execute isto no servidor que o hospeda:',
    manualPickedUp: 'O Hermes usará a nova versão na próxima vez que for iniciado.',
    manualPickedUpBackend: 'O backend usará a nova versão após a conclusão da atualização.',
    guiSkewTitle: 'Atualize o app para desktop',
    guiSkewBody:
      'O backend foi atualizado, mas o pacote deste app para desktop não. Atualize ou reinstale o Hermes Desktop (AppImage, .deb ou .rpm) para manter as versões alinhadas.',
    copy: 'Copiar',
    copied: 'Copiado',
    done: 'Concluído',
    applyingBody:
      'O atualizador do Hermes abrirá em uma janela própria e reabrirá o app automaticamente quando terminar. Não abra o Hermes enquanto a atualização estiver em andamento.',
    applyingBodyBackend:
      'O backend remoto está aplicando a atualização e será reiniciado. O Hermes se conectará novamente quando ele voltar.',
    applyingClose: 'Esta janela será fechada durante a atualização, e o Hermes será reaberto automaticamente.',
    applyingBodyAppInstaller:
      'O Hermes será fechado e o Windows concluirá a atualização. O app abrirá novamente ao terminar — você não precisa fazer nada.',
    applyingCloseAppInstaller:
      'Esta janela será fechada, o Windows concluirá a atualização e o Hermes abrirá novamente.',
    checkUnknownTitleAppInstaller: 'Não foi possível verificar se há atualizações',
    checkUnknownBodyAppInstaller:
      'O Windows não conseguiu verificar se há atualizações agora. Elas também são instaladas automaticamente quando você reinicia o Hermes.',
    errorTitle: 'A atualização não foi concluída',
    errorBody: 'Não se preocupe: nada foi perdido. Você pode tentar novamente agora.',
    blockerTitle: 'Fechar prévias locais para atualizar o Hermes?',
    blockerBody:
      'O Hermes precisa encerrar estas prévias locais antes de atualizar. Seus arquivos não serão alterados nem excluídos.',
    foreignBlockerTitle: 'Feche outros processos para atualizar o Hermes',
    foreignBlockerBody:
      'O Hermes não pode encerrar estes processos com segurança. Feche o app, terminal ou serviço responsável e tente novamente.',
    mixedBlockerBody:
      'O Hermes pode encerrar as prévias locais listadas abaixo. Os outros processos precisam ser encerrados manualmente antes de continuar.',
    closePreviewsAndUpdate: 'Fechar prévias e atualizar',
    closePreviewsAndCheckAgain: 'Fechar prévias e verificar novamente',
    localPreview: 'Prévia local',
    portLabel: port => `Porta ${port}`,
    pidLabel: pid => `PID ${pid}`,
    technicalDetails: 'Detalhes técnicos',
    notNow: 'Agora não',
    clientAlsoBehindTitle: 'O app para desktop está desatualizado',
    clientAlsoBehindMessage:
      'O backend está atualizado, mas este app para desktop ainda usa uma versão anterior. Atualize o app para receber as correções mais recentes.',
    clientAlsoBehindAction: 'Atualizar app para desktop',
    everythingDispatched: 'Atualização iniciada',
    everythingSkipped: 'Ignorada',
    everythingRowFailed: 'Falha na atualização',
    everythingFanoutFailedTitle: 'Não foi possível atualizar outras instâncias',
    changeLogNew: 'Novidades',
    changeLogFixed: 'Corrigido',
    changeLogFaster: 'Mais rápido',
    changeLogImproved: 'Melhorado',
    changeLogOther: 'Outras melhorias',
    changeLogFallbackLabel: 'Nesta atualização',
    changeLogFallbackItem: 'Melhorias e correções',
    applyStatus: {
      preparing: 'Atualizando o backend…',
      pulling: 'Backend em atualização…',
      restarting: 'Reiniciando o backend para carregar a atualização…',
      notAvailable: 'A atualização não está disponível para este backend.',
      failed: 'Falha ao atualizar o backend.',
      noReturn:
        'O backend não voltou a ficar online. Talvez a atualização não tenha sido concluída. Confira o computador que hospeda o backend.'
    },
    appName: 'Hermes',
    version: value => `Versão ${value}`,
    versionUnavailable: 'Versão indisponível',
    checkNow: 'Verificar agora',
    seeWhatsNew: 'Ver novidades',
    releaseNotes: 'Notas da versão',
    onLatest: 'Você está na versão mais recente.',
    installing: 'Uma atualização está sendo instalada.',
    cantReach: 'Não foi possível acessar o servidor de atualizações.',
    tapCheck: 'Selecione “Verificar agora” para procurar atualizações.',
    updateReady: count =>
      `Uma nova atualização está pronta (${count} ${count === 1 ? 'alteração incluída' : 'alterações incluídas'}).`,
    updateReadyUnknown: 'Uma nova atualização está pronta.',
    availableBodyRelease: tag => `A versão ${tag} está pronta para instalar.`,
    lastChecked: age => `Última verificação ${age}`,
    never: 'nunca',
    justNow: 'agora mesmo',
    minAgo: count => `há ${count} min`,
    hoursAgo: count => `há ${count} ${count === 1 ? 'hora' : 'horas'}`,
    daysAgo: count => `há ${count} ${count === 1 ? 'dia' : 'dias'}`,
    justNowSuffix: ' · agora mesmo',
    bundleOutOfSync: 'Versão do app desatualizada',
    bundleOutOfSyncDesc:
      'O runtime do Hermes foi atualizado, mas o app para desktop ainda é uma versão antiga. Atualize o app para receber as correções mais recentes.',
    bundleOutOfSyncAction: 'Baixar o instalador',
    checkingShort: 'Verificando…',
    releaseAvailable: tag => `A versão ${tag} está disponível.`,
    versionDetailsTitle: 'Detalhes da versão',
    versionDetailsBody: 'Esta instalação é gerenciada fora do app. Atualize-a da mesma forma que a instalou.',
    versionDetailsVersion: 'Versão',
    versionDetailsCommit: 'Commit',
    versionDetailsBuildOrigin: 'Origem da compilação',
    versionDetailsDistribution: 'Distribuição',
    versionDetailsDistributionDesktop: 'App para desktop',
    versionDetailsDistributionDesktopMsix: 'App para desktop (MSIX)',
    versionDetailsDistributionDesktopInstaller: 'App para desktop (instalador)',
    versionDetailsDistributionSourceInstaller: 'Código-fonte (script de instalação)',
    versionDetailsDistributionSourceInstallerDesktop: 'Código-fonte (script de instalação) + Hermes Desktop',
    versionDetailsDistributionSource: 'Código-fonte',
    versionDetailsDistributionSourceDesktop: 'Código-fonte + Hermes Desktop',
    versionDetailsDistributionStore: 'Microsoft Store',
    versionDetailsRuntime: 'Ambiente de execução',
    versionDetailsRuntimeEmbedded: 'Ambiente de execução integrado',
    versionDetailsRuntimeExternal: 'Externo (usa o ambiente de execução do computador)',
    versionDetailsInstallId: 'ID da instalação',
    versionDetailsUncommittedChanges: 'alterações não confirmadas'
  },
  install: {
    stageStates: {
      pending: 'Pendente',
      running: 'Instalando',
      succeeded: 'Concluído',
      skipped: 'Ignorado',
      failed: 'Falhou'
    },
    oneTimeTitle: 'O Hermes precisa ser instalado uma vez',
    unsupportedDesc: platform =>
      `A instalação automática na primeira inicialização ainda não está disponível em ${platform}. Abra o Terminal, execute o comando abaixo e inicie o app novamente. Nas próximas vezes, esta etapa será ignorada.`,
    installCommand: 'Comando de instalação',
    copyCommand: 'Copiar comando',
    viewDocs: 'Ver documentação de instalação',
    installTo: 'Será instalado em',
    retryAfterRun: 'Já executei — tentar novamente',
    setupChoiceTitle: 'Configurar o Hermes Desktop',
    setupChoiceDesc:
      'Conecte este app a um gateway Hermes que já esteja em execução ou instale o Hermes neste computador.',
    connectExistingTitle: 'Conectar a um Hermes existente',
    connectExistingShort: 'Conectar a um existente',
    connectExistingDesc:
      'Use um backend remoto com token de sessão ou entre pelo navegador. Nenhuma instalação local será iniciada.',
    installLocalTitle: 'Instalar o Hermes localmente',
    installLocalDesc: 'Baixe o Hermes, crie o ambiente Python e execute o backend neste computador.',
    localStartUnavailable: 'Não foi possível iniciar a instalação local. Reinicie o Hermes Desktop e tente novamente.',
    remoteSetupTitle: 'Conectar a um Hermes existente',
    remoteSetupDesc:
      'Informe o endereço do gateway. O Hermes Desktop detectará se é necessário usar um token ou entrar pelo navegador.',
    remoteUrlTitle: 'URL do gateway',
    remoteUrlDesc: 'Use o endereço base do gateway Hermes. Para conexões remotas, inclua https://.',
    remoteUrlPlaceholder: 'https://gateway.example.com/hermes',
    probing: 'Verificando a autenticação do gateway…',
    probeError:
      'O Hermes não conseguiu acessar esse endereço. Confira a URL e se o Hermes está em execução no outro computador. As opções de login aparecerão quando ele responder.',
    probeErrorDetails: 'Detalhes',
    identityProvider: 'seu provedor de identidade',
    authTitle: 'Autenticação',
    authNeedsOauth: provider => 'Entre com ' + provider + ' antes de testar este gateway.',
    authSignedIn: 'Login pelo navegador concluído.',
    connected: 'Conectado',
    signIn: 'Entrar',
    signInWith: provider => `Entrar com ${provider}`,
    enterUrlFirst: 'Primeiro, informe a URL do gateway.',
    signInIncomplete: 'A janela de login foi fechada antes da conclusão da autenticação.',
    tokenTitle: 'Token de sessão',
    tokenDesc: 'Cole o token de sessão do arquivo .env do gateway remoto.',
    pasteSessionToken: 'Cole o token de sessão',
    incompleteSignInTest: 'Entre antes de testar este gateway que exige OAuth.',
    incompleteTokenTest: 'Informe um token de sessão antes de testar este gateway.',
    testConnection: 'Testar conexão',
    testSucceeded: (baseUrl, version) => `Conectado a ${baseUrl}${version ? ` (${version})` : ''}.`,
    applyRemote: 'Aplicar e reconectar',
    backToSetup: 'Voltar',
    failedTitle: 'Falha na instalação',
    settingUpTitle: 'Configurando o Hermes Agent',
    finishingTitle: 'Finalizando',
    failedDesc:
      'Uma das etapas da configuração não foi concluída. Isso pode acontecer se outra cópia do Hermes estiver em execução, se a conexão cair ou se o antivírus bloquear o instalador. Feche as outras janelas do Hermes e escolha Recarregar e tentar novamente. Se o erro persistir, abra os logs e envie-os ao suporte.',
    activeDesc:
      'Esta configuração é feita uma única vez. O instalador do Hermes está baixando dependências e configurando seu computador. Nas próximas inicializações, esta etapa será ignorada.',
    progress: (completed, total) => `${completed} de ${total} etapas concluídas`,
    currentStage: stage => ` — agora: ${stage}`,
    fetchingManifest: 'Buscando o manifesto do instalador…',
    error: 'Erro',
    hideOutput: 'Ocultar saída do instalador',
    showOutput: 'Mostrar saída do instalador',
    lines: count => `${count} linha${count === 1 ? '' : 's'}`,
    noOutput: 'Nenhuma saída ainda.',
    cancelling: 'Cancelando…',
    cancelInstall: 'Cancelar instalação',
    transcriptSaved: 'Transcrição completa salva em',
    copiedOutput: 'Copiado!',
    copyOutput: 'Copiar saída',
    reloadRetry: 'Recarregar e tentar novamente',
    openLogs: 'Abrir logs'
  },
  rightSidebar: {
    aria: 'Barra lateral direita',
    panelsAria: 'Painéis da barra lateral direita',
    files: 'Arquivos',
    terminal: 'Terminal',
    noFolderSelected: 'Nenhuma pasta selecionada',
    changeCwdTitle: 'Alterar diretório de trabalho',
    remotePickerTitle: 'Escolher pasta remota',
    remotePickerDescription: 'Navegue pelas pastas no backend conectado.',
    remotePickerSelect: 'Selecionar pasta',
    folderTip: cwd => cwd,
    openFolder: 'Abrir pasta',
    refreshTree: 'Atualizar árvore',
    collapseAll: 'Recolher todas as pastas',
    showIgnored: 'Mostrar arquivos ignorados pelo Git',
    hideIgnored: 'Ocultar arquivos ignorados pelo Git',
    previewUnavailable: 'Prévia indisponível',
    couldNotPreview: path => `Não foi possível mostrar a prévia de ${path}`,
    noProjectTitle: 'Nenhum projeto',
    noProjectBody: 'Abra um projeto para navegar pelos arquivos e revisar as alterações.',
    noProjectOpen: 'Nenhum projeto aberto',
    noDiffs: 'Nenhuma diferença',
    unreadableTitle: 'Não foi possível ler',
    unreadableBody: error => `Não foi possível ler esta pasta (${error}).`,
    emptyTitle: 'Vazia',
    emptyBody: 'Esta pasta está vazia.',
    treeErrorTitle: 'Erro na árvore de arquivos',
    treeErrorBody: 'Ocorreu um erro ao exibir esta pasta na árvore de arquivos.',
    tryAgain: 'Tentar novamente',
    loadingTree: 'Carregando árvore de arquivos',
    loadingFiles: 'Carregando arquivos',
    terminalHide: 'Ocultar terminal',
    terminalsAria: 'Terminais',
    terminalNew: 'Novo terminal',
    terminalCloseOthers: 'Fechar os outros',
    terminalCloseAll: 'Fechar todos',
    addToChat: 'Adicionar ao chat'
  },
  preview: {
    tab: 'Prévia',
    closePane: 'Fechar painel de prévia',
    loading: 'Carregando prévia',
    unavailable: 'Prévia indisponível',
    opening: 'Abrindo…',
    hide: 'Ocultar',
    openPreview: 'Abrir prévia',
    openInBrowser: 'Abrir no navegador',
    openInExternal: 'Abrir externamente',
    popIn: 'Recolocar no painel',
    popOut: 'Abrir em outra janela',
    linkHint: '⌘/Ctrl-clique para abrir no painel de prévia',
    sourceLineTitle: 'Clique para selecionar · Shift-clique para ampliar a seleção · arraste para o campo de mensagem',
    source: 'CÓDIGO-FONTE',
    renderedPreview: 'PRÉVIA',
    diff: 'DIFERENÇAS',
    unknownSize: 'tamanho desconhecido',
    binaryTitle: 'Este parece ser um arquivo binário',
    binaryBody: label => `A prévia de ${label} pode exibir texto ilegível.`,
    largeTitle: 'Este arquivo é grande',
    largeBody: (label, size) => `${label} tem ${size}. O Hermes exibirá apenas os primeiros 512 KB.`,
    previewAnyway: 'Mostrar prévia mesmo assim',
    truncated: 'Exibindo os primeiros 512 KB.',
    noInlineTitle: 'Sem prévia incorporada',
    noInlineBody: mimeType => `${mimeType || 'Este tipo de arquivo'} ainda pode ser anexado como contexto.`,
    edit: 'Editar',
    editing: 'Editando',
    unsavedChanges: 'Alterações não salvas',
    saveFailed: message => `Não foi possível salvar: ${message}`,
    diskChangedTitle: 'O arquivo foi alterado no disco',
    diskChangedBody:
      'Este arquivo mudou desde que você o abriu. Quer substituir o arquivo pela sua versão ou descartar suas alterações e recarregar?',
    overwrite: 'Substituir',
    discardReload: 'Descartar e recarregar',
    console: {
      deselect: 'Desmarcar item',
      select: 'Selecionar item',
      copyFailed: 'Não foi possível copiar a saída do console',
      copyEntry: 'Copiar este item',
      sendEntry: 'Enviar este item ao chat',
      messages: count => `${count} mensagem${count === 1 ? '' : 's'} no console`,
      resize: 'Redimensionar console da prévia',
      title: 'Console da prévia',
      selected: count => `${count} selecionado${count === 1 ? '' : 's'}`,
      sendToChat: 'Enviar ao chat',
      copySelected: 'Copiar selecionados para a área de transferência',
      copyAll: 'Copiar tudo para a área de transferência',
      copy: 'Copiar',
      clear: 'Limpar',
      empty: 'Nenhuma mensagem no console ainda.',
      promptHeader: 'Console da prévia:',
      sentTitle: 'Enviado ao chat',
      sentMessage: count =>
        `${count} registro${count === 1 ? '' : 's'} adicionado${count === 1 ? '' : 's'} ao campo de mensagem`
    },
    web: {
      appFailedToBoot: 'Não foi possível iniciar o app da prévia',
      serverNotFound: 'Servidor não encontrado',
      remoteLoopback:
        'Este endereço aponta para o computador que executa seu agente, não para este. O painel do navegador carrega páginas localmente; por isso, um servidor remoto precisa de encaminhamento de porta ou de um nome de host acessível.',
      failedToLoad: 'Não foi possível carregar a prévia',
      tryAgain: 'Tentar novamente',
      restarting: 'O Hermes está reiniciando…',
      askRestart: 'Pedir ao Hermes para reiniciar o servidor',
      lookingRestart: taskId => `O Hermes está procurando um servidor de prévia para reiniciar (${taskId})`,
      restartingTitle: 'Reiniciando o servidor da prévia',
      restartingMessage:
        'O Hermes está trabalhando em segundo plano. Acompanhe o console da prévia para ver o andamento.',
      startRestartFailed: message => `Não foi possível iniciar o reinício do servidor: ${message}`,
      restartFailed: 'Falha ao reiniciar o servidor',
      hideConsole: 'Ocultar console da prévia',
      showConsole: 'Mostrar console da prévia',
      hideDevTools: 'Ocultar DevTools da prévia',
      openDevTools: 'Abrir DevTools da prévia',
      goBack: 'Voltar',
      goForward: 'Avançar',
      reload: 'Recarregar página',
      address: 'Endereço',
      addressPlaceholder: 'Informe o endereço',
      blankPageBody: 'Informe um endereço acima para navegar ou peça ao Hermes para abrir uma página.',
      finishedRestarting: message =>
        `O Hermes terminou de reiniciar o servidor da prévia${message ? `: ${message}` : ''}`,
      failedRestarting: message => `Falha ao reiniciar o servidor: ${message}`,
      unknownError: 'erro desconhecido',
      restartedTitle: 'Servidor da prévia reiniciado',
      reloadingNow: 'Recarregando a prévia.',
      restartFailedTitle: 'Falha ao reiniciar a prévia',
      restartFailedMessage: 'O Hermes não conseguiu reiniciar o servidor.',
      stillWorking:
        'O Hermes ainda está trabalhando, mas não recebeu o resultado do reinício. Talvez o comando do servidor esteja em execução no primeiro plano.',
      workspaceReloading: 'O workspace mudou; recarregando a prévia',
      fileChanged: url => `O arquivo mudou; recarregando a prévia: ${url}`,
      filesChanged: (count, url) =>
        `${count} arquivo${count === 1 ? '' : 's'} alterado${count === 1 ? '' : 's'}; recarregando a prévia: ${url}`,
      watchFailed: message => `Não foi possível monitorar o arquivo da prévia: ${message}`,
      moduleMimeDescription:
        'Os scripts de módulo estão sendo servidos com um tipo MIME incorreto. Isso costuma indicar que um servidor de arquivos estáticos está servindo um app Vite/React em vez do servidor de desenvolvimento do projeto.',
      loadFailedConsole: (code, message) => `Falha ao carregar${code ? ` (${code})` : ''}: ${message}`,
      unreachableDescription: 'Não foi possível acessar a página da prévia.',
      openTarget: url => `Abrir ${url}`,
      fallbackTitle: 'Prévia',
      annotate: 'Anotar',
      annotateOn: 'Parar de anotar',
      annotateNeedPage: 'Abra uma página no navegador do app primeiro.',
      annotateFailed: 'Não foi possível iniciar o modo de anotação',
      commenting: 'Comentando',
      addComments: count => (count === 1 ? 'Adicionar 1 comentário' : `Adicionar ${count} comentários`),
      commentPlaceholder: 'Escreva um comentário…',
      commentTitle: n => `Comentário ${n}`,
      saveComment: 'Salvar',
      cancelComment: 'Cancelar comentário'
    }
  },
  interfaceMode: {
    title: 'Modo da interface',
    hint: 'Altera o que aparece, não o que o Hermes pode fazer.',
    sessionNote:
      'Definido pelo modo Simples. Esta alteração vale só para a sessão atual; mude para Avançado para torná-la permanente.',
    simple: {
      label: 'Simples',
      description: 'Para conversar com o Hermes. Barra lateral e chat, sem terminal, arquivos ou painéis de diferenças.'
    },
    advanced: {
      label: 'Avançado',
      description: 'Para desenvolvimento. Terminal, arquivos, diferenças, barra de status e layouts configuráveis.'
    }
  },
  zones: {
    showTabStrip: 'Mostrar abas',
    hideTabStrip: 'Ocultar abas',
    showStripTab: title => `Mostrar ${title}`,
    hideStripTab: title => `Ocultar ${title}`,
    lastTabKeptTitle: 'A última aba será mantida',
    lastTabKeptBody:
      'Esta área precisa manter pelo menos uma aba visível. Mostre outra aba primeiro ou recolha toda a barra lateral.',
    toggleStripTab: title => `Alternar aba ${title}`,
    minimize: 'Minimizar',
    restore: 'Restaurar',
    closeRunningTitle: 'Fechar a aba em execução?',
    closeRunningBody:
      'Este chat ainda está em andamento ou aguardando sua resposta. Ao fechar a aba, a sessão continuará e poderá ser reaberta pela barra lateral.',
    closeRunningConfirm: 'Fechar aba',
    reload: 'Recarregar',
    closeOthers: 'Fechar outras',
    closeToRight: 'Fechar abas à direita',
    closeAll: 'Fechar todas',
    newSessionTab: 'Nova aba de sessão',
    newTab: 'Nova aba',
    pluginDisabled: pluginId => `Plugin "${pluginId}" desativado`,
    pluginDisabledBody: 'Reative em Recursos → Plugins para restaurar o painel.',
    missingPane: paneId => `painel ausente: ${paneId}`,
    editTitle: 'Layouts',
    editHint: 'Escolha um layout ou arraste os painéis entre as áreas.',
    reset: 'Redefinir',
    templates: 'Modelos',
    custom: 'Personalizado',
    newGridLayout: 'Novo layout em grade',
    saveCurrentAs: 'Salvar disposição atual como modelo',
    nameLayoutPlaceholder: 'Nomeie este layout…',
    deletePreset: name => `Excluir ${name}`,
    zoneEditorTitle: 'Editor de áreas',
    editorHintPre: 'clique para dividir · ',
    editorHintPost:
      ' inverte a linha · arraste entre áreas para uni-las · arraste as bordas compartilhadas para redimensionar',
    templateColumns: 'Colunas',
    templateRows: 'Linhas',
    templateGrid: 'Grade',
    templatePriority: 'Prioridade',
    zoneTag: index => `área ${index}`,
    mergeZones: count => `Unir ${count} áreas`,
    customZoneName: count => `Personalizado (${count} áreas)`,
    layoutNamePlaceholder: fallback => `Nome do layout (${fallback})`,
    saveApply: 'Salvar e aplicar',
    notExpressible: 'esta disposição tem áreas entrelaçadas e ainda não pode ser representada como divisões aninhadas',
    zoneCount: count => `${count} área${count === 1 ? '' : 's'}`,
    tabCount: count => `${count} aba${count === 1 ? '' : 's'}`
  },
  contextMenu: {
    link: {
      openInApp: 'Abrir no navegador do app',
      openExternal: 'Abrir no navegador externo',
      copyUrl: 'Copiar URL',
      copyResolvedUrl: 'Copiar URL final'
    },
    image: {
      copyImage: 'Copiar imagem',
      copyImageAddress: 'Copiar endereço da imagem',
      saveImageAs: 'Salvar imagem como…'
    },
    edit: {
      cut: 'Recortar',
      paste: 'Colar',
      selectAll: 'Selecionar tudo',
      addToDictionary: 'Adicionar ao dicionário'
    },
    page: {
      copyPageUrl: 'Copiar URL da página',
      inspectElement: 'Inspecionar elemento'
    }
  },
  commandCenter: {
    branches: 'Branches',
    pets: { staleBackend: 'Reinicie o Hermes para usar mascotes. Este backend é antigo demais para o recurso.' },
    generatePet: {
      hatchComposing: 'Preparando…',
      hatchSaving: 'Quase lá…',
      staleBackend: 'Atualize o Hermes para gerar mascotes.',
      backgroundHint: 'Você pode fechar esta tela; o Hermes avisará quando terminar.',
      slowProviderHint: 'Isso pode levar alguns minutos.',
      remixConfirmTitle: 'Remixar esta aparência?',
      remixConfirmBody: 'Isso vai gerar novas opções com base nesta aparência. Pode levar alguns minutos.',
      genericError: 'Falha ao gerar. Tente novamente ou escolha uma sugestão.',
      referenceImageTooLarge: 'A imagem de referência é grande demais. Use uma com menos de 16 MB.',
      referenceImageInvalid: 'Não foi possível ler a imagem. Tente PNG, JPG, WebP ou GIF.'
    },
    sectionEntries: {
      sessions: { title: 'Painel de sessões', detail: 'Buscar, fixar e gerenciar sessões' },
      system: { title: 'Painel do sistema', detail: 'Status do gateway, logs, reinício e atualização' },
      usage: { title: 'Painel de uso', detail: 'Tokens, custos e atividade de skills' }
    },
    sharedGatewayRestartTitle: 'Reiniciar o gateway compartilhado?',
    sharedGatewayRestartConfirm: 'Reiniciar todos',
    actionRunning: 'em execução',
    actionDone: 'concluída',
    actionFailed: 'falhou',
    actionStartedWaiting: 'Ação iniciada; aguardando o status…',
    loadingStatus: 'Carregando status…',
    retry: 'Tentar novamente',
    dailyTokens: 'Tokens diários',
    input: 'entrada',
    output: 'saída',
    maintenance: {
      runOps: 'Diagnósticos',
      doctor: 'Verificar instalação',
      doctorDesc: 'Verifica a instalação, as configurações e os provedores',
      securityAudit: 'Auditoria de segurança',
      securityAuditDesc: 'Procura configurações arriscadas nas configurações e nas skills',
      backup: 'Criar backup',
      backupDesc: 'Compacta as configurações, memórias, skills e sessões em um arquivo ZIP',
      debugShare: 'Compartilhar diagnóstico',
      debugShareDesc:
        'Envia um relatório sem dados pessoais e os logs para gerar links compartilháveis (excluídos após 6 horas)',
      debugShareRunning: 'Enviando relatório de diagnóstico…',
      debugShareLinks: 'Links para compartilhar',
      debugShareFailed: 'Falha ao compartilhar o diagnóstico',
      copyLink: 'Copiar link',
      linkCopied: 'Link copiado',
      curator: 'Curadoria de skills',
      curatorDesc: 'Revisão em segundo plano que arquiva skills antigas criadas pelo agente',
      curatorPaused: 'Pausada',
      curatorActive: 'Ativa',
      curatorDisabled: 'Desativada',
      curatorNeverRan: 'Nunca executada',
      pause: 'Pausar',
      resume: 'Retomar',
      runNow: 'Executar agora',
      memoryData: 'Dados da memória',
      memoryDataDesc: 'Arquivos de memória incluídos em todas as sessões',
      builtinMemory: 'integrada',
      memoryFile: 'Memória do agente (MEMORY.md)',
      userFile: 'Perfil do usuário (USER.md)',
      empty: 'vazio',
      resetMemory: 'Redefinir memória',
      resetUser: 'Redefinir perfil',
      resetAll: 'Redefinir ambos',
      resetFailed: 'Falha ao redefinir a memória',
      running: 'Em execução…',
      viewLog: 'Registro de ações'
    }
  },
  profiles: {
    nameHint: 'Use letras minúsculas, números, hífens e sublinhados. Comece com uma letra ou número.',
    fleet: { allOnGateway: 'Todos os perfis neste gateway' },
    remoteOverride: {
      menuItem: 'Conectar a um host remoto…',
      description:
        'As sessões deste perfil serão executadas no Hermes remoto que você indicar, em vez de neste computador.',
      urlLabel: 'Endereço remoto',
      urlPlaceholder: 'https://hermes.example.com',
      urlInvalid: 'Informe um endereço completo começando com http:// ou https://',
      tokenLabel: 'Token de acesso',
      tokenPlaceholder: 'Cole o token de sessão remoto',
      tokenSavedHint: 'Já existe um token salvo. Deixe o campo vazio para mantê-lo.',
      plainTextOptIn:
        'Este computador não tem armazenamento seguro de chaves; o token será salvo sem criptografia no disco. Quer salvá-lo mesmo assim?',
      confirmTitle: 'Conectar este perfil a um host remoto?',
      confirmBack: 'Voltar',
      connect: 'Conectar',
      connecting: 'Conectando…',
      disconnect: 'Remover conexão remota',
      savedTitle: 'Perfil conectado',
      removedTitle: 'Conexão remota removida',
      removeFailed: 'Não foi possível remover a conexão remota',
      authFailedTitle: 'O host remoto rejeitou o token salvo',
      updateToken: 'Informar novo token…'
    },
    colorFor: 'Cor',
    defaultDescription: 'Usado ao abrir o Hermes e para novos chats. As sessões existentes continuam em seus perfis.',
    skillsLabel: 'Skills',
    soulDesc: 'As instruções de personalidade e o prompt de sistema deste perfil.',
    soulPlaceholderCloned: 'clonado',
    soulPlaceholderEmpty: 'vazio',
    emptySoul: 'SOUL.md está vazio — comece a escrever a personalidade…',
    deleteDescPrefix: 'Isso excluirá ',
    deleteDescMid: ' e removerá o diretório ',
    deleteDescSuffix: '. Esta ação não pode ser desfeita.',
    cloneFromDesc: 'Copia as configurações, skills e SOUL.md do perfil de origem selecionado.',
    cloneFromDefaultDesc: 'Copia as configurações, skills e SOUL.md do seu perfil padrão.',
    renameDescPrefix: 'Ao renomear, o diretório do perfil e os scripts de inicialização em ',
    displayNameTitle: 'Dê um nome a este agente',
    displayNameDesc: 'Define o nome exibido no app. O ID interno do perfil continua sendo "default".',
    displayNameLabel: 'Nome de exibição',
    newNameLabel: 'Novo nome'
  },
  sidebar: {
    terminal: 'Terminal',
    logs: 'Logs',
    groupAriaGrouped: 'Mostrar sessões em uma lista única',
    groupAriaUngrouped: 'Agrupar sessões por workspace',
    groupTitleGrouped: 'Desagrupar sessões',
    groupTitleUngrouped: 'Agrupar por workspace',
    allPinned: 'Tudo aqui está fixado. Desafixe um chat para mostrá-lo nas sessões recentes.',
    shiftClickHint: 'Shift-clique em um chat para fixá-lo',
    storageCorrupt: {
      title: 'O banco de dados de sessões está danificado',
      action: 'Feche o Hermes neste perfil. Depois, confira o arquivo sem alterá-lo ou restaure um snapshot:',
      guide: 'Guia de recuperação'
    },
    projects: {
      namePlaceholder: 'por exemplo, Projeto novo',
      ideaPlaceholder: 'Sobre o que é este projeto? (salvo em IDEA.md)',
      staleBackend:
        'Atualize o backend Hermes para criar projetos. Ele é mais antigo que este app para desktop (Configurações → Atualizações → Backend).',
      deleteConfirm:
        'Isso remove o projeto salvo do Hermes. Os arquivos, repositórios git e worktrees permanecem intactos.',
      branchPlaceholder: 'por exemplo, meu-recurso',
      baseBranchPlaceholder: 'Buscar branches…',
      worktreeStaleBackend:
        'Atualize o backend Hermes para criar worktrees nesta conexão remota. Ele não tem a API de worktree do git.',
      worktreeProjectPlaceholder: 'Buscar projetos…',
      convertBranchPlaceholder: 'Buscar branches…',
      removeWorktreeFailed: 'Não foi possível remover o worktree (há alterações não commitadas?)',
      removeWorktreeConfirm:
        'Remova-o do git (o diretório do worktree será excluído, mas a branch continuará) ou apenas oculte a faixa e mantenha o worktree no disco.',
      removeWorktreeDirty:
        'Este worktree tem alterações não commitadas. Force a remoção para descartá-las ou apenas oculte a faixa e mantenha os arquivos no disco.'
    },
    row: {
      unreadFailed: 'Não foi possível atualizar o status de leitura',
      hideTabBar: 'Ocultar barra de abas',
      openInSplit: 'Abrir em um painel dividido'
    }
  },
  statusStack: {
    control: {
      clearCriteriaConfirmTitle: 'Limpar todos os critérios?',
      clearCriteriaConfirmBody: 'Tem certeza de que deseja remover todos os critérios desta meta?',
      loopSelfPaced: 'no seu ritmo',
      stopLoopConfirmBody: 'Tem certeza de que deseja parar esta repetição?',
      loopPromptLabel: 'Prompt',
      loopDeferredNotice: 'Uma meta ativa está controlando esta sessão.',
      loopAwaitingResponse: 'Aguardando resposta',
      heartbeatDueWaitingForIdle: 'agendado — aguardando ficar ocioso',
      clearHeartbeatConfirmBody: 'Tem certeza de que deseja remover este heartbeat?',
      copySuccess: 'Critério copiado para a área de transferência',
      copyFailure: 'Não foi possível copiar o critério para a área de transferência',
      continuationFailed: 'Não foi possível enviar a continuação da meta',
      continuationQueued: 'Meta retomada — continuação na fila até a resposta atual terminar',
      continuationBusy: 'Meta retomada — sessão ocupada; use /interrupt para interromper a resposta atual e continuar'
    },
    coding: {
      revertConfirm:
        'Descartar as alterações deste arquivo e restaurar a versão commitada? Esta ação não pode ser desfeita.',
      revertAllConfirm:
        'Descartar todas as alterações e restaurar os arquivos para a versão commitada? Esta ação não pode ser desfeita.',
      scopeUncommitted: 'Não commitadas',
      scopeBranch: 'Branch',
      scopeLastTurn: 'Última resposta',
      commit: 'Commit',
      ghMissing: 'Instale o GitHub CLI (gh) e entre na sua conta para abrir pull requests.',
      agentShip: 'Pedir ao Hermes para abrir um pull request',
      agentShipUnavailable: 'O chat responsável por estas alterações não está na tela.',
      agentShipPrompt:
        'Revise as alterações atuais, faça um commit com uma mensagem convencional clara, envie a branch e abra um pull request.',
      worktrees: 'Worktrees'
    }
  },
  desktop: {
    audioReadFailed: 'Não foi possível ler o áudio gravado',
    sessionUnavailable: 'Sessão indisponível',
    createSessionFailed: 'Não foi possível criar uma nova sessão',
    promptFailed: 'Falha ao enviar a mensagem',
    providerCredentialRequired: 'Adicione uma credencial de provedor antes de enviar sua primeira mensagem.',
    emptySlashCommand: 'comando slash vazio',
    desktopCommands: 'Comandos do desktop',
    skillCommandsAvailable: count => `${count} comandos de skills disponíveis.`,
    warningLine: message => `aviso: ${message}`,
    yoloArmed: 'YOLO ativado para este chat',
    yoloOff: 'YOLO desativado',
    yoloSystem: active => `YOLO ${active ? 'ativado' : 'desativado'} nesta sessão`,
    yoloTitle: 'YOLO',
    yoloToggleFailed: 'Não foi possível alternar o YOLO',
    profileStatus: current =>
      `Perfil: ${current}. Use /profile <name> ou o seletor "Nova sessão" para iniciar um chat em outro perfil.`,
    unknownProfile: 'Perfil desconhecido',
    noProfileNamed: (target, available) => `Não existe um perfil chamado "${target}". Disponíveis: ${available}`,
    newChatsProfile: name => `Novos chats usarão o perfil ${name}.`,
    setProfileFailed: 'Não foi possível definir o perfil',
    sttDisabled: 'A conversão de fala em texto está desativada nas configurações.',
    stopFailed: 'Não foi possível parar',
    regenerateFailed: 'Não foi possível gerar novamente',
    editFailed: 'Não foi possível editar',
    editTurnUnavailable: 'Esta resposta não está mais no histórico do servidor (talvez tenha sido compactada).',
    resumeFailed: 'Não foi possível retomar a sessão',
    readOnlyTranscriptTitle: 'Aberta em modo somente leitura',
    readOnlyTranscriptBody:
      'Nenhum backend conectado reivindicou este chat antigo, então ele foi aberto como uma transcrição somente leitura. O histórico está intacto; o envio ficará desativado até que um backend o reivindique.',
    readOnlyTranscriptSendBlocked: 'Este chat está aberto como transcrição somente leitura; o envio está desativado.',
    resumeStrandedTitle: 'Não foi possível carregar esta sessão',
    resumeStrandedBody:
      'A conexão com esta sessão falhou e as tentativas automáticas terminaram. Confira se o gateway está em execução e tente novamente.',
    poolSlotTimeoutBody:
      'Há bots demais em execução para o limite deste computador. Aumente o limite em Configurações → Avançado ou aguarde uma tarefa terminar e tente novamente.',
    poolSlotTimeoutOpenSettings: 'Abrir configurações avançadas',
    resumeRetry: 'Tentar novamente',
    nothingToBranch: 'Nada para ramificar',
    branchNeedsChat: 'Inicie ou retome um chat antes de criar uma ramificação.',
    sessionBusy: 'Sessão ocupada',
    branchStopCurrent: 'Pare a resposta atual antes de criar uma ramificação deste chat.',
    branchNoText: 'Esta mensagem não contém texto para criar uma ramificação.',
    branchTitle: n => `Rascunho: ramificação nº ${n}`,
    branchFailed: 'Falha ao criar ramificação',
    deleteFailed: 'Não foi possível excluir',
    archived: 'Arquivado',
    archiveFailed: 'Não foi possível arquivar',
    cwdChangeFailed: 'Não foi possível alterar o diretório de trabalho',
    cwdStagedTitle: 'Diretório de trabalho preparado',
    cwdStagedMessage: 'Reinicie o backend do desktop para aplicar a alteração do diretório de trabalho a esta sessão.',
    modelSwitchConfirmBody: 'Esta troca de modelo precisa de confirmação.',
    modelSwitchConfirmLabel: 'Trocar mesmo assim',
    modelSwitchConfirmTitle: (model: string) => `Trocar para ${model}?`,
    modelSwitchConfirmTitleFallback: 'Trocar de modelo?',
    modelSwitchFailed: 'Falha ao trocar de modelo',
    modelSwitchKeepLabel: 'Manter modelo atual',
    modelSwitchStaleNotice: 'A seleção mudou; a troca de modelo não foi aplicada.',
    hydrationSyncing: (profile: string) => `Sincronizando ${profile}…`,
    sessionExported: 'Sessão exportada',
    sessionExportFailed: 'Não foi possível exportar a sessão',
    imageSaved: 'Imagem salva',
    downloadStarted: 'Download iniciado',
    restartToUseSaveImage: 'Reinicie o Hermes Desktop para usar Salvar imagem.',
    restartToSaveImages: 'Reinicie o Hermes Desktop para salvar imagens',
    imageDownloadFailed: 'Falha ao baixar imagem',
    openImage: 'Abrir imagem',
    downloadImage: 'Baixar imagem',
    savingImage: 'Salvando imagem',
    imagePreviewFailed: 'Falha ao mostrar a prévia da imagem',
    imageAttach: 'Anexar imagem',
    imageWriteFailed: 'Não foi possível gravar a imagem no disco.',
    imageAttachFailed: 'Não foi possível anexar a imagem',
    pastedContent: 'Conteúdo colado',
    pasteAttachFailed: 'Não foi possível anexar o texto colado',
    attachImages: 'Anexar imagens',
    clipboard: 'Área de transferência',
    noClipboardImage: 'Nenhuma imagem encontrada na área de transferência',
    clipboardPasteFailed: 'Falha ao colar da área de transferência',
    dropFiles: 'Solte os arquivos aqui',
    handoff: {
      pickPlatform: 'Escolha para onde enviar',
      success: platform => `Enviado para ${platform}. Você pode retomar por aqui quando quiser.`,
      systemNote: platform => `↻ Enviado para ${platform}; retome por aqui quando quiser.`,
      failed: error => `Falha ao transferir: ${error}`,
      timedOut:
        'O Hermes não conseguiu acessar sua conexão de mensagens. Inicie-a em Configurações → Mensagens e tente novamente.',
      startMessaging: 'Iniciar mensagens'
    }
  },
  modelPicker: {
    title: 'Trocar modelo',
    current: 'atual:',
    unknown: '(desconhecido)',
    search: 'Filtrar provedores e modelos…',
    noModels: 'Nenhum modelo encontrado.',
    addProvider: 'Adicionar provedor',
    loadFailed: 'Não foi possível carregar os modelos',
    loadingIntoMemory: 'Carregando na memória',
    downloading: 'Baixando',
    localDownloadsHeading: 'Locais',
    noAuthenticatedProviders: 'Nenhum provedor autenticado.',
    pro: 'Pro',
    proNeedsSubscription: 'Os modelos Pro exigem uma assinatura paga da Nous.',
    free: 'Gratuito',
    freeTier: 'Plano gratuito',
    priceTitle: 'Preço de entrada/saída por milhão de tokens',
    wasPrice: 'antes'
  }
}
