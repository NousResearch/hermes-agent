import { defineFieldCopy } from '@/app/settings/field-copy'

import { defineLocale, mergeTranslationOverrides, type TranslationOverrides } from './define-locale'

const ptBrBaseOverrides: TranslationOverrides = {
  common: {
    apply: 'Aplicar',
    back: 'Voltar',
    save: 'Salvar',
    saving: 'Salvando…',
    cancel: 'Cancelar',
    change: 'Alterar',
    choose: 'Escolher',
    clear: 'Limpar',
    close: 'Fechar',
    collapse: 'Recolher',
    confirm: 'Confirmar',
    connect: 'Conectar',
    connecting: 'Conectando',
    continue: 'Continuar',
    copied: 'Copiado',
    copy: 'Copiar',
    copyFailed: 'Falha ao copiar',
    delete: 'Excluir',
    docs: 'Documentação',
    done: 'Concluído',
    error: 'Erro',
    expand: 'Expandir',
    failed: 'Falhou',
    formatJson: 'Formatar JSON',
    free: 'Grátis',
    loading: 'Carregando…',
    notSet: 'Não definido',
    refresh: 'Atualizar',
    remove: 'Remover',
    replace: 'Substituir',
    retry: 'Tentar novamente',
    run: 'Executar',
    send: 'Enviar',
    set: 'Definir',
    skip: 'Pular',
    update: 'Atualizar',
    tryHint: term => `Tente “${term}”`,
    on: 'Ativado',
    off: 'Desativado'
  },

  fileMenu: {
    revealFinder: 'Mostrar no Finder',
    revealExplorer: 'Mostrar no Explorador de Arquivos',
    revealFileManager: 'Abrir a pasta que contém o arquivo',
    revealInSidebar: 'Mostrar na árvore de arquivos',
    copyPath: 'Copiar caminho',
    copyRelativePath: 'Copiar caminho relativo',
    rename: 'Renomear…',
    delete: 'Excluir',
    renameTitle: 'Renomear',
    renameLabel: 'Novo nome',
    deleteTitle: (name: string) => `Excluir ${name}?`,
    deleteBody: 'O item será movido para a Lixeira — você pode restaurá-lo de lá.',
    pathCopied: 'Caminho copiado'
  },

  boot: {
    ready: 'O Hermes Desktop está pronto',
    desktopBootFailedWithMessage: message => `Falha na inicialização do desktop: ${message}`,
    steps: {
      connectingGateway: 'Conectando ao gateway do desktop',
      loadingSettings: 'Carregando as configurações do Hermes',
      loadingSessions: 'Carregando as sessões recentes',
      startingDesktopConnection: 'Iniciando a conexão do desktop',
      startingHermesDesktop: 'Iniciando o Hermes Desktop…'
    },
    errors: {
      backgroundExited: 'O processo em segundo plano do Hermes foi encerrado.',
      backgroundExitedDuringStartup: 'O processo em segundo plano do Hermes foi encerrado durante a inicialização.',
      backendStopped: 'Backend parado',
      desktopBootFailed: 'Falha na inicialização do desktop',
      gatewayConnectionLost: 'A conexão com o gateway foi perdida',
      gatewaySignInRequired: 'É necessário entrar no gateway',
      ipcBridgeUnavailable: 'A ponte IPC do desktop está indisponível.',
      gatewayConnectionLostDetail:
        'Ainda tentando reconectar em segundo plano. Você pode continuar lendo e rascunhando — abra as Configurações do Gateway se isso persistir.'
    },
    failure: {
      title: 'O Hermes não conseguiu iniciar',
      description:
        'O gateway em segundo plano não subiu. Tente uma das opções de recuperação abaixo. Nada aqui exclui suas conversas ou configurações.',
      remoteTitle: 'É necessário entrar no gateway remoto',
      remoteDescription:
        'Sua sessão no gateway remoto expirou. Entre novamente para reconectar. Nada aqui exclui suas conversas ou configurações.',
      retry: 'Tentar novamente',
      repairInstall: 'Reparar instalação',
      useLocalGateway: 'Usar gateway local',
      gatewaySettings: 'Configurações do gateway',
      back: 'Voltar',
      openLogs: 'Abrir logs',
      repairHint: 'O reparo executa o instalador de novo e pode levar alguns minutos em uma máquina nova.',
      remoteSignInHint: signInLabel =>
        `Encerra a sessão remota salva no navegador e abre ${signInLabel}. Use o gateway local para trocar pelo backend embutido.`,
      signOutAndSignIn: 'Sair e entrar',
      remoteFailureHint:
        'Verifique a URL do gateway e o login em Configurações do gateway, ou troque para o gateway local.',
      cloudDownTitle: 'O agente Nous Cloud está indisponível',
      cloudDownDescription:
        'O agente em nuvem gerenciado pela Nous ao qual este gateway se conecta está retornando um erro de servidor. Ele não pode ser reiniciado a partir daqui — verifique o status, mude para o gateway local ou solicite suporte.',
      cloudDownHint:
        'Os botões abaixo abrem o Portal Nous (status e controles da instância) e nosso Discord para suporte.',
      cloudDownCheckPortal: 'Verificar status no Portal',
      cloudDownDiscord: 'Obter ajuda no Discord',
      hideRecentLogs: 'Ocultar logs recentes',
      showRecentLogs: 'Mostrar logs recentes',
      signedInTitle: 'Conectado',
      signedInMessage: 'Reconectando ao gateway remoto…',
      signInIncompleteTitle: 'Login incompleto',
      signInIncompleteMessage: 'A janela de login foi fechada antes de a autenticação terminar.',
      signInFailed: 'Falha ao entrar',
      signInToRemoteGateway: 'Entrar no gateway remoto',
      signInWithProvider: provider => `Entrar com ${provider}`,
      identityProvider: 'seu provedor de identidade'
    }
  },

  notifications: {
    region: 'Notificações',
    hide: 'Ocultar',
    show: 'Mostrar',
    more: count => `mais ${count} ${count === 1 ? 'notificação' : 'notificações'}`,
    clearAll: 'Limpar todas',
    dismiss: 'Dispensar notificação',
    details: 'Detalhes',
    copyDetail: 'Copiar detalhe',
    copyDetailFailed: 'Não foi possível copiar o detalhe da notificação',
    backendOutOfDateTitle: 'Backend desatualizado',
    backendOutOfDateMessage:
      'Seu backend do Hermes é mais antigo que esta versão do desktop e pode não funcionar corretamente. Atualize para alinhar os dois.',
    installMethodUnsupportedTitle: 'Método de instalação não suportado',
    updateHermes: 'Atualizar Hermes',
    updateReadyTitle: 'Atualização pronta',
    updateReadyMessage: count =>
      `${count} ${count === 1 ? 'nova alteração disponível' : 'novas alterações disponíveis'}.`,
    seeWhatsNew: 'Ver novidades',
    errors: {
      elevenLabsNeedsKey: 'O STT da ElevenLabs precisa de ELEVENLABS_API_KEY.',
      elevenLabsRejectedKey: 'A ElevenLabs rejeitou a chave de API (401).',
      diskFull: 'Disco cheio — libere espaço e tente novamente.',
      gatewayAuthFailed: 'Falha na autenticação do gateway — verifique sua API_SERVER_KEY.',
      methodNotAllowed:
        'O backend do desktop rejeitou essa requisição (405 Method Not Allowed). Tente reiniciar o Hermes Desktop.',
      microphonePermission: 'A permissão do microfone foi negada.',
      openaiRejectedApiKey: 'A OpenAI rejeitou a chave de API.',
      openaiRejectedApiKeyWithStatus: status => `A OpenAI rejeitou a chave de API (${status} invalid_api_key).`,
      openaiTtsNeedsKey: 'O TTS da OpenAI precisa de VOICE_TOOLS_OPENAI_KEY ou OPENAI_API_KEY.'
    },
    voice: {
      configureSpeechToText: 'Configure a transcrição de voz para usar o modo de voz.',
      couldNotStartSession: 'Não foi possível iniciar a sessão de voz',
      microphoneAccessDenied: 'Acesso ao microfone negado.',
      microphoneConstraintsUnsupported: 'Este dispositivo não suporta as restrições de microfone.',
      microphoneFailed: 'Falha no microfone',
      microphoneInUse: 'O microfone já está sendo usado por outro aplicativo.',
      microphonePermissionDenied: 'A permissão do microfone foi negada.',
      microphoneStartFailed: 'Não foi possível iniciar a gravação pelo microfone.',
      microphoneUnsupported: 'Este ambiente não suporta gravação pelo microfone.',
      noMicrophone: 'Nenhum microfone foi encontrado.',
      noSpeechDetected: 'Nenhuma fala detectada',
      playbackFailed: 'Falha na reprodução de voz',
      recordingFailed: 'Falha na gravação de voz',
      sayStopToEnd: phrase => `Diga "${phrase}" para encerrar a conversa por voz.`,
      transcriptionFailed: 'Falha na transcrição de voz',
      transcriptionUnavailable: 'A transcrição de voz ainda não está disponível.',
      tryRecordingAgain: 'Tente gravar novamente.',
      unavailable: 'Voz indisponível'
    },
    native: {
      approvalTitle: 'Aprovação necessária',
      approveAction: 'Aprovar',
      rejectAction: 'Rejeitar',
      inputTitle: 'Entrada necessária',
      inputBody: 'O Hermes está aguardando sua resposta.',
      turnDoneTitle: 'Hermes concluído',
      turnDoneBody: '',
      turnErrorTitle: 'Falha no turno',
      backgroundDoneTitle: 'Tarefa em segundo plano concluída',
      backgroundFailedTitle: 'Falha na tarefa em segundo plano',
      creditsTitle: 'Créditos'
    }
  },

  remoteDisplayBanner: {
    message: reason =>
      `Renderização por software ativa — display remoto detectado (${reason}). A aceleração por GPU foi desativada para evitar tremulação.`
  },

  billingBlock: {
    titleNous: 'Sem créditos da Nous',
    titleProvider: provider => `Sem créditos — ${provider}`,
    fallbackMessage: 'Sua conta está sem créditos. Adicione créditos para continuar.',
    openBilling: 'Abrir faturamento',
    addCredits: 'Adicionar créditos',
    dismiss: 'Dispensar'
  },

  titlebar: {
    hideSidebar: 'Ocultar barra lateral',
    showSidebar: 'Mostrar barra lateral',
    search: 'Pesquisar',
    searchTitle: 'Pesquisar sessões, visualizações e ações',
    swapSidebarSides: 'Inverter os lados das barras laterais',
    hideRightSidebar: 'Ocultar barra lateral direita',
    showRightSidebar: 'Mostrar barra lateral direita',
    muteHaptics: 'Silenciar resposta tátil',
    unmuteHaptics: 'Reativar resposta tátil',
    openSettings: 'Abrir configurações',
    openStarmap: 'Abrir grafo de memória',
    enterHud: 'Modo HUD',
    exitHud: 'Sair do modo HUD',
    resetHudLayout: 'Redefinir tamanho e posição do HUD',
    layoutEditor: 'Editor de layout',
    layoutEditorTitle: modifier => `Editor de layout — ${modifier}-clique redefine o layout`
  },

  sendDiagnostics: {
    title: 'Enviar diagnósticos para a Nous',
    privacyNotice:
      'Isso envia um pacote de depuração para o armazenamento interno da Nous (não é um texto público). Inclui informações do sistema (SO, versões, provedor, quais chaves de API estão configuradas — nunca as chaves em si) e logs completos do agente, gateway e desktop (até 512 KB cada), que podem conter conteúdo de conversas, saídas de ferramentas e caminhos de arquivos. Segredos são censurados antes do envio. O pacote é visível apenas pela equipe da Nous e moderadores do Discord autorizados, e é excluído automaticamente após 14 dias.',
    upload: 'Enviar',
    uploading: 'Enviando…',
    cancel: 'Cancelar',
    close: 'Fechar',
    copyLink: 'Copiar link',
    uploadIdFallback: id => `Nenhum link retornado — informe o ID de envio ${id} ao suporte`,
    doneTitle: 'Diagnósticos enviados',
    doneDescription:
      'Seu pacote foi enviado de forma privada. Compartilhe o link abaixo no seu tópico de suporte para que a equipe possa analisar os logs.',
    failedTitle: 'Falha no envio',
    failedHint:
      'Você também pode executar `hermes debug share --nous` no terminal, ou `hermes debug share --local` para exibir o relatório sem enviar.',
    handoffLead: 'Continue a conversa em:',
    links: {
      github: 'Issues do GitHub',
      portal: 'Suporte do Portal Nous',
      discord: 'Discord'
    }
  },

  keybinds: {
    title: 'Atalhos de teclado',
    subtitle: open => `Clique em um atalho para redefini-lo · ${open} reabre este painel.`,
    search: 'Pesquisar atalhos…',
    rebind: 'Redefinir atalho',
    reset: 'Restaurar padrão',
    resetAll: 'Restaurar tudo',
    pressKey: 'Pressione uma tecla…',
    set: 'definido',
    conflictWith: label => `Também associado a “${label}”`,
    categories: {
      composer: 'Compositor',
      profiles: 'Perfis',
      session: 'Sessão',
      navigation: 'Navegação',
      view: 'Visualização'
    },
    actions: {
      'keybinds.openPanel': 'Abrir atalhos de teclado',
      'nav.commandPalette': 'Abrir paleta de comandos',
      'nav.commandCenter': 'Abrir central de comandos',
      'nav.settings': 'Abrir configurações',
      'nav.profiles': 'Abrir perfis',
      'nav.skills': 'Abrir habilidades',
      'nav.messaging': 'Abrir mensagens',
      'nav.artifacts': 'Abrir artefatos',
      'nav.cron': 'Abrir tarefas agendadas',
      'nav.agents': 'Abrir agentes',
      'session.new': 'Nova sessão',
      'session.newTab': 'Nova aba de sessão',
      'session.newWindow': 'Nova janela',
      'session.next': 'Próxima sessão',
      'session.prev': 'Sessão anterior',
      'session.slot.1': 'Ir para a sessão recente 1',
      'session.slot.2': 'Ir para a sessão recente 2',
      'session.slot.3': 'Ir para a sessão recente 3',
      'session.slot.4': 'Ir para a sessão recente 4',
      'session.slot.5': 'Ir para a sessão recente 5',
      'session.slot.6': 'Ir para a sessão recente 6',
      'session.slot.7': 'Ir para a sessão recente 7',
      'session.slot.8': 'Ir para a sessão recente 8',
      'session.slot.9': 'Ir para a sessão recente 9',
      'session.focusSearch': 'Pesquisar sessões',
      'session.togglePin': 'Fixar / desafixar a sessão atual',
      'workspace.newWorktree': 'Novo worktree',
      'workspace.openFolder': 'Abrir pasta como projeto',
      'composer.focus': 'Focar o compositor',
      'composer.modelPicker': 'Abrir seletor de modelos',
      'composer.voice': 'Iniciar / parar conversa por voz',
      'view.toggleSidebar': 'Alternar barra lateral de sessões',
      'view.toggleRightSidebar': 'Alternar navegador de arquivos',
      'view.toggleReview': 'Alternar painel de revisão',
      'view.toggleStatusbar': 'Alternar barra de status',
      'view.toggleTabStrip': 'Alternar abas',
      'view.showFiles': 'Mostrar navegador de arquivos',
      'view.toggleHud': 'Alternar modo HUD',
      'hud.snapToPointer': 'Mover o HUD para o ponteiro (global, enquanto o HUD estiver aberto)',
      'view.showTerminal': 'Alternar terminal',
      'view.newTerminal': 'Novo terminal',
      'view.nextTerminal': 'Próximo terminal',
      'view.prevTerminal': 'Terminal anterior',
      'view.closeTerminal': 'Fechar terminal',
      'view.terminalCopy': 'Copiar seleção do terminal',
      'view.selectionToComposer': 'Enviar seleção para o compositor',
      'view.terminalPaste': 'Colar no terminal',
      'view.closeTab': 'Fechar aba',
      'view.reopenTab': 'Reabrir aba fechada',
      'view.flipPanes': 'Inverter os lados das barras laterais',
      'view.findInPage': 'Localizar na página',
      'view.findNext': 'Localizar próxima ocorrência',
      'view.findPrevious': 'Localizar ocorrência anterior',
      'appearance.toggleMode': 'Alternar claro / escuro',
      'profile.default': 'Ir para o perfil padrão',
      'profile.switch.1': 'Ir para o perfil 1',
      'profile.switch.2': 'Ir para o perfil 2',
      'profile.switch.3': 'Ir para o perfil 3',
      'profile.switch.4': 'Ir para o perfil 4',
      'profile.switch.5': 'Ir para o perfil 5',
      'profile.switch.6': 'Ir para o perfil 6',
      'profile.switch.7': 'Ir para o perfil 7',
      'profile.switch.8': 'Ir para o perfil 8',
      'profile.switch.9': 'Ir para o perfil 9',
      'profile.switch.10': 'Ir para o perfil 10',
      'profile.switch.11': 'Ir para o perfil 11',
      'profile.switch.12': 'Ir para o perfil 12',
      'profile.switch.13': 'Ir para o perfil 13',
      'profile.switch.14': 'Ir para o perfil 14',
      'profile.switch.15': 'Ir para o perfil 15',
      'profile.switch.16': 'Ir para o perfil 16',
      'profile.switch.17': 'Ir para o perfil 17',
      'profile.switch.18': 'Ir para o perfil 18',
      'profile.next': 'Próximo perfil',
      'profile.prev': 'Perfil anterior',
      'profile.toggleAll': 'Alternar a visão de todos os perfis',
      'profile.create': 'Criar perfil',
      'composer.send': 'Enviar mensagem',
      'composer.newline': 'Inserir nova linha',
      'composer.steer': 'Direcionar o turno em execução',
      'composer.queue': 'Enfileirar mensagem',
      'composer.sendQueued': 'Enviar o próximo turno da fila',
      'composer.mention': 'Referenciar arquivos, pastas e URLs',
      'composer.slash': 'Paleta de comandos de barra',
      'composer.help': 'Ajuda rápida',
      'composer.history': 'Percorrer o popover / histórico',
      'composer.cancel': 'Fechar popover · cancelar execução'
    }
  },

  findInPage: {
    next: 'Próxima ocorrência',
    previous: 'Ocorrência anterior'
  },

  language: {
    label: 'Idioma',
    description: 'Escolha o idioma da interface do aplicativo.',
    saving: 'Salvando idioma…',
    saveError: 'Falha ao atualizar o idioma',
    switchTo: 'Trocar idioma',
    searchPlaceholder: 'Pesquisar idiomas…',
    noResults: 'Nenhum idioma encontrado'
  },

  settings: {
    closeSettings: 'Fechar configurações',
    exportConfig: 'Exportar configuração',
    importConfig: 'Importar configuração',
    resetToDefaults: 'Restaurar padrões',
    resetConfirm: 'Restaurar todas as configurações aos padrões do Hermes?',
    exportFailed: 'Falha ao exportar',
    resetFailed: 'Falha ao restaurar os padrões',
    nav: {
      providers: 'Provedores',
      providerAccounts: 'Contas',
      providerApiKeys: 'Chaves de API',
      providerCustomEndpoints: 'Endpoints personalizados',
      gateway: 'Gateway',
      apiKeys: 'Ferramentas e chaves',
      keybinds: 'Atalhos de teclado',
      keysTools: 'Ferramentas',
      keysSettings: 'Configurações',
      mcp: 'MCP',
      archivedChats: 'Conversas arquivadas',
      about: 'Sobre',
      billing: 'Faturamento',
      notifications: 'Notificações',
      plugins: 'Plugins'
    },
    plugins: {
      title: 'Plugins do Desktop',
      blurb: 'Embutidos ou colocados na pasta desktop-plugins. Desative para descarregar o plugin imediatamente.',
      count: n => `${n} plugin${n === 1 ? '' : 's'} instalado${n === 1 ? '' : 's'}`,
      openFolder: 'Abrir pasta de plugins',
      rescan: 'Reanalisar',
      reveal: 'Mostrar no gerenciador de arquivos',
      enable: 'Ativar',
      disable: 'Desativar',
      failed: 'Falhou',
      empty: 'Nenhum plugin de desktop instalado ainda.',
      kinds: { bundled: 'embutido', disk: 'em disco', runtime: 'runtime' },
      agent: {
        title: 'Plugins de agente',
        blurb:
          'Executam no backend do Hermes e fornecem ferramentas, habilidades, servidores MCP, hooks e comandos de barra. Os portáteis são pacotes de plugins de agente (habilidades e pacotes MCP que também funcionam em outros agentes). As opções valem para novas sessões.',
        empty: 'Nenhum plugin de agente instalado ainda.',
        loadFailed: 'Não foi possível carregar os plugins do agente',
        portable: 'portátil',
        search: 'Pesquisar plugins…',
        noMatches: 'Nenhum plugin corresponde à sua pesquisa.',
        toggleFailed: (name: string) => `Não foi possível alternar ${name}`,
        updateBackendToManage: 'Atualize o backend do Hermes para gerenciar este plugin pelo Desktop.',
        sources: { bundled: 'embutido', user: 'usuário', git: 'git', project: 'projeto', entrypoint: 'pip' }
      }
    },
    notifications: {
      title: 'Notificações',
      intro:
        'Notificações do sistema operacional — não são avisos dentro do aplicativo. A configuração é feita por dispositivo.',
      enableAll: 'Ativar notificações',
      enableAllDesc: 'Desativar esta opção silencia todas as notificações abaixo.',
      focusedHint: 'Os avisos de conclusão só aparecem enquanto o Hermes está em segundo plano.',
      kinds: {
        approval: {
          label: 'Aprovação necessária',
          description: 'Um comando aguarda sua aprovação ou rejeição.'
        },
        input: {
          label: 'Entrada necessária',
          description: 'O Hermes fez uma pergunta ou precisa de uma senha ou de um segredo.'
        },
        turnDone: {
          label: 'Resposta pronta',
          description: 'Um turno terminou enquanto o Hermes estava em segundo plano.'
        },
        turnError: {
          label: 'Falha no turno',
          description: 'Erros ocorridos em turnos em segundo plano.'
        },
        backgroundDone: {
          label: 'Tarefa em segundo plano concluída',
          description: 'Um comando de terminal em segundo plano foi concluído.'
        },
        credits: {
          label: 'Avisos de crédito',
          description: 'O acesso aos créditos foi pausado ou restabelecido.'
        },
        plugin: {
          label: 'Notificação de plugin',
          description: 'Um plugin do desktop enviou uma notificação enquanto o Hermes estava em segundo plano.'
        }
      },
      test: 'Enviar notificação de teste',
      testTitle: 'Hermes',
      testBody: 'As notificações estão funcionando.',
      testSent:
        'Teste enviado. Se nada aparecer, verifique as permissões de notificação do sistema e o modo Foco/Não Perturbe.',
      testUnsupported: 'Este sistema não suporta notificações nativas.',
      completionSoundTitle: 'Som de conclusão',
      completionSoundDesc: 'Toca quando um turno do agente termina. Escolha uma opção e ouça uma prévia aqui.',
      completionSoundPreview: 'Pré-visualizar'
    },
    sections: {
      model: 'Modelo',
      chat: 'Conversa',
      appearance: 'Aparência',
      workspace: 'Espaço de trabalho',
      safety: 'Segurança',
      memory: 'Memória e contexto',
      voice: 'Voz',
      advanced: 'Avançado'
    },
    searchPlaceholder: {
      about: 'Sobre o Hermes Desktop',
      config: 'Buscar configurações...',
      gateway: 'Conexão com o gateway...',
      keys: 'Pesquisar chaves de API…',
      mcp: 'Pesquisar servidores MCP…',
      sessions: 'Buscar sessões arquivadas...'
    },
    modeOptions: {
      light: { label: 'Claro', description: 'Superfícies claras' },
      dark: { label: 'Escuro', description: 'Espaço de trabalho com pouco brilho' },
      system: { label: 'Sistema', description: 'Seguir a aparência do sistema operacional' }
    },
    appearance: {
      title: 'Aparência',
      intro: 'Apenas no Desktop. O modo controla o brilho; o tema controla a paleta e a aparência da conversa.',
      colorMode: 'Modo de cor',
      colorModeDesc: 'Escolha um modo fixo ou deixe o Hermes seguir a configuração do sistema.',
      toolViewTitle: 'Exibição das chamadas de ferramenta',
      toolViewDesc:
        'O modo Produto oculta os dados brutos das ferramentas; o modo Técnico mostra a entrada e a saída completas.',
      uiScaleTitle: 'Escala da interface',
      uiScaleDesc: (percent: number) =>
        `Redimensiona textos e controles em todo o app. Cmd/Ctrl com +, - e 0 também funciona. Atual: ${percent}%.`,
      terminalFontTitle: 'Fonte do terminal',
      terminalFontDesc:
        'Escolha uma fonte instalada para os terminais do Desktop. Nerd Fonts exibem corretamente o Powerlevel10k e os ícones do shell; deixe em branco para usar a JetBrains Mono incluída.',
      terminalFontPlaceholder: 'MesloLGS NF ou uma pilha de fontes CSS',
      terminalFontPreview: 'Prévia de glifos',
      terminalFontReset: 'Usar padrão',
      translucencyTitle: 'Translucidez da janela',
      translucencyDesc: 'Veja sua área de trabalho através da janela inteira. Apenas macOS e Windows.',
      backdropTitle: 'Plano de fundo do chat',
      backdropDesc: 'A imagem sutil da estátua atrás da conversa.',
      reactionsTitle: 'Reações às mensagens',
      reactionsDesc:
        'Reações com emoji no estilo iMessage — reaja às mensagens, e o Hermes também pode reagir às suas.',
      embedsTitle: 'Conteúdo incorporado',
      embedsDesc:
        'As prévias ricas carregam conteúdo de sites de terceiros (YouTube, X, …). A opção “Perguntar” mostra um espaço reservado até você liberar cada uma; “Sempre” carrega automaticamente; “Desativado” mantém apenas links simples.',
      embedsAsk: 'Perguntar',
      embedsAlways: 'Sempre',
      embedsOff: 'Desativado',
      embedsReset: (count: number) => `Redefinir ${count} ${count === 1 ? 'serviço permitido' : 'serviços permitidos'}`,
      product: 'Produto',
      productDesc: 'Atividade das ferramentas em linguagem acessível, com resumos concisos.',
      technical: 'Técnico',
      technicalDesc: 'Inclui argumentos e resultados brutos das ferramentas e detalhes de baixo nível.',
      themeTitle: 'Tema',
      themeDesc: 'Apenas paletas do Desktop. O modo selecionado é aplicado por cima.',
      themeProfileNote: profile => `Salvo para o perfil ${profile} — cada perfil mantém o próprio tema.`,
      installTitle: 'Instalar tema do VS Code',
      installDesc:
        'Cole o ID de uma extensão do Marketplace (por exemplo, dracula-theme.theme-dracula) para converter o tema de cores dela em uma paleta do Desktop.',
      installPlaceholder: 'publisher.extension',
      installButton: 'Instalar',
      installing: 'Instalando…',
      installError: 'Não foi possível instalar esse tema.',
      installed: name => `Instalado “${name}”.`,
      removeTheme: 'Remover tema',
      importedBadge: 'Importado',
      pet: {
        title: 'Mascote',
        intro:
          'Adote um mascote animado do petdex que flutua sobre o app e reage ao que o Hermes está fazendo — corre enquanto as ferramentas são executadas, comemora nos acertos e fica emburrado nos erros.',
        restartHint:
          'Os mascotes precisam de uma reinicialização rápida — o app em execução foi iniciado antes deste recurso existir. Feche e reabra o Hermes, depois volte aqui.',
        on: 'Ativado',
        off: 'Desativado',
        scaleTitle: 'Tamanho',
        scaleDesc: 'Redimensiona o mascote flutuante. A alteração se aplica imediatamente a todo o aplicativo.',
        roamTitle: 'Passear',
        roamDesc: 'Deixa o mascote circular pela janela sozinho enquanto está ocioso.',
        chooseTitle: 'Escolher um mascote',
        chooseDesc: 'Escolher um instala o mascote, se necessário, e o torna ativo.',
        searchPlaceholder: 'Pesquisar mascotes…',
        unreachable: 'Não foi possível acessar a galeria do petdex. Verifique sua conexão e reabra esta página.',
        noMatch: query => `Nenhum mascote corresponde a "${query}".`,
        installedTag: 'instalado',
        generatedTag: 'Gerado',
        countCapped: (cap, total) => `Mostrando ${cap} de ${total} — digite para refinar.`,
        count: n => `${n} mascote${n === 1 ? '' : 's'}.`,
        uninstall: name => `Desinstalar ${name}`,
        delete: name => `Excluir ${name}`,
        deleteTitle: (name: string) => `Excluir ${name}?`,
        deleteBody: 'Isso exclui o mascote permanentemente — não será possível reinstalá-lo.',
        deleteConfirm: 'Excluir',
        rename: name => `Renomear ${name}`,
        renameTitle: 'Renomear mascote',
        renamePlaceholder: 'Dê um nome ao seu mascote',
        renameSave: 'Salvar',
        exportPet: name => `Exportar ${name}`,
        adoptFailed: slug => `Não foi possível adotar ${slug}`,
        uninstallFailed: slug => `Não foi possível desinstalar ${slug}`,
        renameFailed: slug => `Não foi possível renomear ${slug}`,
        exportFailed: slug => `Não foi possível exportar ${slug}`,
        noneAvailable: 'Nenhum mascote disponível para ativar agora.',
        turnOnFailed: 'Não foi possível ligar o mascote.',
        turnOffFailed: 'Não foi possível desligar o mascote.'
      }
    },
    fieldLabels: defineFieldCopy({
      model: 'Modelo padrão',
      modelContextLength: 'Janela de contexto',
      fallbackProviders: 'Modelos de fallback',
      toolsets: 'Conjuntos de ferramentas ativados',
      timezone: 'Fuso horário',
      display: {
        personality: 'Personalidade',
        showReasoning: 'Blocos de raciocínio'
      },
      desktop: {
        repoScanEnabled: 'Descoberta automática de repositórios',
        repoScanRoots: 'Pastas raiz para descoberta de repositórios',
        repoScanExcludePaths: 'Caminhos de repositório excluídos'
      },
      agent: {
        maxTurns: 'Máximo de etapas do agente',
        imageInputMode: 'Anexos de imagem',
        apiMaxRetries: 'Tentativas da API',
        serviceTier: 'Nível de serviço',
        toolUseEnforcement: 'Obrigatoriedade de uso de ferramentas'
      },
      terminal: {
        cwd: 'Diretório de trabalho',
        backend: 'Backend de execução',
        timeout: 'Tempo limite do comando',
        persistentShell: 'Shell persistente',
        envPassthrough: 'Repasse de variáveis de ambiente',
        dockerImage: 'Imagem Docker',
        singularityImage: 'Imagem Singularity',
        modalImage: 'Imagem Modal',
        daytonaImage: 'Imagem Daytona'
      },
      fileReadMaxChars: 'Limite de leitura de arquivo',
      toolOutput: {
        maxBytes: 'Limite de saída do terminal',
        maxLines: 'Limite de paginação de arquivo',
        maxLineLength: 'Limite de comprimento da linha'
      },
      codeExecution: {
        mode: 'Modo de execução de código'
      },
      approvals: {
        mode: 'Modo de aprovação',
        timeout: 'Tempo limite de aprovação',
        mcpReloadConfirm: 'Confirmar recargas do MCP'
      },
      commandAllowlist: 'Lista de comandos permitidos',
      security: {
        redactSecrets: 'Ocultar segredos',
        allowPrivateUrls: 'Permitir URLs privadas'
      },
      browser: {
        allowPrivateUrls: 'URLs privadas no navegador',
        autoLocalForPrivateUrls: 'Navegador local para URLs privadas',
        useRealProfile: 'Usar meu perfil real do navegador'
      },
      checkpoints: {
        enabled: 'Checkpoints de arquivo',
        maxSnapshots: 'Limite de checkpoints'
      },
      voice: {
        recordKey: 'Atalho de voz',
        maxRecordingSeconds: 'Duração máxima da gravação',
        autoTts: 'Ler as respostas em voz alta'
      },
      stt: {
        enabled: 'Transcrição de voz',
        echoTranscripts: 'Repetir as transcrições',
        provider: 'Provedor de transcrição de voz',
        local: {
          model: 'Modelo local de transcrição',
          language: 'Idioma da transcrição'
        },
        openai: {
          model: 'Modelo STT da OpenAI'
        },
        groq: {
          model: 'Modelo STT da Groq'
        },
        mistral: {
          model: 'Modelo STT da Mistral'
        },
        elevenlabs: {
          modelId: 'Modelo STT da ElevenLabs',
          languageCode: 'Idioma da ElevenLabs',
          tagAudioEvents: 'Marcar eventos de áudio',
          diarize: 'Separação de locutores'
        }
      },
      tts: {
        provider: 'Provedor de síntese de voz',
        edge: {
          voice: 'Voz da Edge'
        },
        openai: {
          model: 'Modelo TTS da OpenAI',
          voice: 'Voz da OpenAI'
        },
        elevenlabs: {
          voiceId: 'Voz da ElevenLabs',
          modelId: 'Modelo da ElevenLabs'
        },
        xai: {
          voiceId: 'Voz da xAI (Grok)',
          language: 'Idioma da xAI',
          speed: 'Velocidade de reprodução da xAI',
          autoSpeechTags: 'Marcadores automáticos de fala da xAI',
          optimizeStreamingLatency: 'Otimização da latência de streaming da xAI',
          sampleRate: 'Taxa de amostragem da xAI',
          bitRate: 'Taxa de bits da xAI'
        },
        minimax: {
          model: 'Modelo TTS da MiniMax',
          voiceId: 'Voz da MiniMax'
        },
        mistral: {
          model: 'Modelo TTS da Mistral',
          voiceId: 'Voz da Mistral'
        },
        gemini: {
          model: 'Modelo TTS do Gemini',
          voice: 'Voz do Gemini'
        },
        neutts: {
          model: 'Modelo NeuTTS',
          device: 'Dispositivo do NeuTTS'
        },
        kittentts: {
          model: 'Modelo KittenTTS',
          voice: 'Voz do KittenTTS'
        },
        piper: {
          voice: 'Voz do Piper'
        },
        deepinfra: {
          model: 'Modelo TTS da DeepInfra',
          voice: 'Voz da DeepInfra'
        }
      },
      memory: {
        memoryEnabled: 'Memória persistente',
        userProfileEnabled: 'Perfil do usuário',
        memoryCharLimit: 'Orçamento de memória',
        userCharLimit: 'Orçamento do perfil',
        provider: 'Provedor de memória'
      },
      context: {
        engine: 'Motor de contexto'
      },
      compression: {
        enabled: 'Compressão automática',
        threshold: 'Limiar de compressão',
        targetRatio: 'Meta de compressão',
        protectLastN: 'Mensagens recentes protegidas'
      },
      delegation: {
        model: 'Modelo dos subagentes',
        provider: 'Provedor dos subagentes',
        maxIterations: 'Limite de turnos dos subagentes',
        maxConcurrentChildren: 'Subagentes em paralelo',
        childTimeoutSeconds: 'Tempo limite dos subagentes',
        reasoningEffort: 'Esforço de raciocínio dos subagentes'
      },
      updates: {
        nonInteractiveLocalChanges: 'Alterações locais durante a atualização pelo app'
      }
    }),
    fieldDescriptions: defineFieldCopy({
      model: 'Usado em novas conversas, a menos que você escolha outro modelo no compositor.',
      modelContextLength: 'Deixe em 0 para usar a janela de contexto detectada do modelo selecionado.',
      fallbackProviders: 'Entradas provedor:modelo de reserva a serem tentadas caso o modelo padrão falhe.',
      display: {
        personality: 'Estilo padrão do assistente em novas sessões.',
        showReasoning: 'Mostra as seções de raciocínio quando o backend as fornece.'
      },
      desktop: {
        repoScanEnabled: 'Procura repositórios Git em pastas locais para exibi-los em Projetos.',
        repoScanRoots: 'Pastas a examinar. Deixe vazio para examinar sua pasta pessoal.',
        repoScanExcludePaths: 'Pastas e seus descendentes a ignorar durante a descoberta de repositórios.'
      },
      timezone: 'Usado quando o Hermes precisa do contexto do horário local. Em branco, usa o fuso horário do sistema.',
      agent: {
        imageInputMode: 'Controla como os anexos de imagem são enviados ao modelo.',
        maxTurns: 'Limite máximo de turnos com chamadas de ferramentas antes de o Hermes encerrar uma execução.'
      },
      terminal: {
        cwd: 'Pasta padrão do projeto para trabalho com ferramentas e terminal.',
        persistentShell: 'Mantém o estado do shell entre comandos quando o backend oferece suporte.',
        envPassthrough: 'Variáveis de ambiente repassadas para a execução das ferramentas.',
        dockerImage: 'Imagem de contêiner usada quando o backend de execução é o Docker.',
        singularityImage: 'Imagem usada quando o backend de execução é o Singularity.',
        modalImage: 'Imagem usada quando o backend de execução é o Modal.',
        daytonaImage: 'Imagem usada quando o backend de execução é o Daytona.'
      },
      codeExecution: {
        mode: 'Define com que rigor a execução de código fica restrita ao projeto atual.'
      },
      fileReadMaxChars: 'Máximo de caracteres que o Hermes pode ler em uma única solicitação de arquivo.',
      approvals: {
        mode: 'Define como o Hermes lida com comandos que exigem aprovação explícita.',
        timeout: 'Tempo que as solicitações de aprovação aguardam antes de expirar.'
      },
      security: {
        redactSecrets: 'Oculta, quando possível, segredos detectados do conteúdo visível ao modelo.'
      },
      browser: {
        useRealProfile:
          'A navegação local usa seus logins reais. O Hermes copia o perfil do navegador padrão — cookies, logins e preferências — para um snapshot gerenciado e o controla com o Chromium integrado; seu perfil ativo nunca é aberto diretamente, e a cópia é atualizada a cada execução. Também permite que o agente abra uma sessão local com perfil real quando solicitado, mesmo se houver um navegador em nuvem configurado. Apenas navegadores Chromium — Chrome, Edge, Brave, Brave Origin e Chromium — são compatíveis; se o navegador padrão não for Chromium, uma mensagem clara será exibida. Desativado por padrão.'
      },
      checkpoints: {
        enabled: 'Cria snapshots para restauração antes de editar arquivos.'
      },
      memory: {
        memoryEnabled: 'Salva memórias persistentes que podem ajudar em sessões futuras.',
        userProfileEnabled: 'Mantém um perfil compacto das preferências do usuário.'
      },
      context: {
        engine: 'Estratégia usada para gerenciar conversas longas próximas do limite de contexto.'
      },
      compression: {
        enabled: 'Resume o contexto mais antigo quando as conversas ficam grandes.'
      },
      voice: {
        autoTts: 'Lê automaticamente em voz alta as respostas do assistente.'
      },
      tts: {
        xai: {
          voiceId: 'ID de voz da xAI (por exemplo, eve) ou um ID de voz personalizado.',
          language: 'Código do idioma falado (por exemplo, en, pt-BR) ou "auto" para detecção automática.',
          speed: 'Velocidade de reprodução. 0,7 = mais lento, 1,0 = normal, 1,5 = mais rápido.',
          autoSpeechTags:
            'Permite que um LLM insira marcadores expressivos de áudio ([laughing], [sighs]) no roteiro antes da síntese.',
          optimizeStreamingLatency: 'Equilíbrio entre latência e qualidade. 0 = melhor qualidade, 2 = menor latência.',
          sampleRate:
            'Taxa de amostragem do áudio em Hz. Valores maiores oferecem melhor qualidade e arquivos maiores.',
          bitRate: 'Taxa de bits do MP3 em bps. Aplica-se apenas quando o codec é mp3.'
        },
        neutts: {
          device: 'Dispositivo de inferência local usado pelo NeuTTS.'
        }
      },
      stt: {
        enabled: 'Ativa a transcrição de fala local ou por provedor.',
        echoTranscripts: 'Publica de volta na conversa a transcrição bruta das mensagens de voz.',
        elevenlabs: {
          languageCode: 'Código de idioma ISO-639-3 opcional. Em branco, a ElevenLabs detecta automaticamente.'
        }
      },
      updates: {
        nonInteractiveLocalChanges:
          'Quando o Hermes é atualizado pelo app sem uma pergunta no terminal, define se as alterações locais no código-fonte serão preservadas (stash) ou descartadas (discard). Atualizações pelo terminal sempre perguntam.'
      }
    }),
    about: {
      heading: 'Hermes Desktop',
      version: value => `versão ${value}`,
      versionUnavailable: 'Versão indisponível',
      updates: 'Atualizações',
      checkNow: 'Verificar agora',
      checking: 'Verificando…',
      seeWhatsNew: 'Ver novidades',
      updateNow: 'Atualizar agora',
      releaseNotes: 'Notas da versão',
      onLatest: 'Você está na versão mais recente.',
      installing: 'Uma atualização está sendo instalada.',
      cantUpdate: 'Esta versão não consegue se atualizar de dentro do app.',
      cantReach: 'Não foi possível acessar o servidor de atualizações.',
      tapCheck: 'Clique em "Verificar agora" para procurar atualizações.',
      updateReady: count =>
        `Uma nova atualização está pronta (${count} ${count === 1 ? 'mudança incluída' : 'mudanças incluídas'}).`,
      lastChecked: age => `Última verificação ${age}`,
      justNowSuffix: ' · agora mesmo',
      automaticUpdates: 'Atualizações automáticas',
      automaticUpdatesDesc:
        'O Hermes procura atualizações automaticamente em segundo plano e avisa quando uma estiver pronta.',
      branchCommit: (branch, commit) => `Branch ${branch} · Commit ${commit}`,
      never: 'nunca',
      justNow: 'agora mesmo',
      minAgo: count => `${count} min atrás`,
      hoursAgo: count => `${count} ${count === 1 ? 'hora' : 'horas'} atrás`,
      daysAgo: count => `${count} ${count === 1 ? 'dia' : 'dias'} atrás`
    },
    config: {
      none: 'Nenhum',
      noneParen: '(nenhum)',
      builtinOnly: 'Somente integrado',
      notSet: 'Não definido',
      commaSeparated: 'valores separados por vírgula',
      searchPlaceholder: 'Pesquisar…',
      noResults: 'Nenhum resultado encontrado',
      systemDefault: 'Padrão do sistema',
      loading: 'Carregando a configuração do Hermes...',
      emptyTitle: 'Nada para configurar',
      emptyDesc: 'Esta seção não tem configurações ajustáveis.',
      failedLoad: 'Falha ao carregar as configurações',
      autosaveFailed: 'Falha ao salvar automaticamente',
      imported: 'Configuração importada',
      invalidJson: 'JSON de configuração inválido',
      keepAwakeTitle: 'Manter o computador ativo',
      keepAwakeDesc:
        'Impede que esta máquina entre em suspensão, para que execuções longas ou noturnas continuem. A tela ainda pode escurecer.',
      attachmentSizeTitle: 'Tamanho máximo para prévias e carregamento de imagens',
      attachmentSizeDesc:
        'Define o tamanho máximo, em MB, de um arquivo local que o Desktop carrega para prévias e anexos de imagem. O padrão é 16. Anexos remotos que não sejam imagens têm um limite separado de 256 MB. Valores muito altos carregam o arquivo inteiro na memória e podem travar ou encerrar o app.',
      attachmentSizeUnit: 'MB',
      attachmentSizeLabel: 'Tamanho máximo, em MB, para prévias e carregamento de imagens'
    },
    quickEntry: {
      enabledTitle: 'Entrada rápida',
      enabledDesc:
        'Abra um pequeno compositor de qualquer lugar com um atalho global e envie um prompt sem abrir o Hermes.',
      shortcutTitle: 'Atalho de entrada rápida',
      shortcutDesc: 'Precisa de pelo menos um modificador, por exemplo CommandOrControl+Shift+Space.',
      active: 'O atalho está ativo.',
      takenBy: 'Outro app já usa este atalho — escolha outro.',
      invalidShortcut: 'Atalho inválido. Inclua pelo menos uma tecla modificadora.'
    },
    credentials: {
      pasteKey: 'Colar chave',
      pasteLabelKey: label => `Colar a chave ${label}`,
      optional: 'Opcional',
      enterValueFirst: 'Digite um valor primeiro.',
      couldNotSave: 'Não foi possível salvar a credencial.',
      remove: 'Remover',
      getKey: 'Obter uma chave',
      saving: 'Salvando'
    },
    envActions: {
      actions: 'Ações',
      manageInKeys: 'Gerenciar em Chaves de API',
      docs: 'Documentação',
      hideValue: 'Ocultar valor',
      revealValue: 'Exibir valor',
      replace: 'Substituir',
      set: 'Definir',
      clear: 'Limpar'
    },
    managedUpdates: {
      title: 'Atualizações gerenciadas',
      intro:
        'Atualize instalações SSH gerenciadas pelo Desktop de forma transacional: sessões são esvaziadas, o repositório remoto é atualizado e cada perfil é restaurado com um recibo correlacionado.',
      sshConnection: 'Instalação SSH gerenciada pelo Desktop',
      update: 'Atualizar',
      updating: 'Atualizando…',
      progress: 'Esvaziando sessões, atualizando a instalação remota e restaurando perfis…',
      updated: 'Atualizado',
      partial: 'Atualizado — falha na restauração',
      refused: 'Recusado',
      failed: 'Falha na atualização',
      alreadyRunning: 'Atualização já em andamento',
      receipt: (id: string, outcome: string) => `Recibo ${id} · ${outcome}`,
      receiptVersions: (pre: string, post: string) => `${pre} → ${post}`,
      scopesRestored: (profiles: string) => `Perfis restaurados: ${profiles}`,
      scopeNotRestored: (profile: string, error: string) => `Perfil “${profile}” não restaurado: ${error}`
    },
    gateway: {
      loading: 'Carregando as configurações do gateway...',
      unavailableTitle: 'Configurações do gateway indisponíveis',
      unavailableDesc: 'A ponte IPC do desktop não expõe as configurações do gateway.',
      title: 'Conexão do gateway',
      envOverride: 'sobrescrito por env',
      intro:
        'Local por padrão. Use remoto quando este app precisar controlar um backend do Hermes em outro lugar. As substituições por perfil ficam abaixo.',

      envOverrideTitle: 'Variáveis de ambiente estão controlando esta sessão do desktop.',
      envOverrideDesc:
        'Remova HERMES_DESKTOP_REMOTE_URL e HERMES_DESKTOP_REMOTE_TOKEN para usar a configuração salva abaixo.',
      modeTitle: 'Modo de conexão',
      localTitle: 'Gateway local',
      localDesc: 'Inicia um backend privado do Hermes em localhost. É o padrão e funciona offline.',

      remoteTitle: 'Gateway remoto',
      remoteDesc: 'Conecta esta interface de desktop a um backend remoto do Hermes.',
      remoteAuthHint:
        'Gateways hospedados usam OAuth ou usuário e senha; os auto-hospedados podem usar um token de sessão.',
      cloudTitle: 'Hermes Cloud',
      cloudDesc:
        'Entre uma vez no Hermes Cloud e escolha entre os agentes da sua conta — sem precisar colar nenhuma URL.',
      cloudSignInTitle: 'Hermes Cloud',
      cloudSignIn: 'Entrar no Hermes Cloud',
      cloudSignedIn: 'Conectado ao Hermes Cloud',
      cloudNeedsSignIn: 'Entre no Hermes Cloud para descobrir os agentes da sua conta.',
      cloudSignedInDesc: 'Você está conectado. Escolha um agente abaixo; a sessão é renovada automaticamente.',
      cloudAgentsTitle: 'Seus agentes',
      cloudOrgPickerTitle: 'Escolher uma organização',
      cloudOrgSelect: 'Selecionar',
      cloudOrgChange: 'Alterar organização',
      cloudOrgRole: role => `função: ${role}`,
      cloudLoadingAgents: 'Carregando seus agentes…',
      cloudNoAgents: {
        before: 'Nenhum agente encontrado nesta conta. Crie um no ',
        linkText: 'portal da Nous',
        after: ' e atualize.'
      },
      cloudRefresh: 'Atualizar',
      cloudConnect: 'Conectar',
      cloudConnecting: 'Conectando…',
      cloudDiscoverFailed: 'Não foi possível carregar seus agentes do Hermes Cloud',
      cloudConnectFailed: 'Não foi possível conectar a esse agente',
      cloudSignInFailed: 'Falha ao entrar no Hermes Cloud',
      cloudSignedOutTitle: 'Desconectado do Hermes Cloud',
      cloudSignedOutMessage: 'A sessão do Hermes Cloud foi limpa.',
      cloudConnectedTitle: 'Conectado',
      cloudConnectedPill: 'Conectado',
      cloudConnectedTo: name => `Conectado a ${name}.`,
      cloudAgentProvisioning: 'Provisionando…',
      cloudStatusLabel: status => `status: ${status}`,
      remoteUrlTitle: 'URL remota',
      remoteUrlDesc:
        'URL base do backend remoto do dashboard. Prefixos de caminho são suportados, por exemplo /hermes.',
      probing: 'Verificando como este gateway autentica…',
      probeError:
        'Ainda não foi possível acessar este gateway. Verifique a URL — o método de autenticação aparece assim que ele responder.',
      signedIn: 'Conectado',
      signIn: 'Entrar',
      signOut: 'Sair',
      signInWith: provider => `Entrar com ${provider}`,
      authTitle: 'Autenticação',
      authSignedInPassword:
        'Este gateway usa usuário e senha. Você está conectado; a sessão é renovada automaticamente.',
      authSignedInOauth: 'Este gateway usa OAuth. Você está conectado; a sessão é renovada automaticamente.',
      authNeedsPassword: 'Este gateway usa usuário e senha. Entre para autorizar este app de desktop.',
      authNeedsOauth: provider => `Este gateway usa OAuth. Entre com ${provider} para autorizar este app de desktop.`,
      tokenTitle: 'Token da sessão',
      tokenDesc:
        'O token de sessão do dashboard usado para acesso REST e WebSocket. Deixe em branco para manter o token salvo.',
      existingToken: value => `Token existente ${value}`,
      savedToken: 'salvo',
      pasteSessionToken: 'Colar token da sessão',
      plainTextConfirmTitle: 'Armazenar o token do gateway em texto simples?',
      plainTextConfirmDesc:
        'Nenhum serviço de chaveiro do sistema operacional foi encontrado nesta máquina, então o token será salvo sem criptografia no arquivo de configurações de conexão do aplicativo, podendo ser lido por qualquer processo executado como este usuário. Instale ou habilite o GNOME Keyring ou o KWallet para armazená-lo de forma criptografada.',
      plainTextConfirmAction: 'Salvar como texto simples',
      plainTextStoredTitle: 'Token armazenado em texto simples',
      plainTextStoredDesc:
        'O armazenamento seguro não está disponível, então o token é armazenado sem criptografia no arquivo de configurações de conexão do aplicativo nesta máquina. Instale ou habilite o GNOME Keyring ou o KWallet para armazená-lo de forma criptografada.',
      keychainEncryptionTitle: 'Criptografar segredos salvos com o keychain do SO',
      keychainEncryptionDesc:
        'Desativado por padrão. Quando ativado, os tokens de gateway e as credenciais de login são criptografados com o keychain do seu sistema (Keychain Access, GNOME Keyring ou Windows DPAPI) — seu sistema pode solicitar permissão ou senha. Quando desativado, eles são armazenados como arquivos de texto simples legíveis apenas pela sua conta de usuário.',
      keychainEncryptionFailed: 'Não foi possível alterar a criptografia de segredos',
      testRemote: 'Testar remoto',
      saveForRestart: 'Salvar para a próxima reinicialização',
      saveAndReconnect: 'Salvar e reconectar',
      diagnostics: 'Diagnóstico',
      diagnosticsDesc:
        'Mostra o desktop.log no seu gerenciador de arquivos — útil quando o gateway não consegue iniciar.',
      openLogs: 'Abrir logs',
      incompleteTitle: 'Gateway remoto incompleto',
      incompleteSignIn: 'Informe uma URL remota e entre antes de trocar para remoto.',
      incompleteToken: 'Informe uma URL remota e o token de sessão antes de trocar para remoto.',
      incompleteSignInTest: 'Informe uma URL remota e entre antes de testar.',
      incompleteTokenTest: 'Informe uma URL remota e o token de sessão antes de testar.',
      enterUrlFirst: 'Informe uma URL remota primeiro.',
      restartingTitle: 'Reiniciando a conexão com o gateway',
      savedTitle: 'Configurações do gateway salvas',
      restartingMessage: 'O Hermes Desktop vai reconectar usando as configurações salvas — a janela continua aberta.',
      savedMessage: 'Salvo para a próxima reinicialização.',
      connectedTo: (baseUrl, version) => `Conectado a ${baseUrl}${version ? ` · Hermes ${version}` : ''}`,
      reachableTitle: 'Gateway remoto acessível',
      signedOutTitle: 'Sessão encerrada',
      signedOutMessage: 'A sessão do gateway remoto foi limpa.',
      failedLoad: 'Falha ao carregar as configurações do gateway',
      signInFailed: 'Falha ao entrar',
      signOutFailed: 'Falha ao sair',
      testFailed: 'Falha ao testar o gateway remoto',
      applyFailed: 'Não foi possível aplicar as configurações do gateway',
      saveFailed: 'Não foi possível salvar as configurações do gateway',
      sshTitle: 'Conectar via SSH',
      sshDesc:
        'O Hermes é iniciado na máquina remota por SSH e tunelado até este app — você não precisa iniciar nem expor nada. Requer acesso SSH por chave funcionando no host.',
      sshTrustHint:
        'A primeira chave de host apresentada é considerada confiável e fica fixada; mudanças posteriores falham por segurança.',
      sshHostTitle: 'Host',
      sshHostDesc: 'usuário@host ou um alias Host do ~/.ssh/config.',
      sshHostPick: 'Selecionar o host…',
      sshHostPickTitle: 'Host',
      sshHostPickDesc: 'Um alias Host do ~/.ssh/config, ou Personalizado para digitar manualmente.',
      sshHostCustom: 'Personalizado (digitar manualmente)…',
      sshUserTitle: 'Usuário',
      sshUserDesc: 'Em branco = ~/.ssh/config ou seu usuário atual.',
      sshUserPlaceholder: 'de ~/.ssh/config',
      sshPortTitle: 'Porta',
      sshPortDesc: 'Em branco = 22 ou a porta do ~/.ssh/config.',
      sshKeyTitle: 'Arquivo de identidade',
      sshKeyDesc: 'Caminho da chave privada. Em branco = ssh-agent ou ~/.ssh/config.',
      sshHermesPathTitle: 'Caminho do Hermes (opcional)',
      sshHermesPathDesc: 'Caminho completo do binário hermes na máquina remota. Em branco = detecção automática.',
      sshHermesPathPlaceholder: 'detecção automática',

      sshTestConnection: 'Testar SSH',
      sshConnect: 'Conectar',
      sshButtonsHint: 'Salvar aplica na próxima inicialização. Conectar reconecta agora.',
      sshReachable: (host, platform) => `Acessível: ${host} (${platform}) — Hermes encontrado`,
      sshIncompleteHost: 'Informe um host SSH antes de conectar.',
      sshErrUnreachable: 'Não foi possível acessar esse host por SSH. Verifique o host, a porta e sua rede.',
      sshErrAuth:
        'Falha na autenticação SSH. Carregue sua chave no ssh-agent (ssh-add) ou defina um IdentityFile no ~/.ssh/config — o Hermes executa o ssh de forma não interativa.',
      sshErrHostKey:
        'A chave do host MUDOU desde a última conexão. Confirme que isso era esperado, execute ssh-keygen -R <host> e reconecte.',
      sshErrNotInstalled:
        'O Hermes não está instalado no host remoto. Instale-o lá (curl -fsSL https://hermes-agent.nousresearch.com/install.sh | sh) ou defina o caminho do Hermes.',
      sshErrPlatform:
        'Plataforma remota não suportada. O modo SSH do Hermes Desktop suporta hosts remotos Linux, macOS e Windows.',
      sshErrTimeout: 'A conexão SSH expirou. O host pode estar inacessível ou suspenso.',
      sshErrUpdateRequired: 'Atualize o Hermes no host remoto antes de conectar pelo SSH do Desktop.',
      sshErrUnknown: 'A conexão SSH falhou.'
    },
    keys: {
      loading: 'Carregando chaves de API e credenciais...',
      failedLoad: 'Falha ao carregar as chaves de API',
      empty: 'Nada configurado nesta categoria ainda.'
    },
    mcp: {
      loading: 'Carregando servidores MCP...',
      failedLoad: 'Falha ao carregar a configuração MCP',
      nameRequiredTitle: 'Nome obrigatório',
      nameRequiredMessage: 'Defina uma chave de configuração para este servidor MCP.',
      objectRequired: 'A configuração do servidor precisa ser um objeto JSON',
      invalidJson: 'JSON do MCP inválido',
      saveFailed: 'Falha ao salvar',
      removeFailed: 'Falha ao remover',
      gatewayUnavailableTitle: 'Gateway indisponível',
      gatewayUnavailableMessage: 'Reconecte o gateway antes de recarregar o MCP.',
      reloadedTitle: 'Ferramentas MCP recarregadas',
      reloadedMessage: 'Os novos schemas de ferramentas serão usados nos próximos turnos.',
      reloadFailed: 'Falha ao recarregar MCP',
      savedTitle: 'Servidor MCP salvo',
      savedMessage: name => `${name} passa a valer depois de recarregar o MCP.`,
      newServer: 'Novo servidor',
      reload: 'Recarregar MCP',
      reloading: 'Recarregando...',
      emptyTitle: 'Nenhum servidor MCP',
      emptyDesc: 'Adicione um servidor stdio ou HTTP para expor ferramentas MCP.',
      disabled: 'desativado',
      editServer: 'Editar servidor',
      name: 'Nome',
      serverJson: 'JSON do servidor',
      remove: 'Remover',
      saveServer: 'Salvar servidor',
      test: 'Testar conexão',
      testing: 'Testando...',
      testOk: count => `Conectado — ${count} ${count === 1 ? 'ferramenta disponível' : 'ferramentas disponíveis'}`,
      testFailed: 'Falha na conexão',
      enableServer: name => `Ativar ${name}`,
      disableServer: name => `Desativar ${name}`,
      serverEnabled: name => `${name} ativado — vale para novas sessões.`,
      serverDisabled: name => `${name} desativado — vale para novas sessões.`,
      toggleFailed: (name, enabled) => `Falha ao ${enabled ? 'ativar' : 'desativar'} ${name}`,
      tabServers: 'Servidores',
      tabCatalog: 'Catálogo',
      catalogLoading: 'Carregando o catálogo do MCP...',
      catalogLoadFailed: 'Falha ao carregar o catálogo MCP',
      catalogEmpty: 'Nenhuma entrada disponível no catálogo.',
      catalogInstalled: 'Instalado',
      catalogEnabled: 'Ativado',
      catalogNeedsInstall: 'Requer instalação',
      catalogInstall: 'Instalar',
      catalogInstalling: 'Instalando...',
      catalogInstallStarted: name => `Instalação de ${name} iniciada… valerá para novas sessões quando terminar.`,
      catalogInstallFailed: name => `Falha ao instalar ${name}`,
      catalogEnvPrompt: name => `${name} exige credenciais`,
      catalogEnvRequired: 'Preencha os valores obrigatórios antes de instalar.',
      capabilitySummary: (tools, prompts, resources) =>
        `${[`${tools} ferramentas`, ...(prompts ? [`${prompts} prompts`] : []), ...(resources ? [`${resources} recursos`] : [])].join(', ')} ativados`,
      statusConnecting: 'Conectando…',
      statusNeedsAuth: 'Requer autenticação',
      statusError: 'Erro',
      statusOff: 'Desativado',
      allServers: 'todos os servidores',
      authenticatedTitle: 'Autenticado',
      authenticatedMessage: (server, count) => `${server}: ${count} ferramentas`,
      waitingForBrowser: 'Aguardando o navegador…',
      authenticate: 'Autenticar',
      unsavedConnect: 'Não salvo — salve o mcp.json para conectar.',
      enableTool: tool => `Ativar ${tool}`,
      disableTool: tool => `Desativar ${tool}`,
      noOutput: 'Nenhuma saída ainda.'
    },
    model: {
      loading: 'Carregando a configuração do modelo...',
      appliesDesc:
        'Vale para novas sessões. Use o seletor de modelos no compositor para trocar o modelo da conversa ativa na hora.',
      provider: 'Provedor',
      model: 'Modelo',
      applying: 'Aplicando...',
      defaultsLabel: 'Padrões',
      reasoning: 'Raciocínio',
      reasoningOff: 'Desativado',
      defaultsFailed: 'Falha ao salvar os padrões do modelo',
      auxiliaryTitle: 'Modelos auxiliares',
      resetAllToMain: 'Redefinir todos para o principal',
      auxiliaryDesc:
        'Por padrão, as tarefas de apoio rodam no modelo principal. Atribua um modelo dedicado a qualquer tarefa para sobrescrever.',
      setToMain: 'Definir como principal',
      change: 'Alterar',
      autoUseMain: 'automático · usa o modelo principal',
      providerDefault: '(Provedor padrão)',
      fallbackAdd: 'Adicionar fallback',
      fallbackEmpty: 'Nenhum modelo de fallback — o modelo padrão é usado a menos que falhe.',
      notInCatalog: 'não está na lista de modelos deste provedor — as chamadas podem recair em um modelo de reserva.',
      tasks: {
        vision: { label: 'Visão', hint: 'Análise de imagem' },
        compression: { label: 'Compactação', hint: 'Compactação de contexto' },
        skills_hub: { label: 'Hub de habilidades', hint: 'Busca de habilidades' },
        approval: { label: 'Aprovação', hint: 'Aprovação automática inteligente' },
        mcp: { label: 'MCP', hint: 'Roteamento de ferramentas MCP' },
        title_generation: { label: 'Geração de título', hint: 'Títulos das sessões' },
        curator: { label: 'Curador', hint: 'Revisão do uso de habilidades' }
      }
    },
    providers: {
      connectAccount: 'Conectar uma conta',
      haveApiKey: 'Prefere usar uma chave de API?',
      intro:
        'Entre com uma assinatura — sem chave de API para copiar. O Hermes faz o login pelo navegador para você, aqui mesmo no app.',
      connected: 'Conectado',
      collapse: 'Recolher',
      connectAnother: 'Conectar outro provedor',
      otherProviders: 'Outros provedores',
      disconnect: 'Desconectar',
      disconnectInTerminal: 'Desconectar (executa o comando de remoção no terminal)',
      removeConfirm: provider => `Remover ${provider}?`,
      removeExternalGeneric: provider => `${provider} é gerenciado pela CLI dele — remova por lá.`,
      removeKeyManaged: provider => `${provider} é configurado por uma chave de API. Remova a chave em Chaves de API.`,
      removeTerminalConfirm: (provider, command) =>
        `Desconectar ${provider}? Isso executa "${command}" no terminal para limpar a credencial.`,
      removeTerminalRunning: provider => `Executando a desconexão de ${provider} no terminal…`,
      removedTitle: 'Conta removida',
      removedMessage: provider => `${provider} foi removido.`,
      failedRemove: provider => `Não foi possível remover ${provider}`,
      noProviderKeys: 'Nenhuma chave de API de provedor disponível.',
      searchKeys: 'Pesquisar provedores…',
      noKeysMatch: 'Nenhum provedor corresponde à sua pesquisa.',
      localEndpoint: {
        title: 'Endpoint local / personalizado',
        description:
          'Aponte o Hermes para qualquer endpoint compatível com a OpenAI (Zyphra, vLLM, llama.cpp, Ollama etc).'
      },
      loading: 'Carregando provedores...'
    },
    sessions: {
      loading: 'Carregando sessões arquivadas…',
      archivedTitle: 'Sessões arquivadas',
      archivedIntro:
        'As conversas arquivadas ficam ocultas na barra lateral, mas mantêm todas as mensagens. Ctrl/⌘-clique em uma conversa na barra lateral para arquivá-la.',
      emptyArchivedTitle: 'Nada arquivado',
      emptyArchivedDesc: 'Arquive uma conversa para ocultá-la aqui.',
      unarchive: 'Desarquivar',
      deletePermanently: 'Excluir permanentemente',
      messages: count => `${count} ${count === 1 ? 'mensagem' : 'mensagens'}`,
      restored: 'Restaurado',
      deleteConfirm: title => `Excluir "${title}" permanentemente? Isso não pode ser desfeito.`,
      autoArchiveTitle: 'Arquivar conversas paradas automaticamente',
      autoArchiveDesc:
        'Arquiva automaticamente as conversas em que você não mexe há um tempo. Conversas fixadas nunca são arquivadas e nada é excluído — as arquivadas apenas vêm para cá.',
      autoArchiveDaysLabel: 'Arquivar após',
      autoArchiveDaysUnit: 'dias de inatividade',
      autoArchiveFailed: 'Não foi possível atualizar o arquivamento automático',
      defaultDirTitle: 'Diretório padrão do projeto',
      defaultDirDesc:
        'As novas sessões começam nesta pasta, a menos que você escolha outra. Deixe em branco para usar sua pasta pessoal.',
      defaultDirUpdated:
        'Pasta padrão de projetos atualizada — inicie uma nova conversa (Ctrl/⌘+N) para que passe a valer',
      defaultsTo: label => `Padrão: ${label}.`,
      change: 'Alterar',
      choose: 'Escolher',
      clear: 'Limpar',
      notSet: 'Não definido',
      failedLoad: 'Não foi possível carregar as sessões arquivadas',
      unarchiveFailed: 'Falha ao desarquivar',
      deleteFailed: 'Falha ao excluir',
      updateDirFailed: 'Não foi possível atualizar o diretório padrão',
      clearDirFailed: 'Não foi possível limpar o diretório padrão'
    },
    toolsets: {
      loadingConfig: 'Carregando a configuração',
      savedTitle: 'Credencial salva',
      savedMessage: key => `${key} atualizada.`,
      removedTitle: 'Credencial removida',
      removedMessage: key => `${key} removida.`,
      failedSave: key => `Falha ao salvar ${key}`,
      failedRemove: key => `Falha ao remover ${key}`,
      failedReveal: key => `Falha ao revelar ${key}`,
      removeConfirm: key => `Remover ${key} de .env?`,
      set: 'Definir',
      notSet: 'Não definido',
      selectedTitle: 'Provedor selecionado',
      selectedMessage: provider => `${provider} está ativo agora.`,
      failedSelect: provider => `Falha ao selecionar ${provider}`,
      failedLoad: 'Falha ao carregar a configuração das ferramentas',
      noProviderOptions:
        'Este conjunto de ferramentas não tem opções de provedor — ative-o para que funcione com a configuração atual.',
      noProviders: 'Nenhum provedor disponível para este conjunto de ferramentas no momento.',
      ready: 'Pronto',
      needsSignIn: 'Requer login',
      needsSetup: 'Configuração necessária',
      activeBackend: 'Ativo',
      activeBackendHint: 'Este é o seu backend ativo',
      useBackend: 'Usar este backend',
      nousIncluded: 'Incluído com uma assinatura Nous — entre no Nous Portal para ativar.',
      nousAuthNeededTitle: 'Entrar no Nous Portal',
      nousAuthNeededMessage: provider =>
        `${provider} está salvo, mas só ficará ativo depois que você entrar no Nous Portal.`,
      nousAuthSignIn: 'Entrar',
      nousAuthDoneTitle: 'Nous Portal conectado',
      nousAuthDoneMessage: 'Os backends da sua assinatura estão ativos agora.',
      nousAuthFailed: 'O login no Nous Portal não foi concluído',
      noApiKeyRequired: 'Nenhuma chave de API necessária.',
      postSetupHint: step =>
        `Este backend precisa de uma instalação única (${step}). É executado nesta máquina — pode levar alguns minutos.`,
      postSetupInstalledHint: 'Instalado. Refaça a configuração apenas se algo estiver quebrado.',
      postSetupRun: 'Executar configuração',
      postSetupRerun: 'Executar novamente',
      postSetupInstalled: 'Instalado',
      postSetupRunning: 'Instalando…',
      postSetupStarting: 'Iniciando…',
      postSetupCompleteTitle: 'Configuração concluída',
      postSetupCompleteMessage: step => `${step} instalado.`,
      postSetupErrorTitle: 'A configuração terminou com erros',
      postSetupErrorMessage: step => `Verifique o log de ${step}.`,
      postSetupFailed: step => `Falha ao executar a configuração de ${step}`,
      webSearchActive: backend => `Pesquisar: ${backend}`,
      webExtractActive: backend => `Extração: ${backend}`,
      webCapabilityUnset: 'não definido',
      webUseForSearch: 'Usar para Busca',
      webUseForExtract: 'Usar para Extração',
      webUsedForSearch: 'Backend de busca',
      webUsedForExtract: 'Backend de extração',
      webCapabilitySelectedMessage: (provider, capability) => `${provider} agora cuida de ${capability} na web.`,
      failedSelectCapability: provider => `Falha ao definir ${provider}`,
      loadingModels: 'Carregando o catálogo de modelos...',
      modelSectionTitle: 'Modelo',
      modelCount: count => `${count} modelo${count === 1 ? '' : 's'}`,
      modelInUse: 'Em uso',
      modelDefault: 'padrão',
      modelInactiveHint: 'Selecione este backend primeiro para trocar o modelo dele.',
      modelSelectedTitle: 'Modelo selecionado',
      modelSelectedMessage: model => `${model} vale para novas sessões.`,
      failedSelectModel: model => `Falha ao selecionar ${model}`,
      terminalBackend: {
        sectionTitle: 'Backend de execução',
        loading: 'Verificando os backends de execução…',
        failedLoad: 'Não foi possível carregar os backends de terminal',
        ready: 'Pronto',
        needsSetup: 'Precisa de configuração',
        unavailable: 'Indisponível',
        inUse: 'Em uso',
        selectedTitle: 'Backend selecionado',
        selectedMessage: backend => `Os comandos de terminal agora rodam via ${backend}. Vale para novas sessões.`,
        failedSelect: backend => `Falha ao selecionar ${backend}`,
        needsSetupHint:
          'Você pode selecionar este backend agora — os comandos vão falhar até a configuração ser concluída.'
      }
    }
  },

  skills: {
    officialCatalog: 'Disponível para instalar',
    officialPill: 'Oficial',
    tabSkills: 'Habilidades',
    tabToolsets: 'Ferramentas',
    tabMcp: 'MCP',

    all: 'Todas',
    searchSkills: 'Buscar habilidades...',
    searchToolsets: 'Buscar ferramentas...',
    refresh: 'Atualizar habilidades',
    refreshing: 'Atualizando habilidades',
    loading: 'Carregando capacidades...',
    noSkillsTitle: 'Nenhuma habilidade encontrada',
    noSkillsDesc: 'Tente uma busca mais ampla ou outra categoria.',
    noToolsetsTitle: 'Nenhum conjunto de ferramentas encontrado',
    noToolsetsDesc: 'Tente uma busca mais ampla.',
    noDescription: 'Sem descrição.',
    configured: 'Configurado',
    needsKeys: 'Precisa de chaves',
    visionModelHint:
      'A visão usa a configuração dos seus modelos auxiliares — o modelo com suporte a imagens é escolhido lá, não nesta configuração de provedor.',
    visionModelLink: 'Escolher o modelo de visão em Configurações → Modelos',
    toolsetsEnabled: (enabled, total) => `${enabled}/${total} conjuntos de ferramentas ativos`,
    configureToolset: label => `Configurar ${label}`,
    toggleToolset: (label, enabled) =>
      `Alternar o conjunto de ferramentas ${label} ${enabled ? 'ligado' : 'desligado'}`,
    skillsLoadFailed: 'Falha ao carregar as habilidades',
    toolsetsRefreshFailed: 'Falha ao atualizar os conjuntos de ferramentas',
    skillEnabled: 'Habilidade ativada',
    skillDisabled: 'Habilidade desativada',
    toolsetEnabled: 'Conjunto de ferramentas ativado',
    toolsetDisabled: 'Conjunto de ferramentas desativado',
    appliesToNewSessions: name => `${name} vale para novas sessões.`,
    failedToUpdate: name => `Falha ao atualizar ${name}`,
    sortMostUsed: 'Mais usadas',
    sortAlpha: 'A–Z',
    sortMostUsedDesc: '↓ Mais usadas',
    sortLeastUsedAsc: '↑ Menos usadas',
    enableAll: 'Ativar todas',
    disableAll: 'Desativar todas',
    disableUnused: 'Desativar as não usadas',
    bulkUpdated: count => `${count === 1 ? '1 item atualizado' : `${count} itens atualizados`} para novas sessões.`,
    bulkNoChange: 'Nada para alterar.',
    usageCount: count => `usada ${count}×`,
    provenance: {
      agent: 'Aprendida',
      bundled: 'Embutida',
      hub: 'Hub'
    },
    emptyNoneFound: noun => `Nenhum ${noun} encontrado`,
    emptyNothingMatches: query => `Nada corresponde a “${query}”.`,
    emptyNoneAvailable: noun => `Nenhum ${noun} disponível ainda.`,
    changesApplyNewSessions: 'As mudanças valem para novas sessões.',
    skillUpdated: 'Habilidade atualizada',
    edit: 'Editar',
    archive: 'Arquivar',
    skillArchivedTitle: 'Habilidade arquivada',
    skillArchivedMessage: 'Pode ser restaurada com hermes curator restore.',
    hub: {
      searchPlaceholder: 'Buscar no hub de habilidades',
      search: 'Buscar',
      searching: 'Buscando...',
      connectingHubs: 'Conectando aos hubs de habilidades...',
      connectedHubs: 'Hubs conectados:',
      featured: 'Habilidades em destaque',
      landingHint:
        'Busque no hub para explorar habilidades instaláveis do índice oficial, do GitHub e de fontes da comunidade.',
      noResults: 'Nenhuma habilidade correspondente encontrada no hub.',
      resultCount: (count, ms) => `${count} resultado${count === 1 ? '' : 's'}${ms !== null ? ` em ${ms}ms` : ''}`,
      timedOut: sources => `Tempo esgotado: ${sources}`,
      installed: 'Instalada',
      install: 'Instalar',
      installing: 'Instalando...',
      uninstall: 'Desinstalar',
      uninstalling: 'Desinstalando...',
      updateAll: 'Atualizar as instaladas',
      updating: 'Atualizando...',
      preview: 'Prévia',
      scan: 'Analisar',
      scanning: 'Analisando...',
      files: 'Arquivos',
      noReadme: 'Esta habilidade não tem prévia de SKILL.md.',
      trust: {
        builtin: 'embutida',
        trusted: 'confiável',
        community: 'comunidade'
      },
      verdictSafe: 'Segura',
      verdictCaution: 'Atenção',
      verdictDangerous: 'Perigosa',
      policyAllow: 'Instalação permitida',
      policyAsk: 'Revise antes de instalar',
      policyBlock: 'Instalação bloqueada por política',
      findings: count => `${count} ${count === 1 ? 'achado' : 'achados'}`,
      noFindings: 'Nenhum achado de segurança.',
      installStarted: name => `Instalando ${name}...`,
      uninstallStarted: name => `Desinstalando ${name}...`,
      updateStarted: 'Atualizando as habilidades instaladas...',
      actionFailed: 'Falha na ação da habilidade',
      actionLog: 'Log da ação',
      loadFailed: 'Falha ao carregar o hub de habilidades',
      previewFailed: 'Falha na prévia da habilidade',
      scanFailed: 'Falha na análise de segurança',
      searchFailed: 'Falha na busca no hub'
    }
  },
  starmap: {
    title: 'Grafo de memória',
    subtitle: (nodes, clusters) => `${nodes} Habilidades across ${clusters} categories`,
    close: 'Fechar o grafo de memória',
    refresh: 'Atualizar',
    memory: 'Memória',
    filterAll: 'Todas',
    filterUsed: 'Usadas',
    filterLearned: 'Aprendidas',
    viewGraph: 'Grafo',
    loadFailed: 'Não foi possível carregar o grafo de memória',
    loading: 'Carregando…',
    emptyTitle: 'Nada aprendido ainda',
    emptyDesc: 'À medida que o Hermes constrói habilidades e memórias para o seu trabalho, elas aparecem aqui.',
    share: 'Compartilhar mapa',
    shareHint:
      'Copie o código para compartilhar este mapa, ou cole um para carregar. Ele inclui apenas o layout, não o texto das suas memórias ou habilidades.',
    shareTitle: 'Importar / exportar mapa',
    sharePlaceholder: 'Cole um código de mapa…',
    copy: 'Copiar código do mapa',
    copied: 'Copiado!',
    importMap: 'Importar um mapa',
    importBtn: 'Carregar',
    importEmpty: 'Cole um código de mapa para carregá-lo.',
    importSuccess: nodes => `Mapa carregado com ${nodes} ${nodes === 1 ? 'nó' : 'nós'}.`,
    importedBadge: 'mapa importado',
    resetToMine: 'Voltar ao meu mapa'
  },
  agents: {
    close: 'Fechar agentes',
    title: 'Árvore de agentes',
    subtitle: 'Atividade ao vivo dos subagentes no turno atual.',
    emptyTitle: 'Nenhum subagente ativo',
    emptyDesc: 'Quando um turno delega trabalho, os agentes filhos transmitem o progresso deles aqui.',
    running: 'Em execução',
    failed: 'Falhou',
    done: 'Concluído',
    streaming: 'Transmitindo',
    files: 'Arquivos',
    moreFiles: count => `+${count} mais arquivos`,
    delegation: index => `Delegação ${index}`,
    workers: count => `${count} ${count === 1 ? 'worker' : 'workers'}`,
    workersActive: count => `${count} ${count === 1 ? 'ativo' : 'ativos'}`,
    agentsCount: count => `${count} ${count === 1 ? 'agente' : 'agentes'}`,
    activeCount: count => `${count} ativo${count === 1 ? '' : 's'}`,
    failedCount: count => `${count} ${count === 1 ? 'falha' : 'falhas'}`,
    toolsCount: count => `${count} ferramentas`,
    filesCount: count => `${count} arquivos`,
    updatedAgo: age => `atualizado ${age}`,
    ageNow: 'agora',
    ageSeconds: seconds => `${seconds}s atrás`,
    ageMinutes: minutes => `${minutes}min atrás`,
    ageHours: hours => `${hours}h atrás`,
    ageDays: days => `${days}d atrás`,
    durationSeconds: seconds => `${seconds}s`,
    durationMinutes: (minutes, seconds) => `${minutes}min ${seconds}s`,
    tokens: value => `${value} tok`
  },

  commandCenter: {
    close: 'Fechar a central de comandos',
    paletteTitle: 'Paleta de comandos',
    back: 'Voltar',
    searchPlaceholder: 'Buscar sessões, telas e ações',
    goTo: 'Ir para',
    goToSession: 'Ir para a sessão',
    branches: 'Branches',
    projects: 'Projetos',
    openFolder: 'Abrir pasta como projeto…',
    openFolderAt: path => `Abrir pasta como projeto — ${path}`,
    newSessionInProject: project => `Nova sessão em ${project}`,
    commands: 'Comandos',
    startInBranch: branch => `Nova conversa em ${branch}`,
    commandCenter: 'Central de comandos',
    appearance: 'Aparência',
    settings: 'Configurações',
    changeTheme: 'Trocar tema',
    changeColorMode: 'Trocar o modo de cor…',
    pets: {
      title: 'Mascotes',
      placeholder: 'Buscar mascotes…',
      loading: 'Carregando a galeria do petdex…',
      error: 'Não foi possível acessar a galeria do petdex.',
      staleBackend: 'Reinicie o Hermes para usar mascotes — o backend é anterior a este recurso.',
      empty: 'Nenhum mascote correspondente.',
      turnOff: 'Desligar',
      turnOn: 'Ligar',
      installed: 'Instalado',
      generatedTag: 'Gerado',
      adoptFailed: 'Não foi possível adotar esse mascote.',
      toggleFailed: enabled => `Não foi possível ${enabled ? 'ligar' : 'desligar'} o mascote.`,
      noneAvailable: 'Nenhum mascote disponível — escolha um abaixo para instalar.'
    },
    generatePet: {
      title: 'Gerar um mascote',
      placeholder: 'Descreva um mascote para gerar…',
      promptHint: 'Digite uma descrição e pressione Enter para esboçar quatro versões.',
      readyHint: 'Pressione Enter para esboçar quatro versões a partir da sua descrição.',
      generate: 'Gerar',
      generating: 'Gerando…',
      retry: 'Tentar de novo',
      hatch: 'Chocar',
      spawning: 'Criando…',
      hatching: 'Seu mascote está saindo do ovo…',
      hatchingSub: 'Dando vida a ele…',
      hatched: 'Seu mascote saiu do ovo!',
      hatchRow: (_state, done, total) => `Desenhando o quadro ${done} de ${total}…`,
      hatchComposing: 'Juntando tudo…',
      hatchSaving: 'Quase lá…',
      namePlaceholder: 'Dê um nome ao seu mascote',
      staleBackend: 'Atualize o Hermes para gerar mascotes.',
      backgroundHint: 'Você pode fechar isto — o Hermes avisa quando terminar.',
      slowProviderHint: 'Isso pode levar vários minutos',
      remix: 'Remixar',
      remixConfirmTitle: 'Remixar esta versão?',
      remixConfirmBody:
        'Isso gera um novo conjunto de rascunhos usando este como ponto de partida. Pode levar vários minutos.',
      genericError: 'Falha ao gerar — tente de novo ou escolha uma sugestão.',
      referenceImageTooLarge: 'A imagem de referência é muito grande. Use uma com menos de 16 MB.',
      referenceImageInvalid: 'Não foi possível ler essa imagem de referência. Tente um PNG, JPG, WebP ou GIF.',
      adopt: 'Adotar',
      startOver: 'Começar de novo'
    },
    installTheme: {
      title: 'Instalar tema…',
      pageTitle: 'Instalar tema',
      placeholder: 'Buscar no Marketplace do VS Code...',
      loading: 'Buscando no Marketplace...',
      error: 'Não foi possível acessar o Marketplace.',
      empty: 'Nenhum tema correspondente.',
      install: 'Instalar',
      installing: 'Instalando...',
      installed: 'Instalado',
      installs: count => `${count} instalações`
    },
    settingsFields: 'Campos de configuração',
    mcpServers: 'Servidores MCP',
    archivedChats: 'Conversas arquivadas',
    sections: { maintenance: 'Manutenção', sessions: 'Sessões', system: 'Sistema', usage: 'Uso' },
    sectionDescriptions: {
      maintenance: 'Diagnóstico, cópias de segurança, curador e dados de memória',
      sessions: 'Buscar e gerenciar sessões',
      system: 'Status, logs e ações do sistema',
      usage: 'Tokens, custo e atividade das habilidades ao longo do tempo'
    },
    nav: {
      newChat: { title: 'Nova sessão', detail: 'Iniciar uma sessão nova' },
      settings: { title: 'Configurações', detail: 'Configurar o Hermes Desktop' },
      skills: { title: 'Capacidades', detail: 'Habilidades, ferramentas e servidores MCP' },
      messaging: { title: 'Mensagens', detail: 'Configurar Telegram, Slack, Discord e mais' },
      artifacts: { title: 'Artefatos', detail: 'Explorar saídas geradas' }
    },
    sectionEntries: {
      sessions: { title: 'Painel de sessões', detail: 'Buscar, fixar e gerenciar sessões' },
      system: { title: 'Painel do sistema', detail: 'Status do gateway, logs, reinício/atualização' },
      usage: { title: 'Painel de uso', detail: 'Tokens, custo e atividade das habilidades' }
    },
    providerNavigate: 'Navegar',
    providerSessions: 'Sessões',
    refresh: 'Atualizar',
    refreshing: 'Atualizando...',
    noResults: 'Nenhum resultado correspondente encontrado.',
    pinSession: 'Fixar sessão',
    unpinSession: 'Desafixar sessão',
    exportSession: 'Exportar sessão',
    deleteSession: 'Excluir sessão',
    noSessions: 'Nenhuma sessão ainda.',
    gatewayRunning: 'Gateway de mensagens em execução',
    gatewayStopped: 'Gateway de mensagens parado',
    hermesActiveSessions: (version, count) => `Hermes ${version} · ${count} sessões ativas`,
    restartGateway: 'Reiniciar o gateway',
    gatewayRestartFailed: 'Falha ao reiniciar o gateway.',
    updateHermes: 'Atualizar o Hermes',
    actionRunning: 'em execução',
    actionDone: 'concluída',
    actionFailed: 'Falhou',
    actionStartedWaiting: 'Ação iniciada, aguardando status...',
    loadingStatus: 'Carregando o status...',
    recentLogs: 'Logs recentes',
    noLogs: 'Nenhum log carregado ainda.',
    days: count => `${count}d`,
    statSessions: 'Sessões',
    statApiCalls: 'Chamadas de API',
    statTokens: 'Tokens de entrada/saída',
    statCost: 'Custo est.',
    actualCost: cost => `real ${cost}`,
    loadingUsage: 'Carregando o uso...',
    noUsage: period => `Nenhum uso nos últimos ${period} dias.`,
    retry: 'Tentar de novo',
    dailyTokens: 'Tokens por dia',
    input: 'entrada',
    output: 'saída',
    noDailyActivity: 'Nenhuma atividade diária.',
    topModels: 'Modelos mais usados',
    noModelUsage: 'Nenhum uso de modelo ainda.',
    topSkills: 'Habilidades mais usadas',
    noSkillActivity: 'Nenhuma atividade de habilidade ainda.',
    actions: count => `${count} ações`,
    logFile: 'Arquivo de log',
    logLevel: 'Nível',
    logSearchPlaceholder: 'Filtrar linhas do log...',
    maintenance: {
      runOps: 'Diagnóstico',
      doctor: 'Executar diagnóstico',
      doctorDesc: 'Verifica a saúde da instalação, da configuração e dos provedores',
      securityAudit: 'Auditoria de segurança',
      securityAuditDesc: 'Analisa a configuração e as habilidades em busca de ajustes arriscados',
      backup: 'Criar cópia de segurança',
      backupDesc: 'Compacta configuração, memórias, habilidades e sessões',
      debugShare: 'Compartilhar depuração',
      debugShareDesc:
        'Envia um relatório com dados sensíveis ocultados, além de logs, e devolve links compartilháveis (que expiram em 6 h)',
      debugShareRunning: 'Enviando relatório de depuração...',
      debugShareLinks: 'Links de compartilhamento',
      debugShareFailed: 'Falha ao compartilhar a depuração',
      copyLink: 'Copiar link',
      linkCopied: 'Link copiado',
      curator: 'Curador de habilidades',
      curatorDesc: 'Revisão em segundo plano que arquiva habilidades criadas pelo agente que ficaram paradas',
      curatorPaused: 'Pausado',
      curatorActive: 'Ativo',
      curatorDisabled: 'Desativado',
      curatorLastRun: when => `Última execução ${when}`,
      curatorNeverRan: 'Nunca executado',
      pause: 'Pausar',
      resume: 'Retomar',
      runNow: 'Executar agora',
      memoryData: 'Dados de memória',
      memoryDataDesc: 'Arquivos de memória embutidos, injetados em todas as sessões',
      memoryProvider: name => `Provedor ativo: ${name}`,
      builtinMemory: 'embutida',
      memoryFile: 'Memória do agente (MEMORY.md)',
      userFile: 'Perfil do usuário (USER.md)',
      bytes: size => size,
      empty: 'vazio',
      resetMemory: 'Zerar memória',
      resetUser: 'Zerar o perfil',
      resetAll: 'Zerar os dois',
      resetConfirm: target => `Excluir ${target}? Isso não pode ser desfeito.`,
      resetDone: files => `Excluído: ${files}.`,
      resetFailed: 'Falha ao zerar a memória',
      actionStarted: name => `${name} iniciada — acompanhando o log...`,
      actionFailed: name => `${name} não conseguiu iniciar`,
      running: 'Executando...',
      viewLog: 'Log da ação'
    }
  },

  messaging: {
    search: 'Buscar mensagens...',
    loading: 'Carregando as plataformas de mensagens...',
    loadFailed: 'Falha ao carregar as plataformas de mensagens',
    states: {
      connected: 'Conectado',
      connecting: 'Conectando',
      disabled: 'Desativado',
      fatal: 'Erro',
      gateway_stopped: 'Gateway de mensagens parado',
      not_configured: 'Precisa de configuração',
      pending_restart: 'Precisa reiniciar',
      retrying: 'Tentando novamente',
      startup_failed: 'Falha na inicialização'
    },
    unknown: 'Desconhecido',
    hintPendingRestart: 'Reinicie o gateway pela barra de status para aplicar esta mudança.',
    hintGatewayStopped: 'Inicie o gateway pela barra de status para conectar.',
    credentialsSet: 'Credenciais definidas',
    needsSetup: 'Precisa de configuração',
    gatewayStopped: 'Gateway de mensagens parado',
    getCredentials: 'Obter suas credenciais',
    openSetupGuide: 'Abrir o guia de configuração',
    required: 'Obrigatório',
    recommended: 'Recomendado',
    advanced: count => `Avançado (${count})`,
    noTokenNeeded:
      'Esta plataforma não precisa de token aqui. Use o guia de configuração acima e depois ative-a abaixo.',
    enabled: 'Ativado',
    disabled: 'Desativado',
    unsavedChanges: 'Alterações não salvas',
    saving: 'Salvando...',
    saveChanges: 'Salvar alterações',
    saved: 'Salvo',
    replaceValue: 'Substituir o valor atual',
    openDocs: 'Abrir a documentação',
    clearField: key => `Limpar ${key}`,
    enableAria: name => `Ativar ${name}`,
    disableAria: name => `Desativar ${name}`,
    platformEnabled: name => `${name} ativado`,
    platformDisabled: name => `${name} desativado`,
    restartToApply: 'Esta mudança passa a valer depois de reiniciar o gateway.',
    setupSaved: name => `Configuração de ${name} salva`,
    restartToReconnect: 'As novas credenciais passam a valer depois de reiniciar o gateway.',
    keyCleared: key => `${key} limpa`,
    setupUpdated: name => `A configuração de ${name} foi atualizada.`,
    failedUpdate: name => `Falha ao atualizar ${name}`,
    failedSave: name => `Falha ao salvar ${name}`,
    failedClear: key => `Falha ao limpar ${key}`,
    pendingRequests: count => `Solicitações pendentes (${count})`,
    pendingAria: count =>
      `${count} ${count === 1 ? 'solicitação de pareamento pendente' : 'solicitações de pareamento pendentes'}`,
    approvedUsers: count => `Usuários aprovados (${count})`,
    approve: 'Aprovar',
    approving: 'Aprovando...',
    revoke: 'Revogar',
    revoking: 'Revogando...',
    revokeAria: name => `Revogar ${name}`,
    revokeTitle: 'Revogar acesso',
    revokeDesc: (name: string) => `${name} perderá o acesso e deixará de ser reconhecido a partir da próxima mensagem.`,
    approvedUser: name => `Acesso aprovado para ${name}`,
    approvedHint: 'Será reconhecido automaticamente na próxima mensagem.',
    revokedUser: name => `Acesso revogado para ${name}`,
    failedApprove: name => `Falha ao aprovar ${name}`,
    failedRevoke: name => `Falha ao revogar ${name}`,
    pairingLockedOut:
      'Muitas aprovações falharam — esta plataforma foi bloqueada temporariamente. Tente novamente mais tarde.',
    waitingSince: minutes => (minutes < 1 ? 'agora mesmo' : `${minutes}min atrás`),
    fieldCopy: {
      TELEGRAM_BOT_TOKEN: {
        label: 'Token do bot',
        help: 'Crie um bot com o @BotFather e cole o token que ele fornecer.',
        placeholder: 'Cole o token do bot do Telegram'
      },
      TELEGRAM_ALLOWED_USERS: {
        label: 'IDs de usuário do Telegram permitidos',
        help: 'Recomendado. IDs numéricos separados por vírgula, obtidos no @userinfobot. Sem isso, qualquer pessoa pode mandar DM para o seu bot.'
      },
      TELEGRAM_PROXY: { label: 'URL do proxy', help: 'Necessário apenas em redes onde o Telegram está bloqueado.' },
      DISCORD_BOT_TOKEN: {
        label: 'Token do bot',
        help: 'Crie uma aplicação no Discord Developer Portal, adicione um bot e cole o token dele.'
      },
      DISCORD_ALLOWED_USERS: {
        label: 'IDs de usuário do Discord permitidos',
        help: 'Recomendado. IDs de usuário do Discord separados por vírgula.'
      },
      DISCORD_REPLY_TO_MODE: { label: 'Estilo de resposta', help: 'first, all ou off.' },
      DISCORD_ALLOW_ALL_USERS: {
        label: 'Permitir todos os usuários do Discord',
        help: 'Apenas para desenvolvimento. Quando definido como verdadeiro, qualquer pessoa pode mandar DM para o bot sem lista de permissões.'
      },
      DISCORD_HOME_CHANNEL: {
        label: 'ID do canal principal',
        help: 'Canal em que o bot envia mensagens proativas (saída de cron, lembretes).'
      },
      DISCORD_HOME_CHANNEL_NAME: {
        label: 'Nome do canal principal',
        help: 'Nome de exibição do canal principal nos logs e na saída de status.'
      },
      BLUEBUBBLES_ALLOW_ALL_USERS: {
        label: 'Permitir todos os usuários do iMessage',
        help: 'Quando definido como verdadeiro, ignora a lista de permissões do BlueBubbles.'
      },
      MATTERMOST_ALLOW_ALL_USERS: { label: 'Permitir todos os usuários do Mattermost' },
      MATTERMOST_HOME_CHANNEL: { label: 'Canal principal' },
      QQ_ALLOW_ALL_USERS: { label: 'Permitir todos os usuários do QQ' },
      QQBOT_HOME_CHANNEL: { label: 'Canal principal do QQ', help: 'Canal ou grupo padrão para entrega do cron.' },
      QQBOT_HOME_CHANNEL_NAME: { label: 'Nome do canal principal do QQ' },
      SLACK_BOT_TOKEN: {
        label: 'Token do bot do Slack',
        help: 'Use o token do bot em OAuth & Permissions depois de instalar o seu app do Slack.',
        placeholder: 'Cole o token do bot do Slack'
      },
      SLACK_APP_TOKEN: {
        label: 'Token do app do Slack',
        help: 'Use o token de nível de app exigido pelo Socket Mode.',
        placeholder: 'Cole o token do app do Slack'
      },
      SLACK_ALLOWED_USERS: {
        label: 'IDs de usuário do Slack permitidos',
        help: 'Recomendado. IDs de usuário do Slack separados por vírgula.'
      },
      MATTERMOST_URL: { label: 'URL do servidor', placeholder: 'https://mattermost.example.com' },
      MATTERMOST_TOKEN: { label: 'Token do bot' },
      MATTERMOST_ALLOWED_USERS: {
        label: 'IDs de usuário permitidos',
        help: 'Recomendado. IDs de usuário do Mattermost separados por vírgula.'
      },
      MATRIX_HOMESERVER: { label: 'URL do homeserver', placeholder: 'https://matrix.org' },
      MATRIX_ACCESS_TOKEN: { label: 'Token de acesso' },
      MATRIX_USER_ID: { label: 'ID de usuário do bot', placeholder: '@hermes:example.org' },
      MATRIX_ALLOWED_USERS: {
        label: 'IDs de usuário do Matrix permitidos',
        help: 'Recomendado. IDs de usuário separados por vírgula, no formato @usuario:servidor.'
      },
      SIGNAL_HTTP_URL: {
        label: 'URL da ponte do Signal',
        placeholder: 'http://127.0.0.1:8080',
        help: 'URL de uma ponte REST do signal-cli em execução.'
      },
      SIGNAL_ACCOUNT: { label: 'Número de telefone', help: 'O número registrado na sua ponte do signal-cli.' },
      SIGNAL_ALLOWED_USERS: {
        label: 'Usuários do Signal permitidos',
        help: 'Recomendado. Identificadores do Signal separados por vírgula.'
      },
      WHATSAPP_ENABLED: {
        label: 'Ativar a ponte do WhatsApp',
        help: 'Definido automaticamente pelo botão abaixo. Não mexa a menos que saiba que precisa.'
      },
      WHATSAPP_MODE: { label: 'Modo da ponte' },
      WHATSAPP_ALLOWED_USERS: {
        label: 'Usuários do WhatsApp permitidos',
        help: 'Recomendado. Números de telefone ou IDs do WhatsApp separados por vírgula.'
      }
    },
    platformIntro: {}
  },

  webhooks: {
    search: 'Buscar webhooks...',
    loading: 'Carregando webhooks...',
    loadFailed: 'Falha ao carregar os webhooks',
    subscriptions: (count: number) => `Assinaturas (${count})`,
    hint: 'As alterações nas assinaturas são recarregadas automaticamente assim que o receptor estiver em execução. Assinaturas desativadas rejeitam eventos recebidos.',
    empty: 'Nenhuma assinatura de webhook ainda.',
    disabledTitle: 'Receptor de webhooks desativado',
    disabledBody:
      'Os webhooks são uma plataforma própria do gateway. Ative-os aqui para aceitar eventos HTTP recebidos; canais de conversa só são necessários quando uma assinatura entrega mensagens no Telegram, Discord, Slack ou outro canal.',
    enable: 'Ativar webhooks',
    enabling: 'Ativando...',
    enabled: (name: string) => `Ativado: "${name}"`,
    disabled: (name: string) => `Desativado: "${name}"`,
    enableRow: 'Ativar',
    disableRow: 'Desativar',
    delete: 'Excluir',
    deleting: 'Excluindo...',
    deleted: 'Webhook excluído',
    deleteTitle: 'Excluir webhook',
    deleteDescPrefix: 'Isso vai remover permanentemente ',
    deleteDescSuffix: '. Isso não pode ser desfeito.',
    deleteFailed: (name: string) => `Falha ao excluir "${name}"`,
    toggleFailed: (name, enabled) => `Falha ao ${enabled ? 'ativar' : 'desativar'} "${name}"`,
    newSubscription: 'Nova assinatura',
    restarting: 'Reiniciando o gateway...',
    restartNeeded: 'Os webhooks estão ativados, mas o gateway ainda precisa reiniciar para o receptor entrar no ar.',
    restartGateway: 'Reiniciar o gateway',
    restartingGateway: 'Reiniciando...',
    restartFailed: (detail: string) => `Falha ao reiniciar o gateway${detail}`,
    enabledRestarting: 'Webhooks ativados; reiniciando o gateway...',
    all: '(todos)',
    deliverOnly: 'apenas entrega',
    createdTitle: 'Assinatura criada',
    createdSecretHint: 'Copie o segredo agora — ele só é mostrado uma vez.',
    webhookUrl: 'URL do webhook',
    secretOnce: 'Segredo (mostrado uma vez)',
    done: 'Concluído',
    fieldName: 'Nome',
    fieldNamePlaceholder: 'ex.: github-push',
    fieldDescription: 'Descrição',
    fieldDescriptionPlaceholder: 'O que este webhook faz (opcional)',
    fieldEvents: 'Eventos',
    fieldEventsPlaceholder: 'separados por vírgula; deixe vazio para incluir todos',
    fieldSkills: 'Habilidades',
    fieldSkillsPlaceholder: 'nomes de habilidades separados por vírgula (opcional)',
    fieldDeliver: 'Entregar em',
    fieldDeliverOnly: 'Entregar apenas os dados recebidos',
    fieldPrompt: 'Prompt',
    fieldPromptPlaceholder: 'Instruções para o agente quando este webhook disparar (opcional)',
    nameRequired: 'Nome obrigatório',
    create: 'Criar',
    creating: 'Criando...',
    created: 'Criado',
    createFailed: (detail: string) => `Falha ao criar: ${detail}`,
    copy: 'Copiar',
    deliverOptions: {
      log: 'Log',
      telegram: 'Telegram',
      discord: 'Discord',
      slack: 'Slack',
      email: 'E-mail',
      github_comment: 'Comentário no GitHub'
    }
  },

  profiles: {
    close: 'Fechar perfis',
    nameHint: 'Letras minúsculas, dígitos, hifens e sublinhados. Precisa começar com letra ou dígito.',
    title: 'Perfis',
    count: count => `${count} ${count === 1 ? 'perfil' : 'perfis'}`,
    search: 'Buscar perfis...',
    loading: 'Carregando perfis...',
    newProfile: 'Novo perfil',
    importProfile: 'Importar perfil…',
    exportProfile: 'Exportar perfil…',
    imported: 'Perfil importado',
    exported: 'Perfil exportado',
    failedImport: 'Falha ao importar o perfil',
    failedExport: 'Falha ao exportar o perfil',
    allProfiles: 'Todos os perfis',
    showAllProfiles: 'Mostrar todos os perfis',
    switchToProfile: name => `Ir para ${name}`,
    manageProfiles: 'Gerenciar perfis…',
    actions: 'Ações',
    color: 'Cor…',
    colorFor: 'Cor',
    setColor: color => `Definir cor ${color}`,
    autoColor: 'Automática',
    noProfiles: 'Nenhum perfil ainda.',
    selectPrompt: 'Selecione um perfil para ver os detalhes dele.',
    refresh: 'Atualizar perfis',
    refreshing: 'Atualizando perfis',
    default: 'padrão',
    skills: count => `${count} ${count === 1 ? 'habilidade' : 'habilidades'}`,
    env: 'env',
    defaultBadge: 'Padrão',
    rename: 'Renomear',
    renameMenu: 'Renomear…',
    editSoul: 'Editar SOUL.md…',
    copySetup: 'Copiar configuração',
    copying: 'Copiando...',
    modelLabel: 'Modelo',
    skillsLabel: 'Habilidades',
    notSet: 'Não definido',
    soulDesc: 'O prompt de sistema e as instruções de persona incorporados a este perfil.',
    soulOptional: 'opcional',
    soulPlaceholder: mode =>
      `O prompt de sistema / persona deste perfil.\nDeixe em branco para manter o padrão ${mode}.`,
    soulPlaceholderCloned: 'clonado',
    soulPlaceholderEmpty: 'vazio',
    unsavedChanges: 'Alterações não salvas',
    loadingSoul: 'Carregando o SOUL.md...',
    emptySoul: 'SOUL.md vazio — comece a escrever a persona...',
    saving: 'Salvando...',
    saveSoul: 'Salvar SOUL.md',
    deleteTitle: 'Excluir perfil?',
    deleteDescPrefix: 'Isso vai excluir ',
    deleteDescMid: ' e remover o diretório ',
    deleteDescSuffix: ' dele. Isso não pode ser desfeito.',
    deleting: 'Excluindo...',
    createDesc: 'Perfis são ambientes independentes do Hermes: configuração, habilidades e SOUL.md separados.',
    nameLabel: 'Nome',
    cloneFrom: 'Clonar de',
    cloneFromNone: 'Nenhum (em branco)',
    cloneFromDesc: 'Copia a configuração, as habilidades e o SOUL.md do perfil de origem selecionado.',
    cloneFromDefault: 'Clonar do padrão',
    cloneFromDefaultDesc: 'Copia a configuração, as habilidades e o SOUL.md do seu perfil padrão.',
    invalidName: hint => `Nome inválido. ${hint}`,
    nameRequired: 'O nome é obrigatório.',
    creating: 'Criando...',
    createAction: 'Criar perfil',
    renameTitle: 'Renomear perfil',
    renameDescPrefix: 'Renomear atualiza o diretório do perfil e quaisquer scripts auxiliares em ',
    renameDescSuffix: '.',
    newNameLabel: 'Novo nome',
    renaming: 'Renomeando...',
    created: 'Perfil criado',
    renamed: 'Perfil renomeado',
    deleted: 'Perfil excluído',
    setupCopied: 'Comando de configuração copiado',
    soulSaved: 'SOUL.md salvo',
    failedLoad: 'Falha ao carregar os perfis',
    failedDelete: 'Falha ao excluir o perfil',
    failedCopy: 'Falha ao copiar o comando de configuração',
    failedLoadSoul: 'Falha ao carregar o SOUL.md',
    failedSaveSoul: 'Falha ao salvar o SOUL.md',
    failedCreate: 'Falha ao criar o perfil',
    failedRename: 'Falha ao renomear o perfil',
    connectGateway: 'Gerenciar gateways…',
    fleet: {
      allOnGateway: 'Todos os perfis neste gateway',
      gateway: (gateway: string) => `Perfis em ${gateway}`,
      gatewayUnreachable: (gateway: string) => `${gateway} · inacessível`,
      onGateway: (name: string, gateway: string) => `${name} · ${gateway}`,
      switchTo: (name: string, gateway: string) => `Alternar para ${name} em ${gateway}`,
      deleteOn: (gateway: string) => ` em ${gateway}`
    },
    remoteOverride: {
      menuItem: 'Conectar a um host remoto…',
      badge: (host: string) => `Executa em ${host}`,
      title: (profile: string) => `Conectar ${profile} a um host remoto`,
      description:
        'As sessões neste perfil serão executadas no Hermes remoto para o qual você apontar, em vez deste computador.',
      urlLabel: 'Endereço remoto',
      urlPlaceholder: 'https://hermes.exemplo.com',
      urlInvalid: 'Insira um endereço completo começando com http:// ou https://',
      tokenLabel: 'Token de acesso',
      tokenPlaceholder: 'Cole o token de sessão remota',
      tokenSavedHint: 'Um token já está salvo. Deixe em branco para mantê-lo.',
      plainTextOptIn:
        'Este computador não possui armazenamento seguro de chaves, portanto o token será salvo sem criptografia no disco. Salvar mesmo assim.',
      collisionWarning: (label: string) =>
        `Um gateway chamado “${label}” já existe nas Configurações. A conexão deste perfil é separada e não irá alterá-lo.`,
      confirmTitle: 'Conectar este perfil a um host remoto?',
      confirmNote: (profile: string, host: string) =>
        `Novos chats em ${profile} serão executados em ${host}. Aquele computador executará comandos e lerá arquivos lá, não neste. Conecte-se apenas a um host em que você confia.`,
      confirmBack: 'Voltar',
      connect: 'Conectar',
      connecting: 'Conectando…',
      disconnect: 'Remover conexão remota',
      savedTitle: 'Perfil conectado',
      savedMessage: (profile: string, host: string) => `${profile} agora executa em ${host}`,
      removedTitle: 'Conexão remota removida',
      removedMessage: (profile: string) => `${profile} agora executa neste computador`,
      removeFailed: 'Não foi possível remover a conexão remota',
      authFailedTitle: 'O host remoto recusou o token salvo',
      authFailedMessage: (profile: string, host: string) =>
        `${host} recusou o token salvo para ${profile}. Ele pode ter sido alterado no lado remoto.`,
      updateToken: 'Inserir novo token…'
    }
  },

  cron: {
    close: 'Fechar o cron',
    title: 'Tarefas agendadas',
    count: count => `${count} ${count === 1 ? 'tarefa' : 'tarefas'}`,
    search: 'Buscar tarefas de cron...',
    loading: 'Carregando as tarefas de cron...',
    states: {
      enabled: 'ativada',
      scheduled: 'agendada',
      running: 'em execução',
      paused: 'pausada',
      disabled: 'desativada',
      error: 'erro',
      completed: 'concluída'
    },
    deliveryLabels: {
      local: 'Este desktop',
      telegram: 'Telegram',
      discord: 'Discord',
      slack: 'Slack',
      email: 'E-mail'
    },
    scheduleLabels: {
      daily: 'Diária',
      weekdays: 'Dias úteis',
      weekly: 'Semanal',
      monthly: 'Mensal',
      hourly: 'De hora em hora',
      'every-15-minutes': 'A cada 15 minutos',
      custom: 'Personalizado'
    },
    scheduleHints: {
      daily: 'Todo dia às 9:00',
      weekdays: 'De segunda a sexta às 9:00',
      weekly: 'Toda segunda-feira às 9:00',
      monthly: 'No primeiro dia de cada mês às 9:00',
      hourly: 'No início de cada hora',
      'every-15-minutes': 'A cada 15 minutos',
      custom: 'Sintaxe cron ou linguagem natural'
    },
    days: {
      '0': 'domingo',
      '1': 'segunda-feira',
      '2': 'terça-feira',
      '3': 'quarta-feira',
      '4': 'quinta-feira',
      '5': 'sexta-feira',
      '6': 'sábado',
      '7': 'domingo'
    },
    dayFallback: value => `dia ${value}`,
    everyDayAt: time => `Todo dia às ${time}`,
    weekdaysAt: time => `Dias úteis às ${time}`,
    everyDayOfWeekAt: (day, time) => `Toda ${day} às ${time}`,
    monthlyOnDayAt: (dayOfMonth, time) => `Todo mês no dia ${dayOfMonth} às ${time}`,
    topOfHour: 'No início de cada hora',
    everyHourAt: minute => `A cada hora, no minuto :${minute}`,
    newCron: 'Novo cron',
    emptyDescNew:
      'Agende um prompt para rodar por uma expressão cron. O Hermes executa e entrega os resultados no destino que você escolher.',
    emptyDescSearch: 'Tente uma busca mais ampla.',
    emptyTitleNew: 'Nenhuma tarefa agendada ainda',
    emptyTitleSearch: 'Nenhuma correspondência',
    last: 'Última:',
    next: 'Próxima:',
    noRuns: 'Nenhuma execução ainda',
    manage: 'Gerenciar',
    showRuns: 'Mostrar execuções',
    hideRuns: 'Ocultar execuções',
    runHistory: 'Histórico de execuções',
    actionsTitle: 'Ações da tarefa de cron',
    resume: 'Retomar o cron',
    pause: 'Pausar o cron',
    resumeTitle: 'Retomar',
    pauseTitle: 'Pausar',
    triggerNow: 'Disparar agora',
    edit: 'Editar cron',
    deleteTitle: 'Excluir a tarefa de cron?',
    deleteDescPrefix: 'Isso vai remover ',
    deleteDescSuffix: ' permanentemente. Ela para de disparar imediatamente.',
    deleting: 'Excluindo...',
    resumed: 'Cron retomado',
    paused: 'Cron pausado',
    triggered: 'Cron disparado',
    deleted: 'Cron excluído',
    created: 'Cron criado',
    updated: 'Cron atualizado',
    failedLoad: 'Falha ao carregar as tarefas de cron',
    failedUpdate: 'Falha ao atualizar a tarefa de cron',
    failedTrigger: 'Falha ao disparar a tarefa de cron',
    failedDelete: 'Falha ao excluir a tarefa de cron',
    failedSave: 'Falha ao salvar a tarefa de cron',
    editTitle: 'Editar a tarefa de cron',
    createTitle: 'Nova tarefa de cron',
    editDesc: 'Atualize o agendamento, o prompt ou o destino de entrega. As mudanças valem na próxima execução.',
    createDesc:
      'Agende um prompt para rodar automaticamente. Use sintaxe cron ou uma frase natural como "a cada 15 minutos".',
    nameLabel: 'Nome',
    namePlaceholder: 'Resumo da manhã',
    promptLabel: 'Prompt',
    promptPlaceholder: 'Resuma minhas conversas não lidas do Slack e me envie as 5 principais por e-mail...',
    frequencyLabel: 'Frequência',
    deliverLabel: 'Entregar em',
    deliverNeedsHomeChannel: 'defina um canal principal primeiro',
    modelLabel: 'Modelo',
    modelDefault: 'Padrão (modelo global)',
    customScheduleLabel: 'Agendamento personalizado',
    customPlaceholder: '0 9 * * * ou dias úteis às 9h',
    customHint: 'Expressão cron, ou frases como "a cada hora" ou "dias úteis às 9h".',
    optional: 'Opcional',
    promptRequired: 'O prompt é obrigatório.',
    promptScheduleRequired: 'O prompt e o agendamento são obrigatórios.',
    scheduleRequired: 'O agendamento é obrigatório.',
    scriptOnlyEditHint: 'Tarefa apenas de script (sem prompt de IA). ID da tarefa:',
    saveChanges: 'Salvar alterações',
    createAction: 'Criar cron',
    tabs: {
      jobs: 'Tarefas',
      blueprints: 'Modelos de automação'
    },
    blueprints: {
      tab: 'Modelos de automação',
      startFrom: 'Começar a partir de',
      custom: 'Personalizado',
      subtitle: 'Automações prontas',
      dialogDesc: 'Preencha os detalhes e agende.',
      scheduleIt: 'Agendar',
      scheduling: 'Agendando...',
      scheduled: 'Modelo de automação agendado',
      loading: 'Carregando modelos de automação...',
      failedLoad: 'Falha ao carregar os modelos de automação',
      catalog: {
        'morning-brief': {
          title: 'Briefing matinal',
          description: 'Um breve resumo diário: agenda de hoje, clima e itens urgentes aguardando você.'
        },
        'important-mail': {
          title: 'Monitor de e-mails importantes',
          description:
            'Verifica sua caixa de entrada periodicamente e notifica APENAS sobre e-mails que realmente precisam de atenção.'
        },
        'weekly-review': {
          title: 'Revisão semanal',
          description:
            'Revisão no domingo à noite ou segunda de manhã: compromissos, tarefas paradas e plano para a próxima semana.'
        },
        'workday-start': {
          title: 'Lembrete de início do dia de trabalho',
          description: 'Defina áreas de foco e as 3 principais prioridades para o dia.'
        },
        'custom-reminder': {
          title: 'Lembrete personalizado',
          description: 'Um lembrete agendado flexível com seu texto personalizado.'
        },
        'evening-winddown': {
          title: 'Desaceleração noturna',
          description: 'Reflita sobre as realizações de hoje e prepare-se para o amanhã.'
        },
        'news-digest': {
          title: 'Resumo de notícias por tópico',
          description: 'Resumo de notícias, pesquisas ou atualizações sobre um tópico de interesse.'
        },
        'bill-renewal-watch': {
          title: 'Lembrete de contas e renovações',
          description: 'Acompanhe contas a vencer, assinaturas e renovações de serviços.'
        },
        'price-watch': {
          title: 'Monitoramento de preço e disponibilidade',
          description: 'Monitore o preço de um produto ou anúncio e alerte quando mudar.'
        },
        'competitor-watch': {
          title: 'Monitoramento de notícias de concorrentes',
          description: 'Acompanhe concorrentes específicos para notícias de produtos ou empresas.'
        },
        'habit-checkin': {
          title: 'Acompanhamento de hábitos',
          description: 'Acompanhe o progresso diário em hábitos e metas pessoais.'
        },
        'hydration-move': {
          title: 'Lembrete de hidratação e movimento',
          description: 'Lembretes amigáveis para manter-se hidratado e fazer pausas para se mover.'
        },
        'meal-plan': {
          title: 'Plano alimentar semanal',
          description: 'Gere um plano alimentar semanal e uma lista de compras consolidada.'
        },
        'learn-daily': {
          title: 'Pílula diária de aprendizado',
          description: 'Receba pequenas pílulas de conhecimento diárias sobre um tópico que você está aprendendo.'
        },
        'gratitude-journal': {
          title: 'Prompt de gratidão e reflexão',
          description: 'Prompt diário para gratidão e reflexão consciente.'
        },
        'on-this-day': {
          title: 'Descoberta "neste dia na história"',
          description: 'Descubra eventos interessantes ou memórias pessoais deste dia na história.'
        }
      },
      emptyTitle: 'Nenhum modelo de automação disponível',
      emptyDesc: 'Nenhum modelo de automação está disponível neste backend.'
    },
    modelImpact: {
      title: 'Tarefas agendadas precisam de revisão',
      message: count =>
        `${count} tarefa${count === 1 ? '' : 's'} agendada${count === 1 ? '' : 's'} será${count === 1 ? '' : 'n'} pulada${count === 1 ? '' : 's'} até que você revise suas configurações de modelo.`,
      detailMore: (names, remaining) => `${names} e ${remaining} mais`,
      review: 'Revisar tarefas agendadas',
      saveFailed: 'O Hermes não salvou essa alteração de modelo.',
      confirmTitle: 'Aviso de Seleção de Modelo',
      confirmDetail: 'Confirme apenas se aceitar essa compensação.',
      confirmAction: 'Confirmar',
      declined: 'Alteração de modelo cancelada — você recusou o aviso de nível de treinamento de dados.'
    }
  },

  artifacts: {
    search: 'Buscar artefatos...',
    refresh: 'Atualizar artefatos',
    refreshing: 'Atualizando artefatos',
    indexing: 'Indexando os artefatos das sessões recentes',
    tabAll: 'Todos',
    tabImages: 'Imagens',
    tabFiles: 'Arquivos',
    tabLinks: 'Links',
    noArtifactsTitle: 'Nenhum artefato encontrado',
    noArtifactsDesc: 'Imagens geradas e arquivos de saída aparecem aqui conforme as sessões os produzem.',
    failedLoad: 'Falha ao carregar os artefatos',
    openFailed: 'Falha ao abrir',
    itemsImage: 'imagens',
    itemsLink: 'links',
    itemsFile: 'arquivos',
    itemsGeneric: 'itens',
    zero: '0',
    rangeOf: (start, end, total) => `${start}-${end} de ${total}`,
    goToPage: (itemLabel, page) => `Ir para a página ${page} de ${itemLabel}`,
    colTitleLink: 'Título do link',
    colTitleFile: 'Nome',
    colTitleDefault: 'Título / nome',
    colLocationLink: 'URL',
    colLocationFile: 'Caminho',
    colLocationDefault: 'Local',
    colSession: 'Sessão',
    kindImage: 'imagem',
    kindFile: 'arquivo',
    kindLink: 'link',
    chat: 'Conversa',
    copyUrl: 'Copiar URL',
    copyPath: 'Copiar caminho'
  },

  artifactCard: {
    kind: { code: 'Código', html: 'Página interativa', svg: 'Gráfico' },
    generating: lines => `Gerando… ${lines} linhas`,
    versionBadge: count => `${count} versões`,
    open: 'Abrir'
  },

  artifactPreview: {
    versionOf: (current, total) => `v${current} de ${total}`,
    olderVersion: 'Versão anterior',
    newerVersion: 'Versão mais recente',
    latest: 'Mais recente',
    copyContent: 'Copiar conteúdo',
    download: 'Baixar',
    openInBrowser: 'Abrir no navegador',
    openInBrowserFailed: 'Não foi possível abrir no navegador',
    missingTitle: 'Artefato indisponível',
    missingBody: 'Este artefato não está mais no registro local.'
  },

  sidebar: {
    nav: {
      'new-session': 'Nova sessão',
      skills: 'Capacidades',
      messaging: 'Mensagens',
      artifacts: 'Artefatos'
    },
    searchAria: 'Buscar sessões',
    searchPlaceholder: 'Buscar sessões…',
    clearSearch: 'Limpar a busca',
    noMatch: query => `Nenhuma sessão corresponde a “${query}”.`,
    results: 'Resultados',
    pinned: 'Fixadas',
    sessions: 'Sessões',
    cronJobs: 'Tarefas de cron',
    groupAriaGrouped: 'Mostrar as sessões como uma lista única',
    groupAriaUngrouped: 'Agrupar as sessões por espaço de trabalho',
    showProjects: 'Mostrar projetos',
    showSessions: 'Mostrar sessões',
    groupTitleGrouped: 'Desagrupar as sessões',
    groupTitleUngrouped: 'Agrupar por espaço de trabalho',
    allPinned: 'Tudo aqui está fixado. Desafixe uma conversa para vê-la nas recentes.',
    shiftClickHint: 'Shift-clique em uma conversa para fixar',
    noWorkspace: 'Nenhum espaço de trabalho',
    projectEmpty: 'Nenhuma sessão ainda',
    noSessions: 'Nenhuma sessão ainda',
    noFilterMatches: 'Nenhuma sessão corresponde a estes filtros',
    projects: {
      sectionLabel: 'Projetos',
      home: 'Início',
      newButton: 'Novo projeto',
      createTitle: 'Novo projeto',
      createDesc: 'Dê um nome ao espaço de trabalho e adicione uma ou mais pastas.',
      renameTitle: 'Renomear projeto',
      addFolderTitle: 'Adicionar pasta',
      namePlaceholder: 'ex.: Skunkworks',
      foldersLabel: 'Pastas',
      ideaLabel: 'Ideia',
      ideaPlaceholder: 'Do que se trata este projeto? (salvo no IDEA.md)',
      ideaGenerate: 'Gerar ideia',
      ideaGenerating: 'Gerando…',
      ideaShuffle: 'Embaralhar modelos',
      noFolders: 'Nenhuma pasta adicionada ainda.',
      addFolder: 'Adicionar pasta',
      primaryBadge: 'principal',
      removeFolder: 'Remover',
      create: 'Criar',
      menu: 'Ações',
      menuRename: 'Renomear',
      menuAppearance: 'Aparência',
      noColor: 'Sem cor',
      menuAddFolder: 'Adicionar pasta',
      menuSetActive: 'Tornar ativo',
      menuDelete: 'Excluir',
      moveToProject: 'Mover para projeto',
      movedTo: name => `Movido para ${name}`,
      moveFailed: 'Não foi possível mover a sessão',
      moveNoFolder: 'Esse projeto não tem nenhuma pasta para receber a sessão',
      moveNoProjects: 'Nenhum outro projeto',
      reveal: 'Mostrar na pasta',
      copyPath: 'Copiar caminho',
      removeFromSidebar: 'Ocultar da barra lateral',
      createFailed: 'Não foi possível criar o projeto',
      staleBackend:
        'Atualize o backend do Hermes para criar projetos — seu backend é mais antigo que este app de desktop (Configurações → Atualizações → Backend).',
      deleteConfirm:
        'Isso remove o projeto salvo do Hermes. Arquivos, repositórios git e worktrees permanecem intactos.',
      startWork: 'Novo worktree',
      newWorktreeTitle: 'Novo worktree',
      newWorktreeDesc: 'Dê um nome à branch deste worktree.',
      branchPlaceholder: 'ex.: minha-feature',
      branchOff: () => ({ after: '', before: 'ramificar de ' }),
      baseBranchPlaceholder: 'Pesquisar branches…',
      baseBranchNone: 'Nenhuma branch encontrada',
      startWorkFailed: 'Não foi possível criar o worktree',
      worktreeProjectLabel: 'Projeto',
      worktreeProjectPlaceholder: 'Pesquisar projetos…',
      worktreeProjectNone: 'Nenhum projeto com a pasta',
      convertBranch: 'Converter uma branch…',
      convertBranchTitle: 'Converter uma branch',
      convertBranchDesc: 'Abra branches já em checkout, ou crie um worktree para uma branch livre.',
      convertBranchPlaceholder: 'Pesquisar branches…',
      convertBranchInstead: 'Converter uma branch existente',
      branchOpenExisting: 'abrir',
      branchSwitchHome: 'trocar a principal',
      branchCreateWorktree: 'Novo worktree',
      branchTrackRemote: 'rastrear remota',
      branchesLoading: 'Carregando branches…',
      noBranches: 'Nenhuma branch encontrada',
      removeWorktree: 'Remover worktree',
      removeWorktreeFailed: 'Não foi possível remover o worktree (há alterações não commitadas?)',
      removeWorktreeConfirm:
        'Remova do git (exclui o diretório do worktree; a branch permanece), ou apenas oculte a faixa da barra lateral e deixe o worktree em disco.',
      removeWorktreeDirty:
        'Este worktree tem mudanças não commitadas. Remova à força (descarta essas mudanças), ou apenas oculte a faixa e mantenha em disco.',
      forceRemove: 'Remover à força',
      enter: label => `Abrir ${label}`,
      reorder: label => `Reordenar ${label}`,
      toggle: (label, open) => `${open ? 'Mostrar' : 'Ocultar'} as sessões de ${label}`,
      back: 'Todos os projetos'
    },
    newSessionIn: label => `Nova sessão em ${label}`,
    showMoreIn: (count, label) => `Mostrar mais ${count} em ${label}`,
    loading: 'Carregando…',
    loadMore: 'Carregar mais',
    loadCount: step => `Carregar mais ${step}`,
    row: {
      pin: 'Fixar',
      unpin: 'Desafixar',
      copyId: 'Copiar ID',
      export: 'Exportar',
      branchFrom: 'Ramificar',
      rename: 'Renomear',
      archive: 'Arquivar',
      newWindow: 'Nova janela',
      hideTabBar: 'Ocultar barra de abas',
      openInNewTab: 'Abrir em uma nova aba',
      openInSplit: 'Abrir em tela dividida',
      copyIdFailed: 'Não foi possível copiar o ID da sessão',
      sessionActions: 'Ações da sessão',
      sessionRunning: 'Sessão em execução',
      needsInput: 'Precisa da sua resposta',
      waitingForAnswer: 'Aguardando a sua resposta',
      finishedUnread: 'Concluída — não lida',
      backgroundRunning: 'Tarefa em segundo plano em execução',
      draftSession: 'Rascunho — nada enviado ainda',
      handoffOrigin: platform => `Transferida do ${platform}`,
      ownedByProfile: profile => `Perfil: ${profile}`,
      renamed: 'Renomeado',
      renameFailed: 'Falha ao renomear',
      renameTitle: 'Renomear sessão',
      renameDesc: 'Deixe vazio para limpar.',
      untitledPlaceholder: 'Sessão sem título',
      untitledChat: id => `chat ${id}`,
      ageNow: 'agora',
      ageDay: 'd',
      ageHour: 'h',
      ageMin: 'min',
      messageCount: count => `${count} ${count === 1 ? 'mensagem' : 'mensagens'}`,
      todoProgress: ''
    },
    dateDivider: {
      today: 'Mais cedo hoje',
      yesterday: 'Ontem',
      thisWeek: 'Mais cedo esta semana',
      lastWeek: 'Semana passada',
      thisMonth: 'Mais cedo este mês'
    },
    statusDivider: {
      working: 'funcionando',
      done: 'Concluído'
    }
  },

  composer: {
    message: 'Mensagem',
    wakingProfile: profile => `Acordando ${profile}…`,
    placeholderStarting: 'Iniciando o Hermes...',
    placeholderReconnecting: 'Reconectando ao Hermes…',
    placeholderFollowUp: 'Enviar complemento',
    newSessionPlaceholders: [
      'O que vamos construir?',
      'Dê uma tarefa ao Hermes',
      'O que está passando pela sua cabeça?',
      'Descreva o que você precisa',
      'O que vamos resolver?',
      'Pergunte qualquer coisa',
      'Comece com um objetivo'
    ],
    followUpPlaceholders: [
      'Envie um complemento',
      'Acrescente mais contexto',
      'Refine a solicitação',
      'E agora?',
      'Continue',
      'Vá mais fundo',
      'Ajuste ou continue'
    ],
    startVoice: 'Iniciar conversa por voz',
    openDirective: 'abrir',
    queueMessage: 'Enfileirar mensagem',
    steer: 'Direcionar a execução atual',
    stop: 'Parar',
    send: 'Enviar',
    speaking: 'Falando',
    transcribing: 'Transcrevendo',
    thinking: 'Pensando',
    muted: 'Mudo',
    listening: 'Ouvindo',
    muteMic: 'Silenciar o microfone',
    unmuteMic: 'Reativar o microfone',
    stopListening: 'Parar de ouvir e enviar',
    stopShort: 'Parar',
    endConversation: 'Encerrar a conversa por voz',
    endShort: 'Encerrar',
    stopDictation: 'Parar o ditado',
    transcribingDictation: 'Transcrevendo o ditado',
    voiceDictation: 'Ditado por voz',
    speakReplies: 'Ler as respostas em voz alta',
    stopSpeakingReplies: 'Parar de ler as respostas em voz alta',
    wakeWordListening: phrase => `Palavra de ativação: "${phrase}" — ouvindo`,
    wakeWordOff: phrase => `Palavra de ativação: "${phrase}" — desligada`,
    wakeWordPausedVoice: phrase => `Palavra de ativação: "${phrase}" — pausada durante a conversa por voz`,
    lookupLoading: 'Buscando…',
    lookupNoMatches: 'Nenhum resultado.',
    lookupTry: 'Tente',
    lookupOr: 'ou',
    commonCommands: 'Comandos comuns',
    hotkeys: 'Atalhos',
    helpFooter: 'abre o painel completo · backspace dispensa',
    commandDescs: {
      '/help': 'lista completa de comandos e atalhos',
      '/clear': 'iniciar uma nova sessão',
      '/resume': 'retomar uma sessão anterior',
      '/details': 'controlar o nível de detalhes da transcrição',
      '/copy': 'copiar a seleção ou a última mensagem do assistente',
      '/quit': 'sair do Hermes'
    },
    hotkeyDescs: {
      'composer.mention': 'referenciar arquivos, pastas, URLs e Git',
      'composer.slash': 'paleta de comandos de barra',
      'composer.help': 'esta ajuda rápida (Delete para fechar)',
      'composer.sendNewline': 'enviar · Shift+Enter para nova linha',
      'composer.sendQueued': 'enviar o próximo turno da fila',
      'keybinds.openPanel': 'todos os atalhos de teclado',
      'composer.cancel': 'fechar popover · cancelar execução',
      'composer.history': 'percorrer o popover / histórico'
    },
    attachUrlTitle: 'Anexar uma URL',
    attachUrlDesc: 'O Hermes vai buscar a página e incluí-la como contexto deste turno.',
    urlPlaceholder: 'https://exemplo.com/post',
    urlHintPre: 'Inclua a URL completa, por exemplo ',
    attach: 'Anexar',
    queued: count => `${count} na fila`,
    queuedPaused: count => `${count} na fila — pausados`,
    attachmentOnly: 'Turno só com anexo',
    emptyTurn: 'Turno vazio',
    attachments: count => `${count} anexo${count === 1 ? '' : 's'}`,
    editingInComposer: 'Editando no compositor',
    editingQueuedInComposer: 'Editando o turno da fila no compositor',
    queueEdit: 'editar',
    queueSendNext: 'Próximo',
    queueSteer: 'Direcionar — altere a interação atual agora',
    queueSend: 'enviar',
    queueDelete: 'Excluir',
    queueResume: 'Retomar',
    queueResumeTip: 'Pausado pelo botão “Parar” — retome o envio dos turnos da fila',
    queueStuckTitle: 'Mensagem da fila não enviada',
    queueStuckBody: 'Um turno da fila falhou repetidamente ao enviar. Ele continua na fila — tente enviar de novo.',
    previewUnavailable: 'Prévia indisponível',
    previewLabel: label => `Pré-visualizar ${label}`,
    couldNotPreview: label => `Não foi possível pré-visualizar ${label}`,
    removeAttachment: label => `Remover ${label}`,
    dictating: 'Ditando',
    preparingAudio: 'Preparando o áudio',
    speakingResponse: 'Falando a resposta',
    readingAloud: 'Lendo em voz alta',
    themeSuggestions: 'Sugestões de tema do Desktop',
    noMatchingThemes: 'Nenhum tema correspondente.',
    themeTryPre: 'Tente ',
    themeTryPost: '.',
    attachLabel: 'Anexar',
    files: 'Arquivos…',
    folder: 'Pasta…',
    images: 'Imagens…',
    pasteImage: 'Colar imagem',
    url: 'URL…',
    promptSnippets: 'Trechos de prompt…',
    tipPre: 'Dica: digite ',
    tipPost: ' para referenciar arquivos inline.',
    snippetsTitle: 'Trechos de prompt',
    snippetsDesc: 'Escolha um prompt inicial para colocar no compositor.',
    dropFiles: 'Solte arquivos para anexar',
    dropSession: 'Solte para vincular esta conversa',
    mcpSuggestions: {
      label: server => `Adicionar ${server}`,
      tip: keyword => `Sugerido porque você mencionou “${keyword}” — clique para conectar`,
      connecting: server => `Conectando ${server}…`,
      cancelTip: 'Clique para cancelar',
      added: server => `${server} adicionado`,
      addedTip: 'Conectado — as ferramentas dele estão prontas nesta conversa',
      connectFailed: server => `Não foi possível conectar ${server}`
    },
    skillSuggestions: {
      label: skill => `Usar habilidade: ${skill}`,
      tip: skill => `Você mencionou “${skill}” — clique para usar essa habilidade primeiro`,
      done: skill => `/${skill} adicionado`,
      doneTip: 'A habilidade será carregada quando você enviar'
    },
    repairSuggestions: {
      label: server => `Reconectar ${server}`,
      tip: server => `Uma chamada para ${server} acabou de falhar devido a um erro de conexão`,
      working: server => `Reconectando ${server}…`,
      workingTip: 'Clique para cancelar',
      done: server => `${server} reconectado`,
      doneTip: 'As credenciais atualizadas estão ativas nesta conversa',
      failed: server => `Não foi possível reconectar ${server}`
    },
    cronSuggestions: {
      label: 'Agendar isto',
      tip: phrase => `“${phrase}” parece ser algo recorrente — agende isso conforme uma programação`,
      prefix: 'Configurar isto como uma tarefa agendada:',
      done: 'Marcado para agendamento',
      doneTip: 'Envie e o agente criará a tarefa'
    },
    snippets: {
      codeReview: {
        label: 'Revisão de código',
        description: 'Audita a mudança atual em busca de regressões, casos de borda não cobertos e testes ausentes.',
        text: 'Por favor, revise isto em busca de bugs, regressões e testes ausentes.'
      },
      implementationPlan: {
        label: 'Plano de implementação',
        description: 'Descreve uma abordagem antes de mexer no código para que o diff fique focado.',
        text: 'Por favor, faça um plano de implementação conciso antes de alterar o código.'
      },
      explainThis: {
        label: 'Explique isto',
        description: 'Explica como funciona o código selecionado e aponta os arquivos principais.',
        text: 'Por favor, explique como isto funciona e me aponte os arquivos principais.'
      }
    }
  },

  statusStack: {
    agents: 'agentes',
    background: count => `${count} em segundo plano`,
    goalActive: 'Objetivo ativo',
    goalDone: 'Objetivo concluído',
    goalPaused: 'Objetivo pausado',
    goalWaiting: 'Objetivo em espera',
    subagents: count => `${count} subagente${count === 1 ? '' : 's'}`,
    todos: (done, total) => `Tarefas ${done}/${total}`,
    running: 'Executando',
    stop: 'Parar',
    dismiss: 'Dispensar',
    exit: code => `saída ${code}`,
    coding: {
      title: 'Árvore de trabalho',
      noBranch: 'Nenhuma branch',
      detached: 'desanexado',
      clean: 'Limpo',
      changed: count => `${count} arquivo${count === 1 ? '' : 's'} alterado${count === 1 ? '' : 's'}`,
      ahead: count => `${count} à frente`,
      behind: count => `${count} atrás`,
      review: 'Revisar',
      openChanges: 'Abrir alterações',
      openFile: 'Abrir arquivo',
      stage: 'Adicionar ao stage',
      unstage: 'Tirar do stage',
      stageAll: 'Adicionar todos ao stage',
      viewAsTree: 'Ver como árvore',
      viewAsList: 'Ver como lista',
      revert: 'Reverter',
      revertAll: 'Reverter tudo',
      revertConfirm:
        'Descartar as alterações deste arquivo e restaurá-lo ao estado commitado? Isso não pode ser desfeito.',
      revertAllConfirm:
        'Descartar todas as alterações e restaurar os arquivos ao estado commitado? Isso não pode ser desfeito.',
      staged: 'No stage',
      noChanges: 'Nenhuma alteração',
      notRepo: 'Não é um repositório git',
      noDiff: 'Nenhum diff para mostrar',
      scopeUncommitted: 'Não commitado',
      scopeBranch: 'Branch',
      scopeLastTurn: 'Último turno',
      commit: 'Commit',
      commitAndPush: 'Commit e Push',
      commitPlaceholder: shortcut => `Mensagem (${shortcut} para fazer commit)`,
      generateCommitMessage: 'Gerar mensagem de commit',
      stopGenerating: 'Parar de gerar',
      createPr: 'Criar PR',
      openPr: 'Abrir PR',
      ghMissing: 'Instale a CLI do GitHub (gh) e faça login para abrir PRs',
      agentShip: 'Pedir ao Hermes para abrir o PR',
      agentShipPrompt:
        'Revise as alterações atuais, faça o commit com uma mensagem clara no padrão conventional commits, envie a branch e abra um pull request.',
      newBranch: 'Novo branch',
      branchOffFrom: base => `Novo branch de ${base}`,
      switchTo: branch => `Trocar para ${branch}`,
      switchFailed: branch => `Não foi possível trocar para ${branch}`,
      worktrees: 'Worktrees'
    }
  },

  updates: {
    stages: {
      idle: 'Preparando…',
      prepare: 'Preparando…',
      fetch: 'Baixando…',
      pull: 'Quase lá…',
      pydeps: 'Finalizando…',
      update: 'Atualizando Hermes…',
      rebuild: 'Reconstruindo o app de desktop…',
      restart: 'Reiniciando o Hermes…',
      done: 'Atualização concluída',
      manual: 'Atualizar pelo terminal',
      guiSkew: 'Atualizar o app de desktop',
      error: 'Atualização pausada'
    },
    checking: 'Procurando atualizações…',
    checkFailedTitle: 'Não foi possível procurar atualizações',
    tryAgain: 'Tentar de novo',
    notAvailableTitle: 'Atualização indisponível',
    unsupportedMessage: 'Esta versão do Hermes não consegue se atualizar de dentro do app.',
    connectionRetry: 'Verifique sua conexão e tente de novo.',
    latestBody: 'Você está na versão mais recente.',
    latestBodyBackend: 'O backend está na versão mais recente.',
    allSetTitle: 'Tudo certo',
    availableTitle: 'Nova atualização disponível',
    availableBody: 'Uma nova versão do Hermes está pronta para instalar.',
    availableTitleBackend: 'Atualização do backend disponível',
    availableBodyBackend: 'Uma versão mais nova do backend do Hermes conectado está pronta para instalar.',
    availableBodyNoChangelog:
      'Uma versão mais nova está pronta. As notas da versão não estão disponíveis para este tipo de instalação.',
    updateNow: 'Atualizar agora',
    maybeLater: 'Talvez depois',
    moreChanges: count => `+ ${count} alteração${count === 1 ? '' : 'ções'} incluída${count === 1 ? '' : 's'}.`,
    manualTitle: 'Atualizar pelo terminal',
    manualBody:
      'Você instalou o Hermes pela linha de comando, então as atualizações também rodam por lá. Cole isto no seu terminal:',
    manualPickedUp: 'O Hermes vai carregar a nova versão na próxima vez que você abri-lo.',
    guiSkewTitle: 'Atualizar o app de desktop',
    guiSkewBody:
      'O backend foi atualizado, mas o pacote deste app de desktop não mudou. Atualize ou reinstale o app de desktop do Hermes (seu AppImage / .deb / .rpm) para ficar compatível.',
    copy: 'copiar',
    copied: 'Copiado',
    done: 'Concluído',
    applyingBody:
      'O atualizador do Hermes assume o controle em uma janela própria e reabre o Hermes automaticamente quando terminar. Não reabra o Hermes enquanto ele estiver atualizando.',
    applyingBodyBackend:
      'O backend remoto está aplicando a atualização e vai reiniciar. O Hermes reconecta automaticamente quando ele voltar.',
    applyingClose: 'Esta janela vai fechar durante a atualização e o Hermes reabre sozinho.',
    errorTitle: 'A atualização não foi concluída',
    errorBody: 'Nada foi perdido. Você pode tentar de novo agora.',
    notNow: 'Agora não',
    clientAlsoBehindTitle: 'O app de desktop está desatualizado',
    clientAlsoBehindMessage:
      'O backend está atualizado, mas este app de desktop ainda usa uma versão antiga. Atualize-o para obter as correções mais recentes.',
    clientAlsoBehindAction: 'Atualizar o app de desktop',
    everythingDispatched: 'Atualização enviada',
    everythingSkipped: 'Ignorada',
    everythingRowFailed: 'Falha na atualização',
    everythingFanoutFailedTitle: 'Não foi possível atualizar outras instâncias',
    applyStatus: {
      preparing: 'Atualizando backend…',
      pulling: 'Atualizando o backend…',
      restarting: 'Reiniciando o backend para carregar a atualização…',
      notAvailable: 'Atualização indisponível para este backend.',
      failed: 'Falha na atualização do backend.',
      noReturn:
        'O backend não voltou a ficar online. A atualização pode não ter sido concluída — verifique o host do backend.'
    }
  },

  install: {
    stageStates: {
      pending: 'Pendente',
      running: 'Instalando',
      succeeded: 'Concluído',
      skipped: 'Pulado',
      failed: 'Falhou'
    },
    oneTimeTitle: 'O Hermes precisa de uma instalação única',
    unsupportedDesc: platform =>
      `A instalação automática no primeiro uso ainda não está disponível em ${platform}. Abra o Terminal, execute o comando abaixo e relance este app. As próximas execuções pulam esta etapa.`,
    installCommand: 'Comando de instalação',
    copyCommand: 'Copiar o comando',
    viewDocs: 'Ver a documentação de instalação',
    installTo: 'Vai instalar em',
    retryAfterRun: 'Já executei — tentar novamente',
    setupChoiceTitle: 'Configurar o Hermes Desktop',
    setupChoiceDesc:
      'Conecte este app a um gateway do Hermes que você já executa, ou instale o Hermes localmente neste computador.',
    connectExistingTitle: 'Conectar a um Hermes existente',
    connectExistingShort: 'Conectar a um Hermes existente',
    connectExistingDesc:
      'Usa um backend remoto com token de sessão ou login pelo navegador. Nenhuma instalação local será iniciada.',
    installLocalTitle: 'Instalar o Hermes localmente',
    installLocalDesc: 'Baixa o Hermes, cria o ambiente Python dele e roda o backend neste computador.',
    localStartUnavailable: 'A instalação local não pôde iniciar. Reinicie o Hermes Desktop e tente de novo.',
    remoteSetupTitle: 'Conectar a um Hermes existente',
    remoteSetupDesc:
      'Informe a URL do seu gateway. O Hermes Desktop detecta se ele precisa de token ou de login pelo navegador.',
    remoteUrlTitle: 'URL do gateway',
    remoteUrlDesc: 'Use a URL base do gateway do Hermes, incluindo https:// quando for remoto.',
    remoteUrlPlaceholder: 'https://gateway.exemplo.com/hermes',
    probing: 'Detectando a autenticação do gateway...',
    probeError: 'Não foi possível acessar esse gateway do Hermes.',
    identityProvider: 'seu provedor de identidade',
    authTitle: 'Autenticação',
    authNeedsOauth: provider => `Entre com ${provider} antes de testar este gateway.`,
    authSignedIn: 'Login pelo navegador concluído.',
    connected: 'Conectado',
    signIn: 'Entrar',
    signInWith: provider => `Entrar com ${provider}`,
    enterUrlFirst: 'Informe uma URL de gateway primeiro.',
    signInIncomplete: 'A janela de login foi fechada antes de a autenticação terminar.',
    tokenTitle: 'Token da sessão',
    tokenDesc: 'Cole o token de sessão do arquivo .env do gateway remoto.',
    pasteSessionToken: 'Colar token da sessão',
    incompleteSignInTest: 'Entre antes de testar este gateway protegido por OAuth.',
    incompleteTokenTest: 'Informe um token de sessão antes de testar este gateway.',
    testConnection: 'Testar conexão',
    testSucceeded: (baseUrl, version) => `Conectado a ${baseUrl}${version ? ` (${version})` : ''}.`,
    applyRemote: 'Aplicar e reconectar',
    backToSetup: 'Voltar',
    failedTitle: 'Falha na instalação',
    settingUpTitle: 'Configurando o Hermes Agent',
    finishingTitle: 'Finalizando',
    failedDesc:
      'Uma das etapas de instalação falhou. No Windows, isso pode acontecer se outra CLI ou instância de desktop do Hermes estiver rodando. Encerre qualquer instância do Hermes em execução e tente de novo. Verifique os detalhes abaixo ou o log do desktop para a transcrição completa.',
    activeDesc:
      'Esta é uma configuração única. O instalador do Hermes está baixando dependências e configurando sua máquina. As próximas execuções pulam esta etapa.',
    progress: (completed, total) => `${completed} de ${total} etapas concluídas`,
    currentStage: stage => ` -- agora: ${stage}`,
    fetchingManifest: 'Buscando o manifesto do instalador...',
    error: 'Erro',
    hideOutput: 'Ocultar a saída do instalador',
    showOutput: 'Mostrar a saída do instalador',
    lines: count => `${count} linha${count === 1 ? '' : 's'}`,
    noOutput: 'Nenhuma saída ainda.',
    cancelling: 'Cancelando...',
    cancelInstall: 'Cancelar instalação',
    transcriptSaved: 'Transcrição completa salva em',
    copiedOutput: 'Copiado!',
    copyOutput: 'Copiar a saída',
    reloadRetry: 'recarregar e tentar novamente'
  },

  onboarding: {
    headerTitle: 'Vamos configurar você no Hermes Agent',
    headerDesc: 'Conecte um provedor de modelos para começar a conversar. A maioria das opções leva um clique.',
    preparingInstall:
      'O Hermes está terminando a instalação. Isso normalmente leva menos de um minuto na primeira vez.',
    starting: 'Iniciando o Hermes…',
    lookingUpProviders: 'Buscando provedores...',
    collapse: 'Recolher',
    otherProviders: 'Outros provedores',
    haveApiKey: 'Eu tenho uma chave de API',
    chooseLater: 'Escolho um provedor depois',
    recommended: 'Recomendado',
    connected: 'Conectado',
    featuredPitch: 'Uma assinatura, mais de 300 modelos de ponta — a forma recomendada de rodar o Hermes',
    fireworksPitch: 'API direta de modelos — modelos avançados hospedados pela Fireworks',
    openRouterPitch: 'Uma chave, centenas de modelos — uma ótima opção padrão',
    apiKeyOptions: {
      fireworks: {
        short: 'API direta de modelos',
        description: 'Acesso direto a modelos hospedados pela Fireworks AI.'
      },
      openrouter: {
        short: 'uma chave, muitos modelos',
        description:
          'Hospeda centenas de modelos atrás de uma única chave. Uma boa opção padrão para novas instalações.'
      },
      openai: { short: 'Modelos da família GPT', description: 'Acesso direto aos modelos da OpenAI.' },
      gemini: { short: 'Modelos Gemini', description: 'Acesso direto aos modelos Google Gemini.' },
      xai: { short: 'Modelos Grok', description: 'Acesso direto aos modelos Grok da xAI.' },
      local: {
        short: 'auto-hospedado',
        description:
          'Aponte o Hermes para um endpoint local ou auto-hospedado compatível com a OpenAI (vLLM, llama.cpp, Ollama etc).'
      }
    },
    backToSignIn: 'Voltar ao login',
    getKey: 'Obter uma chave',
    replaceCurrent: 'Substituir valor atual',
    pasteApiKey: 'Colar API chave',
    localApiKeyPlaceholder: 'Chave de API (opcional — só se o seu endpoint exigir)',
    couldNotSave: 'Não foi possível salvar a credencial.',
    connecting: 'Conectando',
    update: 'Atualizar',
    flowSubtitles: {
      pkce: 'Abre o navegador para entrar e depois continua aqui',
      device_code: 'Abre uma página de verificação no seu navegador — o Hermes conecta automaticamente',
      external: 'Entre uma vez pelo terminal e depois volte à conversa'
    },
    startingSignIn: provider => `Iniciando o login em ${provider}...`,
    verifyingCode: provider => `Verificando seu código com ${provider}...`,
    connectedProvider: provider => `${provider} conectado`,
    connectedPicking: provider => `${provider} conectado. Escolhendo um modelo padrão...`,
    signInFailed: 'Falha ao entrar. Tente de novo.',
    pickDifferentProvider: 'Escolher outro provedor',
    signInWith: provider => `Entrar com ${provider}`,
    openedBrowser: provider => `Abrimos ${provider} no seu navegador.`,
    authorizeThere: 'Autorize o Hermes por lá.',
    copyAuthCode: 'Copie o código de autorização e cole abaixo.',
    pasteAuthCode: 'Colar código de autorização',
    reopenAuthPage: 'Reabrir a página de autorização',
    autoBrowser: provider =>
      `Abrimos ${provider} no seu navegador. Autorize o Hermes por lá e você será conectado automaticamente — nada para copiar ou colar.`,
    reopenSignInPage: 'Reabrir a página de login',
    waitingAuthorize: 'Aguardando você autorizar...',
    externalPending: provider =>
      `${provider} faz o login pela CLI própria. Execute este comando em um terminal, depois volte e escolha "Já entrei":`,
    signedIn: 'Já entrei',
    deviceCodeOpened: provider => `Abrimos ${provider} no seu navegador. Informe este código lá:`,
    reopenVerification: 'Reabrir a página de verificação',
    copy: 'copiar',
    defaultModel: 'Modelo padrão',
    freeTier: 'Plano gratuito',
    pro: 'Pro',
    free: 'Gratuito',
    price: (input, output) => `${input} entrada / ${output} saída por Mtok`,
    change: 'Alterar',
    startChatting: 'Começar',
    docs: provider => `Documentação do ${provider}`
  },

  modelPicker: {
    title: 'Trocar de modelo',
    current: 'atual:',
    unknown: '(desconhecido)',
    search: 'Filtrar provedores e modelos...',
    noModels: 'Nenhum modelo encontrado.',
    addProvider: 'Adicionar provedor',
    loadFailed: 'Não foi possível carregar modelos',
    noAuthenticatedProviders: 'Nenhum provedor autenticado.',
    pro: 'Pro',
    proNeedsSubscription: 'Modelos Pro precisam de uma assinatura paga da Nous.',
    free: 'Gratuito',
    freeTier: 'Plano gratuito',
    priceTitle: 'Preço de entrada / saída por milhão de tokens',
    wasPrice: 'era'
  },

  modelVisibility: {
    title: 'Modelos',
    search: 'Pesquisar modelos',
    noAuthenticatedProviders: 'Nenhum provedor autenticado.',
    addProvider: 'Adicionar provedor…'
  },

  shell: {
    windowControls: 'Controles da janela',
    paneControls: 'Controles do painel',
    appControls: 'Controles do app',
    modelMenu: {
      search: 'Pesquisar modelos',
      noModels: 'Nenhum modelo encontrado',
      editModels: 'editar Modelos…',
      refreshModels: 'Atualizar Modelos',
      fast: 'Rápido'
    },
    modelOptions: {
      noOptions: 'Nenhuma opção para este modelo',
      options: 'Opções',
      thinking: 'Raciocínio',
      fast: 'Rápido',
      effort: 'Esforço',
      minimal: 'Mínimo',
      low: 'Baixo',
      medium: 'Médio',
      high: 'Alto',
      xhigh: 'Muito alto',
      max: 'Máximo',
      ultra: 'Ultra',
      updateFailed: 'Falha ao atualizar a opção do modelo',
      fastFailed: 'Falha ao atualizar o modo rápido'
    },
    gatewayMenu: {
      gateway: 'Gateway',
      connected: 'Conectado',
      connecting: 'Conectando',
      offline: 'Offline',
      inferenceReady: 'Inferência pronta',
      inferenceNotReady: 'Inferência não está pronta',
      checkingInference: 'Verificando inferência',
      disconnected: 'Desconectado',
      openSystem: 'Abrir painel do sistema',
      connection: label => `Conexão: ${label}`,
      recentActivity: 'Atividade recente',
      viewAllLogs: 'Ver todos os logs →',
      messagingPlatforms: 'Plataformas de mensagens'
    },
    approvalMode: {
      title: 'Modo de aprovação',
      ariaLabel: mode => `Modo de aprovação: ${mode}`,
      manual: 'Manual',
      manualDescription: 'Perguntar antes de ações que exigem aprovação',
      smart: 'Inteligente',
      smartDescription: 'Avalia as ações automaticamente e pergunta quando necessário',
      off: 'Desligado',
      offDescription: 'Executa sem pedir aprovação'
    },
    statusbar: {
      unknown: 'desconhecido',
      restart: 'reiniciar',
      update: 'atualização',
      updateInProgress: 'Atualização em andamento',
      commitsBehind: (count, branch) => `${count} commit${count === 1 ? '' : 's'} atrás de ${branch}`,
      desktopVersion: version => `Hermes Desktop v${version}`,
      backendVersion: version => `Backend v${version}`,
      clientLabel: version => `cliente v${version}`,
      connectionSsh: host => `SSH: ${host}`,
      connectionRemote: host => `Remoto: ${host}`,
      connectionCloud: host => `Cloud: ${host}`,
      connectionCloudTooltip: host => `Hermes Cloud · ${host}`,
      connectionSshTooltip: host => `SSH · ${host}`,
      connectionRemoteTooltip: host => `Remoto · ${host}`,
      backendLabel: version => `backend v${version}`,
      commit: sha => `commit ${sha}`,
      branch: branch => `branch ${branch}`,
      closeCommandCenter: 'Fechar Central de comandos',
      openCommandCenter: 'Abrir Central de comandos',
      showTerminal: 'Mostrar terminal',
      hideTerminal: 'Ocultar terminal',
      gateway: 'Gateway',
      gatewayReady: 'pronto',
      gatewayNeedsSetup: 'precisa de configuração',
      gatewayUnavailable: 'inferência indisponível',
      gatewayChecking: 'verificando',
      gatewayConnecting: 'conectando',
      gatewayOffline: 'offline',
      gatewayRestarting: 'reiniciando…',
      gatewayTitle: 'Gateway',
      customizeTitle: 'Mostrar na barra de status',
      hideStatusbar: 'Ocultar barra de status',
      toggleApprovalMode: 'Aprovações',
      toggleBackendVersion: 'Versão do backend',
      toggleCommandCenter: 'Central de comandos',
      toggleContextUsage: 'Medidor de contexto',
      toggleRunningTimer: 'Cronômetro do turno',
      toggleSessionTimer: 'Cronômetro da sessão',
      toggleTerminal: 'Terminal',
      toggleVersion: 'Versão e atualizações',
      toggleWorkspace: 'Espaço de trabalho',
      agents: 'agentes',
      closeAgents: 'Fechar agentes',
      openAgents: 'Abrir agentes',
      subagents: count => `${count} subagente${count === 1 ? '' : 's'}`,
      failed: count => `${count} ${count === 1 ? 'falha' : 'falhas'}`,
      running: count => `${count} executando`,
      cron: 'Cron',
      openCron: 'Abrir tarefas de cron',
      webhooks: 'Webhooks',
      openWebhooks: 'Abrir webhooks',
      starmap: 'Grafo de memória',
      openStarmap: 'Abrir grafo de memória',
      turnRunning: 'Executando',
      contextUsage: 'Uso de contexto',
      contextUsagePanel: {
        categories: {
          conversation: 'conversa',
          mcp: 'MCP',
          memory: 'Memória',
          rules: 'Regras',
          skills: 'Habilidades',
          subagent_definitions: 'Definições de subagentes',
          system_prompt: 'Prompt de sistema',
          tool_definitions: 'Definições de ferramentas'
        },
        empty: 'Nenhum dado de contexto ainda',
        loading: 'Carregando detalhamento…',
        percentFull: percent => `${percent}% cheio`,
        title: 'Uso de contexto',
        tokenSummary: (used, max) => `${used} / ${max} tokens`
      },
      session: 'Sessão',
      yoloOn: 'YOLO ligado — aprovando comandos perigosos automaticamente. Shift+clique alterna globalmente.',
      yoloOff: 'YOLO desligado. Shift+clique alterna globalmente.',
      modelNone: 'nenhum',
      noModel: 'sem modelo',
      switchModel: 'Trocar de modelo',
      openModelPicker: 'Abrir seletor de modelos',
      modelPinned: 'fixado por você; as novas conversas usam este em vez do padrão das Configurações',
      modelTitle: (provider, model) => `Modelo · ${provider}: ${model}`,
      providerModelTitle: (provider, model) => `${provider} · ${model}`,
      resetStatusbar: ''
    }
  },

  rightSidebar: {
    aria: 'Barra lateral direita',
    panelsAria: 'Painéis da barra lateral direita',
    files: 'Sistema de arquivos',
    terminal: 'Terminal',
    noFolderSelected: 'Nenhuma pasta selecionada',
    changeCwdTitle: 'Alterar diretório de trabalho',
    remotePickerTitle: 'Escolher a pasta remota',
    remotePickerDescription: 'Navegue pelas pastas do backend conectado.',
    remotePickerSelect: 'Selecionar pasta',
    folderTip: cwd => cwd,
    openFolder: 'Abrir pasta',
    refreshTree: 'Atualizar árvore',
    collapseAll: 'Recolher todas as pastas',
    previewUnavailable: 'Prévia indisponível',
    couldNotPreview: path => `Não foi possível visualizar ${path}`,
    noProjectTitle: 'Nenhum projeto',
    noProjectBody: 'Abra um projeto para navegar pelos arquivos e revisar as alterações.',
    noProjectOpen: 'Nenhum projeto aberto',
    noDiffs: 'Nenhum diff',
    unreadableTitle: 'Ilegível',
    unreadableBody: error => `Não foi possível ler esta pasta (${error}).`,
    emptyTitle: 'Vazio',
    emptyBody: 'Esta pasta está vazia.',
    treeErrorTitle: 'Erro na árvore',
    treeErrorBody: 'A árvore de arquivos deu erro ao renderizar esta pasta.',
    tryAgain: 'Tentar de novo',
    loadingTree: 'Carregando a árvore de arquivos',
    loadingFiles: 'Carregando arquivos',
    terminalHide: 'Ocultar terminal',
    terminalsAria: 'Terminais',
    terminalNew: 'Novo terminal',
    terminalCloseOthers: 'Fechar os outros',
    terminalCloseAll: 'Fechar todos',
    addToChat: 'Adicionar à conversa'
  },

  preview: {
    tab: 'Prévia',
    closePane: 'Fechar o painel de prévia',
    loading: 'Carregando a prévia',
    unavailable: 'Prévia indisponível',
    opening: 'Abrindo...',
    hide: 'Ocultar',
    openPreview: 'Abrir prévia',
    openInBrowser: 'Abrir no navegador',
    openInExternal: 'Abrir no aplicativo externo',
    popIn: 'Acoplar janela',
    popOut: 'Destacar em janela própria',
    linkHint: '⌘/Ctrl-clique para abrir no painel de prévia',
    sourceLineTitle: 'Clique para selecionar · shift-clique para estender · arraste para o compositor',
    source: 'FONTE',
    renderedPreview: 'prévia renderizada',
    diff: 'DIFF',
    unknownSize: 'tamanho desconhecido',
    binaryTitle: 'Isto parece ser um arquivo binário',
    binaryBody: label => `A prévia de ${label} pode mostrar texto ilegível.`,
    largeTitle: 'Este arquivo é grande',
    largeBody: (label, size) => `${label} tem ${size}. O Hermes vai mostrar apenas os primeiros 512 KB.`,
    previewAnyway: 'Visualizar mesmo assim',
    truncated: 'Mostrando os primeiros 512 KB.',
    noInlineTitle: 'Sem prévia inline',
    noInlineBody: mimeType => `${mimeType || 'Este tipo de arquivo'} ainda pode ser anexado como contexto.`,
    edit: 'Editar',
    editing: 'Editando',
    unsavedChanges: 'Alterações não salvas',
    saveFailed: message => `Não foi possível salvar: ${message}`,
    diskChangedTitle: 'O arquivo mudou no disco',
    diskChangedBody:
      'Este arquivo mudou desde que você o abriu. Sobrescrever com a sua versão, ou descartar suas edições e recarregar?',
    overwrite: 'Sobrescrever',
    discardReload: 'Descartar e recarregar',
    console: {
      deselect: 'Desmarcar a entrada',
      select: 'Selecionar a entrada',
      copyFailed: 'Não foi possível copiar a saída do console',
      copyEntry: 'Copiar esta entrada',
      sendEntry: 'Enviar esta entrada para a conversa',
      messages: count => `${count} mensagens do console`,
      resize: 'Redimensionar o console da prévia',
      title: 'Console da prévia',
      selected: count => `${count} selecionadas`,
      sendToChat: 'Enviar para a conversa',
      copySelected: 'Copiar selecionados para a área de transferência',
      copyAll: 'Copiar tudo para a área de transferência',
      copy: 'copiar',
      clear: 'Limpar',
      empty: 'Nenhuma mensagem do console ainda.',
      promptHeader: 'Console da prévia:',
      sentTitle: 'Enviado para a conversa',
      sentMessage: count =>
        `${count} ${count === 1 ? 'entrada de log adicionada' : 'entradas de log adicionadas'} ao compositor`
    },
    web: {
      appFailedToBoot: 'O app de prévia falhou ao iniciar',
      serverNotFound: 'Servidor não encontrado',
      failedToLoad: 'Falha ao carregar a prévia',
      tryAgain: 'Tentar de novo',
      restarting: 'O Hermes está reiniciando...',
      askRestart: 'Pedir ao Hermes para reiniciar o servidor',
      lookingRestart: taskId => `O Hermes está procurando um servidor de prévia para reiniciar (${taskId})`,
      restartingTitle: 'Reiniciando o servidor de prévia',
      restartingMessage: 'O Hermes está trabalhando em segundo plano. Acompanhe o progresso no console da prévia.',
      startRestartFailed: message => `Não foi possível iniciar o reinício do servidor: ${message}`,
      restartFailed: 'Falha ao reiniciar o servidor',
      hideConsole: 'Ocultar console da prévia',
      showConsole: 'Mostrar console da prévia',
      hideDevTools: 'Ocultar DevTools da prévia',
      openDevTools: 'Abrir DevTools da prévia',
      finishedRestarting: message =>
        `O Hermes terminou de reiniciar o servidor de prévia${message ? `: ${message}` : ''}`,
      failedRestarting: message => `Falha ao reiniciar o servidor: ${message}`,
      unknownError: 'erro desconhecido',
      restartedTitle: 'Servidor de prévia reiniciado',
      reloadingNow: 'Recarregando a prévia agora.',
      restartFailedTitle: 'Falha ao reiniciar a prévia',
      restartFailedMessage: 'O Hermes não conseguiu reiniciar o servidor.',
      stillWorking:
        'O Hermes ainda está trabalhando, mas nenhum resultado do reinício chegou. O comando do servidor pode estar rodando em primeiro plano.',
      workspaceReloading: 'O espaço de trabalho mudou; recarregando a prévia',
      fileChanged: url => `Arquivo alterado, recarregando a prévia: ${url}`,
      filesChanged: (count, url) => `${count} arquivos alterados, recarregando a prévia: ${url}`,
      watchFailed: message => `Não foi possível monitorar o arquivo da prévia: ${message}`,
      moduleMimeDescription:
        'Os scripts de módulo estão sendo servidos com o tipo MIME incorreto. Isso normalmente significa que um servidor de arquivos estáticos está servindo um app Vite/React em vez do servidor de desenvolvimento do projeto.',
      loadFailedConsole: (code, message) => `Falha ao carregar${code ? ` (${code})` : ''}: ${message}`,
      unreachableDescription: 'Não foi possível acessar a página da prévia.',
      openTarget: url => `Abrir ${url}`,
      fallbackTitle: 'Pré-visualizar'
    }
  },

  zones: {
    showTabStrip: 'Mostrar abas',
    hideTabStrip: 'Ocultar abas',
    showStripTab: title => `Mostrar ${title}`,
    hideStripTab: title => `Ocultar ${title}`,
    lastTabKeptTitle: 'A última aba permanece',
    lastTabKeptBody:
      'Esta zona precisa de pelo menos uma aba visível. Mostre outra aba primeiro ou recolha toda a barra lateral.',
    toggleStripTab: title => `Alternar aba ${title}`,
    minimize: 'Minimizar',
    restore: 'restaurar',
    closeRunningTitle: 'Fechar a aba em execução?',
    closeRunningBody:
      'Esta conversa ainda está trabalhando (ou aguardando a sua resposta). Fechar a aba apenas a oculta — a sessão mantém o progresso e pode ser reaberta pela barra lateral.',
    closeRunningConfirm: 'Fechar aba',
    reload: 'recarregar',
    closeOthers: 'Fechar as outras',
    closeToRight: 'Fechar as da direita',
    closeAll: 'Fechar todos',
    newSessionTab: 'Nova aba de sessão',
    newTab: 'Nova aba',
    pluginDisabled: pluginId => `Plugin "${pluginId}" desativado`,
    pluginDisabledBody: 'Reative em Configurações → Plugins para trazer o painel de volta.',
    missingPane: paneId => `painel ausente: ${paneId}`,
    editTitle: 'Layouts',
    editHint: 'Escolha um layout ou arraste painéis entre as zonas.',
    reset: 'redefinir',
    templates: 'Modelos',
    custom: 'personalizado',
    newGridLayout: 'Novo layout em grade',
    saveCurrentAs: 'Salvar o arranjo atual como modelo',
    nameLayoutPlaceholder: 'Dê um nome a este layout…',
    deletePreset: name => `Excluir ${name}`,
    zoneEditorTitle: 'Editor de zonas',
    editorHintPre: 'clique para dividir · ',
    editorHintPost:
      ' inverte a linha · arraste entre zonas para mesclar · arraste as bordas compartilhadas para redimensionar',
    templateColumns: 'Colunas',
    templateRows: 'Linhas',
    templateGrid: 'Grade',
    templatePriority: 'Prioridade',
    zoneTag: index => `zona ${index}`,
    mergeZones: count => `Mesclar ${count} zonas`,
    customZoneName: count => `Personalizado (${count} zonas)`,
    layoutNamePlaceholder: fallback => `Nome do layout (${fallback})`,
    saveApply: 'Salvar e aplicar',
    notExpressible: 'este arranjo se entrelaça (cata-vento) — ainda não é expressável como divisões aninhadas',
    zoneCount: count => `${count} zonas`,
    tabCount: count => `${count} abas`
  },

  assistant: {
    thread: {
      loadingSession: 'Carregando a sessão',
      showEarlier: 'Mostrar mensagens anteriores',
      loadingResponse: 'O Hermes está carregando uma resposta',
      resumeWhenBackgroundDone: count =>
        count === 1
          ? 'Vai retomar quando a tarefa em segundo plano terminar'
          : `Vai retomar quando ${count} tarefas em segundo plano terminarem`,
      thinking: 'Pensando',
      thought: 'Pensou',
      thoughtBriefly: 'Pensou brevemente',
      thoughtFor: duration => `Pensou por ${duration}`,
      today: time => `hoje, ${time}`,
      yesterday: time => `Ontem, ${time}`,
      copy: 'copiar',
      refresh: 'Atualizar',
      moreActions: 'mais ações',
      branchNewChat: 'Ramificar em uma nova conversa',
      react: 'Reagir',
      dismissError: 'Dispensar erro',
      errorLayers: {
        auth: 'Erro de autenticação',
        billing: 'Sem créditos',
        disk: 'Disco cheio',
        endpoint: 'Erro no endpoint personalizado',
        gateway: 'Erro no gateway',
        generic: 'Falha no turno',
        provider: 'Erro do provedor',
        runtime: 'Erro no runtime local',
        streaming: 'Erro na conexão de transmissão (streaming)'
      },
      errorRetry: 'Tentar novamente',
      errorSwitchProvider: 'Trocar provedor',
      errorOpenLogs: 'Abrir logs',
      errorOpenLogsFailed: 'Não foi possível abrir a pasta de logs',
      errorOpenDesktopLogs: 'Abrir logs do Desktop',
      errorCopyDiagnostics: 'Copiar detalhes do erro',
      errorSendDiagnostics: 'Enviar diagnósticos',
      filesChanged: count => (count === 1 ? '1 arquivo alterado' : `${count} arquivos alterados`),
      reviewChanges: 'Revisar',
      readAloudFailed: 'Falha ao ler em voz alta',
      preparingAudio: 'Preparando o áudio...',
      stopReading: 'Parar a leitura',
      readAloud: 'Ler em voz alta',
      editMessage: 'Editar a mensagem',
      expandMessage: 'Expandir a mensagem',
      scrollToBottom: 'Ir para o final',
      stop: 'Parar',
      restorePrevious: 'Restaurar o checkpoint anterior',
      restoreCheckpoint: 'Restaurar checkpoint',
      restoreFromHere: 'Restaurar o checkpoint — executar novamente a partir deste prompt',
      restoreTitle: 'Restaurar para este checkpoint?',
      restoreBody: 'Tudo depois deste prompt é removido da conversa, e o prompt roda de novo a partir daqui.',
      restoreConfirm: 'Restaurar e executar novamente',
      restoreNext: 'Restaurar o próximo checkpoint',
      goForward: 'Avançar',
      sendEdited: 'Enviar a mensagem editada',
      attachingFile: 'Anexando…'
    },
    approval: {
      gatewayDisconnected: 'O gateway do Hermes não está conectado',
      sendFailed: 'Não foi possível enviar a resposta de aprovação',
      run: 'Executar',
      command: 'Comando',
      moreOptions: 'Mais opções de aprovação',
      allowSession: 'Permitir nesta sessão',
      alwaysAllowMenu: 'Permitir sempre…',
      jumpToApproval: 'Aprovação necessária',
      reject: 'Rejeitar',
      alwaysTitle: 'Permitir sempre este comando?',
      alwaysDescription: pattern =>
        `Isso adiciona o padrão “${pattern}” à sua lista de permissões permanente (~/.hermes/config.yaml). O Hermes não vai perguntar de novo para comandos assim — nesta sessão nem em nenhuma futura.`,
      alwaysAllow: 'Permitir sempre'
    },
    clarify: {
      notReady: 'O pedido de esclarecimento ainda não está pronto',
      gatewayDisconnected: 'O gateway do Hermes não está conectado',
      sendFailed: 'Não foi possível enviar a resposta de esclarecimento',
      loadingQuestion: 'Carregando a pergunta…',
      other: 'Outro (digite sua resposta)',
      placeholder: 'Digite sua resposta…',
      skip: 'Pular',
      skipped: 'Pulado',
      continueLabel: 'Continuar',
      lateAnswer: (question, choice) => `Sobre "${question}" — minha resposta: ${choice}`,
      lateAnswerTip: 'Rascunhar esta resposta como mensagem de complemento',
      lateAnswerHint:
        'Este prompt não está mais aguardando. Escolha uma opção para rascunhá-la como mensagem de complemento.'
    },
    mcpSetup: {
      installTitle: server => `Adicionar o servidor MCP ${server}?`,
      enableTitle: server => `Ativar o servidor MCP ${server}?`,
      authorizeTitle: server => `Autorizar o servidor MCP ${server}?`,
      installAction: 'Instalar',
      enableAction: 'Ativar',
      authorizeAction: 'Autorizar',
      decline: 'Agora não',
      declined: 'Recusado',
      installed: server => `${server} instalado`,
      enabled: server => `${server} ativado`,
      authorized: server => `${server} autorizado`,
      failed: server => `Falha na configuração de ${server}`,
      unanswered: 'Sem resposta',
      toolCount: count => (count === 1 ? '1 ferramenta' : `${count} ferramentas`),
      notInCatalog: server => `“${server}” não está no catálogo MCP`,
      catalogSource: 'Do catálogo aprovado pela Nous',
      envRequired: 'Preencha primeiro as credenciais obrigatórias',
      sendFailed: 'Não foi possível enviar a resposta da configuração MCP',
      reloadFailed:
        'Servidor salvo, mas não foi possível recarregar as ferramentas MCP — elas serão carregadas na próxima sessão',
      gatewayDisconnected: 'O gateway Hermes não está conectado'
    },
    tool: {
      copyCode: 'Copiar o código',
      renderingImage: 'Renderizando a imagem',
      copyOutput: 'Copiar a saída',
      copyCommand: 'Copiar o comando',
      copyContent: 'Copiar o conteúdo',
      copyUrl: 'Copiar URL',
      copyResults: 'Copiar resultados',
      copyQuery: 'Copiar a consulta',
      copyFile: 'Copiar o arquivo',
      copyPath: 'Copiar caminho',
      outputAlt: 'Saída da ferramenta',
      rawResponse: 'Resposta bruta',
      copyActivity: 'Copiar atividade',
      recoveredOne: 'Recuperado após 1 etapa com falha',
      recoveredMany: (count: number) => `Recuperado após ${count} etapas com falha`,
      failedOne: '1 etapa falhou',
      failedMany: (count: number) => `${count} etapas falharam`,
      statusRunning: 'Executando',
      statusError: 'Erro',
      statusRecovered: 'Recuperado',
      statusDone: 'Concluído',
      memoryWriteNoted: 'Gravação na memória registrada',
      actions: {
        read: 'Leu',
        reading: 'Lendo',
        opened: 'Abriu',
        opening: 'Abrindo',
        failedToOpen: 'Falha ao abrir',
        searched: 'Buscou',
        searching: 'Pesquisando',
        ran: 'Executou',
        running: 'Executando',
        ranCode: 'Executou o código',
        runningCode: 'Executando o código'
      },
      prefixes: {
        browser: 'Navegador',
        web: 'Web'
      },
      titleTemplates: {
        actionCommand: (action, command) => `${action} ${command}`,
        actionQuoted: (action, value) => `${action} “${value}”`,
        actionTarget: (action, target) => `${action} ${target}`,
        prefixedDone: (prefix, action) => `${prefix} ${action}`,
        runningPrefixedTool: (prefix, action) => `Executando ${prefix.toLowerCase()} ${action.toLowerCase()}`,
        runningTool: action => `Executando ${action.toLowerCase()}`
      },
      titles: {
        browser_click: {
          done: 'Clicou no elemento da página',
          pending: 'Clicando no elemento da página',
          pendingAction: 'Clicando'
        },
        browser_fill: {
          done: 'Preencheu o campo do formulário',
          pending: 'Preenchendo o campo do formulário',
          pendingAction: 'Preenchendo'
        },
        browser_navigate: { done: 'Abriu a página', pending: 'Abrindo a página', pendingAction: 'Abrindo' },
        browser_snapshot: {
          done: 'Capturou o estado da página',
          pending: 'Capturando o estado da página',
          pendingAction: 'Capturando'
        },
        browser_take_screenshot: {
          done: 'Capturou a tela',
          pending: 'Capturando a tela',
          pendingAction: 'Capturando'
        },
        browser_type: { done: 'Digitou na página', pending: 'Digitando na página', pendingAction: 'Digitando' },
        clarify: { done: 'Fez uma pergunta', pending: 'Fazendo uma pergunta', pendingAction: 'Perguntando' },
        cronjob: { done: 'Tarefa de cron', pending: 'Agendando a tarefa de cron', pendingAction: 'Agendando' },
        edit_file: { done: 'Editou o arquivo', pending: 'Editando o arquivo', pendingAction: 'Editando' },
        execute_code: { done: 'Executou o código', pending: 'Executando o código', pendingAction: 'Executando' },
        image_generate: { done: 'Gerou a imagem', pending: 'Gerando a imagem', pendingAction: 'Gerando' },
        list_files: { done: 'Listou os arquivos', pending: 'Listando os arquivos', pendingAction: 'Listando' },
        memory: { done: 'Salvou na memória', pending: 'Salvando na memória', pendingAction: 'Salvando' },
        patch: {
          done: 'Aplicou patch no arquivo',
          pending: 'Aplicando patch no arquivo',
          pendingAction: 'Aplicando patch'
        },
        read_file: { done: 'Leu o arquivo', pending: 'Lendo o arquivo', pendingAction: 'Lendo' },
        search_files: { done: 'Buscou nos arquivos', pending: 'Buscando nos arquivos', pendingAction: 'Buscando' },
        session_search_recall: {
          done: 'Buscou no histórico da sessão',
          pending: 'Pesquisando o histórico da sessão',
          pendingAction: 'Pesquisando'
        },
        terminal: { done: 'Executou o comando', pending: 'Executando o comando', pendingAction: 'Executando' },
        todo: { done: 'Atualizou as tarefas', pending: 'Atualizando as tarefas', pendingAction: 'Atualizando' },
        vision_analyze: { done: 'Analisou a imagem', pending: 'Analisando a imagem', pendingAction: 'Analisando' },
        web_extract: { done: 'Leu a página web', pending: 'Lendo a página web', pendingAction: 'Lendo' },
        web_search: { done: 'Buscou na web', pending: 'Buscando na web', pendingAction: 'Buscando' },
        write_file: { done: 'Editou o arquivo', pending: 'Editando o arquivo', pendingAction: 'Editando' }
      }
    }
  },

  prompts: {
    gatewayDisconnected: 'O gateway do Hermes não está conectado',
    sudoSendFailed: 'Não foi possível enviar a senha do sudo',
    secretSendFailed: 'Não foi possível enviar o segredo',
    sudoTitle: 'Senha de administrador',
    sudoDesc:
      'O Hermes precisa da sua senha do sudo para executar um comando privilegiado. Ela é enviada apenas ao seu agente local.',
    sudoPlaceholder: 'senha do sudo',
    secretTitle: 'Segredo necessário',
    secretDesc: 'O Hermes precisa de uma credencial para continuar.',
    secretPlaceholder: 'valor do segredo'
  },

  desktop: {
    audioReadFailed: 'Não foi possível ler o áudio gravado',
    sessionUnavailable: 'Sessão indisponível',
    createSessionFailed: 'Não foi possível criar uma nova sessão',
    promptFailed: 'Falha no prompt',
    providerCredentialRequired: 'Adicione uma credencial de provedor antes de enviar sua primeira mensagem.',
    emptySlashCommand: 'comando de barra vazio',
    desktopCommands: 'Comandos do desktop',
    skillCommandsAvailable: count =>
      `${count} ${count === 1 ? 'comando de habilidade disponível' : 'comandos de habilidades disponíveis'}.`,
    warningLine: message => `aviso: ${message}`,
    yoloArmed: 'YOLO ativado para esta conversa',
    yoloOff: 'YOLO desligado',
    yoloSystem: active => `YOLO ${active ? 'ligado' : 'desligado'} nesta sessão`,
    yoloTitle: 'YOLO',
    yoloToggleFailed: 'Não foi possível alternar o YOLO',
    profileStatus: current =>
      `Perfil: ${current}. Use /profile <nome> ou o seletor "Nova sessão" para iniciar uma conversa em outro perfil.`,
    unknownProfile: 'Perfil desconhecido',
    noProfileNamed: (target, available) => `Nenhum perfil chamado "${target}". Disponíveis: ${available}`,
    newChatsProfile: name => `As novas conversas vão usar o perfil ${name}.`,
    setProfileFailed: 'Falha ao definir o perfil',
    sttDisabled: 'A transcrição de voz está desativada nas configurações.',
    stopFailed: 'Falha ao parar',
    regenerateFailed: 'Falha ao regenerar',
    editFailed: 'Falha ao editar',
    resumeFailed: 'Falha ao retomar',
    resumeStrandedTitle: 'Não foi possível carregar esta sessão',
    resumeStrandedBody:
      'A conexão com esta sessão falhou e as tentativas automáticas desistiram. Verifique se o gateway está rodando e tente de novo.',
    resumeRetry: 'Tentar de novo',
    nothingToBranch: 'Nada para ramificar',
    branchNeedsChat: 'Inicie ou retome uma conversa antes de ramificar.',
    sessionBusy: 'Sessão ocupada',
    branchStopCurrent: 'Pare o turno atual antes de ramificar esta conversa.',
    branchNoText: 'Esta mensagem não tem texto para ramificar.',
    branchTitle: n => `Rascunho: Ramificação #${n}`,
    branchFailed: 'Falha ao ramificar',
    deleteFailed: 'Falha ao excluir',
    archived: 'Arquivada',
    archiveFailed: 'Falha ao arquivar',
    cwdChangeFailed: 'Falha ao trocar o diretório de trabalho',
    cwdStagedTitle: 'Diretório de trabalho preparado',
    cwdStagedMessage: 'Reinicie o backend do desktop para aplicar as mudanças de diretório nesta sessão ativa.',
    modelSwitchFailed: 'Falha ao trocar de modelo',
    hydrationSyncing: (profile: string) => `Sincronizando ${profile}…`,
    sessionExported: 'Sessão exportada',
    sessionExportFailed: 'Não foi possível exportar a sessão',
    imageSaved: 'Imagem salva',
    downloadStarted: 'Download iniciado',
    restartToUseSaveImage: 'Reinicie o Hermes Desktop para usar Salvar imagem.',
    restartToSaveImages: 'Reinicie o Hermes Desktop para salvar imagens',
    imageDownloadFailed: 'Falha ao baixar a imagem',
    openImage: 'Abrir imagem',
    downloadImage: 'Baixar imagem',
    savingImage: 'Salvando a imagem',
    imagePreviewFailed: 'Falha na prévia da imagem',
    imageAttach: 'Anexar imagem',
    imageWriteFailed: 'Falha ao gravar a imagem em disco.',
    imageAttachFailed: 'Falha ao anexar a imagem',
    attachImages: 'Anexar imagens',
    clipboard: 'Área de transferência',
    noClipboardImage: 'Nenhuma imagem encontrada na área de transferência',
    clipboardPasteFailed: 'Falha ao colar da área de transferência',
    dropFiles: 'Soltar arquivos',
    handoff: {
      pickPlatform: 'Escolha um destino',
      success: platform => `Encaminhado para ${platform}. Retome aqui quando quiser.`,
      systemNote: platform => `↻ Encaminhado para ${platform} — retome aqui quando quiser.`,
      failed: error => `Falha no encaminhamento: ${error}`,
      timedOut: 'Tempo esgotado aguardando o gateway. O `hermes gateway` está rodando?'
    }
  },

  errors: {
    genericFailure: 'Algo deu errado',
    boundaryTitle: 'Algo quebrou na interface',
    boundaryDesc: 'A tela encontrou um erro inesperado. Suas conversas e configurações estão seguras.',
    reloadWindow: 'Recarregar a janela',
    openLogs: 'Abrir os logs'
  },

  ui: {
    search: {
      clear: 'Limpar a busca'
    },
    pagination: {
      label: 'paginação',
      previous: 'Anterior',
      previousAria: 'Ir para a página anterior',
      next: 'Próxima',
      nextAria: 'Ir para a próxima página'
    },
    sidebar: {
      title: 'Barra lateral',
      description: 'Exibe a barra lateral em dispositivos móveis.',
      toggle: (open: boolean) => `${open ? 'Mostrar' : 'Ocultar'} barra lateral`
    }
  }
}

// Current-contract additions are kept separately so a future catalog refresh
// can reconcile them against the corresponding English groups without hiding
// them in a giant generated-looking object.
const ptBrCurrentOverrides: TranslationOverrides = {
  onboarding: {
    localModelsTitle: 'Execute modelos localmente',
    localModelsPitch: 'Não precisa de conta — baixe um modelo e execute-o nesta máquina',
    signInExpired:
      'O login expirou aguardando autorização. Isso normalmente significa que a página de login travou na aba aberta (problema no servidor) — conclua o login nela e tente novamente. Se continuar falhando, use uma chave de API ou a alternativa pela CLI.'
  },
  modelPicker: {
    loadingIntoMemory: 'Carregando na memória',
    downloading: 'Baixando',
    localDownloadsHeading: 'Local'
  },
  fileMenu: {
    download: 'Baixar',
    downloadSaved: 'Download salvo',
    downloadFailed: 'Falha no download'
  },
  boot: {
    steps: {
      retryingRemoteBackend: 'Reconectando ao backend remoto do Hermes…'
    }
  },
  notifications: {
    updateReadyMessageUnknown: 'Uma nova atualização está disponível.',
    errors: {
      codeSkewRestartRequired:
        'Este backend está executando código antigo após uma atualização. Reinicie-o para carregar o código novo.'
    },
    mcp: {
      needsAuthTitle: 'O servidor MCP precisa de nova autenticação',
      needsAuthMessage: name => `O servidor MCP ${name} precisa de nova autenticação.`,
      errorTitle: 'Servidor MCP inacessível',
      errorMessage: name => `O servidor MCP ${name} falhou na verificação de saúde.`,
      signIn: 'Entrar',
      view: 'Ver'
    }
  },
  titlebar: {
    unreadSessions: count => `${count} ${count === 1 ? 'sessão não lida' : 'sessões não lidas'}`
  },
  keybinds: {
    actions: {
      'session.archive': 'Arquivar sessão atual',
      'view.showBrowser': 'Abrir navegador'
    }
  },
  settings: {
    nav: {
      providerLocalModels: 'Modelos locais'
    },
    model: {
      loadFailed: 'Não foi possível carregar os modelos',
      restartRequired:
        'Este backend está executando código antigo após uma atualização. Reinicie-o para carregar o código novo.',
      restartBackend: 'Reiniciar backend',
      restartingBackend: 'Reiniciando backend…',
      restartFailed: 'Não foi possível reiniciar o backend',
      tasks: {
        review: {
          label: 'Revisão',
          hint: '/review subagente revisor'
        }
      }
    },
    toolsets: {
      browserRealProfile: {
        label: 'Usar meu perfil real do navegador',
        description:
          'Copia os logins e cookies do seu navegador padrão para um snapshot gerenciado que o agente usa para navegar. Seu perfil ativo nunca é aberto diretamente. Aplica-se a novas sessões.',
        enabledTitle: 'Navegação com perfil real ativada',
        enabledMessage: 'Novas sessões navegarão com um snapshot do seu perfil padrão do navegador.',
        disabledTitle: 'Navegação com perfil real desativada',
        disabledMessage: 'O snapshot do perfil será excluído; novas sessões usarão um navegador limpo.',
        failedSave: 'Não foi possível salvar a configuração do perfil real',
        prompt: {
          title: 'Permaneça conectado aos seus sites',
          body: 'Permita que o Hermes navegue com um snapshot do seu perfil padrão do navegador, para que os sites sejam abertos já conectado.',
          bulletSnapshot: 'Cookies e logins são copiados para um snapshot gerenciado.',
          bulletLiveProfile: 'Seu perfil de navegador ativo nunca é aberto diretamente.',
          bulletLocal: 'Nada sai deste computador.',
          dontShowAgain: 'Não mostrar novamente',
          notNow: 'Agora não',
          enable: 'Usar meu perfil'
        }
      }
    },
    plugins: {
      agent: {
        appliesTo: 'Aplica-se a:'
      },
      installModal: {
        title: 'Instalar plugin',
        description: 'Revise o conteúdo deste repositório antes de instalar qualquer coisa.',
        repoLabel: 'Repositório',
        includesHeading: 'Este pacote inclui',
        agentLabel: 'Plugin de agente',
        desktopLabel: 'Interface do Desktop',
        agentTargetLocal: profile => `Instala no backend local “${profile}” (~/.hermes/plugins/)`,
        agentTargetRemote: profile => `Instala no backend remoto conectado “${profile}”`,
        desktopTarget: 'Instala na pasta local desktop-plugins deste aplicativo',
        desktopOnlyNote: 'Pacotes exclusivos do Desktop não instalam um plugin de agente no backend.',
        insecureWarning:
          'Esta URL usa um esquema inseguro ou local. Prefira https:// ou git@ para instalações de produção.',
        securityHeading: 'Antes de instalar',
        securityIntro:
          'Instale somente de fontes confiáveis — revise o repositório abaixo para ver o que será adicionado.',
        sourceHeading: 'Código-fonte',
        viewRepository: 'Ver repositório',
        viewPluginFiles: 'Ver arquivos do plugin',
        gitCloneLabel: 'URL de clone Git',
        enableAgent: 'Ativar plugin de agente após a instalação',
        forceReinstall: 'Forçar reinstalação (substituir se já estiver instalado)',
        install: 'Instalar',
        installing: 'Instalando…',
        probing: 'Inspecionando repositório…',
        probeUnavailable: 'A inspeção de plugins não está disponível neste ambiente.',
        desktopUnavailable: 'A instalação de plugins do Desktop não está disponível neste ambiente.',
        selectComponent: 'Selecione pelo menos um componente para instalar.',
        agentSuccess: name => `Plugin de agente ${name} instalado`,
        desktopSuccess: name => `Plugin do Desktop ${name} instalado`,
        agentFailed: 'Falha ao instalar o plugin de agente',
        desktopFailed: 'Falha ao instalar o plugin do Desktop',
        missingEnv: vars => `Variáveis de ambiente ausentes: ${vars}. Adicione-as em Configurações → Chaves.`
      }
    },
    appearance: {
      resumeLastSessionTitle: 'Reabrir último chat ao iniciar',
      resumeLastSessionDesc:
        'Quando ativado, o aplicativo reabre seu chat mais recente ao iniciar do zero. Desative para sempre começar com um novo chat.',
      tipsTitle: 'Dicas no aplicativo',
      tipsDesc:
        'Um pequeno balão que aponta para uma parte do aplicativo, exibido ocasionalmente enquanto você não está interagindo e pelo Hermes quando for útil. Fechar uma dica a oculta permanentemente.',
      tipsReset: count => `Restaurar ${count} ${count === 1 ? 'dica fechada' : 'dicas fechadas'}`,
      toursTitle: 'Tours guiados',
      toursDesc: 'Deixe o Hermes guiar você pelo aplicativo, escurecendo a tela e destacando cada etapa.',
      reasoningCollapsedTitle: 'Recolher raciocínio por padrão',
      reasoningCollapsedDesc: 'Mantenha o raciocínio transmitido disponível sem expandi-lo até que você o abra.',
      sessionDensityTitle: 'Densidade da lista de sessões',
      sessionDensityDesc: 'Escolha quanto contexto aparece abaixo dos títulos das sessões na barra lateral.',
      sessionDensityCompact: 'Compacta',
      sessionDensityComfortable: 'Confortável',
      sessionDensityDetailed: 'Detalhada',
      tabStripTitle: 'Faixa de abas',
      tabStripDesc:
        'Mostra abas acima de uma zona. No modo automático, elas ficam ocultas quando a zona contém um único painel.',
      tabStripAuto: 'Automático',
      tabStripAlways: 'Sempre',
      tabStripNever: 'Nunca',
      translucencyGlassDesc:
        'Vidro fosco: o Desktop aparece como um desfoque suave enquanto o texto permanece nítido. Ajustado separadamente para os temas claro e escuro.',
      translucencyModeClear: 'Transparente',
      translucencyModeGlass: 'Vidro',
      translucencyTintTitle: 'Tonalidade',
      translucencyFadeTitle: 'Esmaecimento',
      translucencyFrostTitle: 'Fosco',
      translucencyFrost: {
        'under-window': 'Profundo',
        popover: 'Suave',
        titlebar: 'Brilhante',
        header: 'Reflexo'
      },
      translucencyScopeTitle: 'Área',
      translucencyScope: {
        window: 'Janela inteira',
        sidebar: 'Somente barra lateral'
      },
      introSplashTitle: 'Tela de abertura',
      introSplashDesc: 'A marca e o prompt exibidos em um chat vazio.',
      composerPopoutTitle: 'Compositor flutuante',
      composerPopoutDesc:
        'Permitir arrastar o compositor para fora da área encaixada. Desative para mantê-lo preso na parte inferior.',
      vibeHeartsTitle: 'Corações de Vibe',
      vibeHeartsDesc:
        'Corações flutuantes quando você envia obrigado, te amo, bom bot ou um coração. Separado das Reações de Mensagem acima.'
    },
    about: {
      bundleOutOfSync: 'Build do aplicativo desatualizado',
      bundleOutOfSyncDesc:
        'O runtime do Hermes foi atualizado, mas o aplicativo Desktop ainda usa um build antigo — novos recursos da interface, como o Modo Bot, não aparecerão até que ele seja atualizado. Execute a atualização abaixo para reconstruir o aplicativo. Se o aviso continuar, reinstale usando o instalador Desktop mais recente.',
      bundleOutOfSyncAction: 'Obter o instalador',
      updateReadyUnknown: 'Uma nova atualização está pronta.',
      bundleSwapPending: 'Reinicie para concluir a atualização',
      bundleSwapPendingDesc:
        'O aplicativo atualizado já está instalado — o Hermes só precisa reiniciar para carregá-lo. Seus chats e configurações não serão alterados.',
      bundleSwapPendingAction: 'Reiniciar Hermes'
    },
    localModels: {
      title: 'Modelos locais',
      runtimeTitle: 'Runtime local',
      runtimeReady: backend => `Pronto · ${backend}`,
      serverRunning: 'Em execução',
      runtimeInstalled: 'Runtime do llama.cpp instalado',
      runtimeInstalledDetail: (tag, backend) =>
        `Build ${tag}, backend ${backend}. O Hermes inicia e gerencia o servidor para você.`,
      installTitle: 'Instalar o runtime local',
      installDetail:
        'Baixa o mecanismo de inferência llama.cpp (algumas centenas de MB). Os modelos que você baixar executam inteiramente nesta máquina — sem conta e sem dados saindo do computador.',
      installAction: 'Instalar runtime',
      installing: 'Instalando runtime…',
      installFailed: 'Falha ao instalar o runtime',
      hardwareTitle: 'Esta máquina',
      hardwareLoading: 'Verificando seu hardware…',
      vram: label => `${label} de memória da GPU`,
      ram: label => `${label} de RAM`,
      unifiedMemory: 'Memória unificada',
      modelsTitle: 'Modelos',
      recommended: 'Recomendado',
      recommendedReason: {
        'best-quality-resident':
          'O modelo de maior qualidade que executa inteiramente na sua GPU em velocidade total. A escolha equilibra qualidade e velocidade prevista neste hardware.',
        'speed-gated-quality':
          'Um modelo de maior qualidade cabe nesta máquina, mas responderia lentamente com a largura de banda disponível; este é o melhor modelo que continua rápido.',
        'fastest-resident':
          'Nenhum modelo alcança velocidade total neste hardware; este chega mais perto enquanto executa inteiramente na memória da GPU.',
        'least-painful-spilled':
          'Nenhum modelo cabe inteiramente na memória da GPU; este executa melhor usando a RAM do sistema.'
      },
      downloaded: 'Baixado',
      downloadAction: size => `Baixar · ${size}`,
      downloadProgress: (done, total) => `Baixando ${done} de ${total}`,
      downloadDoneToast: model => `${model} está pronto.`,
      installDoneToast: 'Runtime local instalado e pronto.',
      quickstartTitle: 'Executar um modelo nesta máquina',
      quickstartDetail: (model, size) =>
        `Um clique configura tudo: o mecanismo local, ${model} (${size} de download) e o padrão para novos chats. Nada sai deste computador.`,
      quickstartDetailReady: model =>
        `Um clique torna ${model} o padrão para novos chats. Tudo é executado nesta máquina.`,
      quickstartAction: 'Configurar para mim',
      quickstartConfigure: 'Configurar…',
      quickstartDoneToast: model => `${model} está configurado — novos chats são executados nesta máquina.`,
      quickstartFailed: 'Falha ao configurar o modelo local',
      quickstartStageEngine: 'Mecanismo',
      quickstartStageModel: 'Modelo',
      quickstartStageFinish: 'Concluir',
      useAction: 'Usar',
      activePill: 'Padrão',
      updateTitle: 'Atualização do mecanismo disponível',
      updateDetail: (next, current) =>
        `Uma compilação mais recente do llama.cpp (${next}) está pronta para instalar — você usa ${current}. Os modelos continuam funcionando durante o download.`,
      updateAction: 'Atualizar mecanismo',
      updating: 'Atualizando mecanismo…',
      upToDateTitle: 'Mecanismo atualizado',
      upToDateDetail: (tag, backend) =>
        `Executando llama.cpp ${tag} (${backend}) — a versão mais recente distribuída pelo Hermes.`,
      updateToast: next =>
        `Uma compilação mais recente do mecanismo local (${next}) está disponível. Atualize em Configurações → Modelos locais.`,
      activeDetail: 'Novos chats usam este modelo — ele carrega quando você envia a primeira mensagem',
      activeNotLoaded: 'Carrega na sua primeira mensagem',
      loadedPill: 'Na memória',
      placementResident: 'inteiro na GPU',
      placementSpilled: 'parcialmente na RAM',
      placementResidentTip: 'Executa inteiramente na memória da GPU nesta janela de contexto — velocidade total.',
      placementSpilledTip:
        'Parte deste modelo executa na RAM do sistema — funciona, mas é mais lento. Uma compilação mais compacta ou contexto menor caberia por inteiro.',
      loadingPill: 'Carregando…',
      ejectTip: 'Liberar memória da GPU (carrega novamente na próxima mensagem)',
      ejected: 'Modelo descarregado — memória da GPU liberada.',
      ejectFailed: 'Não foi possível descarregar o modelo',
      stopServer: 'Desligar',
      startServer: 'Ligar',
      runtimeRunningDetail:
        'O servidor local está em execução. Desligá-lo libera toda a memória da GPU e impede que novos chats usem modelos locais até você ligá-lo novamente.',
      serverStopped: 'Servidor local interrompido — memória da GPU liberada.',
      serverStarted: 'Servidor local em execução.',
      serverStopFailed: 'Não foi possível interromper o servidor local',
      serverStartFailed: 'Não foi possível iniciar o servidor local',
      activating: 'Iniciando…',
      activateFailed: model => `Não foi possível alternar para ${model}`,
      activateDoneToast: model => `Novos chats usam ${model}.`,
      downloadFailed: model => `Falha ao baixar ${model}`,
      pillFitsGpu: 'Cabe na sua GPU',
      pillUsesRam: 'Usa RAM do sistema',
      pillTooBig: 'Grande demais para esta máquina',
      browseTitle: 'Encontrar mais modelos',
      browseHint:
        'Pesquise em todo o Hugging Face. Os modelos baixados aqui têm tamanho ajustado automaticamente à sua máquina, mas não são testados por nós.',
      browsePlaceholder: 'Pesquisar modelos por nome ou autor…',
      browseSearching: 'Pesquisando no Hugging Face',
      browseListing: 'Lendo arquivos do modelo',
      browseShowFiles: 'Mostrar arquivos',
      browseRefresh: 'Atualizar',
      browseDownloads: 'downloads',
      browseLikes: 'curtidas',
      browseGated: 'requer login no Hugging Face',
      browseNoGguf: 'Nenhum arquivo de modelo compatível foi encontrado.',
      browseFitUnknown: 'Compatibilidade desconhecida',
      browseAlreadyDownloaded: 'Já baixado.',
      addedByYou: 'Adicionado por você',
      browseDownloadStarted: 'Baixando {name}',
      browseDownloadAria: 'Baixar {name}',
      sideloadButton: 'Adicionar arquivo de modelo',
      sideloadTitle: 'Escolha um arquivo de modelo GGUF',
      sideloadDone: 'Adicionado {name}.',
      sideloadAlreadyPresent: 'Já está na sua biblioteca.',
      pillFullContext: max => `Contexto completo de ${max}`,
      pillFullContextTip: 'Executa com a janela de contexto completa do modelo desde o início',
      pillUpTo: max => `Até ${max} de contexto`,
      pillGrowsTip: 'Aumenta automaticamente conforme sua conversa precisa de mais espaço',
      pillVision: 'Enxerga imagens',
      deleteAction: 'Excluir modelo',
      deleteConfirm: model => `Excluir ${model} do disco?`,
      deleted: model => `${model} excluído.`,
      deleteFailed: 'Falha ao excluir'
    },
    config: {
      toolsetsWipeConfirm:
        'Remover todos os conjuntos de ferramentas ativados? Isso desativa memória, terminal, busca na web, delegação e a maioria das outras ferramentas até que você as ative novamente.',
      disableF12Title: 'Desativar DevTools com F12',
      disableF12Desc: 'Impedir que F12 abra as Ferramentas do desenvolvedor. Ctrl+Shift+I continua funcionando.'
    },
    connections: {
      title: 'Gateways registrados',
      intro:
        'Gerencie este dispositivo e todos os gateways Hermes que ele pode alcançar por conexões remotas, SSH ou Cloud.',
      stagedNote:
        'Troque de gateway em Sessões. Perfis, chats, messaging e tarefas agendadas permanecem com seus gateways; o trabalho nos outros gateways continua.',
      launchModeTitle: 'Na inicialização, voltar a Sessões no gateway usado por último',
      launchModeDesc: 'Quando desativado, Sessões abre no gateway Principal.',
      searchPlaceholder: 'Pesquisar gateways…',
      noSearchResults: 'Nenhum gateway corresponde à sua pesquisa.',
      loadFailed: 'Não foi possível carregar as conexões',
      currentPill: 'Atual',
      primaryPill: 'Principal',
      managedPill: 'Gerenciado pelo aplicativo',
      addConnection: 'Adicionar conexão',
      editConnection: 'Editar',
      removeConnection: 'Remover',
      removeConfirmTitle: 'Remover esta conexão?',
      removeConfirmDesc: label =>
        `“${label}” será removida deste aplicativo. A instância em si não será alterada — você poderá adicioná-la novamente quando quiser.`,
      makePrimary: 'Tornar principal',
      testConnection: 'Testar',
      testOk: 'Acessível',
      testFailed: 'O teste de conexão falhou',
      saveFailed: 'Não foi possível salvar a conexão',
      removeFailed: 'Não foi possível remover a conexão',
      updateAll: 'Atualizar todas as instâncias',
      updateAllRunning: 'Atualizando todas as instâncias…',
      updateAllDone: 'Atualizações iniciadas',
      updateAllFailed: 'Falha ao atualizar todas as instâncias',
      updateSkippedCloud: 'Gerenciado pelo Hermes Cloud',
      kindLocal: 'Local',
      kindRemote: 'Gateway remoto',
      kindCloud: 'Hermes Cloud',
      kindSsh: 'SSH',
      kindLocalDesc: 'O runtime Hermes gerenciado por este aplicativo.',
      kindRemoteDesc: 'Um gateway Hermes acessível por HTTP(S) — rede local, Tailscale ou internet.',
      kindCloudDesc: 'Uma instância hospedada descoberta pela sua conta Hermes Cloud.',
      kindSshDesc: 'Uma instalação do Hermes acessada por SSH.',
      labelTitle: 'Nome',
      labelDesc:
        'Obrigatório. Exibido em todos os lugares onde esta instância aparece; deve ser exclusivo, por exemplo “Homelab” ou “Notebook de trabalho”.',
      labelPlaceholder: 'Homelab',
      urlTitle: 'URL do gateway',
      sshHostTitle: 'Host SSH',
      headersTitle: 'Cabeçalhos extras do gateway',
      headersDesc:
        'Enviados com cada solicitação HTTP e WebSocket para este gateway — por exemplo, para proxies de acesso. Os cabeçalhos gerenciados pelo Hermes são ignorados. Os valores são armazenados criptografados.',
      headerValuePlaceholder: 'Valor',
      headerValueSaved: 'Salvo — deixe em branco para manter',
      headerAdd: 'Adicionar cabeçalho',
      headerRemove: 'Remover',
      duplicateLocal: 'Este aplicativo já gerencia uma conexão local — só pode existir uma.',
      duplicateUrl: label => `Já existe uma conexão com esta URL de gateway (“${label}”).`,
      duplicateSsh: label => `Já existe uma conexão com este host SSH (“${label}”).`,
      sameBackendHint: label => `Mesmo backend que “${label}”`,
      localAddHint: 'O modo local está indisponível: a conexão local gerenciada já existe.',
      cloudAddHint:
        'Dica: entrar no Hermes Cloud acima descobre seus agentes automaticamente — use este formulário apenas para registrar manualmente uma URL conhecida.',
      save: 'Salvar conexão',
      saving: 'Salvando…',
      cancel: 'Cancelar',
      empty: 'Nenhuma conexão registrada ainda.'
    },
    search: {
      placeholder: 'Pesquisar em todas as configurações…',
      pill: 'Pesquisar'
    },
    profileScope: {
      appliesTo: 'Aplica-se a',
      editsProfile: profile => `As alterações nesta página aplicam-se ao perfil “${profile}”.`
    },
    mcp: {
      costTokens: value => `~${value} tokens/chamada`,
      usage30d: value => `${value} usos/30 dias`,
      unusedPill: 'não usado',
      deepLinkTitle: 'Adicionar servidor MCP?',
      deepLinkDescription:
        'Um link solicitou a adição deste servidor MCP ao Hermes. Revise a configuração exata abaixo — ela vem do link, não do Hermes.',
      deepLinkStdioWarning:
        'Este servidor executa um processo local na sua máquina com o comando mostrado abaixo. Continue somente se confiar na origem.',
      deepLinkConfirm: 'Adicionar servidor',
      deepLinkNameInvalid:
        'Os nomes devem ter de 1 a 64 caracteres, formados por letras, números, pontos, hífens ou sublinhados.',
      deepLinkNameConflict: name => `Já existe um servidor chamado ${name} — escolha outro nome ou cancele.`,
      deepLinkErrorTitle: 'Link de instalação MCP rejeitado',
      deepLinkErrorName: 'O nome do servidor no link está ausente ou é inválido.',
      deepLinkErrorConfig: 'A configuração do link não é um JSON válido codificado em base64.',
      deepLinkErrorShape: 'A configuração deve ser um objeto JSON com um campo `url` ou `command` do tipo string.',
      deepLinkErrorUrl: 'Somente URLs de servidor http:// e https:// são permitidas.',
      deepLinkErrorTooLarge: 'A configuração excede o limite de 32 KB.',
      importButton: 'Importar',
      importPlaceholder: 'Cole um trecho mcp.json, comando npx/docker, linha claude mcp add, URL ou link do Cursor…',
      importNoMatch: 'Nenhuma configuração de servidor foi reconhecida no texto colado.',
      importConfirm: 'Adicionar ao mcp.json',
      importConfirmMany: count => `Adicionar ${count} servidores ao mcp.json`
    }
  },
  skills: {
    configuringProfile: 'Configurando:',
    hub: {
      close: 'Fechar',
      alreadyInstalled: name => `“${name}” já está instalado`,
      pickerTitle: 'Skills Hub',
      pickerBrowse: 'Explorar o hub completo',
      pickerHide: 'Ocultar navegador do hub',
      pickerHint:
        'Clique em “+ Adicionar a este agente” em qualquer skill — ela será instalada e aparecerá na lista acima.'
    }
  },
  commandCenter: {
    openBrowser: 'Abrir navegador',
    reloadWindow: 'Recarregar janela'
  },
  profiles: {
    switchToConnection: name => `Trocar para ${name}`,
    switchConnectionFailed: name => `Não foi possível conectar a ${name}`,
    connectGateway: 'Gerenciar gateways…',
    displayNameTitle: 'Nomear este agente',
    displayNameDesc: 'Define um nome exibido em todo o aplicativo. O ID interno do perfil continua sendo “default”.',
    displayNameLabel: 'Nome exibido',
    exportMenu: 'Exportar…'
  },
  sidebar: {
    nav: {
      cron: 'Tarefas agendadas'
    },
    projects: {
      worktreeStaleBackend:
        'Atualize o backend Hermes para criar worktrees nesta conexão remota — ele é anterior à API de worktrees do Git.'
    },
    messageCount: count => `${count} ${count === 1 ? 'mensagem' : 'mensagens'}`,
    toolCallCount: count => `${count} ${count === 1 ? 'chamada de ferramenta' : 'chamadas de ferramenta'}`,
    row: {
      markUnread: 'Marcar como não lida',
      markRead: 'Marcar como lida',
      unreadFailed: 'Não foi possível atualizar o estado de leitura',
      openInTerminal: 'Abrir no terminal',
      deleteTitle: 'Excluir sessão?',
      deleteDesc: name => `Isso excluirá permanentemente “${name}”. Não é possível desfazer.`,
      deleting: 'Excluindo…',
      deleted: 'Sessão excluída'
    },
    markAllRead: 'Marcar todas como lidas'
  },
  composer: {
    voiceControls: 'Voz',
    githubSuggestions: {
      label: 'Configurar GitHub',
      tip: 'O GitHub funciona por meio das skills da CLI gh — clique para conectar sua conta',
      done: 'Adicionado /github-auth',
      doneTip: 'Envie a mensagem e o agente orientará você no login do GitHub'
    }
  },
  statusStack: {
    coding: {
      close: 'Fechar',
      agentShipUnavailable: 'O chat que contém essas alterações não está na tela.'
    }
  },
  updates: {
    blockerTitle: 'Fechar previews locais para atualizar o Hermes?',
    blockerBody:
      'O Hermes precisa parar estes previews locais antes de atualizar. Seus arquivos não serão modificados nem excluídos.',
    foreignBlockerTitle: 'Fechar outros processos para atualizar o Hermes',
    foreignBlockerBody:
      'O Hermes não pode fechar estes processos com segurança. Feche o aplicativo, terminal ou serviço responsável por cada um e tente atualizar novamente.',
    mixedBlockerBody:
      'O Hermes pode fechar os previews locais listados abaixo. Outros processos precisam ser fechados manualmente antes de continuar.',
    closePreviewsAndUpdate: 'Fechar previews e atualizar',
    closePreviewsAndCheckAgain: 'Fechar previews e verificar novamente',
    localPreview: 'Preview local',
    portLabel: port => `Porta ${port}`,
    pidLabel: pid => `PID ${pid}`,
    technicalDetails: 'Detalhes técnicos'
  },
  shell: {
    gatewayMenu: {
      reconnectGateway: 'Reconectar gateway'
    },
    statusbar: {
      toggleCacheHitRate: 'Taxa de acerto do cache',
      toggleTokensPerSecond: 'Tokens por segundo',
      cacheHitRateTitle:
        'Taxa de acerto do cache de prompts nesta sessão — tokens em cache custam menos, portanto valores maiores são mais econômicos',
      tokensPerSecondTitle: 'Tokens de saída por segundo, calculados pela média das últimas 10 chamadas ao modelo',
      systemResources: {
        title: 'Recursos do sistema',
        loading: 'Recursos…',
        gpuUtilization: 'Uso da GPU',
        gpuMemory: 'Memória da GPU',
        ram: 'RAM',
        unifiedNote: 'Memória unificada — a GPU e o sistema compartilham este conjunto.',
        toggle: 'Recursos do sistema'
      }
    }
  },
  preview: {
    web: {
      remoteLoopback:
        'Este endereço aponta para a máquina que executa seu agente, não para esta máquina. O painel de navegador carrega páginas localmente; um servidor remoto precisa de encaminhamento de porta ou de um host acessível.',
      goBack: 'Voltar',
      goForward: 'Avançar',
      reload: 'Recarregar página',
      address: 'Endereço',
      addressPlaceholder: 'Digite um endereço',
      blankPageBody: 'Digite um endereço acima para navegar ou peça ao Hermes para abrir uma página.',
      annotate: 'Anotar',
      annotateOn: 'Parar anotações',
      annotateNeedPage: 'Abra primeiro uma página no navegador integrado.',
      annotateFailed: 'Não foi possível iniciar o modo de anotação',
      commenting: 'Comentando',
      addComments: count => (count === 1 ? 'Adicionar 1 comentário' : `Adicionar ${count} comentários`),
      commentPlaceholder: 'Adicionar um comentário…',
      commentTitle: n => `Comentário ${n}`,
      saveComment: 'Salvar',
      cancelComment: 'Cancelar comentário'
    }
  },
  zones: {
    showStripTab: title => `Mostrar ${title}`,
    hideStripTab: title => `Ocultar ${title}`,
    lastTabKeptTitle: 'A última aba permanece',
    lastTabKeptBody:
      'Esta zona precisa de pelo menos uma aba visível. Mostre outra aba primeiro ou recolha a barra lateral inteira.',
    toggleStripTab: title => `Alternar aba ${title}`
  },
  contextMenu: {
    link: {
      openInApp: 'Abrir no navegador do aplicativo',
      openExternal: 'Abrir no navegador externo',
      copyUrl: 'Copiar URL',
      copyResolvedUrl: 'Copiar URL resolvida'
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
  assistant: {
    thread: {
      turnDuration: duration => `Esta rodada levou ${duration}`,
      loadingLocalModel: model => `Carregando ${model} na memória`,
      processingPrompt: 'Processando prompt'
    },
    clarify: {
      confirmAndContinueLabel: 'Confirmar e continuar',
      answeredBadge: 'Respondida',
      questionProgress: (answered, total) => `${answered} de ${total} respondidas`
    }
  },
  desktop: {
    editTurnUnavailable: 'Esta rodada não está mais no histórico do servidor — talvez tenha sido compactada.',
    readOnlyTranscriptTitle: 'Aberto somente para leitura',
    readOnlyTranscriptBody:
      'Nenhum backend conectado assumiu este chat mais antigo ainda, por isso ele foi aberto como uma transcrição somente para leitura. O histórico está intacto; o envio fica desativado até que um backend o assuma.',
    readOnlyTranscriptSendBlocked:
      'Este chat está aberto como uma transcrição somente para leitura — o envio está desativado.'
  },
  tips: {
    close: 'Não mostrar esta dica novamente',
    items: {
      'new-session': {
        title: 'Comece do zero',
        text: 'Um novo chat tem seu próprio contexto, terminal e diretório de trabalho.'
      },
      skills: {
        title: 'Ensine uma vez',
        text: 'Skills são pastas de instruções que o Hermes carrega quando o trabalho precisa delas.'
      },
      messaging: {
        title: 'Hermes longe da sua mesa',
        text: 'Conecte Telegram, Discord, Slack e mais — o mesmo agente, a mesma memória.'
      },
      artifacts: {
        title: 'Tudo que o Hermes criou',
        text: 'Imagens, arquivos e links de todas as sessões, indexados em um só lugar.'
      },
      cron: {
        title: 'Trabalho que se executa sozinho',
        text: 'Agende um prompt por hora, à noite ou usando uma expressão cron.'
      },
      'command-palette': {
        title: 'Uma caixa para tudo',
        text: 'Sessões, configurações, skills e comandos estão todos na paleta.'
      },
      profiles: {
        title: 'Perfis são separados',
        text: 'Cada um é seu próprio Hermes — suas próprias chaves, memória e sessões.'
      },
      'composer-mentions': {
        title: 'Anexe e comande',
        text: 'Digite @ para trazer um arquivo à conversa e / para executar um comando.'
      },
      'local-setup': {
        title: 'Esta máquina pode executar modelos localmente',
        text: 'Seu hardware pode servir um modelo local. Os chats permanecem no seu computador e não custam nada.',
        action: 'Configurar'
      },
      'right-pane': {
        title: 'O painel de trabalho',
        text: 'Arquivos, terminal, revisão e o navegador integrado ocupam o lado direito.'
      }
    }
  }
}

export const ptBrOverrides = mergeTranslationOverrides(ptBrBaseOverrides, ptBrCurrentOverrides)
export const ptBr = defineLocale(ptBrOverrides)
