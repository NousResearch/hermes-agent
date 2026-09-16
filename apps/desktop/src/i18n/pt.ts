import { defineLocale } from './define-locale'

// Brazilian Portuguese desktop catalog, written as partial overrides merged over
// `en`: keys left out fall back to English, and every key here stays type-checked
// against `Translations`, so en.ts additions can't drift silently. Shares the `pt`
// id with the web catalog and the backend's `locales/pt.yaml`.
export const pt = defineLocale({
  catalog: {
    listView: 'Visualização em lista',
    cardView: 'Visualização em cartões',
    installTitle: name => `Instalar “${name}”?`,
    installDescription: 'Esta skill ficará disponível em novas sessões. Instale apenas fontes nas quais você confia.',
    installTo: 'Instalar em',
    thisComputer: 'Este computador',
    installing: 'Instalando…',
    installComplete: name => `“${name}” instalado`,
    destinationChanged: 'O destino mudou. Feche esta caixa de diálogo e abra o link de instalação novamente.',
    browse: 'Explorar',
    installed: 'Instalados',
    searchSkills: 'Pesquisar skills',
    searchPlugins: 'Pesquisar plugins',
    allSources: 'Todas as fontes',
    allCategories: 'Todas as categorias',
    about: 'Sobre',
    author: 'Autor',
    source: 'Fonte',
    category: 'Categoria',
    version: 'Versão',
    platforms: 'Plataformas',
    requires: 'Requisitos',
    tools: 'Ferramentas',
    hooks: 'Hooks',
    repository: 'Repositório',
    documentation: 'Documentação',
    noResults: 'Sem resultados',
    tryAnother: 'Tente outra pesquisa ou limpe os filtros.',
    clearFilters: 'Limpar filtros',
    loadFailed: 'Não foi possível carregar o catálogo',
    retry: 'Tentar novamente',
    more: 'Mostrar mais',
    pinned: 'Commit revisado',
    snapshotHint: 'Do catálogo do Hermes. A navegação nunca consulta os repositórios de origem.',
    installHint: 'Revise a origem antes de instalar. As alterações valem para novas sessões.',
    results: count => `${count.toLocaleString('pt-BR')} resultado${count === 1 ? '' : 's'}`,
    back: 'Voltar aos resultados'
  },
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
    failed: 'Falha',
    formatJson: 'Formatar JSON',
    free: 'Gratuito',
    loading: 'Carregando…',
    notSet: 'Não definido',
    refresh: 'Atualizar',
    remove: 'Remover',
    replace: 'Substituir',
    retry: 'Repetir',
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
    revealFileManager: 'Abrir pasta do arquivo',
    revealInSidebar: 'Mostrar na árvore de arquivos',
    copyPath: 'Copiar Caminho',
    copyRelativePath: 'Copiar Caminho Relativo',
    download: 'Baixar',
    downloadSaved: 'Salvo',
    downloadFailed: 'Falha no download',
    rename: 'Renomear…',
    delete: 'Excluir',
    renameTitle: 'Renomear',
    renameLabel: 'Novo nome',
    deleteTitle: name => `Excluir ${name}?`,
    deleteBody: 'O arquivo será movido para a Lixeira — você pode restaurá-lo de lá.',
    pathCopied: 'Caminho copiado'
  },
  boot: {
    ready: 'Hermes Desktop está pronto',
    desktopBootFailedWithMessage: message => `A inicialização do Desktop falhou: ${message}`,
    steps: {
      connectingGateway: 'Conectando gateway desktop ao vivo',
      loadingSettings: 'Carregando configurações do Hermes',
      loadingSessions: 'Carregando sessões recentes',
      retryingRemoteBackend: 'Reconectando ao backend remoto do Hermes…',
      startingDesktopConnection: 'Iniciando conexão do desktop',
      startingHermesDesktop: 'Iniciando Hermes Desktop…'
    },
    errors: {
      backgroundExited: 'O processo em segundo plano do Hermes foi encerrado.',
      backgroundExitedDuringStartup: 'O processo em segundo plano do Hermes foi encerrado durante a inicialização.',
      backendStopped: 'Backend parado',
      desktopBootFailed: 'Falha na inicialização do Desktop',
      gatewayConnectionLost: 'Conexão com o gateway perdida',
      gatewaySignInRequired: 'Login no gateway necessário',
      ipcBridgeUnavailable: 'A ponte IPC do Desktop está indisponível.',
      gatewayConnectionLostDetail:
        'Ainda tentando reconectar em segundo plano. Você pode continuar lendo e escrevendo — abra Configurações do gateway se isso persistir.'
    },
    failure: {
      title: 'O Hermes não pôde ser iniciado',
      description:
        'O gateway em segundo plano não iniciou. Tente uma das etapas de recuperação abaixo. Nenhuma das opções exclui seus chats ou configurações.',
      remoteTitle: 'Login remoto no gateway necessário',
      remoteDescription:
        'Sua sessão no gateway remoto expirou. Faça login novamente. Nenhuma das opções exclui seus chats ou configurações.',
      retry: 'Repetir',
      repairInstall: 'Reparar instalação',
      useLocalGateway: 'Usar gateway local',
      gatewaySettings: 'Configurações do gateway',
      back: 'Voltar',
      openLogs: 'Abrir logs',
      repairHint: 'Reparar executa o instalador novamente e pode levar alguns minutos.',
      remoteSignInHint: signInLabel =>
        `Encerra a sessão remota salva e abre ${signInLabel}. Use Usar gateway local para alternar para o backend embutido.`,
      signOutAndSignIn: 'Sair e Entrar',
      remoteFailureHint:
        'Verifique a URL do gateway e o login em Configurações do gateway, ou alterne para o gateway local.',
      cloudDownTitle: 'O agente da Nous Cloud está inativo',
      cloudDownDescription:
        'O agente gerenciado pela Nous retornou um erro. Ele não pode ser reiniciado por aqui — verifique o status ou obtenha suporte.',
      cloudDownHint: 'Os botões abaixo abrem o Portal Nous e nosso Discord para suporte.',
      cloudDownCheckPortal: 'Verificar status no Portal',
      cloudDownDiscord: 'Obter ajuda no Discord',
      hideRecentLogs: 'Ocultar logs recentes',
      showRecentLogs: 'Mostrar logs recentes',
      signedInTitle: 'Conectado',
      signedInMessage: 'Reconectando ao gateway remoto…',
      signInIncompleteTitle: 'Login incompleto',
      signInIncompleteMessage: 'A janela de login fechou antes da conclusão da autenticação.',
      signInFailed: 'Falha no login',
      signInToRemoteGateway: 'Fazer login no gateway remoto',
      signInWithProvider: provider => `Entrar com ${provider}`,
      identityProvider: 'seu provedor de identidade'
    }
  },
  notifications: {
    region: 'Notificações',
    hide: 'Ocultar',
    show: 'Mostrar',
    more: count => `Mais ${count} ${count === 1 ? 'notificação' : 'notificações'}`,
    clearAll: 'Limpar tudo',
    dismiss: 'Dispensar notificação',
    details: 'Detalhes',
    copyDetail: 'Copiar detalhes',
    copyDetailFailed: 'Não foi possível copiar os detalhes da notificação',
    backendOutOfDateTitle: 'Backend desatualizado',
    backendOutOfDateMessage:
      'O backend do Hermes é mais antigo que esta compilação do desktop. Atualize para alinhá-los.',
    installMethodUnsupportedTitle: 'Método de instalação não suportado',
    updateHermes: 'Atualizar Hermes',
    updateReadyTitle: 'Atualização pronta',
    updateReadyMessage: count =>
      `${count} nova${count === 1 ? '' : 's'} mudança${count === 1 ? '' : 's'} disponível${count === 1 ? '' : 'is'}.`,
    updateReadyMessageUnknown: 'Uma nova atualização está disponível.',
    seeWhatsNew: 'Ver o que há de novo',
    mcp: {
      needsAuthTitle: 'O servidor MCP precisa de reautenticação',
      needsAuthMessage: name => `O MCP ${name} precisa de reautenticação.`,
      errorTitle: 'Servidor MCP inacessível',
      errorMessage: name => `O MCP ${name} falhou na verificação de saúde.`,
      signIn: 'Entrar',
      view: 'Visualizar'
    },
    errors: {
      elevenLabsNeedsKey: 'ElevenLabs STT precisa de ELEVENLABS_API_KEY.',
      elevenLabsRejectedKey: 'ElevenLabs rejeitou a chave de API (401).',
      diskFull: 'Disco cheio — libere espaço e tente novamente.',
      gatewayAuthFailed: 'Falha na autenticação do Gateway — verifique API_SERVER_KEY.',
      methodNotAllowed: 'O backend do desktop rejeitou a solicitação (405). Tente reiniciar o aplicativo.',
      microphonePermission: 'A permissão do microfone foi negada.',
      openaiRejectedApiKey:
        'A OpenAI não aceitou sua chave de API. Atualize-a em Configurações → Chaves e tente novamente.',
      openaiTtsNeedsKey: 'OpenAI TTS precisa da VOICE_TOOLS_OPENAI_KEY ou OPENAI_API_KEY.',
      codeSkewRestartRequired:
        'Este backend está executando código antigo após uma atualização. Reinicie-o para carregar o novo código.'
    },
    voice: {
      configureSpeechToText: 'Configure a conversão fala-para-texto para usar o modo de voz.',
      couldNotStartSession: 'Não foi possível iniciar a sessão de voz',
      microphoneAccessDenied: 'Acesso ao microfone negado.',
      microphoneConstraintsUnsupported: 'Restrições do microfone não são suportadas por este dispositivo.',
      microphoneFailed: 'Falha no microfone',
      microphoneInUse: 'O microfone já está sendo usado por outro aplicativo.',
      microphonePermissionDenied: 'A permissão do microfone foi negada.',
      microphoneStartFailed: 'Não foi possível iniciar a gravação do microfone.',
      microphoneUnsupported: 'O runtime não suporta gravação de microfone.',
      noMicrophone: 'Nenhum microfone encontrado.',
      noSpeechDetected: 'Nenhuma fala detectada',
      playbackFailed: 'Falha na reprodução de voz',
      recordingFailed: 'Falha na gravação de voz',
      sayStopToEnd: phrase => `Diga "${phrase}" para encerrar o chat de voz.`,
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
      turnDoneTitle: 'Hermes terminou',
      turnErrorTitle: 'O turno falhou',
      backgroundDoneTitle: 'Tarefa em segundo plano concluída',
      backgroundFailedTitle: 'A tarefa em segundo plano falhou',
      creditsTitle: 'Créditos'
    }
  },
  remoteDisplayBanner: {
    message: reason =>
      `Renderização por software ativa — monitor remoto detectado (${reason}). A aceleração de GPU está desativada para evitar cintilação.`
  },
  billingBlock: {
    titleNous: 'Sem créditos Nous',
    titleProvider: provider => `Sem créditos — ${provider}`,
    fallbackMessage: 'Sua conta está sem créditos. Adicione mais para continuar.',
    openBilling: 'Abrir faturamento',
    addCredits: 'Adicionar créditos',
    dismiss: 'Dispensar'
  },
  sendDiagnostics: {
    title: 'Enviar diagnósticos para a Nous',
    privacyNotice:
      'Isto faz o upload de um pacote de depuração para o armazenamento interno da Nous (não é público). Inclui informações do sistema (SO, versões, provedor, quais chaves de API estão configuradas — mas nunca as chaves em si) e logs completos do agente, gateway e desktop (até 512 KB cada), que provavelmente contêm conteúdo de conversas, saídas de ferramentas e caminhos de arquivos. Segredos são ocultados antes do envio. O pacote é visível apenas pela equipe da Nous e moderadores autorizados do Discord, e é excluído automaticamente após 14 dias.',
    upload: 'Enviar',
    uploading: 'Enviando…',
    cancel: 'Cancelar',
    close: 'Fechar',
    copyLink: 'Copiar link',
    uploadIdFallback: id => `Nenhum link de visualização retornado — cite o ID de upload ${id} para o suporte`,
    doneTitle: 'Diagnósticos enviados',
    doneDescription:
      'Seu pacote foi enviado com privacidade. Compartilhe o link abaixo em seu tópico de suporte para que a equipe possa ver seus logs.',
    failedTitle: 'Falha no upload',
    failedHint:
      'Você também pode executar `hermes debug share --nous` em um terminal, ou `hermes debug share --local` para imprimir o relatório sem fazer o upload.',
    handoffLead: 'Continue a discussão em:',
    links: {
      portal: 'Suporte do Portal Nous'
    }
  },
  titlebar: {
    hideSidebar: 'Ocultar barra lateral',
    showSidebar: 'Mostrar barra lateral',
    search: 'Pesquisar',
    searchTitle: 'Pesquisar sessões, exibições e ações',
    swapSidebarSides: 'Inverter lados da barra lateral',
    hideRightSidebar: 'Ocultar barra lateral direita',
    showRightSidebar: 'Mostrar barra lateral direita',
    unreadSessions: count => (count === 1 ? '1 sessão não lida' : `${count} sessões não lidas`),
    muteHaptics: 'Silenciar hápticos',
    unmuteHaptics: 'Ativar hápticos',
    openSettings: 'Abrir configurações',
    openStarmap: 'Abrir gráfico de memória',
    enterHud: 'Modo HUD',
    exitHud: 'Sair do modo HUD',
    layoutEditor: 'Editor de layout',
    layoutEditorTitle: mod => `Editor de layout — clique com ${mod} para redefinir o layout`,
    resetHudLayout: 'Redefinir tamanho e posição do HUD'
  },
  keybinds: {
    title: 'Atalhos de teclado',
    subtitle: open => `Clique em um atalho para reatribuir · ${open} reabre este painel.`,
    search: 'Pesquisar atalhos…',
    rebind: 'Reatribuir',
    reset: 'Redefinir para padrão',
    resetAll: 'Redefinir tudo',
    pressKey: 'Pressione uma tecla…',
    set: 'definir',
    conflictWith: label => `Também atribuído a “${label}”`,
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
      'nav.skills': 'Abrir skills',
      'nav.messaging': 'Abrir mensageria',
      'nav.artifacts': 'Abrir artefatos',
      'nav.cron': 'Abrir tarefas agendadas',
      'nav.agents': 'Abrir agentes',
      'session.new': 'Nova sessão',
      'session.newTab': 'Nova aba de sessão',
      'session.newWindow': 'Nova janela',
      'session.next': 'Próxima sessão',
      'session.prev': 'Sessão anterior',
      'session.slot.1': 'Alternar para sessão recente 1',
      'session.slot.2': 'Alternar para sessão recente 2',
      'session.slot.3': 'Alternar para sessão recente 3',
      'session.slot.4': 'Alternar para sessão recente 4',
      'session.slot.5': 'Alternar para sessão recente 5',
      'session.slot.6': 'Alternar para sessão recente 6',
      'session.slot.7': 'Alternar para sessão recente 7',
      'session.slot.8': 'Alternar para sessão recente 8',
      'session.slot.9': 'Alternar para sessão recente 9',
      'session.focusSearch': 'Pesquisar sessões',
      'session.togglePin': 'Fixar / desafixar sessão atual',
      'session.archive': 'Arquivar sessão atual',
      'workspace.newWorktree': 'Nova worktree',
      'workspace.openFolder': 'Abrir pasta como projeto',
      'composer.focus': 'Focar compositor',
      'composer.modelPicker': 'Abrir seletor de modelo',
      'composer.voice': 'Iniciar / parar conversa por voz',
      'view.toggleSidebar': 'Alternar barra lateral de sessões',
      'view.toggleRightSidebar': 'Alternar navegador de arquivos',
      'view.toggleReview': 'Alternar painel de revisão',
      'view.toggleStatusbar': 'Alternar barra de status',
      'view.toggleTabStrip': 'Alternar abas',
      'view.showFiles': 'Mostrar navegador de arquivos',
      'view.showBrowser': 'Abrir navegador',
      'view.toggleHud': 'Alternar modo HUD',
      'hud.snapToPointer': 'Mover HUD para o ponteiro (global, enquanto HUD está aberto)',
      'view.showTerminal': 'Alternar terminal',
      'view.newTerminal': 'Novo terminal',
      'view.nextTerminal': 'Próximo terminal',
      'view.prevTerminal': 'Terminal anterior',
      'view.closeTerminal': 'Fechar terminal',
      'view.terminalCopy': 'Copiar seleção do terminal',
      'view.terminalPaste': 'Colar no terminal',
      'view.closeTab': 'Fechar aba',
      'view.reopenTab': 'Reabrir aba fechada',
      'view.flipPanes': 'Inverter lados da barra lateral',
      'view.findInPage': 'Localizar na página',
      'view.findNext': 'Localizar próxima ocorrência',
      'view.findPrevious': 'Localizar ocorrência anterior',
      'appearance.toggleMode': 'Alternar modo claro / escuro',
      'profile.default': 'Alternar para o perfil padrão',
      'profile.switch.1': 'Alternar para o perfil 1',
      'profile.switch.2': 'Alternar para o perfil 2',
      'profile.switch.3': 'Alternar para o perfil 3',
      'profile.switch.4': 'Alternar para o perfil 4',
      'profile.switch.5': 'Alternar para o perfil 5',
      'profile.switch.6': 'Alternar para o perfil 6',
      'profile.switch.7': 'Alternar para o perfil 7',
      'profile.switch.8': 'Alternar para o perfil 8',
      'profile.switch.9': 'Alternar para o perfil 9',
      'profile.switch.10': 'Alternar para o perfil 10',
      'profile.switch.11': 'Alternar para o perfil 11',
      'profile.switch.12': 'Alternar para o perfil 12',
      'profile.switch.13': 'Alternar para o perfil 13',
      'profile.switch.14': 'Alternar para o perfil 14',
      'profile.switch.15': 'Alternar para o perfil 15',
      'profile.switch.16': 'Alternar para o perfil 16',
      'profile.switch.17': 'Alternar para o perfil 17',
      'profile.switch.18': 'Alternar para o perfil 18',
      'profile.next': 'Próximo perfil',
      'profile.prev': 'Perfil anterior',
      'profile.toggleAll': 'Alternar visualização de todos os perfis',
      'profile.create': 'Criar perfil',
      'composer.send': 'Enviar mensagem',
      'composer.newline': 'Inserir quebra de linha',
      'composer.steer': 'Direcionar o turno em andamento',
      'composer.queue': 'Colocar mensagem na fila',
      'composer.sendQueued': 'Enviar próximo turno na fila',
      'composer.mention': 'Referenciar arquivos, pastas, URLs',
      'composer.slash': 'Paleta de comandos slash',
      'composer.help': 'Ajuda rápida',
      'composer.history': 'Navegar histórico / popover',
      'composer.cancel': 'Fechar popover · cancelar execução',
      'view.selectionToComposer': 'Enviar seleção para o compositor'
    }
  },
  findInPage: {
    next: 'Próxima ocorrência',
    previous: 'Ocorrência anterior'
  },
  language: {
    label: 'Idioma',
    description: 'Escolha o idioma da interface do desktop.',
    saving: 'Salvando idioma…',
    saveError: 'Falha ao atualizar idioma',
    switchTo: 'Mudar idioma',
    searchPlaceholder: 'Pesquisar idiomas…',
    noResults: 'Nenhum idioma encontrado'
  },
  settings: {
    closeSettings: 'Fechar configurações',
    exportConfig: 'Exportar configurações',
    importConfig: 'Importar configurações',
    resetToDefaults: 'Redefinir para padrão',
    resetConfirm: 'Redefinir todas as configurações para o padrão do Hermes?',
    exportFailed: 'Falha ao exportar',
    resetFailed: 'Falha ao redefinir',
    nav: {
      providers: 'Provedores',
      providerAccounts: 'Contas',
      providerApiKeys: 'Chaves de API',
      providerCustomEndpoints: 'Endpoints Personalizados',
      apiKeys: 'Ferramentas & Chaves',
      keybinds: 'Atalhos de Teclado',
      keysTools: 'Ferramentas',
      keysSettings: 'Configurações',
      archivedChats: 'Chats Arquivados',
      about: 'Sobre',
      billing: 'Faturamento',
      notifications: 'Notificações',
      providerLocalModels: 'Modelos Locais'
    },
    plugins: {
      title: 'Plugins do Desktop',
      blurb:
        'Estendem este aplicativo, não um agente — instalados uma única vez para todo o app, em qualquer perfil, gateway ou máquina à qual você se conectar. Nativos ou colocados na pasta desktop-plugins; as alternâncias valem imediatamente.',
      count: n => `${n} instalados`,
      openFolder: 'Abrir pasta de plugins',
      rescan: 'Reescanear',
      reveal: 'Mostrar no explorador de arquivos',
      enable: 'Ativar',
      disable: 'Desativar',
      failed: 'falhou',
      empty: 'Nenhum plugin do desktop instalado ainda.',
      kinds: {
        bundled: 'nativo',
        disk: 'no disco',
        runtime: 'em execução'
      },
      installModal: {
        title: 'Instalar plugin',
        description: 'Revise o que este repositório contém antes de instalar qualquer coisa.',
        repoLabel: 'Repositório',
        includesHeading: 'Este pacote inclui',
        agentLabel: 'Plugin do agente',
        desktopLabel: 'Interface Desktop',
        agentTargetLocal: profile => `Instala no backend ${profile} (~/.hermes/plugins/)`,
        agentTargetRemote: profile => `Instala no backend ${profile} conectado`,
        desktopTarget: 'Instala na pasta local desktop-plugins deste aplicativo',
        desktopOnlyNote: 'Pacotes apenas para desktop não instalam um plugin de agente no backend.',
        insecureWarning:
          'Esta URL usa um esquema inseguro ou local. Prefira https:// ou git@ para instalações de produção.',
        securityHeading: 'Antes de instalar',
        securityIntro:
          'Instale apenas de fontes confiáveis — revise o repositório abaixo se quiser ver o que será adicionado.',
        sourceHeading: 'Código-fonte',
        viewRepository: 'Ver repositório',
        viewPluginFiles: 'Ver arquivos do plugin',
        gitCloneLabel: 'URL clone Git',
        enableAgent: 'Ativar plugin do agente após instalar',
        forceReinstall: 'Forçar reinstalação (substituir se já instalado)',
        install: 'Instalar',
        installing: 'Instalando…',
        probing: 'Inspecionando repositório…',
        probeUnavailable: 'Inspeção de plugin não está disponível neste ambiente.',
        desktopUnavailable: 'Instalação de plugin de desktop não está disponível neste ambiente.',
        selectComponent: 'Selecione pelo menos um componente para instalar.',
        agentSuccess: name => `Plugin de agente ${name} instalado`,
        desktopSuccess: name => `Plugin de desktop ${name} instalado`,
        agentFailed: 'Falha ao instalar o plugin do agente',
        desktopFailed: 'Falha ao instalar o plugin do desktop',
        missingEnv: vars => `Variáveis de ambiente ausentes: ${vars}. Adicione-as em Configurações → Chaves.`
      }
    },
    notifications: {
      title: 'Notificações',
      intro: 'Notificações do sistema (não alertas internos). Por dispositivo.',
      enableAll: 'Habilitar notificações',
      enableAllDesc: 'Quando desativado, silencia todas as notificações abaixo.',
      focusedHint: 'Alertas de conclusão só disparam enquanto o Hermes está em segundo plano.',
      kinds: {
        approval: {
          label: 'Aprovação necessária',
          description: 'Um comando está aguardando você aprovar ou rejeitar.'
        },
        input: {
          label: 'Entrada necessária',
          description: 'O Hermes fez uma pergunta ou precisa de uma senha/segredo.'
        },
        turnDone: {
          label: 'Resposta pronta',
          description: 'Um turno terminou enquanto o Hermes estava em segundo plano.'
        },
        turnError: {
          label: 'O turno falhou',
          description: 'Erros de turno em segundo plano.'
        },
        backgroundDone: {
          label: 'Tarefa em segundo plano concluída',
          description: 'Um comando de terminal em segundo plano foi concluído.'
        },
        credits: {
          label: 'Alertas de crédito',
          description: 'Acesso a créditos foi pausado ou restaurado.'
        },
        plugin: {
          label: 'Notificações de plugin',
          description: 'Um plugin enviou uma notificação enquanto o Hermes estava em segundo plano.'
        }
      },
      test: 'Enviar notificação de teste',
      testBody: 'As notificações estão funcionando.',
      testSent:
        'Teste enviado. Se nada aparecer, verifique as permissões de notificação do seu SO e modos de Foco/Não Perturbe.',
      testUnsupported: 'Este sistema não suporta notificações nativas.',
      completionSoundTitle: 'Som de Conclusão',
      completionSoundDesc: 'Toca quando o turno do agente termina. Escolha um som e teste aqui.',
      completionSoundPreview: 'Visualizar'
    },
    sections: {
      model: 'Modelo',
      appearance: 'Aparência',
      safety: 'Segurança',
      memory: 'Memória & Contexto',
      voice: 'Voz',
      advanced: 'Avançado'
    },
    searchPlaceholder: {
      about: 'Sobre o Hermes Desktop',
      config: 'Pesquisar configurações...',
      gateway: 'Conexão com gateway...',
      keys: 'Pesquisar chaves de API...',
      mcp: 'Pesquisar servidores MCP...',
      sessions: 'Pesquisar sessões arquivadas...'
    },
    modeOptions: {
      light: {
        label: 'Claro',
        description: 'Superfícies claras no desktop'
      },
      dark: {
        label: 'Escuro',
        description: 'Espaço de trabalho com baixo reflexo'
      },
      system: {
        label: 'Sistema',
        description: 'Seguir a aparência do SO'
      }
    },
    appearance: {
      title: 'Aparência',
      intro: 'Apenas Desktop. Modo de cor e tema da interface.',
      colorMode: 'Modo de Cor',
      colorModeDesc: 'Escolha um modo fixo ou siga as configurações do sistema.',
      toolViewTitle: 'Exibição de Ferramentas',
      toolViewDesc: 'Produto oculta os payloads; Técnico exibe entrada/saída completas.',
      reasoningCollapsedTitle: 'Ocultar raciocínio por padrão',
      reasoningCollapsedDesc: 'Mantém o raciocínio em segundo plano até você abrir.',
      uiScaleTitle: 'Escala da Interface',
      uiScaleDesc: percent => `Ajusta o tamanho de toda a interface. Atual: ${percent}%.`,
      sessionDensityTitle: 'Densidade da Lista de Sessões',
      sessionDensityDesc: 'Escolha quanto contexto aparece abaixo dos títulos das sessões na barra lateral.',
      sessionDensityCompact: 'Compacta',
      sessionDensityComfortable: 'Confortável',
      sessionDensityDetailed: 'Detalhada',
      tabStripTitle: 'Barra de Abas',
      tabStripDesc: 'Mostra abas acima de uma zona. Oculta automaticamente quando há apenas um painel.',
      tabStripAlways: 'Sempre',
      tabStripNever: 'Nunca',
      terminalFontTitle: 'Fonte do Terminal',
      terminalFontDesc:
        'Escolha uma fonte instalada para terminais do Desktop. Nerd Fonts renderizam ícones de shell e Powerlevel10k; deixe em branco para usar a JetBrains Mono padrão.',
      terminalFontPlaceholder: 'MesloLGS NF ou uma pilha de fontes CSS',
      terminalFontPreview: 'Pré-visualização de glifos',
      terminalFontReset: 'Usar padrão',
      translucencyTitle: 'Translucidez da Janela',
      translucencyDesc:
        'Veja sua área de trabalho através de toda a janela. Ajustado separadamente para temas claro e escuro.',
      translucencyGlassDesc:
        'Vidro fosco: o desktop aparece como um desfoque suave enquanto o texto permanece nítido. Ajustado separadamente.',
      translucencyModeClear: 'Transparente',
      translucencyModeGlass: 'Vidro Fosco',
      translucencyTintTitle: 'Tonalidade',
      translucencyFadeTitle: 'Desvanecimento',
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
        sidebar: 'Apenas barra lateral'
      },
      backdropTitle: 'Plano de Fundo do Chat',
      backdropDesc: 'A imagem suave da estátua atrás da conversa.',
      introSplashTitle: 'Tela de Abertura',
      introSplashDesc: 'A marca e o prompt exibidos em um chat vazio.',
      reactionsTitle: 'Reações de Mensagem',
      reactionsDesc: 'Reações de emoji estilo iMessage — reaja a mensagens, e o Hermes pode reagir às suas.',
      composerPopoutTitle: 'Compositor Flutuante',
      composerPopoutDesc:
        'Permite arrastar a caixa de mensagens para fora da doca. Desative para mantê-la fixa no fundo.',
      embedsTitle: 'Visualizações Integradas',
      embedsDesc:
        'Visualizações ricas carregam de sites de terceiros (YouTube, X, …). "Perguntar" mostra um espaço reservado até você permitir cada um; "Sempre" carrega automaticamente; "Desativado" mantém links simples.',
      embedsAsk: 'Perguntar',
      embedsAlways: 'Sempre',
      embedsOff: 'Desativado',
      embedsReset: count => `Redefinir ${count} serviço${count === 1 ? ' permitido' : 's permitidos'}`,
      product: 'Produto',
      productDesc: 'Atividade de ferramentas amigável com resumos concisos.',
      technical: 'Técnico',
      technicalDesc: 'Inclui argumentos reais, resultados puros e detalhes técnicos.',
      themeTitle: 'Tema',
      themeDesc: 'Apenas paletas do desktop. O modo selecionado é aplicado por cima.',
      themeProfileNote: profile => `Salvo para o perfil ${profile} — cada perfil mantém seu próprio tema.`,
      installTitle: 'Instalar do VS Code',
      installDesc:
        'Cole o ID de uma extensão do Marketplace (ex. dracula-theme.theme-dracula) para converter seu tema de cores em uma paleta de desktop.',
      installButton: 'Instalar',
      installing: 'Instalando…',
      installError: 'Não foi possível instalar esse tema.',
      installed: name => `Instalado “${name}”.`,
      removeTheme: 'Remover tema',
      importedBadge: 'Importado',
      pet: {
        title: 'Mascote',
        intro:
          'Adote um mascote animado que flutua sobre o aplicativo e reage ao que o Hermes faz — corre quando as ferramentas executam, comemora o sucesso e fica chateado com erros.',
        scaleTitle: 'Tamanho',
        scaleDesc: 'Redimensiona o mascote flutuante. Aplicado instantaneamente.',
        roamTitle: 'Vagar livremente',
        roamDesc: 'Deixa o mascote vagar pela janela sozinho quando inativo.',
        chooseTitle: 'Escolha um mascote',
        chooseDesc: 'Escolher um instala-o (se necessário) e o torna ativo.'
      },
      tipsDesc:
        'Dicas ocasionais do app e do Hermes. Cada dica aparece uma vez. Desativa automaticamente após os seus primeiros 30 dias; você pode reativar.',
      tipsReset: count => `Mostrar ${count} ${count === 1 ? 'dica' : 'dicas'} novamente`,
      toursDesc:
        'Deixe o Hermes destacar cada etapa enquanto guia você pelo app. Desativa automaticamente após os seus primeiros 30 dias; você pode reativar.'
    },
    about: {
      versionUnavailable: 'Versão indisponível',
      bundleOutOfSync: 'Build do app desatualizado',
      bundleOutOfSyncDesc:
        'O runtime do Hermes foi atualizado, mas o app desktop ainda é uma versão antiga — novas funcionalidades (como o Bot Mode) não aparecerão até que seja atualizado. Rode a atualização abaixo para recompilar o app. Se o aviso continuar, reinstale a partir do instalador mais recente.',
      bundleOutOfSyncAction: 'Baixar instalador',
      updates: 'Atualizações',
      checkNow: 'Verificar agora',
      checking: 'Verificando…',
      seeWhatsNew: 'Ver o que há de novo',
      updateNow: 'Atualizar agora',
      releaseNotes: 'Notas de lançamento',
      onLatest: 'Você está na versão mais recente.',
      installing: 'Uma atualização está sendo instalada.',
      cantUpdate: 'Este build não pode se atualizar dentro do aplicativo.',
      cantReach: 'Não conseguimos alcançar o servidor de atualizações.',
      tapCheck: 'Toque em "Verificar agora" para buscar atualizações.',
      updateReady: count =>
        `Uma nova atualização está pronta (${count} ${count === 1 ? 'mudança incluída' : 'mudanças incluídas'}).`,
      updateReadyUnknown: 'Uma nova atualização está pronta.',
      automaticUpdates: 'Atualizações automáticas',
      automaticUpdatesDesc:
        'O Hermes verifica atualizações automaticamente em segundo plano e avisa quando uma está pronta.'
    },
    credentials: {
      optional: 'Opcional',
      remove: 'Remover'
    },
    envActions: {
      docs: 'Documentação',
      clear: 'Limpar'
    },
    connections: {
      saving: 'Salvando…',
      cancel: 'Cancelar'
    },
    gateway: {
      unavailableTitle: 'Configurações do gateway indisponíveis',
      unavailableDesc: 'As configurações de conexão só podem ser alteradas no app Hermes Desktop do computador que o executa.',
      envOverrideTitle: 'Esta conexão foi fixada pelo modo como o Hermes foi iniciado.',
      envOverrideDesc:
        'Uma configuração de inicialização fora do app escolheu esta conexão, então as opções abaixo são somente leitura. Reinicie o Hermes sem essa configuração — ou peça a quem a definiu — para alterá-la aqui.',
      cloudDesc: 'Entre uma vez no Hermes Cloud e escolha entre os agentes da sua conta — nenhuma URL para colar.',
      cloudSignIn: 'Entrar no Hermes Cloud',
      cloudSignedIn: 'Conectado ao Hermes Cloud',
      cloudNeedsSignIn: 'Entre no Hermes Cloud para descobrir os agentes da sua conta.',
      probeError:
        'O Hermes não consegue acessar esse endereço. Verifique a URL e se o outro computador está executando o Hermes — as opções de login aparecem assim que ele responder.',
      signedIn: 'Conectado',
      signIn: 'Entrar',
      signInWith: provider => `Entrar com ${provider}`,
      authNeedsPassword: 'Este gateway usa nome de usuário e senha. Entre para autorizar este app desktop.',
      authNeedsOauth: provider => `Este gateway usa OAuth. Entre com ${provider} para autorizar este app desktop.`,
      openLogs: 'Abrir logs',
      savedTitle: 'Configurações do gateway salvas',
      failedLoad: 'Falha ao carregar as configurações do gateway',
      signInFailed: 'Falha no login',
      sshErrUpdateRequired: 'Atualize o Hermes no host remoto antes de conectar com o Desktop SSH.'
    },
    mcp: {
      disabled: 'desativado',
      remove: 'Remover'
    },
    model: {
      appliesDesc: 'Aplica-se a novas sessões. Use o seletor de modelo no chat para trocar o modelo atual.',
      auxiliaryTitle: 'Modelos auxiliares',
      resetAllToMain: 'Redefinir para o principal',
      auxiliaryDesc:
        'Tarefas auxiliares rodam no modelo principal por padrão. Atribua um modelo dedicado a qualquer tarefa para sobrescrever.',
      setToMain: 'Definir para principal',
      change: 'Alterar',
      autoUseMain: 'auto · usar modelo principal',
      tasks: {
        compression: {
          label: 'Compressão',
          hint: 'Compactação de contexto'
        },
        skills_hub: {
          label: 'Central de skills',
          hint: 'Busca de skills'
        },
        approval: {
          label: 'Aprovação',
          hint: 'Aprovação automática'
        },
        mcp: {
          hint: 'Roteamento de ferramentas MCP'
        },
        curator: {
          label: 'Curador',
          hint: 'Revisão de uso de skills'
        }
      },
      loadFailed: 'Não foi possível carregar os modelos',
      restartRequired:
        'Este backend está executando código antigo após uma atualização. Reinicie-o para carregar o novo código.'
    },
    providers: {
      intro:
        'Entre com uma assinatura — nenhuma chave de API para copiar. O Hermes executa o login pelo navegador para você, aqui mesmo no app.',
      collapse: 'Recolher'
    },
    sessions: {
      loading: 'Carregando sessões arquivadas…',
      archivedTitle: 'Sessões arquivadas',
      archivedIntro:
        'Chats arquivados ficam ocultos na barra lateral, mas guardam todas as mensagens. Segure Ctrl/⌘ e clique num chat na barra para arquivá-lo.',
      emptyArchivedTitle: 'Nada arquivado',
      emptyArchivedDesc: 'Arquive um chat para ocultá-lo aqui.',
      unarchive: 'Desarquivar',
      deleteConfirm: title => `Excluir permanentemente "${title}"? Isso não pode ser desfeito.`,
      autoArchiveTitle: 'Auto-arquivar chats inativos',
      autoArchiveDesc:
        'Arquiva automaticamente chats inativos. Chats fixados nunca são arquivados, e nada é excluído — eles apenas vêm para cá.',
      autoArchiveDaysLabel: 'Arquivar após',
      autoArchiveDaysUnit: 'dias de inatividade',
      autoArchiveFailed: 'Não foi possível atualizar o auto-arquivamento',
      defaultDirTitle: 'Diretório padrão do projeto',
      defaultDirDesc:
        'Novas sessões começam nesta pasta, exceto se você escolher outra. Deixe vazio para usar seu diretório raiz.',
      defaultDirUpdated: 'Diretório padrão do projeto atualizado — inicie um novo chat (Ctrl/⌘+N) para entrar em vigor',
      change: 'Alterar',
      choose: 'Escolher',
      clear: 'Limpar'
    },
    toolsets: {
      nousIncluded: 'Incluído em uma assinatura Nous — entre com sua conta Nous para ativar.',
      nousAuthNeededTitle: 'Entre com sua conta Nous',
      nousAuthNeededMessage: provider => `${provider} está salvo, mas só funcionará depois que você entrar com sua conta Nous.`,
      nousAuthSignIn: 'Entrar',
      nousAuthDoneTitle: 'Conta Nous conectada',
      nousAuthFailed: 'O login na Nous não foi concluído',
      postSetupErrorMessage: step => `Verifique o log de ${step}.`,
      terminalBackend: {
        needsSetupHint: 'Você pode selecionar esta opção agora — os comandos falharão até que a configuração seja concluída.'
      },
      browserRealProfile: {
        label: 'Usar Meu Perfil de Navegador Real',
        description:
          'Copia os logins e cookies do seu navegador padrão para um snapshot gerenciado com o qual o agente navega. Seu perfil ativo nunca é aberto diretamente. Aplica-se a novas sessões.',
        enabledTitle: 'Navegação com perfil real ativada',
        enabledMessage: 'Novas sessões navegarão com um snapshot do perfil do seu navegador padrão.',
        disabledTitle: 'Navegação com perfil real desativada',
        disabledMessage: 'O snapshot do perfil será excluído; novas sessões usarão um navegador limpo.',
        failedSave: 'Não foi possível salvar a configuração do perfil real',
        prompt: {
          title: 'Permaneça logado em seus sites',
          body: 'Permita que o Hermes navegue com um snapshot do perfil do seu navegador padrão, para que os sites abram já logados.',
          bulletSnapshot: 'Cookies e logins são copiados para um snapshot gerenciado.',
          bulletLiveProfile: 'Seu perfil de navegador ativo nunca é aberto diretamente.',
          bulletLocal: 'Nada sai deste computador.',
          dontShowAgain: 'Não mostrar novamente',
          notNow: 'Agora não',
          enable: 'Usar meu perfil'
        }
      }
    },
    localModels: {
      quickstartConfigure: 'Configure…',
      upToDateDetail: (tag, backend) => `Executando llama.cpp ${tag} (${backend}) — a build configurada.`
    }
  },
  skills: {
    skillDisabled: 'Skill desativada',
    toolsetEnabled: 'Toolset ativado',
    toolsetDisabled: 'Toolset desativado',
    appliesToNewSessions: name => `${name} se aplica a novas sessões.`,
    failedToUpdate: name => `Falha ao atualizar ${name}`,
    sortMostUsed: 'Mais usados',
    officialCatalog: 'Disponível para instalação',
    officialPill: 'Oficial',
    emptyNoneFound: noun => `Nenhum ${noun} encontrado`,
    emptyNothingMatches: query => `Nada corresponde a “${query}”.`,
    emptyNoneAvailable: noun => `Nenhum ${noun} disponível ainda.`,
    changesApplyNewSessions: 'Mudanças se aplicam a novas sessões.',
    skillUpdated: 'Skill atualizada',
    edit: 'Editar',
    archive: 'Arquivar',
    hub: {
      search: 'Pesquisar',
      installed: 'Instalado',
      install: 'Instalar',
      installing: 'Instalando...',
      close: 'Fechar'
    }
  },
  starmap: {
    refresh: 'Atualizar',
    loading: 'Carregando…'
  },
  agents: {
    failed: 'Falha',
    done: 'Concluído'
  },
  commandCenter: {
    back: 'Voltar',
    searchPlaceholder: 'Buscar sessões, visualizações e ações',
    settings: 'Configurações',
    pets: {
      installed: 'Instalado'
    },
    generatePet: {
      retry: 'Repetir',
      staleBackend: 'Atualize o Hermes para gerar mascotes.'
    },
    installTheme: {
      install: 'Instalar',
      installing: 'Instalando...',
      installed: 'Instalado',
      installs: count => `${count} instalações`
    },
    settingsFields: 'Campos de configuração',
    mcpServers: 'Servidores MCP',
    archivedChats: 'Chats arquivados',
    sections: {
      maintenance: 'Manutenção',
      sessions: 'Sessões',
      system: 'Sistema',
      usage: 'Uso'
    },
    sectionDescriptions: {
      maintenance: 'Diagnósticos, backups, curadoria e memória',
      sessions: 'Buscar e gerenciar sessões',
      system: 'Status, logs e ações do sistema',
      usage: 'Tokens, custos e atividade de skills no tempo'
    },
    nav: {
      skills: {
        detail: 'Skills, ferramentas, servidores MCP e plugins'
      },
      artifacts: {
        title: 'Artefatos',
        detail: 'Navegar nas saídas geradas'
      }
    },
    sectionEntries: {
      sessions: {
        title: 'Painel de sessões',
        detail: 'Buscar, fixar e gerenciar sessões'
      },
      system: {
        title: 'Painel do Sistema',
        detail: 'Status do gateway, logs, reiniciar/atualizar'
      },
      usage: {
        title: 'Painel de Uso',
        detail: 'Tokens, custo e atividade das skills'
      }
    },
    providerNavigate: 'Navegar',
    providerSessions: 'Sessões',
    refresh: 'Atualizar',
    refreshing: 'Atualizando...',
    noResults: 'Nenhum resultado correspondente.',
    unpinSession: 'Desafixar sessão',
    exportSession: 'Exportar sessão',
    deleteSession: 'Excluir sessão',
    noSessions: 'Nenhuma sessão ainda.',
    gatewayRunning: 'Gateway de mensagens em execução',
    gatewayStopped: 'Gateway de mensagens parado',
    hermesActiveSessions: (version, count) => `Hermes ${version} · ${count} sessões ativas`,
    restartGateway: 'Reiniciar gateway',
    openBrowser: 'Abrir navegador',
    gatewayRestartFailed: 'Falha ao reiniciar o gateway.',
    updateHermes: 'Atualizar Hermes',
    recentLogs: 'Logs recentes',
    noLogs: 'Nenhum log carregado.',
    statSessions: 'Sessões',
    statApiCalls: 'Chamadas de API',
    statTokens: 'Tokens (in/out)',
    statCost: 'Custo est.',
    retry: 'Repetir',
    maintenance: {
      pause: 'Pausar',
      resume: 'Retomar'
    }
  },
  messaging: {
    states: {
      connecting: 'Conectando'
    },
    unknown: 'Desconhecido',
    gatewayStopped: 'Gateway de mensagens parado',
    saveChanges: 'Salvar alterações',
    saved: 'Salvo',
    approvedUsers: count => `Usuários aprovados (${count})`,
    approve: 'Aprovar'
  },
  webhooks: {
    delete: 'Excluir',
    deleting: 'Excluindo...',
    restartGateway: 'Reiniciar gateway',
    done: 'Concluído',
    copy: 'Copiar'
  },
  profiles: {
    fleet: {
      allOnGateway: 'Todos os perfis neste gateway',
      gateway: gateway => `Perfis em ${gateway}`,
      gatewayUnreachable: gateway => `${gateway} — inacessível`,
      onGateway: (name, gateway) => `${name} — ${gateway}`,
      switchTo: (name, gateway) => `Mudar para ${name} em ${gateway}`,
      deleteOn: gateway => ` em ${gateway}`
    },
    remoteOverride: {
      menuItem: 'Conectar a um host remoto…',
      badge: host => `Executa em ${host}`,
      title: profile => `Conectar ${profile} a um host remoto`,
      description: 'As sessões deste perfil rodarão no Hermes remoto que você apontar, em vez deste computador.',
      urlLabel: 'Endereço remoto',
      urlInvalid: 'Insira um endereço completo começando com http:// ou https://',
      tokenLabel: 'Token de acesso',
      tokenPlaceholder: 'Cole o token da sessão remota',
      tokenSavedHint: 'Um token já está salvo. Deixe em branco para mantê-lo.',
      plainTextOptIn:
        'Este computador não possui armazenamento seguro, portanto, o token seria armazenado em texto puro na configuração.'
    },
    exportMenu: 'Exportar…',
    rename: 'Renomear',
    renameMenu: 'Renomear…',
    modelLabel: 'Modelo',
    deleting: 'Excluindo...',
    nameLabel: 'Nome',
    newNameLabel: 'Novo nome'
  },
  cron: {
    close: 'Fechar',
    title: 'Tarefas agendadas',
    count: count => `${count} ${count === 1 ? 'tarefa' : 'tarefas'}`,
    modelImpact: {
      title: 'Tarefas precisam de revisão',
      message: count =>
        `${count} ${count === 1 ? 'tarefa será ignorada' : 'tarefas serão ignoradas'} até que você revise as configurações do modelo.`,
      detailMore: (names, remaining) => `${names} e mais ${remaining}`,
      review: 'Revisar tarefas',
      saveFailed: 'Hermes não pôde salvar a alteração de modelo.'
    },
    search: 'Pesquisar tarefas...',
    loading: 'Carregando tarefas...',
    states: {
      enabled: 'ativado',
      scheduled: 'agendado',
      running: 'rodando',
      paused: 'pausado',
      disabled: 'desativado',
      error: 'erro',
      completed: 'concluído'
    },
    deliveryLabels: {
      local: 'Este desktop'
    },
    scheduleLabels: {
      daily: 'Diário',
      weekdays: 'Dias úteis',
      weekly: 'Semanal',
      monthly: 'Mensal',
      hourly: 'Por hora',
      'every-15-minutes': 'A cada 15 minutos',
      custom: 'Personalizado'
    },
    scheduleHints: {
      daily: 'Todos os dias às 9:00',
      weekdays: 'Segunda a sexta às 9:00',
      weekly: 'Toda segunda às 9:00',
      monthly: 'Primeiro dia do mês às 9:00',
      hourly: 'Toda hora exata',
      'every-15-minutes': 'A cada 15 minutos',
      custom: 'Sintaxe Cron ou linguagem natural'
    },
    days: {
      '0': 'Domingo',
      '1': 'Segunda',
      '2': 'Terça',
      '3': 'Quarta',
      '4': 'Quinta',
      '5': 'Sexta',
      '6': 'Sábado',
      '7': 'Domingo'
    },
    dayFallback: value => `dia ${value}`,
    everyDayAt: time => `Todo dia às ${time}`,
    weekdaysAt: time => `Dias úteis às ${time}`,
    everyDayOfWeekAt: (day, time) => `Toda ${day} às ${time}`,
    monthlyOnDayAt: (dayOfMonth, time) => `Mensalmente dia ${dayOfMonth} às ${time}`,
    topOfHour: 'Na hora exata',
    everyHourAt: minute => `Toda hora aos :${minute}`,
    newCron: 'Nova tarefa',
    emptyDescNew:
      'Agende um prompt para rodar em expressão cron. Hermes o executará e enviará os resultados para onde você escolher.',
    emptyDescSearch: 'Tente uma pesquisa mais abrangente.',
    emptyTitleNew: 'Nenhuma tarefa agendada',
    emptyTitleSearch: 'Sem resultados',
    last: 'Última:',
    next: 'Próxima:',
    noRuns: 'Nenhuma execução',
    manage: 'Gerenciar',
    showRuns: 'Mostrar execuções',
    hideRuns: 'Ocultar execuções',
    runHistory: 'Histórico de execuções',
    actionsTitle: 'Ações da tarefa',
    resume: 'Retomar tarefa',
    pause: 'Pausar tarefa',
    resumeTitle: 'Retomar',
    pauseTitle: 'Pausar',
    triggerNow: 'Executar agora',
    edit: 'Editar tarefa',
    deleteTitle: 'Excluir tarefa?',
    deleteDescPrefix: 'Isto removerá ',
    deleteDescSuffix: ' permanentemente. Irá parar de rodar imediatamente.',
    deleting: 'Excluindo...',
    resumed: 'Tarefa retomada',
    paused: 'Tarefa pausada',
    triggered: 'Tarefa acionada',
    deleted: 'Tarefa excluída',
    created: 'Tarefa criada',
    updated: 'Tarefa atualizada',
    failedLoad: 'Falha ao carregar tarefas',
    failedUpdate: 'Falha ao atualizar tarefa',
    failedTrigger: 'Falha ao acionar tarefa',
    failedDelete: 'Falha ao excluir tarefa',
    failedSave: 'Falha ao salvar tarefa',
    editTitle: 'Editar tarefa',
    createTitle: 'Nova tarefa',
    editDesc: 'Atualize o agendamento, prompt ou destino.',
    createDesc: 'Agende um prompt para rodar automaticamente. Use sintaxe cron ou uma frase natural como "a cada 15 minutos".',
    nameLabel: 'Nome',
    namePlaceholder: 'Resumo matinal',
    promptPlaceholder: 'Resuma minhas threads não lidas do Slack e me envie as 5 principais por e-mail...',
    frequencyLabel: 'Frequência',
    deliverLabel: 'Entregar em',
    deliverNeedsHomeChannel: 'defina um canal primeiro',
    modelLabel: 'Modelo',
    modelDefault: 'Padrão (global)',
    customScheduleLabel: 'Agendamento personalizado',
    customPlaceholder: '0 9 * * * ou dias úteis às 9:00',
    customHint: 'Sintaxe Cron ou linguagem natural.',
    optional: 'Opcional',
    promptRequired: 'Prompt é obrigatório.',
    promptScheduleRequired: 'Prompt e agendamento obrigatórios.',
    scheduleRequired: 'Agendamento é obrigatório.',
    scriptOnlyEditHint: 'Tarefa apenas com script (sem IA). Id:',
    saveChanges: 'Salvar alterações',
    createAction: 'Criar tarefa',
    tabs: {
      jobs: 'Tarefas'
    },
    blueprints: {
      startFrom: 'Começar de',
      custom: 'Personalizado',
      subtitle: 'Automações prontas',
      dialogDesc: 'Preencha os detalhes para agendar.',
      scheduleIt: 'Agendar',
      scheduling: 'Agendando...',
      scheduled: 'Blueprint agendado',
      loading: 'Carregando blueprints...',
      failedLoad: 'Falha ao carregar blueprints',
      emptyTitle: 'Sem blueprints disponíveis',
      emptyDesc: 'Nenhum blueprint de automação disponível neste servidor.'
    }
  },
  artifacts: {
    copyUrl: 'Copiar URL',
    copyPath: 'Copiar caminho'
  },
  artifactCard: {
    open: 'Abrir'
  },
  artifactPreview: {
    copyContent: 'Copiar conteúdo',
    download: 'Baixar',
    openInBrowser: 'Abrir no navegador'
  },
  sidebar: {
    nav: {
      'new-session': 'Nova Sessão',
      skills: 'Capacidades',
      messaging: 'Mensageria',
      artifacts: 'Artefatos',
      cron: 'Tarefas Agendadas'
    },
    searchAria: 'Pesquisar sessões',
    searchPlaceholder: 'Pesquisar sessões…',
    clearSearch: 'Limpar pesquisa',
    noMatch: query => `Nenhuma sessão com “${query}”.`,
    results: 'Resultados',
    pinned: 'Fixados',
    sessions: 'Sessões',
    cronJobs: 'Tarefas agendadas',
    groupAriaGrouped: 'Mostrar sessões como lista única',
    groupAriaUngrouped: 'Agrupar sessões por workspace',
    showProjects: 'Mostrar projetos',
    showSessions: 'Mostrar sessões',
    groupTitleGrouped: 'Desagrupar sessões',
    groupTitleUngrouped: 'Agrupar por workspace',
    allPinned: 'Tudo aqui está fixado. Desafixe um chat para exibi-lo nos recentes.',
    shiftClickHint: 'Shift-click para fixar um chat',
    noWorkspace: 'Sem workspace',
    projectEmpty: 'Sem sessões',
    noSessions: 'Nenhuma sessão ainda',
    noFilterMatches: 'Nenhuma sessão corresponde a esses filtros',
    projects: {
      sectionLabel: 'Projetos',
      menuRename: 'Renomear',
      copyPath: 'Copiar caminho',
      startWork: 'Nova árvore de trabalho',
      newWorktreeTitle: 'Nova árvore de trabalho'
    },
    loading: 'Carregando…',
    row: {
      export: 'Exportar',
      rename: 'Renomear',
      archive: 'Arquivar'
    },
    statusDivider: {
      done: 'Concluído'
    }
  },
  composer: {
    stop: 'Parar',
    hotkeyDescs: {
      'keybinds.openPanel': 'todos os atalhos de teclado'
    }
  },
  statusStack: {
    stop: 'Parar',
    coding: {
      close: 'Fechar',
      openChanges: 'Abrir alterações',
      openFile: 'Abrir arquivo'
    }
  },
  updates: {
    connectionRetry:
      'O Hermes não conseguiu acessar o servidor de atualizações. Verifique sua conexão com a internet e tente novamente. Se você usa um Hermes remoto, verifique se ele está online.',
    updateNow: 'Atualizar agora',
    copy: 'Copiar',
    copied: 'Copiado',
    done: 'Concluído'
  },
  install: {
    stageStates: {
      failed: 'Falha'
    },
    copyCommand: 'Copiar comando',
    probeError:
      'O Hermes não consegue acessar esse endereço. Verifique a URL e se o outro computador está executando o Hermes — as opções de login aparecem assim que ele responder.',
    identityProvider: 'seu provedor de identidade',
    authNeedsOauth: provider => `Entre com ${provider} antes de testar este gateway.`,
    signIn: 'Entrar',
    signInWith: provider => `Entrar com ${provider}`,
    incompleteSignInTest: 'Entre antes de testar este gateway protegido por OAuth.',
    failedDesc:
      'Uma das etapas da instalação não foi concluída. Isso pode acontecer quando outra cópia do Hermes está em execução, a conexão com a internet caiu ou o antivírus bloqueou o instalador. Feche outras janelas do Hermes, escolha Recarregar e tente novamente. Se falhar de novo, abra os logs e envie-os ao suporte.',
    error: 'Erro',
    copyOutput: 'Copiar saída'
  },
  onboarding: {
    collapse: 'Recolher',
    connecting: 'Conectando',
    update: 'Atualizar',
    flowSubtitles: {
      external: 'Entre uma vez no seu terminal e volte para conversar'
    },
    signInFailed: 'Falha no login. Tente novamente.',
    signInWith: provider => `Entrar com ${provider}`,
    copy: 'Copiar',
    change: 'Alterar',
    signInExpired:
      'A página de login expirou antes de você concluir. Tente novamente e complete a etapa do navegador em alguns minutos, ou use uma chave de API.'
  },
  modelPicker: {
    title: 'Mudar modelo',
    loadFailed: 'Não foi possível carregar os modelos',
    wasPrice: 'era'
  },
  modelVisibility: {
    search: 'Buscar modelos'
  },
  shell: {
    modelMenu: {
      search: 'Buscar modelos',
      editModels: 'Editar modelos…',
      refreshModels: 'Atualizar modelos'
    },
    modelOptions: {
      noOptions: 'Nenhuma opção para este modelo'
    },
    gatewayMenu: {
      connecting: 'Conectando',
      openSystem: 'Abrir painel do sistema'
    },
    approvalMode: {
      manualDescription: 'Perguntar antes de ações que requerem aprovação',
      smartDescription: 'Avaliar ações automaticamente e perguntar quando necessário'
    },
    statusbar: {
      restart: 'reiniciar',
      updateInProgress: 'Atualização em andamento',
      openCommandCenter: 'Abrir Central de Comandos',
      customizeTitle: 'Mostrar na barra de status',
      resetStatusbar: 'Redefinir para padrões',
      openAgents: 'Abrir agentes',
      openCron: 'Abrir tarefas cron',
      openWebhooks: 'Abrir webhooks',
      openStarmap: 'Abrir grafo de memória',
      session: 'Sessão',
      switchModel: 'Mudar modelo',
      openModelPicker: 'Abrir seletor de modelo',
      modelPinned: 'fixado por você; novos chats usam este em vez do padrão das Configurações'
    }
  },
  rightSidebar: {
    noFolderSelected: 'Nenhuma pasta selecionada',
    remotePickerTitle: 'Escolher pasta remota',
    remotePickerDescription: 'Navegar pelas pastas no backend conectado.',
    remotePickerSelect: 'Selecionar pasta',
    openFolder: 'Abrir pasta',
    noProjectTitle: 'Nenhum projeto',
    noProjectBody: 'Abra um projeto para navegar nos arquivos e revisar as alterações.',
    noProjectOpen: 'Nenhum projeto aberto',
    emptyBody: 'Esta pasta está vazia.',
    treeErrorBody: 'A árvore de arquivos encontrou um erro ao renderizar esta pasta.',
    terminalNew: 'Novo terminal',
    addToChat: 'Adicionar ao chat'
  },
  preview: {
    openPreview: 'Abrir visualização',
    openInBrowser: 'Abrir no navegador',
    binaryTitle: 'Este parece ser um arquivo binário',
    largeTitle: 'Este arquivo é grande',
    edit: 'Editar',
    console: {
      copyFailed: 'Não foi possível copiar a saída do console',
      copyEntry: 'Copiar esta entrada',
      sendEntry: 'Enviar esta entrada ao chat',
      sendToChat: 'Enviar ao chat',
      copySelected: 'Copiar selecionado para a área de transferência',
      copyAll: 'Copiar tudo para a área de transferência',
      copy: 'Copiar',
      clear: 'Limpar',
      sentTitle: 'Enviado ao chat'
    },
    web: {
      appFailedToBoot: 'O aplicativo de visualização falhou ao iniciar',
      failedToLoad: 'A visualização falhou ao carregar',
      restarting: 'O Hermes está reiniciando...',
      askRestart: 'Pedir ao Hermes para reiniciar o servidor',
      restartingMessage:
        'O Hermes está trabalhando em segundo plano. Acompanhe o console de visualização para ver o progresso.',
      restartFailed: 'Falha ao reiniciar o servidor',
      openDevTools: 'Abrir DevTools de visualização',
      blankPageBody: 'Digite um endereço acima para navegar, ou peça ao Hermes para abrir uma página.',
      finishedRestarting: message => `O Hermes terminou de reiniciar o servidor de visualização${message ? `: ${message}` : ''}`,
      reloadingNow: 'Recarregando a visualização agora.',
      restartFailedTitle: 'Falha ao reiniciar a visualização',
      restartFailedMessage: 'O Hermes não conseguiu reiniciar o servidor.',
      unreachableDescription: 'A página de visualização não pôde ser acessada.',
      annotateNeedPage: 'Abra uma página no navegador integrado primeiro.',
      annotateFailed: 'Não foi possível iniciar o modo de anotação'
    },
    openInExternal: 'Abrir no externo'
  },
  zones: {
    lastTabKeptBody:
      'Esta zona precisa de pelo menos uma aba visível. Mostre outra aba primeiro, ou recolha toda a barra lateral.',
    closeToRight: 'Fechar à direita',
    newSessionTab: 'Nova aba de sessão',
    pluginDisabledBody: 'Reative em Configurações → Plugins para trazer o painel de volta.',
    editHint: 'Escolha um layout ou arraste painéis entre zonas.',
    reset: 'Redefinir',
    custom: 'Personalizado',
    newGridLayout: 'Novo layout de grade',
    newTab: 'Nova aba'
  },
  contextMenu: {
    link: {
      openInApp: 'Abrir no navegador integrado',
      openExternal: 'Abrir no navegador externo',
      copyUrl: 'Copiar URL',
      copyResolvedUrl: 'Copiar URL resolvida'
    },
    image: {
      copyImage: 'Copiar imagem',
      copyImageAddress: 'Copiar endereço da imagem'
    },
    edit: {
      addToDictionary: 'Adicionar ao dicionário'
    },
    page: {
      copyPageUrl: 'Copiar URL da página'
    }
  },
  assistant: {
    thread: {
      loadingResponse: 'O Hermes está carregando uma resposta',
      copy: 'Copiar',
      refresh: 'Atualizar',
      branchNewChat: 'Ramificar em novo chat',
      errorLayers: {
        auth: 'Erro de autenticação',
        billing: 'Sem créditos',
        endpoint: 'Erro de endpoint personalizado',
        gateway: 'Erro do gateway',
        generic: 'O turno falhou',
        provider: 'Erro do provedor',
        runtime: 'Erro do runtime local',
        streaming: 'Erro de conexão de streaming'
      },
      errorSwitchProvider: 'Mudar provedor',
      errorOpenLogs: 'Abrir logs',
      errorOpenLogsFailed: 'Não foi possível abrir a pasta de logs',
      errorOpenDesktopLogs: 'Abrir logs do Desktop',
      errorCopyDiagnostics: 'Copiar detalhes do erro',
      errorSendDiagnostics: 'Enviar diagnósticos',
      scrollToBottom: 'Rolar para o fim',
      stop: 'Parar',
      restoreTitle: 'Restaurar para este checkpoint?',
      sendEdited: 'Enviar mensagem editada'
    },
    approval: {
      gatewayDisconnected: 'O gateway do Hermes não está conectado',
      sendFailed: 'Não foi possível enviar resposta de aprovação',
      allowSession: 'Permitir esta sessão',
      jumpToApproval: 'Aprovação necessária',
      reject: 'Rejeitar',
      alwaysTitle: 'Sempre permitir este comando?'
    },
    clarify: {
      notReady: 'A solicitação de esclarecimento ainda não está pronta',
      gatewayDisconnected: 'O gateway do Hermes não está conectado',
      sendFailed: 'Não foi possível enviar resposta de esclarecimento',
      skip: 'Pular',
      confirmAndContinueLabel: 'Confirmar e continuar',
      lateAnswerTip: 'Rascunhe esta resposta como mensagem de acompanhamento',
      lateAnswerHint:
        'Este prompt não está mais aguardando. Escolha uma opção para rascunhá-lo como mensagem de acompanhamento.'
    },
    mcpSetup: {
      installTitle: 'Adicionar servidores MCP',
      enableTitle: 'Ativar servidores MCP',
      authorizeTitle: 'Autorizar servidores MCP',
      envRequired: 'Preencha as credenciais necessárias primeiro',
      sendFailed: 'Não foi possível enviar resposta de configuração MCP',
      gatewayDisconnected: 'O gateway do Hermes não está conectado'
    },
    tool: {
      copyCode: 'Copiar código',
      copyOutput: 'Copiar saída',
      copyCommand: 'Copiar comando',
      copyContent: 'Copiar conteúdo',
      copyUrl: 'Copiar URL',
      copyResults: 'Copiar resultados',
      copyQuery: 'Copiar consulta',
      copyFile: 'Copiar arquivo',
      copyPath: 'Copiar caminho',
      copyActivity: 'Copiar atividade',
      actions: {
        failedToOpen: 'Falha ao abrir'
      }
    }
  },
  prompts: {
    gatewayDisconnected: 'O gateway do Hermes não está conectado',
    sudoSendFailed: 'Não foi possível enviar a senha sudo',
    secretSendFailed: 'Não foi possível enviar o segredo',
    sudoDesc:
      'O Hermes precisa da sua senha sudo para executar um comando privilegiado. Ela é enviada apenas para o seu agente local.',
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
    emptySlashCommand: 'comando barra (slash) vazio',
    desktopCommands: 'Comandos do desktop',
    skillCommandsAvailable: count => `${count} comandos de skill disponíveis.`,
    warningLine: message => `aviso: ${message}`,
    yoloArmed: 'YOLO armado para este chat',
    yoloOff: 'YOLO desligado',
    yoloSystem: active => `YOLO ${active ? 'ligado' : 'desligado'} para esta sessão`,
    yoloToggleFailed: 'Não foi possível alternar o YOLO',
    profileStatus: current =>
      `Perfil: ${current}. Use /profile <nome> ou o seletor de "Nova sessão" para iniciar um chat em outro perfil.`,
    unknownProfile: 'Perfil desconhecido',
    noProfileNamed: (target, available) => `Nenhum perfil chamado "${target}". Disponíveis: ${available}`,
    newChatsProfile: name => `Novos chats usarão o perfil ${name}.`,
    setProfileFailed: 'Falha ao definir o perfil',
    sttDisabled: 'Fala-para-texto (STT) está desativado nas configurações.',
    stopFailed: 'Falha ao parar',
    regenerateFailed: 'Falha ao regenerar',
    editFailed: 'Falha ao editar',
    editTurnUnavailable: 'Este turno não está mais no histórico do servidor (pode ter sido compactado).',
    resumeFailed: 'Falha ao retomar',
    resumeStrandedTitle: 'Não foi possível carregar esta sessão',
    resumeStrandedBody:
      'A conexão com esta sessão falhou e as tentativas automáticas desistiram. Verifique se o gateway está rodando e tente novamente.',
    resumeRetry: 'Tentar novamente',
    nothingToBranch: 'Nada para ramificar',
    branchNeedsChat: 'Inicie ou retome um chat antes de ramificar.',
    sessionBusy: 'Sessão ocupada',
    branchStopCurrent: 'Pare o turno atual antes de ramificar este chat.',
    branchNoText: 'Esta mensagem não tem texto para ramificar.',
    branchTitle: n => `Rascunho: Ramo #${n}`,
    branchFailed: 'Falha ao ramificar',
    deleteFailed: 'Falha ao deletar',
    archived: 'Arquivado',
    archiveFailed: 'Falha ao arquivar',
    cwdChangeFailed: 'Falha na mudança do diretório de trabalho',
    cwdStagedTitle: 'Diretório de trabalho preparado',
    cwdStagedMessage: 'Reinicie o backend do desktop para aplicar as mudanças de diretório nesta sessão ativa.',
    modelSwitchFailed: 'Falha ao trocar o modelo',
    sessionExported: 'Sessão exportada',
    sessionExportFailed: 'Não foi possível exportar a sessão',
    imageSaved: 'Imagem salva',
    downloadStarted: 'Download iniciado',
    restartToUseSaveImage: 'Reinicie o Hermes Desktop para usar Salvar Imagem.',
    restartToSaveImages: 'Reinicie o Hermes Desktop para salvar imagens',
    imageDownloadFailed: 'Falha no download da imagem',
    openImage: 'Abrir imagem',
    downloadImage: 'Baixar imagem',
    savingImage: 'Salvando imagem',
    imagePreviewFailed: 'Falha na visualização da imagem',
    imageAttach: 'Anexar imagem',
    imageWriteFailed: 'Falha ao escrever a imagem no disco.',
    imageAttachFailed: 'Falha ao anexar imagem',
    attachImages: 'Anexar imagens',
    clipboard: 'Área de transferência',
    noClipboardImage: 'Nenhuma imagem encontrada na área de transferência',
    clipboardPasteFailed: 'Falha ao colar da área de transferência',
    dropFiles: 'Soltar arquivos',
    handoff: {
      pickPlatform: 'Escolha um destino',
      success: platform => `Transferido para ${platform}. Retome aqui a qualquer momento.`,
      systemNote: platform => `↻ Transferido para ${platform} — retome aqui a qualquer momento.`,
      failed: error => `Falha na transferência: ${error}`,
      timedOut: 'Tempo esgotado esperando pelo gateway. O `hermes gateway` está rodando?'
    },
    readOnlyTranscriptBody:
      'Nenhum backend conectado reivindica este chat mais antigo ainda, então foi aberto como transcrição somente leitura. O histórico está intacto; o envio está desativado até que um backend o reivindique.'
  },
  errors: {
    genericFailure: 'Algo deu errado',
    boundaryTitle: 'Algo quebrou na interface',
    boundaryDesc: 'A visualização encontrou um erro inesperado. Seus chats e configurações estão seguros.',
    reloadWindow: 'Recarregar janela',
    openLogs: 'Abrir logs'
  },
  ui: {
    search: {
      clear: 'Limpar pesquisa'
    },
    pagination: {
      label: 'paginação',
      previous: 'Anterior',
      previousAria: 'Ir para a página anterior',
      next: 'Próximo',
      nextAria: 'Ir para a próxima página'
    },
    sidebar: {
      title: 'Barra lateral',
      description: 'Exibe a barra lateral móvel.',
      toggle: open => `${open ? 'Mostrar' : 'Ocultar'} barra lateral`
    }
  },
  tips: {
    items: {
      'new-session': {
        text: 'Um novo chat tem seu próprio contexto, terminal e diretório de trabalho.'
      },
      skills: {
        text: 'Skills são pastas de instruções que o Hermes carrega quando o trabalho exige.'
      },
      messaging: {
        title: 'Hermes longe da sua mesa'
      },
      artifacts: {
        text: 'Imagens, arquivos e links de cada sessão, indexados em um único lugar.'
      },
      cron: {
        title: 'Trabalho que se executa sozinho',
        text: 'Agende um prompt de hora em hora, à noite ou com uma expressão cron.'
      },
      'command-palette': {
        title: 'Uma caixa para tudo',
        text: 'Sessões, configurações, skills e comandos todos respondem à paleta.'
      },
      profiles: {
        title: 'Perfis são separados'
      },
      'composer-mentions': {
        title: 'Anexar e comandar',
        text: 'Digite @ para trazer um arquivo para a conversa, / para executar um comando.'
      },
      'local-setup': {
        title: 'Esta máquina pode executar modelos localmente',
        text: 'Seu hardware pode servir um modelo local. Os chats ficam no seu computador e não custam nada.'
      },
      'right-pane': {
        title: 'O painel de trabalho',
        text: 'Arquivos, terminal, revisão e o navegador integrado compartilham o lado direito.'
      }
    }
  }
})
