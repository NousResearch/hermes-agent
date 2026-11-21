import type { TranslationOverrides } from './define-locale'
import { ptBrCatalogInstall } from './pt-br_catalog_install'
import { ptBrClarify } from './pt-br_clarify'
import { ptBrErrorCodes } from './pt-br_error_codes'
import { ptBrErrorLayers } from './pt-br_error_layers'
import { ptBrModelMenu } from './pt-br_model_menu'
import { ptBrPreviewWeb } from './pt-br_preview_web'
import { ptBrThemePicker } from './pt-br_theme_picker'

export const ptBrShellOverrides = {
  shell: {
    windowControls: 'Controles da janela',
    paneControls: 'Controles do painel',
    appControls: 'Controles do aplicativo',
    modelMenu: ptBrModelMenu,
    modelOptions: {
      noOptions: 'Nenhuma opção para este modelo',
      options: 'Opções',
      thinking: 'Raciocínio',
      fast: 'Rápido',
      ultrafast: 'Ultrarrápido',
      useStandardSpeed: 'Usar velocidade padrão',
      effort: 'Esforço',
      minimal: 'Mínimo',
      low: 'Baixo',
      medium: 'Médio',
      high: 'Alto',
      xhigh: 'Muito alto',
      max: 'Máximo',
      ultra: 'Ultra',
      sendsOnRoute: (level: string) => `envia ${level} nesta rota`,
      updateFailed: 'Falha ao atualizar a opção do modelo',
      fastFailed: 'Falha ao atualizar o modo rápido',
      daybreak: 'Daybreak',
      daybreakFailed: 'Falha ao atualizar Daybreak'
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
      reconnectGateway: 'Reconectar gateway',
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
      smartDescription: 'Avaliar ações automaticamente e perguntar quando necessário',
      off: 'Desativado',
      offDescription: 'Executar sem pedidos de aprovação'
    },
    statusbar: {
      unknown: 'desconhecido',
      restart: 'reiniciar',
      update: 'atualizar',
      updateInProgress: 'Atualização em andamento',
      commitsBehind: (count, branch) => `${count} commit${count === 1 ? '' : 's'} atrás de ${branch}`,
      releaseAvailable: (tag: string) => `A versão ${tag} está disponível.`,
      desktopVersion: version => `Hermes Desktop v${version}`,
      backendVersion: version => `Backend v${version}`,
      clientLabel: version => `cliente v${version}`,
      connectionSsh: host => `SSH: ${host}`,
      connectionRemote: host => `Remoto: ${host}`,
      connectionCloud: host => `Nuvem: ${host}`,
      connectionCloudTooltip: host => `Hermes Cloud · ${host}`,
      connectionSshTooltip: host => `SSH · ${host}`,
      connectionRemoteTooltip: host => `Remoto · ${host}`,
      backendLabel: version => `backend v${version}`,
      commit: sha => `commit ${sha}`,
      branch: branch => `branch ${branch}`,
      closeCommandCenter: 'Fechar central de comandos',
      openCommandCenter: 'Abrir central de comandos',
      showTerminal: 'Mostrar terminal',
      hideTerminal: 'Ocultar terminal',
      gateway: 'Gateway',
      backend: 'Backend',
      messagingStopped: 'mensagens interrompidas',
      messagingDegraded: name => `${name} indisponível`,
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
      resetStatusbar: 'Restaurar padrões',
      toggleApprovalMode: 'Aprovações',
      toggleBackendVersion: 'Versão do backend',
      toggleCacheHitRate: 'Taxa de acertos do cache',
      toggleCommandCenter: 'Central de comandos',
      toggleContextUsage: 'Medidor de contexto',
      toggleRunningTimer: 'Cronômetro do turno',
      toggleSessionTimer: 'Cronômetro da sessão',
      toggleTerminal: 'Terminal',
      toggleTokensPerSecond: 'Tokens por segundo',
      toggleVersion: 'Versão e atualizações',
      toggleFreeTier: 'Plano gratuito',
      toggleWorkspace: 'Área de trabalho',
      cacheHitRateTitle:
        'Taxa de acertos do cache de prompts nesta sessão — tokens em cache custam menos, então quanto maior, mais barato',
      tokensPerSecondTitle: 'Tokens de saída por segundo, em média nas últimas 10 chamadas ao modelo',
      togglePlanUsage: 'Uso do plano',
      planUsageTitle: (used, resetsIn) =>
        `${used} do uso do seu plano consumido${resetsIn ? ` · renova em ${resetsIn}` : ''} · abrir Faturamento`,
      agents: 'Agentes',
      closeAgents: 'Fechar agentes',
      openAgents: 'Abrir agentes',
      subagents: count => `${count} subagente${count === 1 ? '' : 's'}`,
      failed: count => `${count} com falha`,
      running: count => `${count} em execução`,
      cron: 'Cron',
      openCron: 'Abrir tarefas cron',
      webhooks: 'Webhooks',
      openWebhooks: 'Abrir webhooks',
      starmap: 'Grafo de memória',
      openStarmap: 'Abrir grafo de memória',
      turnRunning: 'Em execução',
      contextUsage: 'Uso de contexto',
      compressions: count => `Compactações: ${count}`,
      systemResources: {
        title: 'Recursos do sistema',
        loading: 'Recursos…',
        gpuUtilization: 'Uso da GPU',
        gpuMemory: 'Memória da GPU',
        ram: 'RAM',
        unifiedNote: 'Memória unificada — a GPU e o sistema compartilham esta memória.',
        toggle: 'Recursos do sistema'
      },
      contextUsagePanel: {
        categories: {
          conversation: 'Conversa',
          mcp: 'MCP',
          memory: 'Memória',
          rules: 'Regras',
          skills: 'Habilidades',
          subagent_definitions: 'Definições de subagentes',
          system_prompt: 'Prompt do sistema',
          tool_definitions: 'Definições de ferramentas'
        },
        compact: {
          action: 'Compactar conversa',
          busyHint: 'Conclua ou interrompa a resposta atual primeiro',
          confirm: 'Compactar',
          hint: 'Resuma o contexto antigo para liberar espaço. Continue nesta mesma conversa.',
          question: 'Compactar esta conversa em um resumo?',
          questionNamed: title => `Compactar “${title}” em um resumo?`,
          working: 'Compactando…'
        },
        empty: 'Ainda não há dados de contexto',
        loading: 'Carregando detalhamento…',
        percentFull: percent => `${percent}% ocupado`,
        title: 'Uso de contexto',
        tokenSummary: (used, max) => `${used} / ${max} tokens`
      },
      focusedSince: 'Em foco há',
      focusedSinceTitle: 'Tempo desde que esta conversa recebeu foco — não é a duração de um turno em execução',
      yoloOn: 'YOLO ativado — aprovação automática de comandos perigosos. Shift+clique alterna globalmente.',
      yoloOff: 'YOLO desativado. Shift+clique alterna globalmente.',
      modelNone: 'nenhum',
      noModel: 'nenhum modelo',
      switchModel: 'Trocar modelo',
      openModelPicker: 'Abrir seletor de modelos',
      modelPinned: 'fixado por você; novas conversas usam este modelo em vez do padrão das Configurações',
      modelTitle: (provider, model) => `Modelo · ${provider}: ${model}`,
      providerModelTitle: (provider, model) => `${provider} · ${model}`
    }
  },

  rightSidebar: {
    terminalReadOnly: 'Saída somente leitura',
    terminalReadOnlyHelp:
      'Para responder às solicitações, interrompa o comando em segundo plano e execute-o em um novo terminal. O novo terminal abre um shell separado; ele não se conecta a este processo.',
    terminalOpenInteractive: 'Abrir novo terminal',
    aria: 'Barra lateral direita',
    panelsAria: 'Painéis da barra lateral direita',
    files: 'Sistema de arquivos',
    terminal: 'Terminal',
    noFolderSelected: 'Nenhuma pasta selecionada',
    changeCwdTitle: 'Alterar diretório de trabalho',
    remotePickerTitle: 'Escolher pasta remota',
    remotePickerDescription: 'Navegue pelas pastas no backend conectado.',
    remotePickerSelect: 'Selecionar pasta',
    remotePickerNewFolder: 'Nova pasta',
    remotePickerFolderName: 'Nome da pasta',
    remotePickerCreateFolder: 'Criar pasta',
    remotePickerInvalidFolderName: 'Digite apenas um nome de pasta, sem barras.',
    remotePickerCreateFolderFailed: error => `Não foi possível criar a pasta (${error}).`,
    folderTip: cwd => cwd,
    openFolder: 'Abrir pasta',
    refreshTree: 'Atualizar árvore',
    collapseAll: 'Recolher todas as pastas',
    showIgnored: 'Mostrar arquivos ignorados pelo Git',
    hideIgnored: 'Ocultar arquivos ignorados pelo Git',
    previewUnavailable: 'Prévia indisponível',
    couldNotPreview: path => `Não foi possível visualizar ${path}`,
    noProjectTitle: 'Nenhum projeto',
    noProjectBody: 'Abra um projeto para navegar pelos arquivos e revisar alterações.',
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
    terminalNewOn: 'Novo terminal em…',
    terminalGatewayRemoved: 'Gateway removido',
    terminalGatewayRemovedBody:
      'O gateway em que este terminal era executado foi removido, então não é possível reabri-lo. Feche esta aba.',
    terminalGatewayNoShellBody:
      'Este gateway não tem acesso ao shell, então não é possível reabrir este terminal. Feche esta aba.',
    terminalDisconnectedBody:
      'Terminal desconectado. Pressione Enter para iniciar um novo shell; o histórico de saída será preservado.',
    terminalDropUploadFailed: name => `Não foi possível enviar ${name}`,
    terminalDropTooLargeTitle: 'Muito grande para enviar',
    terminalDropTooLarge: (names, limit) => `Cada arquivo pode ter até ${limit}. Não enviados: ${names}`,
    terminalDropControlTitle: 'Não inserido no terminal',
    terminalDropControl: paths =>
      `Estes nomes contêm caracteres de controle que o shell interpretaria como teclas: ${paths}`,
    terminalDropUnquotable: (shell, paths) => `Não há uma forma segura de digitar estes nomes em ${shell}: ${paths}`,
    addToChat: 'Adicionar à conversa'
  },

  preview: {
    tab: 'Prévia',
    pin: 'Fixar na área de trabalho',
    unpin: 'Desafixar da área de trabalho',
    closePane: 'Fechar painel de prévia',
    loading: 'Carregando prévia',
    unavailable: 'Prévia indisponível',
    missingTarget: 'Esse caminho não existe neste computador',
    missingTitle: 'O arquivo não existe mais',
    missingBody: label =>
      `${label} foi excluído, movido ou seu local temporário foi limpo. Esta aba não será restaurada na próxima inicialização.`,
    opening: 'Abrindo...',
    hide: 'Ocultar',
    openPreview: 'Abrir prévia',
    openInBrowser: 'Abrir no navegador',
    openInExternal: 'Abrir em aplicativo externo',
    popIn: 'Integrar à janela',
    popOut: 'Abrir em janela separada',
    linkHint: '⌘/Ctrl+clique para abrir no painel de prévia',
    sourceLineTitle: 'Clique para selecionar · Shift+clique para ampliar a seleção · arraste para a caixa de mensagem',
    source: 'CÓDIGO-FONTE',
    renderedPreview: 'PRÉVIA',
    diff: 'DIFERENÇAS',
    unknownSize: 'tamanho desconhecido',
    binaryTitle: 'Este parece ser um arquivo binário',
    binaryBody: label => `A prévia de ${label} pode mostrar texto ilegível.`,
    largeTitle: 'Este arquivo é grande',
    largeBody: (label, size) => `${label} tem ${size}. O Hermes mostrará apenas os primeiros 512 KB.`,
    previewAnyway: 'Visualizar mesmo assim',
    truncated: 'Mostrando os primeiros 512 KB.',
    noInlineTitle: 'Sem prévia integrada',
    noInlineBody: mimeType => `${mimeType || 'Este tipo de arquivo'} ainda pode ser anexado como contexto.`,
    edit: 'Editar',
    editing: 'Editando',
    unsavedChanges: 'Alterações não salvas',
    saveFailed: message => `Não foi possível salvar: ${message}`,
    saveScopeChanged: 'Volte à conexão e ao perfil originais para salvar este rascunho.',
    diskChangedTitle: 'Arquivo alterado no disco',
    diskChangedBody:
      'Este arquivo foi alterado desde que você o abriu. Substituir pela sua versão ou descartar suas edições e recarregar?',
    overwrite: 'Substituir',
    discardReload: 'Descartar e recarregar',
    console: {
      deselect: 'Desmarcar entrada',
      select: 'Selecionar entrada',
      copyFailed: 'Não foi possível copiar a saída do console',
      copyEntry: 'Copiar esta entrada',
      sendEntry: 'Enviar esta entrada à conversa',
      messages: count => `${count} mensagens do console`,
      resize: 'Redimensionar console de prévia',
      title: 'Console de prévia',
      selected: count => `${count} selecionadas`,
      sendToChat: 'Enviar à conversa',
      copySelected: 'Copiar seleção para a área de transferência',
      copyAll: 'Copiar tudo para a área de transferência',
      copy: 'Copiar',
      clear: 'Limpar',
      empty: 'Ainda não há mensagens no console.',
      promptHeader: 'Console de prévia:',
      sentTitle: 'Enviado à conversa',
      sentMessage: count =>
        `${count} registro${count === 1 ? ' de log adicionado' : 's de log adicionados'} à caixa de mensagem`
    },
    web: ptBrPreviewWeb
  },

  interfaceMode: {
    title: 'Modo da interface',
    hint: 'Altera o que é exibido, não o que o Hermes pode fazer.',
    sessionNote:
      'Definido pelo modo Simples. Uma alteração aqui vale para esta sessão; mude para Avançado para personalizar.',
    simple: {
      label: 'Simples',
      description:
        'Para conversar com o Hermes. Barra lateral e conversa; sem painéis de terminal, arquivos ou diferenças.'
    },
    advanced: {
      label: 'Avançado',
      description:
        'Para desenvolvedores. Terminal, arquivos, diferenças, barra de status e layouts, como você preferir.'
    }
  },

  zones: {
    showTabStrip: 'Mostrar abas',
    hideTabStrip: 'Ocultar abas',
    showStripTab: title => `Mostrar ${title}`,
    hideStripTab: title => `Ocultar ${title}`,
    zoneMenuLabel: title => `Opções da zona de ${title}`,
    lastTabKeptTitle: 'A última aba permanece',
    lastTabKeptBody:
      'Esta zona precisa de pelo menos uma aba visível. Mostre outra aba primeiro ou recolha toda a barra lateral.',
    toggleStripTab: title => `Alternar aba ${title}`,
    minimize: 'Minimizar',
    restore: 'Restaurar',
    closeRunningTitle: 'Fechar aba em execução?',
    closeRunningBody:
      'Esta conversa ainda está em andamento (ou aguardando sua resposta). Fechar a aba apenas a oculta — a sessão mantém o progresso e pode ser reaberta pela barra lateral.',
    closeRunningConfirm: 'Fechar aba',
    reload: 'Recarregar',
    closeOthers: 'Fechar as outras',
    closeToRight: 'Fechar à direita',
    closeAll: 'Fechar todas',
    newSessionTab: 'Nova aba de sessão',
    newTab: 'Nova aba',
    pluginDisabled: pluginId => `Plugin "${pluginId}" desativado`,
    pluginDisabledBody: 'Reative-o em Recursos → Plugins para trazer o painel de volta.',
    missingPane: paneId => `painel ausente: ${paneId}`,
    editTitle: 'Layouts',
    editHint: 'Escolha um layout ou arraste painéis entre as zonas.',
    reset: 'Redefinir',
    templates: 'Modelos',
    custom: 'Personalizado',
    newGridLayout: 'Novo layout em grade',
    saveCurrentAs: 'Salvar organização atual como modelo',
    nameLayoutPlaceholder: 'Dê um nome a este layout…',
    deletePreset: name => `Excluir ${name}`,
    zoneEditorTitle: 'Editor de zonas',
    editorHintPre: 'clique para dividir · ',
    editorHintPost:
      ' inverte a linha · arraste entre zonas para uni-las · arraste bordas compartilhadas para redimensionar',
    templateColumns: 'Colunas',
    templateRows: 'Linhas',
    templateGrid: 'Grade',
    templatePriority: 'Prioridade',
    zoneTag: index => `zona ${index}`,
    mergeZones: count => `Unir ${count} zonas`,
    customZoneName: count => `Personalizado com ${count} zonas`,
    layoutNamePlaceholder: fallback => `Nome do layout (${fallback})`,
    saveApply: 'Salvar e aplicar',
    notExpressible:
      'esta organização se entrelaça (em cata-vento) — ainda não é possível representá-la como divisões aninhadas',
    zoneCount: count => `${count} zonas`,
    tabCount: count => `${count} aba${count === 1 ? '' : 's'}`,
    tabActions: title => `Ações para ${title}`
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
      loadingSession: 'Carregando sessão',
      showEarlier: 'Mostrar mensagens anteriores',
      loadingResponse: 'O Hermes está carregando uma resposta',
      loadingLocalModel: model => `Carregando ${model} na memória`,
      processingPrompt: 'Processando prompt',
      resumeWhenBackgroundDone: count =>
        count === 1
          ? 'Será retomado quando a tarefa em segundo plano terminar'
          : `Será retomado quando ${count} tarefas em segundo plano terminarem`,
      thinking: 'Pensando',
      thought: 'Pensou',
      thoughtBriefly: 'Pensou brevemente',
      thoughtFor: duration => `Pensou por ${duration}`,
      turnDuration: duration => `Este turno levou ${duration}`,
      today: time => `Hoje, ${time}`,
      yesterday: time => `Ontem, ${time}`,
      copy: 'Copiar',
      refresh: 'Atualizar',
      moreActions: 'Mais ações',
      branchNewChat: 'Ramificar em nova conversa',
      react: 'Reagir',
      dismissError: 'Dispensar erro',
      responseStopped: 'Resposta interrompida',
      contentRefusedNotice: 'Esta mensagem foi recusada pelo filtro de segurança do modelo — tente outra solicitação.',
      ...ptBrErrorLayers,
      errorCodes: ptBrErrorCodes,
      errorAuthKinds: {
        api_key: {
          title: provider => `${provider} rejeitou sua chave de API`,
          body: () => 'A chave salva é inválida ou foi revogada. Atualize-a e envie sua mensagem novamente.'
        },
        oauth: {
          title: provider => `Entre em ${provider} novamente`
        }
      },
      errorDetails: 'Ver detalhes do erro',
      errorHideDetails: 'Ocultar detalhes do erro',
      errorDetailsCopy: 'Copiar detalhes',
      errorCopyRaw: 'Copiar erro',
      errorContinue: 'Continuar',
      errorOpenProviderSettings: provider => `Abrir configurações de ${provider}`,
      errorOpenSetupGuide: 'Abrir guia de configuração',
      errorSetupGuideLink: 'Guia de configuração',
      errorOpenModelSettings: 'Abrir configurações de modelos',
      errorStartNewChatLink: 'Iniciar uma nova conversa',
      errorChooseModelLink: 'Escolher outro modelo',
      errorRetryAtResetShort: time => `Tentar novamente às ${time}`,
      errorRateLimitResetBody: (provider, time) =>
        `Você atingiu o limite de requisições de ${provider}. Ele será renovado às ${time}.`,
      errorGenericProvider: 'O serviço de IA',
      errorToastTitle: 'O Hermes não conseguiu concluir a resposta',
      errorRetry: 'Tentar novamente',
      errorRetryScheduled: (time, wait) => `Nova tentativa às ${time} — em ${wait}`,
      errorRetryScheduledCancel: 'Cancelar',
      errorStartNewSession: 'Iniciar nova sessão',
      errorSwitchProvider: 'Trocar provedor',
      errorChooseModel: 'Escolher um modelo',
      errorSwitchToModel: (model, note) => (note ? `Trocar para ${model} (${note})` : `Trocar para ${model}`),
      errorSwitchModelFailed: 'Não foi possível trocar o modelo',
      errorUseCredits: 'Ativar Continuar com créditos',
      errorUseCreditsFailed: 'Não foi possível ativar Continuar com créditos',
      errorAddCredits: 'Adicionar créditos',
      errorOpenBillingSettings: 'Abrir configurações de faturamento',
      errorCompressConversation: 'Compactar conversa',
      errorCompressFailed: 'Não foi possível compactar a conversa',
      errorOpenHermesFolder: 'Abrir pasta do Hermes',
      errorOpenHermesFolderFailed: 'Não foi possível abrir a pasta do Hermes',
      errorUpdateApiKey: 'Atualizar chave de API',
      errorSignInAgain: provider => `Entre em ${provider} novamente`,
      errorSignInFreeTier: 'Entrar com uma conta Nous',
      errorOauthExpired: provider =>
        `Seu acesso a ${provider} expirou ou foi revogado. Entre novamente para continuar conversando.`,
      errorOpenLogs: 'Abrir logs',
      errorOpenLogsFailed: 'Não foi possível abrir a pasta de logs',
      errorOpenDesktopLogs: 'Abrir logs do Desktop',
      errorSendDiagnostics: 'Enviar diagnóstico',
      filesChanged: count => (count === 1 ? '1 arquivo alterado' : `${count} arquivos alterados`),
      reviewChanges: 'Revisar',
      readAloudFailed: 'Falha na leitura em voz alta',
      preparingAudio: 'Preparando áudio...',
      stopReading: 'Parar leitura',
      readAloud: 'Ler em voz alta',
      copyFullResponse: 'Copiar resposta completa',
      readAloudFullResponseHint: 'Shift+clique: ler a resposta completa',
      editMessage: 'Editar mensagem',
      expandMessage: 'Expandir mensagem',
      scrollToBottom: 'Rolar até o fim',
      stop: 'Parar',
      restorePrevious: 'Restaurar ponto de controle anterior',
      restoreCheckpoint: 'Restaurar ponto de controle',
      restoreFromHere: 'Restaurar ponto de controle — executar novamente a partir deste prompt',
      restoreTitle: 'Restaurar este ponto de controle?',
      restoreBody:
        'Tudo após este prompt será removido da conversa e o prompt será executado novamente a partir daqui.',
      restoreConfirm: 'Restaurar e executar novamente',
      restoreNext: 'Restaurar próximo ponto de controle',
      goForward: 'Avançar',
      sendEdited: 'Enviar mensagem editada',
      attachingFile: 'Anexando…'
    },
    approval: {
      gatewayDisconnected:
        'O Hermes está offline no momento. O comando ainda aguarda sua resposta (até o prazo de aprovação expirar). Reconecte e envie novamente.',
      sendFailed: 'Não foi possível enviar sua resposta',
      reconnect: 'Reconectar',
      timedOutSystemLine:
        'O prazo de aprovação expirou — o comando não foi executado. Peça ao Hermes para tentar novamente ou aumente o limite em Configurações → Segurança → Tempo limite de aprovação.',
      openSafetySettings: 'Abrir configurações de segurança',
      run: 'Executar',
      command: 'Comando',
      commandDetails: 'Detalhes do comando',
      moreOptions: 'Mais opções de aprovação',
      allowSession: 'Permitir nesta sessão',
      alwaysAllowMenu: 'Sempre permitir…',
      jumpToApproval: 'Aprovação necessária',
      reject: 'Rejeitar',
      alwaysTitle: 'Sempre permitir este comando?',
      alwaysDescription: pattern =>
        `Isso adiciona o padrão “${pattern}” à sua lista permanente de permissões (~/.hermes/config.yaml). O Hermes não perguntará novamente sobre comandos como este — nem nesta sessão, nem nas próximas.`,
      alwaysAllow: 'Sempre permitir'
    },
    clarify: ptBrClarify,
    themePicker: ptBrThemePicker,
    catalogInstall: ptBrCatalogInstall,
    mcpSetup: {
      installTitle: 'Adicionar servidores MCP',
      enableTitle: 'Ativar servidores MCP',
      authorizeTitle: 'Autorizar servidores MCP',
      installAction: 'Instalar',
      enableAction: 'Ativar',
      authorizeAction: 'Autorizar',
      installed: server => `${server} instalado`,
      enabled: server => `${server} ativado`,
      authorized: server => `${server} autorizado`,
      failed: server => `Falha na configuração de ${server}`,
      toolCount: count => (count === 1 ? '1 ferramenta' : `${count} ferramentas`),
      launchCommand: target => `Executa: ${target}`,
      launchRemote: target => `Conecta a: ${target}`,
      launchEnv: keys => `Ambiente: ${keys}`,
      launchCwd: dir => `Diretório de trabalho: ${dir}`,
      envRequired: 'Preencha as credenciais obrigatórias primeiro',
      sendFailed: 'Não foi possível enviar a resposta de configuração do MCP',
      reloadFailed:
        'Servidor salvo, mas houve falha ao recarregar as ferramentas MCP — elas serão carregadas na próxima sessão',
      gatewayDisconnected: 'O Hermes está offline no momento. Reconecte e envie novamente.'
    },
    tool: {
      copyCode: 'Copiar código',
      renderingImage: 'Renderizando imagem',
      copyOutput: 'Copiar saída',
      copyCommand: 'Copiar comando',
      copyContent: 'Copiar conteúdo',
      copyUrl: 'Copiar URL',
      copyResults: 'Copiar resultados',
      copyQuery: 'Copiar consulta',
      copyFile: 'Copiar arquivo',
      copyPath: 'Copiar caminho',
      failedCalls: (count: number) => `${count} chamada${count === 1 ? '' : 's'} de ferramenta com falha`,
      skillActivity: {
        loading: 'Carregando habilidade',
        loaded: 'Habilidade carregada',
        loadFailed: 'Falha ao carregar habilidade',
        readingResource: 'Lendo recurso da habilidade',
        readResource: 'Recurso da habilidade lido',
        resourceFailed: 'Falha ao ler recurso da habilidade',
        listing: 'Listando habilidades',
        listed: 'Habilidades listadas',
        listFailed: 'Falha ao listar habilidades',
        unavailable: 'Resultado da habilidade indisponível'
      },
      outputAlt: 'Saída da ferramenta',
      rawResponse: 'Resposta bruta',
      copyActivity: 'Copiar atividade',
      recoveredOne: 'Recuperado após 1 etapa com falha',
      recoveredMany: count => `Recuperado após ${count} etapas com falha`,
      failedOne: '1 etapa com falha',
      failedMany: count => `${count} etapas com falha`,
      statusRunning: 'Em execução',
      statusError: 'Erro',
      statusRecovered: 'Recuperado',
      statusDone: 'Concluído',
      resultUnavailable: 'Resultado indisponível',
      resultInterrupted: 'Interrompido',
      memoryWriteNoted: 'Gravação na memória registrada',
      actions: {
        read: 'Leu',
        reading: 'Lendo',
        opened: 'Abriu',
        opening: 'Abrindo',
        failedToOpen: 'Falha ao abrir',
        searched: 'Pesquisou',
        searching: 'Pesquisando',
        ran: 'Executou',
        running: 'Executando',
        ranCode: 'Executou código',
        runningCode: 'Executando script'
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
          done: 'Clicou em elemento da página',
          pending: 'Clicando em elemento da página',
          pendingAction: 'Clicando'
        },
        browser_fill: {
          done: 'Preencheu campo do formulário',
          pending: 'Preenchendo campo do formulário',
          pendingAction: 'Preenchendo'
        },
        browser_navigate: { done: 'Abriu página', pending: 'Abrindo página', pendingAction: 'Abrindo' },
        browser_snapshot: {
          done: 'Capturou estado da página',
          pending: 'Capturando estado da página',
          pendingAction: 'Capturando'
        },
        browser_take_screenshot: {
          done: 'Capturou tela',
          pending: 'Capturando tela',
          pendingAction: 'Capturando'
        },
        browser_type: { done: 'Digitou na página', pending: 'Digitando na página', pendingAction: 'Digitando' },
        clarify: { done: 'Fez uma pergunta', pending: 'Fazendo uma pergunta', pendingAction: 'Perguntando' },
        cronjob: { done: 'Tarefa cron', pending: 'Agendando tarefa cron', pendingAction: 'Agendando' },
        edit_file: { done: 'Editou arquivo', pending: 'Editando arquivo', pendingAction: 'Editando' },
        execute_code: { done: 'Executou código', pending: 'Executando script', pendingAction: 'Executando script' },
        image_generate: { done: 'Gerou imagem', pending: 'Gerando imagem', pendingAction: 'Gerando' },
        list_files: { done: 'Listou arquivos', pending: 'Listando arquivos', pendingAction: 'Listando' },
        memory: { done: 'Salvou na memória', pending: 'Salvando na memória', pendingAction: 'Salvando' },
        patch: {
          done: 'Aplicou patch ao arquivo',
          pending: 'Aplicando patch ao arquivo',
          pendingAction: 'Aplicando patch'
        },
        read_file: { done: 'Leu arquivo', pending: 'Lendo arquivo', pendingAction: 'Lendo' },
        search_files: { done: 'Pesquisou arquivos', pending: 'Pesquisando arquivos', pendingAction: 'Pesquisando' },
        session_search_recall: {
          done: 'Pesquisou histórico de sessões',
          pending: 'Pesquisando histórico de sessões',
          pendingAction: 'Pesquisando'
        },
        terminal: { done: 'Executou comando', pending: 'Executando comando', pendingAction: 'Executando' },
        todo: { done: 'Atualizou tarefas', pending: 'Atualizando tarefas', pendingAction: 'Atualizando' },
        vision_analyze: { done: 'Analisou imagem', pending: 'Analisando imagem', pendingAction: 'Analisando' },
        web_extract: { done: 'Leu página web', pending: 'Lendo página web', pendingAction: 'Lendo' },
        web_search: { done: 'Pesquisou na web', pending: 'Pesquisando na web', pendingAction: 'Pesquisando' },
        write_file: { done: 'Editou arquivo', pending: 'Editando arquivo', pendingAction: 'Editando' }
      }
    }
  },

  prompts: {
    gatewayDisconnected: 'O Hermes está offline no momento. Reconecte e envie novamente.',
    reconnect: 'Reconectar',
    sudoSendFailed: 'Não foi possível enviar a senha do sudo',
    secretSendFailed: 'Não foi possível enviar o segredo',
    sudoTitle: 'Senha de administrador',
    sudoDesc:
      'Revise o comando antes de digitar sua senha do sudo. Sua senha é enviada ao agente que executa o comando e fica em cache nesta sessão.',
    sudoCommandUnavailable: 'Este agente não informou o comando. Cancele se você não puder verificá-lo na conversa.',
    sudoInstallDesc:
      'O Hermes precisa da sua senha do sudo para instalar os pacotes do Bot Screen (TigerVNC + Xfce) no host do gateway. Ela é enviada apenas a esse host.',
    sudoPlaceholder: 'senha do sudo',
    secretTitle: 'Segredo necessário',
    secretDesc: 'O Hermes precisa de uma credencial para continuar.',
    secretPlaceholder: 'valor do segredo',
    vaultUnlockSendFailed: 'Não foi possível enviar a senha mestra',
    vaultUnlockTitle: name => `Desbloquear ${name}`,
    vaultUnlockDesc: name =>
      `O agente quer entrar em um site com um login salvo em ${name}. Digite sua senha mestra para desbloqueá-lo nesta sessão — ela vai diretamente para ${name} nesta máquina e nunca é armazenada nem mostrada ao agente.`,
    vaultUnlockPlaceholder: 'Senha mestra',
    vaultUnlockKeepLocked: 'Manter bloqueado',
    vaultUnlockConfirm: 'Desbloquear',
    vaultSaveSendFailed: 'Não foi possível salvar o login',
    vaultSaveTitle: site => `Salvar seu login de ${site}?`,
    vaultSaveDesc: origin =>
      `O Hermes chegou a uma página de login em ${origin} e não tem um login para ela. Informe-o uma vez aqui; ele será criptografado nesta máquina e preenchido na página sem que o modelo veja a senha.`,
    vaultSaveIdentifierLabel: 'E-mail ou nome de usuário',
    vaultSaveIdentifierPlaceholder: 'voce@exemplo.com',
    vaultSavePasswordPlaceholder: 'Senha',
    vaultSaveFootnote: 'Gerencie logins salvos em Configurações → Senhas e logins.',
    vaultSaveDecline: 'Não salvar',
    vaultSaveConfirm: 'Salvar e entrar',
    vaultCodeSendFailed: 'Não foi possível enviar o código',
    vaultCodeTitle: site => `Código de verificação de ${site}`,
    vaultCodeDesc: site =>
      `${site} está pedindo um código de uso único (por SMS, e-mail ou aplicativo autenticador). Digite-o aqui e o Hermes o preencherá na página; o modelo nunca o verá.`,
    vaultCodeLabel: 'Código',
    vaultCodeFootnote:
      'Dica: salve a chave do autenticador com este login em Configurações → Senhas e logins para o Hermes preencher os códigos por você.',
    vaultCodeSkip: 'Pular',
    vaultCodeConfirm: 'Inserir código'
  },

  desktop: {
    audioReadFailed: 'Não foi possível ler o áudio gravado',
    sessionUnavailable: 'Sessão indisponível',
    createSessionFailed: 'Não foi possível criar uma nova sessão',
    promptFailed: 'Falha no prompt',
    providerCredentialRequired: 'Adicione uma credencial de provedor antes de enviar sua primeira mensagem.',
    emptySlashCommand: 'comando de barra vazio',
    slashCommandIgnoredTitle: 'Comando não enviado',
    slashCommandIgnoredBody:
      'Comandos de barra não podem ser combinados com anexos. Remova o anexo ou envie o comando separadamente.',
    desktopCommands: 'Comandos do Desktop',
    skillCommandsAvailable: count => `${count} comandos de habilidades disponíveis.`,
    warningLine: message => `aviso: ${message}`,
    yoloArmed: 'YOLO armado para esta conversa',
    yoloOff: 'YOLO desativado',
    yoloSystem: active => `YOLO ${active ? 'ativado' : 'desativado'} nesta sessão`,
    yoloTitle: 'YOLO',
    yoloToggleFailed: 'Não foi possível alternar o YOLO',
    profileStatus: current =>
      `Perfil: ${current}. Use /profile <name> ou o seletor "Nova sessão" para iniciar uma conversa em outro perfil.`,
    unknownProfile: 'Perfil desconhecido',
    noProfileNamed: (target, available) => `Nenhum perfil chamado "${target}". Disponíveis: ${available}`,
    newChatsProfile: name => `Novas conversas usarão o perfil ${name}.`,
    setProfileFailed: 'Falha ao definir perfil',
    sttDisabled: 'A transcrição de voz está desativada nas configurações.',
    stopFailed: 'Falha ao parar',
    regenerateFailed: 'Falha ao gerar novamente',
    editFailed: 'Falha ao editar',
    editTurnUnavailable: 'Este turno não está mais no histórico do servidor (pode ter sido removido pela compactação).',
    resumeFailed: 'Falha ao retomar',
    readOnlyTranscriptTitle: 'Aberta em modo somente leitura',
    readOnlyTranscriptBody:
      'Nenhum backend conectado assumiu esta conversa antiga ainda, então ela foi aberta como uma transcrição somente leitura. O histórico está intacto; o envio fica desativado até que um backend a assuma.',
    readOnlyTranscriptSendBlocked:
      'Esta conversa está aberta como uma transcrição somente leitura — o envio está desativado.',
    resumeStrandedTitle: 'Não foi possível carregar esta sessão',
    resumeStrandedBody:
      'A conexão com esta sessão falhou e as tentativas automáticas foram encerradas. Verifique se o gateway está em execução e tente novamente.',
    poolSlotTimeoutBody:
      'Há mais bots em execução ao mesmo tempo do que o limite deste computador permite. Aumente o limite em Configurações → Avançado ou espere um deles terminar e tente novamente.',
    poolSlotTimeoutOpenSettings: 'Abrir configurações avançadas',
    resumeRetry: 'Tentar novamente',
    nothingToBranch: 'Nada para ramificar',
    branchNeedsChat: 'Inicie ou retome uma conversa antes de ramificar.',
    sessionBusy: 'Sessão ocupada',
    branchStopCurrent: 'Interrompa o turno atual antes de ramificar esta conversa.',
    branchNoText: 'Esta mensagem não tem texto para ramificar.',
    branchTitle: n => `Rascunho: Ramificação #${n}`,
    branchFailed: 'Falha ao ramificar',
    deleteFailed: 'Falha ao excluir',
    archived: 'Arquivada',
    archiveFailed: 'Falha ao arquivar',
    restored: 'Restaurada',
    unarchiveFailed: 'Falha ao desarquivar',
    cwdChangeFailed: 'Falha ao alterar diretório de trabalho',
    cwdStagedTitle: 'Diretório de trabalho preparado',
    cwdStagedMessage: 'Reinicie o backend do Desktop para aplicar as alterações de cwd a esta sessão ativa.',
    modelSwitchConfirmBody: 'Esta troca de modelo precisa de confirmação.',
    modelSwitchConfirmLabel: 'Trocar mesmo assim',
    modelSwitchConfirmTitle: (model: string) => `Trocar para ${model}?`,
    modelSwitchConfirmTitleFallback: 'Trocar de modelo?',
    modelSwitchFailed: 'Falha ao trocar modelo',
    modelSwitchKeepLabel: 'Manter modelo atual',
    modelSwitchStaleNotice: 'A seleção mudou — a troca de modelo não foi aplicada.',
    hydrationSyncing: (profile: string) => `Sincronizando ${profile}\u2026`,
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
    zoomIn: 'Ampliar',
    zoomOut: 'Reduzir',
    resetZoom: 'Redefinir zoom',
    imagePreviewFailed: 'Falha na prévia da imagem',
    imageAttach: 'Anexo de imagem',
    imageWriteFailed: 'Falha ao gravar imagem no disco.',
    imageAttachFailed: 'Falha ao anexar imagem',
    attachFailed: 'Falha ao anexar',
    pastedContent: 'Conteúdo colado',
    pasteAttachFailed: 'Não foi possível anexar o texto colado',
    attachImages: 'Anexar imagens',
    clipboard: 'Área de transferência',
    noClipboardImage: 'Nenhuma imagem encontrada na área de transferência',
    clipboardPasteFailed: 'Falha ao colar da área de transferência',
    dropFiles: 'Solte arquivos',
    dropFilesMore: (count: number) => `e mais ${count}`,
    handoff: {
      pickPlatform: 'Escolher um destino',
      success: platform => `Transferida para ${platform}. Retome aqui a qualquer momento.`,
      systemNote: platform => `↻ Transferida para ${platform} — retome aqui a qualquer momento.`,
      failed: error => `Falha na transferência: ${error}`,
      timedOut:
        'O Hermes não conseguiu acessar sua conexão de mensagens. Inicie-a em Configurações → Mensagens e tente transferir novamente.',
      startMessaging: 'Iniciar mensagens'
    }
  },

  tips: {
    close: 'Não mostrar esta dica novamente',
    items: {
      'new-session': {
        title: 'Comece do zero',
        text: 'Uma nova conversa tem seu próprio contexto, terminal e diretório de trabalho.'
      },
      skills: {
        title: 'Ensine uma vez',
        text: 'Habilidades são pastas de instruções que o Hermes carrega quando o trabalho exige.'
      },
      messaging: {
        title: 'Hermes longe da sua mesa',
        text: 'Conecte Telegram, Discord, Slack e outros — o mesmo agente, a mesma memória.'
      },
      artifacts: {
        title: 'Tudo o que o Hermes criou',
        text: 'Imagens, arquivos e links de todas as sessões, indexados em um só lugar.'
      },
      cron: {
        title: 'Trabalho que se executa sozinho',
        text: 'Agende um prompt a cada hora, todas as noites ou com uma expressão cron.'
      },
      'command-palette': {
        title: 'Uma caixa para tudo',
        text: 'Acesse sessões, configurações, habilidades e comandos pela paleta.'
      },
      profiles: {
        title: 'Os perfis são separados',
        text: 'Cada um é seu próprio Hermes — com suas chaves, memória e sessões.'
      },
      'composer-mentions': {
        title: 'Anexe e comande',
        text: 'Digite @ para incluir um arquivo na conversa e / para executar um comando.'
      },
      'local-runtime-update': {
        title: 'Uma atualização do mecanismo local está disponível',
        text: 'Atualize o mecanismo que executa seus modelos locais. Solicitações locais em andamento podem ser interrompidas.',
        action: 'Atualizar agora'
      },
      'right-pane': {
        title: 'O painel de trabalho',
        text: 'Arquivos, terminal, revisão e o navegador do aplicativo compartilham o lado direito.'
      }
    }
  },

  errors: {
    genericFailure: 'Algo deu errado',
    boundaryTitle: 'Algo falhou na interface',
    boundaryDesc: 'A visualização encontrou um erro inesperado. Suas conversas e configurações estão seguras.',
    boundaryDetails: 'Detalhes',
    sendDiagnostics: 'Enviar diagnóstico',
    reloadWindow: 'Recarregar janela',
    openLogs: 'Abrir logs'
  },

  ui: {
    search: {
      clear: 'Limpar busca'
    },
    logs: {
      bottom: 'Fim do log',
      search: 'Buscar nos logs…',
      top: 'Início do log'
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
      toggle: open => `${open ? 'Mostrar' : 'Ocultar'} barra lateral`
    }
  }
} satisfies TranslationOverrides
