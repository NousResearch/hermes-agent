import type { Translations } from './types'

export const ptBr: Translations = {
  // O conteúdo editorial de introdução usa o fallback seguro em inglês.
  intro: { stock: {}, custom: () => [] },
  connectors: {
    title: 'Conecte seus aplicativos',
    connect: 'Conectar',
    skip: 'Agora não',
    cancel: 'Parar de esperar',
    retry: 'Tentar novamente',
    grant: 'Reconectar',
    connected: 'Conectado',
    checking: 'Verificando seus aplicativos…',
    notConnected: 'Não conectado',
    skipped: 'Ignorado',
    disabled: 'Indisponível',
    failed: 'Não foi possível conectar',
    needsAuth: 'Acesso expirado',
    opening: 'Abrindo a tela de login…',
    waiting: 'Aguardando seu navegador…',
    timeout: 'Ainda aguardando autorização.',
    refresh: 'Atualizar status',
    connectError: 'Não foi possível iniciar a autorização. Tente novamente.',
    connectErrorFor: app => `Não foi possível iniciar a autorização para ${app}.`,
    unavailable: 'Os conectores estão indisponíveis nesta sessão.',
    ownerMissing: 'Reabra esta conversa para gerenciar as conexões.',
    search: 'Encontrar um aplicativo',
    empty: 'Nenhum aplicativo correspondente',
    disclaimer: 'A conexão é opcional. Autorize apenas os aplicativos que você quer que o Hermes use.',
    execution: 'Ferramentas do conector',
    setup: server => `Configurar ${server}`,
    openInBrowser: 'Abrir no navegador',
    setupCancel: 'Cancelar',
    authorizedToolsUnavailable: 'Autorizado. Ferramentas indisponíveis.',
    required: 'Obrigatório'
  },
  connectorsPage: {
    title: 'Conectores',
    searchPlaceholder: count => `Pesquisar em ${count} aplicativos`,
    filterCategory: 'Categoria',
    categoryAll: 'Todas as categorias',
    uncategorised: 'Sem categoria',
    residencyLocal: 'Neste dispositivo',
    segment: {
      all: 'Todos',
      available: 'Disponíveis',
      connected: 'Conectados',
      off: 'Desativados'
    },
    group: {
      connected: 'Conectados',
      connectedNote: 'Conexões com problemas primeiro.',
      available: 'Disponíveis',
      off: 'Desativados',
      offNote: 'Os logins são mantidos.'
    },
    card: {
      kindManaged: 'Gerenciado',
      kindCatalog: 'MCP · Catálogo',
      kindCustom: 'MCP · Personalizado',
      kindPlugin: plugin => `MCP · Plugin ${plugin}`,
      inCatalog: 'No catálogo do Hermes',
      hostedTwin: 'Versão gerenciada disponível',
      alsoLocal: 'Também é executado neste dispositivo',
      open: name => `Abrir ${name}`,
      turnServerOn: name => `Ativar ${name}`,
      turnServerOff: name => `Desativar ${name}`,
      state: {
        accessExpired: 'Acesso expirado',
        available: 'Disponível',
        connected: 'Conectado',
        connecting: 'Conectando',
        connectionUnknown: 'Estado desconhecido',
        couldNotConnect: 'Não foi possível conectar',
        offByYourOrganisation: 'Desativado pela sua organização',
        offForYou: 'Desativado para você',
        serverConnecting: 'Conectando…',
        serverError: 'Erro',
        serverNeedsAuth: 'Requer autenticação',
        serverOff: 'Desativado',
        serverOn: 'Ativado',
        serverOnUnused: 'Ativado, sem uso'
      },
      fact: {
        tools: count => `${count} ferramenta${count === 1 ? '' : 's'}`,
        toolsOff: count => `${count} ferramenta${count === 1 ? '' : 's'} desativada${count === 1 ? '' : 's'}`,
        toolsOn: count => `${count} ferramenta${count === 1 ? '' : 's'} ativada${count === 1 ? '' : 's'}`,
        toolsSomeOn: (total, on) => `${total} ferramentas, ${on} ativadas`
      },
      verb: {
        authenticate: 'Autenticar',
        connect: 'Conectar',
        install: 'Instalar',
        openLogs: 'Abrir logs',
        reconnect: 'Reconectar',
        stopWaiting: 'Parar de esperar',
        tryAgain: 'Tentar novamente',
        turnBackOn: 'Reativar'
      },
      reason: {
        finishSignIn: 'Conclua o login no navegador.',
        reconnect: 'Reconecte para manter este aplicativo funcionando.',
        serverError: 'O servidor recusou a conexão.',
        serverNeedsAuth: 'Entre para permitir que este servidor responda.'
      }
    },
    page: {
      loading: 'Lendo o catálogo e os servidores neste computador',
      emptyTitle: 'Ainda não há aplicativos aqui. Adicione um servidor neste computador para começar.',
      noMatchTitle: 'Nenhum aplicativo correspondente',
      noMatchBody: 'Nada aqui corresponde à busca. Aponte o Hermes para seu próprio servidor MCP para adicioná-lo.',
      clearSearch: 'Limpar a pesquisa',
      hostedFailedTitle: 'Não foi possível acessar os aplicativos hospedados.',
      hostedFailedBody:
        'Os servidores neste computador não foram afetados e continuam em execução. Nada foi desativado.',
      retry: 'Tentar novamente',
      matchesElsewhere: count => `Mais ${count} correspondência${count === 1 ? '' : 's'} em outros grupos.`,
      showAllMatches: 'Mostrar todas as correspondências',
      segmentNoMatch: segment =>
        `Nenhuma correspondência em ${segment}; por isso, todas as correspondências são exibidas.`,
      freeTierNote: 'As conexões permanecem neste computador até você entrar.',
      signInLine: 'Entre na Nous para usar aplicativos gerenciados.',
      signIn: 'Entrar',
      managedUnavailable: 'Os aplicativos gerenciados ainda não estão disponíveis para esta conta.',
      writeFailed: 'Essa alteração não foi salva.',
      refreshFailed: 'A lista de ferramentas não foi atualizada.',
      disconnectNoAccount: 'O Hermes não tem nenhuma conta para desconectar aqui. Atualize a página e tente novamente.',
      disconnectRefused:
        'A Nous não conseguiu remover este login agora. Desative o aplicativo usando o botão ou tente novamente mais tarde.'
    },
    add: {
      action: 'Adicionar o seu',
      title: 'Conectar a um MCP personalizado',
      hint: 'uma nova entrada em mcp.json neste dispositivo',
      pasteLabel: 'Cole um comando ou trecho de código',
      pastePlaceholder: 'npx -y @modelcontextprotocol/server-filesystem /path/to/dir',
      pasteNoMatch: 'Nada aqui parece ser um servidor. Preencha os campos abaixo.',
      name: 'Nome',
      nameTaken: 'Esse nome já está em uso.',
      type: 'Tipo',
      typeStdio: 'STDIO',
      typeHttp: 'HTTP com streaming',
      command: 'Comando para iniciar',
      args: 'Argumentos',
      addArg: '+ Adicionar argumento',
      envVars: 'Variáveis de ambiente',
      addEnvVar: '+ Adicionar variável de ambiente',
      passthrough: 'Repasse de variável de ambiente',
      addPassthrough: '+ Adicionar variável',
      cwd: 'Diretório de trabalho',
      url: 'URL',
      headers: 'Cabeçalhos',
      addHeader: '+ Adicionar cabeçalho',
      auth: 'Autenticação',
      authNone: 'Nenhuma',
      authOauth: 'OAuth',
      authBearer: 'Token Bearer',
      keyPlaceholder: 'KEY',
      valuePlaceholder: 'valor',
      removeRow: 'Remover esta linha',
      editJson: 'Editar mcp.json',
      saveFailed: 'Esse servidor não foi salvo.'
    },
    dialog: {
      disconnect: 'Desconectar',
      disconnectTitle: name => `Desconectar ${name}?`,
      disconnectBody: 'O Hermes deixa de atuar como esta conta. Você pode se conectar novamente a qualquer momento.',
      menuRefreshTools: 'Atualizar ferramentas',
      moreActions: 'Mais ações',
      removeServerTitle: name => `Remover ${name}?`,
      removeServerBody: 'A entrada permanece no mcp.json deste computador. Nada mais é excluído.',
      appSwitch: name => `O Hermes pode usar ${name}`,
      waysTitle: name => `Onde ${name} é executado`,
      wayNotConnected: name => `Ainda não conectado. Entre em ${name} no seu navegador.`,
      wayHosted: 'Gerenciado',
      bothOn: name => `Ambos estão ativados, então o Hermes vê cada ferramenta de ${name} duas vezes.`,
      turnOffLocal: 'Desativar o servidor local',
      providedByPlugin: plugin => `Fornecido pelo plugin ${plugin}`,
      openPlugins: 'Abrir a aba Plugins',
      nousLine: 'Os aplicativos da Nous acompanham sua conta, não o perfil.',
      rulesReadOnly: 'As regras não podem ser alteradas agora.',
      rulesAppOff: name => `Ative ${name} para alterar as ferramentas.`,
      rulesSignIn: 'Entre para alterar o que o Hermes pode fazer aqui.',
      orgNote: count => `Sua organização desativou ${count} ferramenta${count === 1 ? '' : 's'}.`,
      orgLink: 'Abrir a administração de conectores',
      connectEnded: 'O login não foi concluído.',
      connectOpenAgain: 'Abrir o link novamente',
      tokensPerCall: 'tokens por chamada',
      usesPerMonth: 'usos em 30 dias',
      advanced: 'Avançado',
      advancedHint: 'a entrada do mcp.json e os logs'
    },
    tools: {
      title: 'Ferramentas',
      notInstalledBody: 'Instale neste dispositivo para ver as ferramentas incluídas.',
      summaryTitle: name => `O que o Hermes pode fazer com ${name}`,
      summaryPreviewTitle: name => `O que o Hermes poderá fazer com ${name} após a conexão`,
      summaryCount: count => `${count} ferramenta${count === 1 ? '' : 's'}`,
      summaryAllTools: 'Todas as ferramentas',
      summaryOther: 'Outras',
      allToolsSwitch: 'Ativar ou desativar todas as ferramentas',
      summaryAllOn: 'todas ativadas',
      summarySomeOn: (on, total) => `${on} de ${total} ativadas`,
      summaryOff: 'desativadas',
      showAllTools: count => `Mostrar todas as ${count} ferramenta${count === 1 ? '' : 's'}`,
      showSummary: 'Mostrar resumo',
      facetSwitch: facet => `Ativar ou desativar as ferramentas de ${facet}`,
      moreHints: count => `+${count}`,
      staleSignIn: 'Entre para ver a lista de ferramentas mais recente.',
      searchCountPlaceholder: count => `Pesquisar em ${count} ferramentas`,
      toolList: name => `Ferramentas de ${name}`,
      categorySelect: count => `${count} categorias`,
      showDeprecated: count => `Mostrar ${count} obsoleta${count === 1 ? '' : 's'}`,
      hideDeprecated: count => `Ocultar ${count} obsoleta${count === 1 ? '' : 's'}`,
      quickReadOnly: 'Somente leitura',
      quickNoDestructive: 'Desativar as destrutivas',
      quickEverythingOn: 'Ativar tudo',
      lockedHint: 'desativada pela sua organização',
      turnToolOn: tool => `Ativar ${tool}`,
      turnToolOff: tool => `Desativar ${tool}`,
      showDetails: tool => `Mostrar o que ${tool} faz`,
      hideDetails: tool => `Ocultar o que ${tool} faz`,
      noMatch: 'Nenhuma ferramenta corresponde a estes filtros.',
      loading: 'Lendo a lista de ferramentas',
      unavailableLine: 'Lista de ferramentas indisponível.',
      needsAuthTitle: name => `Entre em ${name} para ver as ferramentas.`,
      needsAuthBody: 'O login permanece neste computador. Nada sai dele.',
      retry: 'Tentar novamente',
      goneTitle: name => `${name} saiu do catálogo.`,
      goneBody:
        'O Hermes não pode mais fazer chamadas para esse conector. A linha permanece até você removê-la, para que nada desapareça.',
      remove: 'Remover',
      offTitle: name => `${name} está desativado.`,
      offBody: 'Ative usando o botão acima para ver as ferramentas incluídas.',
      signedOutTitle: 'Entre na Nous para ver a lista de ferramentas.',
      signedOutBody: 'Seus servidores neste computador não serão afetados.',
      conflictTitle: 'Alguém alterou esta regra enquanto você a editava.',
      conflictBody: (theyOff, theyOn) => {
        const they = [
          theyOff > 0 ? `desativaram ${theyOff} ferramenta${theyOff === 1 ? '' : 's'} que você ativou` : '',
          theyOn > 0
            ? `deixaram ativada${theyOn === 1 ? '' : 's'} ${theyOn} ferramenta${theyOn === 1 ? '' : 's'} que você desativou`
            : ''
        ].filter(Boolean)

        return `${they.length > 0 ? `Eles ${they.join(' e ')}. ` : ''}Suas edições continuam na tela; nada foi salvo.`
      },
      conflictReload: 'Recarregar a versão deles',
      conflictSave: 'Salvar sobre a versão deles',
      saveFailed: 'Essas regras de ferramentas não foram salvas.',
      footerDirty: (off, backOn) =>
        `${off} ferramenta${off === 1 ? '' : 's'} desativada${off === 1 ? '' : 's'}, ${backOn === 0 ? 'nenhuma' : backOn} reativada${backOn === 1 ? '' : 's'}`,
      discard: 'Descartar',
      save: 'Salvar alterações',
      saving: 'Salvando...'
    },
    vocabulary: {
      facetRead: {
        label: 'Leitura',
        long: 'Lê dados deste aplicativo. Não altera nada.'
      },
      facetWrite: {
        label: 'Gravação',
        long: 'Cria ou altera algo neste aplicativo.'
      },
      facetDestructive: {
        label: 'Destrutiva',
        long: 'Pode remover algo deste aplicativo de forma permanente.'
      },
      facetUnclassified: {
        label: 'Efeito desconhecido',
        long: 'O aplicativo não informou o que esta ferramenta faz.'
      },
      hintReadOnly: {
        label: 'Somente leitura',
        long: 'A ferramenta declara que apenas lê dados.'
      },
      hintCreate: {
        label: 'Cria',
        long: 'Cria algo novo.'
      },
      hintUpdate: {
        label: 'Atualiza',
        long: 'Altera algo que já existe.'
      },
      hintDelete: {
        label: 'Exclui',
        long: 'Remove algo.'
      },
      hintDestructive: {
        label: 'Destrutiva',
        long: 'A alteração feita não pode ser desfeita aqui.'
      },
      hintIdempotent: {
        label: 'Repetível',
        long: 'Executar duas vezes produz o mesmo resultado que executar uma vez.'
      },
      hintOpenWorld: {
        label: 'Externa',
        long: 'Acessa algo fora deste aplicativo.'
      }
    }
  },
  sessionImport: {
    title: 'Continuar de outro aplicativo',
    subtitle: 'Traga uma conversa para o Hermes e continue de onde parou.',
    action: 'Importar sessão',
    readingFrom: 'Lendo de',
    connectedComputer: 'o computador conectado',
    destination: 'Importar para',
    all: 'Todas',
    search: 'Pesquisar nas sessões carregadas',
    scanning: 'Procurando conversas',
    scanError: 'Não foi possível encontrar sessões',
    scanHelp:
      'Verifique a conexão com o backend e tente novamente. Backends mais antigos talvez precisem de uma atualização.',
    empty: 'Nenhuma conversa encontrada',
    emptyHelp: 'As sessões do Claude Code e do Codex neste backend aparecerão aqui.',
    noMatches: 'Nenhuma conversa correspondente',
    searchHelp: 'Tente outro título ou pasta, ou carregue mais sessões.',
    skipped: 'Alguns logs estavam vazios, ilegíveis ou eram grandes demais para a prévia.',
    more: 'Carregar mais sessões',
    messages: 'mensagens',
    choose: 'Uma conversa que vale a pena continuar',
    chooseHelp: 'Escolha uma sessão para ler o histórico antes de trazê-la para o Hermes.',
    previewLoading: 'Abrindo prévia',
    previewError: 'Prévia indisponível',
    previewHelp: 'A origem pode ter sido movida ou alterada. Atualize a lista e tente novamente.',
    previewLimit: 'A prévia foi reduzida para facilitar a leitura. A conversa completa será importada.',
    you: 'Você',
    snapshot: 'Esta conversa já está no Hermes. Abra a cópia existente para continuar.',
    copyNotice:
      'Copia o texto da conversa. Os arquivos de origem permanecem inalterados. A saída de ferramentas e o raciocínio não são transferidos.',
    importing: 'Importando…',
    open: 'Abrir no Hermes',
    continue: 'Continuar no Hermes',
    importError: 'Não foi possível importar esta conversa.'
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
    bots: 'Bots',
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
    download: 'Baixar',
    downloadSaved: 'Download salvo',
    downloadFailed: 'Falha no download',
    rename: 'Renomear…',
    delete: 'Excluir',
    renameTitle: 'Renomear',
    renameLabel: 'Novo nome',
    deleteTitle: name => `Excluir ${name}?`,
    deleteBody: 'O item será movido para a Lixeira — você pode restaurá-lo de lá.',
    pathCopied: 'Caminho copiado',
    revealMissing: 'Essa pasta não está neste computador',
    revealUnavailable:
      'Esse caminho não está neste computador — ele fica na máquina do backend. Use “Mostrar na árvore de arquivos”.'
  },
  boot: {
    ready: 'O Hermes Desktop está pronto',
    desktopBootFailedWithMessage: message => `Falha na inicialização do desktop: ${message}`,
    steps: {
      connectingGateway: 'Conectando ao gateway do desktop',
      loadingSettings: 'Carregando as configurações do Hermes',
      loadingSessions: 'Carregando as sessões recentes',
      retryingRemoteBackend: 'Reconectando ao backend remoto do Hermes…',
      startingDesktopConnection: 'Iniciando a conexão do desktop',
      startingHermesDesktop: 'Iniciando o Hermes Desktop…'
    },
    errors: {
      backgroundExited: 'O processo em segundo plano do Hermes foi encerrado.',
      backgroundExitedDuringStartup: 'O processo em segundo plano do Hermes foi encerrado durante a inicialização.',
      backendStopped: 'Backend parado',
      restartHermes: 'Reiniciar o Hermes',
      openLogs: 'Abrir logs',
      desktopBootFailed: 'Falha na inicialização do desktop',
      gatewayConnectionLost: 'A conexão com o gateway foi perdida',
      gatewayConnectionLostDetail:
        'Ainda tentando reconectar em segundo plano. Você pode continuar lendo e rascunhando — abra as Configurações do Gateway se isso persistir.',
      reconnectNow: 'Reconectar agora',
      connectionSettings: 'Configurações de conexão',
      gatewaySignInRequired: 'É necessário entrar no gateway',
      gatewaySignInRequiredDetail:
        'Entre novamente para restabelecer a conexão. Suas conversas e configurações estão seguras.',
      signInAgain: 'Entrar novamente',
      ipcBridgeUnavailable: 'A ponte IPC do desktop está indisponível.'
    },
    causes: {
      exitedEarly: 'O serviço em segundo plano do Hermes parou logo após iniciar.',
      timedOut: 'O serviço em segundo plano do Hermes não respondeu a tempo.',
      permission: 'O Hermes não conseguiu gravar na pasta de dados por causa de um problema de permissão.',
      diskFull: 'O disco está cheio, por isso o Hermes não pôde iniciar.',
      portInUse: 'Outro programa está usando a porta de rede necessária para o Hermes.',
      installMissing: 'Parte da instalação do Hermes está ausente. Selecione Reparar instalação para restaurá-la.'
    },
    failure: {
      title: 'O Hermes não conseguiu iniciar',
      description:
        'O gateway em segundo plano não subiu. Tente uma das opções de recuperação abaixo. Nada aqui exclui suas conversas ou configurações.',
      details: 'Detalhes',
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
    updateReadyMessageUnknown: 'Uma nova atualização está disponível.',
    seeWhatsNew: 'Ver novidades',
    mcp: {
      needsAuthTitle: 'O servidor MCP precisa de nova autenticação',
      needsAuthMessage: name => `O servidor MCP ${name} precisa de nova autenticação.`,
      errorTitle: 'Servidor MCP inacessível',
      errorMessage: name => `O servidor MCP ${name} falhou na verificação de saúde.`,
      signIn: 'Entrar',
      view: 'Ver',
      disable: 'Desativar',
      disabledMessage: name => `${name} MCP desativado. Reative-o a qualquer momento em Recursos → MCP.`,
      disableFailed: name => `Não foi possível desativar o MCP ${name}.`
    },
    errors: {
      elevenLabsNeedsKey: 'O STT da ElevenLabs precisa de ELEVENLABS_API_KEY.',
      elevenLabsRejectedKey: 'A ElevenLabs rejeitou a chave de API (401).',
      diskFull: 'Disco cheio — libere espaço e tente novamente.',
      storageFailure:
        'O Hermes não conseguiu salvar na pasta de dados. Abra Manutenção para verificar e corrigir o problema.',
      gatewayAuthFailed: 'Falha na autenticação do gateway — verifique sua API_SERVER_KEY.',
      methodNotAllowed:
        'O backend do desktop rejeitou essa requisição (405 Method Not Allowed). Tente reiniciar o Hermes Desktop.',
      microphonePermission: 'A permissão do microfone foi negada.',
      openaiRejectedApiKey:
        'A OpenAI não aceitou sua chave de API. Atualize-a em Configurações → Chaves e tente novamente.',
      openaiTtsNeedsKey: 'O TTS da OpenAI precisa de VOICE_TOOLS_OPENAI_KEY ou OPENAI_API_KEY.',
      codeSkewRestartRequired:
        'Este backend está executando código antigo após uma atualização. Reinicie-o para carregar o código novo.',
      rpcOutOfSync: 'O aplicativo e o backend estão em versões diferentes. Atualize ambos.',
      restartHermesFailed: 'Não foi possível reiniciar o Hermes'
    },
    actions: {
      restartHermes: 'Reiniciar o Hermes',
      openKeys: 'Abrir chaves',
      openGateways: 'Abrir gateways',
      openMaintenance: 'Abrir Manutenção'
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
      unavailable: 'Voz indisponível',
      liveEnded: 'Sessão de voz ao vivo encerrada',
      liveEndedConnectionLost: 'A conexão da sessão de voz ao vivo foi perdida.',
      liveEndedClosed: 'A sessão de voz ao vivo foi encerrada pelo serviço.',
      liveError: 'Voz ao vivo',
      liveDelegationFailed: 'Não foi possível encaminhar a solicitação ao Hermes',
      liveUnavailable: reason =>
        `O chat por voz GPT-Live não está disponível: ${reason}. Usando conversão de fala em texto.`
    },
    native: {
      approvalTitle: 'Aprovação necessária',
      approvalTitleNamed: session => `Aprovação necessária — ${session}`,
      approveAction: 'Aprovar',
      rejectAction: 'Rejeitar',
      inputTitle: 'Entrada necessária',
      inputTitleNamed: session => `Entrada necessária — ${session}`,
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
  titlebar: {
    hideSidebar: 'Ocultar barra lateral',
    showSidebar: 'Mostrar barra lateral',
    search: 'Pesquisar',
    searchTitle: 'Pesquisar sessões, visualizações e ações',
    swapSidebarSides: 'Inverter os lados das barras laterais',
    hideRightSidebar: 'Ocultar barra lateral direita',
    showRightSidebar: 'Mostrar barra lateral direita',
    unreadSessions: count => `${count} ${count === 1 ? 'sessão não lida' : 'sessões não lidas'}`,
    muteHaptics: 'Silenciar resposta tátil',
    unmuteHaptics: 'Reativar resposta tátil',
    openSettings: 'Abrir configurações',
    openStarmap: 'Abrir grafo de memória',
    enterHud: 'Modo HUD',
    exitHud: 'Sair do modo HUD',
    resetHudLayout: 'Redefinir tamanho e posição do HUD',
    layoutEditor: 'Editor de layout',
    layoutEditorTitle: mod => `Editor de layout — ${mod}-clique redefine o layout`
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
      'nav.capabilities': 'Abrir recursos',
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
      'session.archive': 'Arquivar sessão atual',
      'workspace.newWorktree': 'Novo worktree',
      'workspace.openFolder': 'Abrir pasta como projeto',
      'composer.focus': 'Focar o compositor',
      'composer.modelPicker': 'Abrir seletor de modelos',
      'composer.voice': 'Iniciar / parar conversa por voz',
      'view.toggleSidebar': 'Alternar barra lateral de sessões',
      'view.cycleSidebarGrouping': 'Alternar agrupamento de sessões',
      'view.toggleRightSidebar': 'Alternar navegador de arquivos',
      'view.toggleReview': 'Alternar painel de revisão',
      'view.toggleStatusbar': 'Alternar barra de status',
      'view.toggleTabStrip': 'Alternar abas',
      'view.toggleProfileRail': 'Alternar barra de perfis',
      'view.toggleSimpleMode': 'Alternar modo Simples',
      'view.showFiles': 'Mostrar navegador de arquivos',
      'view.showBrowser': 'Abrir navegador',
      'view.toggleHud': 'Alternar modo HUD',
      'hud.snapToPointer': 'Mover o HUD para o ponteiro (global, enquanto o HUD estiver aberto)',
      'view.showTerminal': 'Alternar terminal',
      'view.newTerminal': 'Novo terminal',
      'view.nextTerminal': 'Próximo terminal',
      'view.prevTerminal': 'Terminal anterior',
      'view.closeTerminal': 'Fechar terminal',
      'view.selectionToComposer': 'Enviar seleção para o compositor',
      'view.terminalCopy': 'Copiar seleção do terminal',
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
    subpages: {
      appearanceTheme: 'Tema',
      appearanceTypography: 'Tipografia',
      appearanceWindowLayout: 'Janela e layout',
      appearanceChatDisplay: 'Exibição do chat',
      appearancePet: 'Mascote',
      appearanceGeneral: 'Geral',
      modelMain: 'Modelo principal',
      modelAuxiliary: 'Modelos auxiliares',
      modelMoa: 'Mistura de Agentes',
      modelFallbacks: 'Modelos alternativos',
      chatBehavior: 'Comportamento',
      chatAttachments: 'Anexos',
      workspaceProjects: 'Projetos e descoberta',
      workspaceShell: 'Ambiente do shell',
      workspaceFiles: 'Arquivos e execução',
      safetyApprovals: 'Aprovações',
      safetyPrivacy: 'Privacidade e rede',
      safetyCheckpoints: 'Pontos de restauração',
      browserProfile: 'Perfil do navegador',
      browserNetwork: 'URLs locais e privadas',
      memoryPersistent: 'Memória persistente',
      memoryContext: 'Contexto e compactação',
      voiceConversation: 'Conversa por voz',
      voiceTranscription: 'Fala em texto',
      voiceSpeech: 'Texto em fala',
      advancedRuntime: 'Limites do agente',
      advancedTools: 'Acesso a ferramentas',
      advancedTerminal: 'Backend do terminal',
      advancedOutput: 'Limites de saída',
      advancedDelegation: 'Subagentes',
      advancedDesktop: 'Desktop e inicialização',
      gatewayConnection: 'Esta janela',
      gatewayDevices: 'Conexões salvas',
      gatewayManagedUpdates: 'Atualizações remotas',
      gatewayManagedUpdatesUnavailable:
        'As atualizações remotas exigem uma versão para desktop compatível com atualizações gerenciadas via SSH.',
      gatewayManagedUpdatesEmpty:
        'Adicione uma conexão SSH em Conexões salvas para gerenciar as atualizações dela aqui.',
      keyboardShortcuts: 'Atalhos de teclado',
      hudGesture: 'Gesto do HUD',
      screenCapture: 'Captura de tela',
      notificationAlerts: 'Alertas do desktop',
      notificationSounds: 'Sons',
      archivedSessions: 'Arquivamento e retenção',
      defaultDirectory: 'Pasta padrão de projetos',
      vaultCredentials: 'Credenciais salvas',
      vaultSources: 'Gerenciadores de senhas',
      appUpdates: 'Versão e atualizações',
      uninstall: 'Desinstalar',
      billingOverview: 'Visão geral',
      billingPlans: 'Planos'
    },
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
      providerLocalModels: 'Modelos locais',
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
      vault: 'Senhas e logins'
    },
    plugins: {
      title: 'Plugins do Desktop',
      blurb:
        'Embutidos ou colocados na pasta desktop-plugins. Desative para remover o plugin imediatamente da memória.',
      count: n => `${n} plugin${n === 1 ? '' : 's'} instalado${n === 1 ? '' : 's'}`,
      openFolder: 'Abrir pasta de plugins',
      rescan: 'Reanalisar',
      reveal: 'Mostrar no gerenciador de arquivos',
      enable: 'Ativar',
      disable: 'Desativar',
      failed: 'Falhou',
      empty: 'Nenhum plugin de desktop instalado ainda.',
      kinds: {
        bundled: 'embutido',
        disk: 'em disco',
        runtime: 'runtime'
      },
      agentHalfMissing: 'componente do agente ausente aqui',
      agentHalfMissingTip:
        'Este é o componente para desktop de um plugin integrado, mas o componente do agente não está instalado no backend/perfil conectado no momento. Instale-o em Recursos → Plugins.',
      installModal: {
        installFromGit: 'Instalar pelo Git',
        reviewRepository: 'Analisar repositório',
        repoPlaceholder: 'https://github.com/owner/repo',
        title: 'Instalar plugin',
        description: 'Revise o conteúdo deste repositório antes de instalar qualquer coisa.',
        repoLabel: 'Repositório',
        includesHeading: 'Este pacote inclui',
        agentLabel: 'Plugin de agente',
        desktopLabel: 'Interface do Desktop',
        profileLabel: 'Instalar para o perfil',
        agentTargetLocal: (profile, dir) => `Instala no backend ${profile} (${dir})`,
        agentTargetRemote: profile => `Instala no backend remoto conectado “${profile}”`,
        catalogPinned: (name, sha) =>
          `Entrada "${name}" do catálogo do Hermes — o componente do agente é instalado no commit fixado e analisado${sha ? ` ${sha}` : ''}, não na ponta da branch.`,
        reviewedHeading: 'Entrada analisada do catálogo',
        reviewedIntro:
          'Esta entrada foi analisada por uma pessoa no commit fixado. Você ainda pode inspecionar o código exato abaixo.',
        toolsConnected: n => (n === 1 ? '1 ferramenta conectada' : `${n} ferramentas conectadas`),
        skillsReady: names => (names.length === 1 ? `skill ${names[0]} pronta` : `${names.length} skills prontas`),
        nextChat: 'mais ferramentas disponíveis no seu próximo chat',
        serverNotConnected: (server, reason) =>
          `O servidor MCP ${server} não está conectado${reason ? `: ${reason}` : '.'}`,
        missingEnvAction: 'Configurar',
        alreadyInstalled: name => `${name} já está instalado.`,
        desktopTarget: 'Instala na pasta local desktop-plugins deste aplicativo',
        desktopTargetFromPackage: 'Carregado neste aplicativo pelo pacote acima — igual para todos os perfis',
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
        pinToCommit: 'Fixar no commit (opcional)',
        pinToCommitPlaceholder: 'SHA completo do commit com 40 caracteres',
        pinToCommitHint:
          'Todos que instalarem este SHA receberão o mesmo código; depois disso, o plugin recusará atualizações até que outro commit seja fixado. Deixe em branco para usar o commit mais recente.',
        pinToCommitInvalid: 'Deve ser um SHA completo de commit com 40 caracteres (branches e tags não são aceitas).',
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
        missingEnv: (name, vars) =>
          `${name} está instalado, mas precisa de uma chave para funcionar: ${vars}. Adicione-a agora; caso contrário, as ferramentas do plugin falharão.`
      }
    },
    vault: {
      title: 'Senhas e logins',
      blurb:
        'Diga "entre no GitHub" e o agente fará login para você. Na primeira vez que encontrar uma página de login, ele solicitará os dados ali mesmo; depois disso, tudo funcionará automaticamente. As senhas são criptografadas nesta máquina e preenchidas diretamente na página — o modelo nunca as vê.',
      count: n => `${n} salvos`,
      loadFailed: 'Não foi possível carregar os itens do cofre',
      empty: 'Nada salvo ainda',
      emptyDesc:
        'Você não precisa adicionar nada aqui. Peça ao agente para entrar em um site e ele solicitará os dados de login uma única vez, na hora. Use Adicionar se preferir inseri-los antecipadamente.',
      add: 'Adicionar',
      addTitle: 'Adicionar login, cartão ou endereço',
      addDescription: 'Armazenado de forma criptografada nesta máquina. O agente nunca vê a senha.',
      added: 'Salvo.',
      adding: 'Salvando…',
      addConfirm: 'Salvar',
      kindField: 'Tipo',
      kinds: {
        login: 'Login',
        payment: 'Cartão de pagamento',
        address: 'Endereço'
      },
      labelField: 'Rótulo',
      labelPlaceholder: 'Ex.: conta corporativa do GitHub',
      labelRequired: 'É necessário informar um rótulo.',
      originField: 'Origem do site',
      originPlaceholder: 'https://github.com',
      originPlaceholderCheckout: 'https://shop.example.com',
      originInvalid: 'Insira uma URL válida, como https://example.com.',
      identifierTypeField: 'Tipo de identificador',
      identifierTypes: {
        email: 'E-mail',
        phone: 'Telefone',
        username: 'Nome de usuário'
      },
      identifierField: 'Identificador',
      identifierShown: identifier => identifier,
      passwordField: 'Senha',
      loginFieldsRequired: 'O identificador e a senha são obrigatórios.',
      cardNumberField: 'Número do cartão',
      cardNameField: 'Nome no cartão',
      expMonthField: 'Mês de validade',
      expYearField: 'Ano de validade',
      cvcField: 'CVC',
      postalField: 'CEP',
      addressLine1Field: 'Linha 1 do endereço',
      addressLine2Field: 'Linha 2 do endereço',
      cityField: 'Cidade',
      stateField: 'Estado/região',
      countryField: 'País',
      optional: '(opcional)',
      createdOn: date => `Adicionado em ${date}`,
      deleteAction: 'Remover item salvo',
      otpField: 'Chave do autenticador',
      otpPlaceholder: 'Segredo Base32 ou link otpauth://',
      otpHint:
        'A "chave de configuração" exibida pelo site quando você ativa a autenticação de dois fatores. Com ela salva, o Hermes gera os códigos automaticamente.',
      twoFactorBadge: '2FA automático',
      deleteTitle: 'Excluir este item?',
      deleteDescription: label => `"${label}" será removido. Essa ação não pode ser desfeita.`,
      deleteConfirm: 'Excluir',
      sources: {
        title: 'Gerenciadores de senhas',
        blurb:
          'Os gerenciadores de senhas instalados são detectados automaticamente. O agente pede que você desbloqueie um deles na primeira vez que precisar de um login armazenado nele (uma vez por sessão); apenas um token de sessão permanece na memória, e o agente nunca vê sua senha mestra nem nenhum login.',
        toggleFailed: 'Não foi possível atualizar o gerenciador de senhas',
        notInstalled: name =>
          `Não detectado. Instale a ferramenta de linha de comando do ${name} e faça login nela; o Hermes a detectará automaticamente.`,
        disabledDesc: 'Detectado, mas desativado para o Hermes.',
        lockedDesc:
          'Detectado. O agente pedirá que você o desbloqueie quando precisar de um login, ou você pode desbloqueá-lo agora.',
        unlockedDesc:
          'Desbloqueado para esta sessão. Será bloqueado automaticamente após 30 minutos de inatividade ou quando o Hermes for fechado.',
        statusLocked: 'Bloqueado',
        statusNotDetected: 'Não detectado',
        statusOff: 'Desativado',
        statusUnlocked: 'Desbloqueado',
        unlock: 'Desbloquear',
        unlocking: 'Desbloqueando…',
        lock: 'Bloquear',
        unlocked: name => `${name} foi desbloqueado para esta sessão.`,
        unlockTitle: name => `Desbloquear ${name}`,
        unlockDescription:
          'Digite sua senha mestra. Ela é repassada ao gerenciador de senhas neste computador e descartada — nunca é armazenada, registrada em logs nem exibida ao agente.',
        masterPasswordPlaceholder: 'Senha mestra'
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
      light: {
        label: 'Claro',
        description: 'Superfícies claras'
      },
      dark: {
        label: 'Escuro',
        description: 'Espaço de trabalho com pouco brilho'
      },
      system: {
        label: 'Sistema',
        description: 'Seguir a aparência do sistema operacional'
      }
    },
    appearance: {
      title: 'Aparência',
      intro: 'Apenas no Desktop. O modo controla o brilho; o tema controla a paleta e a aparência da conversa.',
      colorMode: 'Modo de cor',
      colorModeDesc: 'Escolha um modo fixo ou deixe o Hermes seguir a configuração do sistema.',
      toolViewTitle: 'Exibição das chamadas de ferramenta',
      toolViewDesc:
        'O modo Produto oculta os dados brutos das ferramentas; o modo Técnico mostra a entrada e a saída completas.',
      hideCodeDiffsTitle: 'Ocultar diffs de código',
      hideCodeDiffsDesc:
        'Mostrar edições de arquivos como linhas de ferramenta integradas, com a contagem de linhas adicionadas e removidas, sem exibir o código.',
      hideThreadTimelineTitle: 'Ocultar barras da linha do tempo das conversas',
      hideThreadTimelineDesc: 'Ocultar as barras de navegação na borda direita de cada conversa.',
      reasoningCollapsedTitle: 'Recolher raciocínio por padrão',
      reasoningCollapsedDesc: 'Mantenha o raciocínio transmitido disponível sem expandi-lo até que você o abra.',
      uiScaleTitle: 'Escala da interface',
      uiScaleDesc: percent =>
        `Redimensiona textos e controles em todo o app. Cmd/Ctrl com +, - e 0 também funciona. Atual: ${percent}%.`,
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
      appActionsTitle: 'Ações do aplicativo',
      appActionsDesc:
        'Onde Configurações, Layout e HUD aparecem na barra de título. À direita, sobra espaço para as abas à esquerda.',
      appActionsLeft: 'Esquerda',
      appActionsRight: 'Direita',
      terminalFontTitle: 'Fonte do terminal',
      terminalFontDesc:
        'Escolha uma fonte instalada para os terminais do Desktop. Nerd Fonts exibem corretamente o Powerlevel10k e os ícones do shell; deixe em branco para usar a JetBrains Mono incluída.',
      terminalFontPlaceholder: 'MesloLGS NF ou uma pilha de fontes CSS',
      terminalFontPreview: 'Prévia de glifos',
      terminalFontReset: 'Usar padrão',
      chatFontTitle: 'Fonte do chat',
      chatFontDesc:
        'Escolha uma fonte instalada para o chat e o restante do aplicativo. É útil para fontes que facilitam a leitura, como OpenDyslexic; deixe em branco para usar a fonte do tema.',
      chatFontPlaceholder: 'OpenDyslexic ou uma pilha de fontes CSS',
      chatFontPreview: 'Prévia',
      chatFontSample: 'A rápida raposa marrom salta sobre o cão preguiçoso. 0123456789',
      chatFontReset: 'Usar fonte do tema',
      translucencyTitle: 'Translucidez da janela',
      translucencyDesc: 'Veja sua área de trabalho através da janela inteira. Apenas macOS e Windows.',
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
      backdropTitle: 'Plano de fundo do chat',
      backdropDesc: 'A imagem sutil da estátua atrás da conversa.',
      userBubbleTitle: 'Balão de mensagem',
      userBubbleDesc:
        'Define a transparência das suas próprias mensagens. Totalmente opacas em 0; em 100, resta apenas o contorno.',
      introSplashTitle: 'Tela de abertura',
      introSplashDesc: 'A marca e o prompt exibidos em um chat vazio.',
      reactionsTitle: 'Reações às mensagens',
      reactionsDesc:
        'Reações com emoji no estilo iMessage — reaja às mensagens, e o Hermes também pode reagir às suas.',
      tipsTitle: 'Dicas no aplicativo',
      tipsDesc:
        'Um pequeno balão que aponta para uma parte do aplicativo, exibido ocasionalmente enquanto você não está interagindo e pelo Hermes quando for útil. Fechar uma dica a oculta permanentemente.',
      tipsReset: count => `Restaurar ${count} ${count === 1 ? 'dica fechada' : 'dicas fechadas'}`,
      toursTitle: 'Tours guiados',
      toursDesc: 'Deixe o Hermes guiar você pelo aplicativo, escurecendo a tela e destacando cada etapa.',
      composerPopoutTitle: 'Compositor flutuante',
      composerPopoutDesc:
        'Permitir arrastar o compositor para fora da área encaixada. Desative para mantê-lo preso na parte inferior.',
      vibeHeartsTitle: 'Corações de Vibe',
      vibeHeartsDesc:
        'Corações flutuantes quando você envia obrigado, te amo, bom bot ou um coração. Separado das Reações de Mensagem acima.',
      embedsTitle: 'Conteúdo incorporado',
      embedsDesc:
        'As prévias ricas carregam conteúdo de sites de terceiros (YouTube, X, …). A opção “Perguntar” mostra um espaço reservado até você liberar cada uma; “Sempre” carrega automaticamente; “Desativado” mantém apenas links simples.',
      embedsAsk: 'Perguntar',
      embedsAlways: 'Sempre',
      embedsOff: 'Desativado',
      embedsReset: count => `Redefinir ${count} ${count === 1 ? 'serviço permitido' : 'serviços permitidos'}`,
      resumeLastSessionTitle: 'Reabrir último chat ao iniciar',
      resumeLastSessionDesc:
        'Quando ativado, o aplicativo reabre seu chat mais recente ao iniciar do zero. Desative para sempre começar com um novo chat.',
      product: 'Produto',
      productDesc: 'Atividade das ferramentas em linguagem acessível, com resumos concisos.',
      technical: 'Técnico',
      technicalDesc: 'Inclui argumentos e resultados brutos das ferramentas e detalhes de baixo nível.',
      themeTitle: 'Tema',
      themeDesc: 'Apenas paletas do Desktop. O modo selecionado é aplicado por cima.',
      themeSearchPlaceholder: 'Pesquise nos seus temas ou no VS Code Marketplace…',
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
        deleteTitle: name => `Excluir ${name}?`,
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
    fieldLabels: {
      model: 'Modelo padrão',
      modelContextLength: 'Janela de contexto',
      fallbackProviders: 'Modelos de fallback',
      toolsets: 'Conjuntos de ferramentas ativados',
      timezone: 'Fuso horário',
      'display.personality': 'Personalidade',
      'display.showReasoning': 'Blocos de raciocínio',
      'desktop.repoScanEnabled': 'Descoberta automática de repositórios',
      'desktop.repoScanRoots': 'Pastas raiz para descoberta de repositórios',
      'desktop.repoScanExcludePaths': 'Caminhos de repositório excluídos',
      'agent.maxTurns': 'Máximo de etapas do agente',
      'agent.imageInputMode': 'Anexos de imagem',
      'agent.apiMaxRetries': 'Tentativas da API',
      'agent.serviceTier': 'Nível de serviço',
      'agent.toolUseEnforcement': 'Obrigatoriedade de uso de ferramentas',
      'terminal.cwd': 'Diretório de trabalho',
      'terminal.backend': 'Backend de execução',
      'terminal.timeout': 'Tempo limite do comando',
      'terminal.persistentShell': 'Shell persistente',
      'terminal.envPassthrough': 'Repasse de variáveis de ambiente',
      'terminal.dockerImage': 'Imagem Docker',
      'terminal.singularityImage': 'Imagem Singularity',
      'terminal.modalImage': 'Imagem Modal',
      'terminal.daytonaImage': 'Imagem Daytona',
      fileReadMaxChars: 'Limite de leitura de arquivo',
      'toolOutput.maxBytes': 'Limite de saída do terminal',
      'toolOutput.maxLines': 'Limite de paginação de arquivo',
      'toolOutput.maxLineLength': 'Limite de comprimento da linha',
      'codeExecution.mode': 'Modo de execução de código',
      'approvals.mode': 'Modo de aprovação',
      'approvals.timeout': 'Tempo limite de aprovação',
      'approvals.mcpReloadConfirm': 'Confirmar recargas do MCP',
      commandAllowlist: 'Lista de comandos permitidos',
      'security.redactSecrets': 'Ocultar segredos',
      'security.allowPrivateUrls': 'Permitir URLs privadas',
      'browser.allowPrivateUrls': 'URLs privadas no navegador',
      'browser.autoLocalForPrivateUrls': 'Navegador local para URLs privadas',
      'browser.useRealProfile': 'Usar meu perfil real do navegador',
      'checkpoints.enabled': 'Checkpoints de arquivo',
      'checkpoints.maxSnapshots': 'Limite de checkpoints',
      'voice.recordKey': 'Atalho de voz',
      'voice.maxRecordingSeconds': 'Duração máxima da gravação',
      'voice.autoTts': 'Ler as respostas em voz alta',
      'voice.voiceChatMode': 'Modo de chat por voz',
      'voice.gptLive.voice': 'Voz do GPT-Live',
      'voice.gptLive.instructions': 'Persona do GPT-Live',
      'stt.enabled': 'Transcrição de voz',
      'stt.echoTranscripts': 'Repetir as transcrições',
      'stt.provider': 'Provedor de transcrição de voz',
      'stt.local.model': 'Modelo local de transcrição',
      'stt.local.language': 'Idioma da transcrição',
      'stt.openai.model': 'Modelo STT da OpenAI',
      'stt.groq.model': 'Modelo STT da Groq',
      'stt.mistral.model': 'Modelo STT da Mistral',
      'stt.elevenlabs.modelId': 'Modelo STT da ElevenLabs',
      'stt.elevenlabs.languageCode': 'Idioma da ElevenLabs',
      'stt.elevenlabs.tagAudioEvents': 'Marcar eventos de áudio',
      'stt.elevenlabs.diarize': 'Separação de locutores',
      'tts.provider': 'Provedor de síntese de voz',
      'tts.edge.voice': 'Voz da Edge',
      'tts.openai.model': 'Modelo TTS da OpenAI',
      'tts.openai.voice': 'Voz da OpenAI',
      'tts.elevenlabs.voiceId': 'Voz da ElevenLabs',
      'tts.elevenlabs.modelId': 'Modelo da ElevenLabs',
      'tts.xai.voiceId': 'Voz da xAI (Grok)',
      'tts.xai.language': 'Idioma da xAI',
      'tts.xai.speed': 'Velocidade de reprodução da xAI',
      'tts.xai.autoSpeechTags': 'Marcadores automáticos de fala da xAI',
      'tts.xai.optimizeStreamingLatency': 'Otimização da latência de streaming da xAI',
      'tts.xai.sampleRate': 'Taxa de amostragem da xAI',
      'tts.xai.bitRate': 'Taxa de bits da xAI',
      'tts.minimax.model': 'Modelo TTS da MiniMax',
      'tts.minimax.voiceId': 'Voz da MiniMax',
      'tts.mistral.model': 'Modelo TTS da Mistral',
      'tts.mistral.voiceId': 'Voz da Mistral',
      'tts.gemini.model': 'Modelo TTS do Gemini',
      'tts.gemini.voice': 'Voz do Gemini',
      'tts.neutts.model': 'Modelo NeuTTS',
      'tts.neutts.device': 'Dispositivo do NeuTTS',
      'tts.kittentts.model': 'Modelo KittenTTS',
      'tts.kittentts.voice': 'Voz do KittenTTS',
      'tts.piper.voice': 'Voz do Piper',
      'tts.deepinfra.model': 'Modelo TTS da DeepInfra',
      'tts.deepinfra.voice': 'Voz da DeepInfra',
      'memory.memoryEnabled': 'Memória persistente',
      'memory.userProfileEnabled': 'Perfil do usuário',
      'memory.memoryCharLimit': 'Orçamento de memória',
      'memory.userCharLimit': 'Orçamento do perfil',
      'memory.provider': 'Provedor de memória',
      'context.engine': 'Motor de contexto',
      'compression.enabled': 'Compressão automática',
      'compression.threshold': 'Limiar de compressão',
      'compression.codexGpt55Autoraise': 'Aumento automático da compressão do Codex',
      'compression.targetRatio': 'Meta de compressão',
      'compression.protectLastN': 'Mensagens recentes protegidas',
      'auxiliary.compression.timeout': 'Tempo limite da compressão',
      'delegation.model': 'Modelo dos subagentes',
      'delegation.provider': 'Provedor dos subagentes',
      'delegation.maxIterations': 'Limite de turnos dos subagentes',
      'delegation.maxConcurrentChildren': 'Subagentes em paralelo',
      'delegation.childTimeoutSeconds': 'Tempo limite dos subagentes',
      'delegation.reasoningEffort': 'Esforço de raciocínio dos subagentes',
      'updates.nonInteractiveLocalChanges': 'Alterações locais durante a atualização pelo app'
    },
    fieldDescriptions: {
      model: 'Usado em novas conversas, a menos que você escolha outro modelo no compositor.',
      modelContextLength:
        'Substitui a janela de contexto detectada apenas do modelo PRINCIPAL do chat (tokens). Deixe em 0 para usar o valor detectado do modelo selecionado. Não afeta modelos auxiliares nem MoA.',
      fallbackProviders: 'Entradas provedor:modelo de reserva a serem tentadas caso o modelo padrão falhe.',
      'display.personality': 'Estilo padrão do assistente em novas sessões.',
      'display.showReasoning': 'Mostra as seções de raciocínio quando o backend as fornece.',
      'desktop.repoScanEnabled': 'Procura repositórios Git em pastas locais para exibi-los em Projetos.',
      'desktop.repoScanRoots': 'Pastas a examinar. Deixe vazio para examinar sua pasta pessoal.',
      'desktop.repoScanExcludePaths': 'Pastas e seus descendentes a ignorar durante a descoberta de repositórios.',
      timezone: 'Identificador de fuso horário IANA. Em branco, usa o fuso horário do sistema.',
      'agent.imageInputMode': 'Controla como os anexos de imagem são enviados ao modelo.',
      'agent.maxTurns': 'Limite máximo de turnos com chamadas de ferramentas antes de o Hermes encerrar uma execução.',
      'terminal.cwd': 'Pasta padrão do projeto para trabalho com ferramentas e terminal.',
      'terminal.persistentShell': 'Mantém o estado do shell entre comandos quando o backend oferece suporte.',
      'terminal.envPassthrough': 'Variáveis de ambiente repassadas para a execução das ferramentas.',
      'terminal.dockerImage': 'Imagem de contêiner usada quando o backend de execução é o Docker.',
      'terminal.singularityImage': 'Imagem usada quando o backend de execução é o Singularity.',
      'terminal.modalImage': 'Imagem usada quando o backend de execução é o Modal.',
      'terminal.daytonaImage': 'Imagem usada quando o backend de execução é o Daytona.',
      'codeExecution.mode': 'Define com que rigor a execução de código fica restrita ao projeto atual.',
      fileReadMaxChars: 'Máximo de caracteres que o Hermes pode ler em uma única solicitação de arquivo.',
      'approvals.mode': 'Define como o Hermes lida com comandos que exigem aprovação explícita.',
      'approvals.timeout': 'Tempo que as solicitações de aprovação aguardam antes de expirar.',
      'security.redactSecrets': 'Oculta, quando possível, segredos detectados do conteúdo visível ao modelo.',
      'browser.useRealProfile':
        'A navegação local usa seus logins reais. O Hermes copia o perfil do navegador padrão — cookies, logins e preferências — para um snapshot gerenciado e o controla com o Chromium integrado; seu perfil ativo nunca é aberto diretamente, e a cópia é atualizada a cada execução. Também permite que o agente abra uma sessão local com perfil real quando solicitado, mesmo se houver um navegador em nuvem configurado. Apenas navegadores Chromium — Chrome, Edge, Brave, Brave Origin e Chromium — são compatíveis; se o navegador padrão não for Chromium, uma mensagem clara será exibida. Desativado por padrão.',
      'checkpoints.enabled': 'Cria snapshots para restauração antes de editar arquivos.',
      'memory.memoryEnabled': 'Salva memórias persistentes que podem ajudar em sessões futuras.',
      'memory.userProfileEnabled': 'Mantém um perfil compacto das preferências do usuário.',
      'context.engine': 'Estratégia usada para gerenciar conversas longas próximas do limite de contexto.',
      'compression.enabled': 'Resume o contexto mais antigo quando as conversas ficam grandes.',
      'compression.codexGpt55Autoraise': 'Aumenta a compressão para 85% em modelos OAuth ChatGPT Codex compatíveis.',
      'auxiliary.compression.timeout':
        'Segundos de espera por chamada para o modelo auxiliar de compressão (padrão: 120). Aumente para modelos locais lentos.',
      'voice.autoTts': 'Lê automaticamente em voz alta as respostas do assistente.',
      'voice.voiceChatMode':
        'chained: conversão de voz em texto → Hermes → conversão de texto em voz com os provedores abaixo. gpt-live: um modelo de voz full-duplex da OpenAI (gpt-live-1) ouve e fala, e encaminha todas as solicitações reais ao Hermes — qualquer modelo selecionado responde com o conjunto completo de ferramentas. Exige uma chave de API da OpenAI; a camada de voz custa US$ 0,05 por minuto.',
      'voice.gptLive.voice': 'Voz do modo GPT-Live. IDs de voz personalizados são aceitos.',
      'voice.gptLive.instructions':
        'Frases adicionais para a persona de voz ao vivo (tom, ritmo, idioma). O Hermes mantém o próprio prompt do sistema.',
      'tts.xai.voiceId': 'ID de voz da xAI (por exemplo, eve) ou um ID de voz personalizado.',
      'tts.xai.language': 'Código do idioma falado (por exemplo, en, pt-BR) ou "auto" para detecção automática.',
      'tts.xai.speed': 'Velocidade de reprodução. 0,7 = mais lento, 1,0 = normal, 1,5 = mais rápido.',
      'tts.xai.autoSpeechTags':
        'Permite que um LLM insira marcadores expressivos de áudio ([laughing], [sighs]) no roteiro antes da síntese.',
      'tts.xai.optimizeStreamingLatency':
        'Equilíbrio entre latência e qualidade. 0 = melhor qualidade, 2 = menor latência.',
      'tts.xai.sampleRate':
        'Taxa de amostragem do áudio em Hz. Valores maiores oferecem melhor qualidade e arquivos maiores.',
      'tts.xai.bitRate': 'Taxa de bits do MP3 em bps. Aplica-se apenas quando o codec é mp3.',
      'tts.neutts.device': 'Dispositivo de inferência local usado pelo NeuTTS.',
      'stt.enabled': 'Ativa a transcrição de fala local ou por provedor.',
      'stt.echoTranscripts': 'Publica de volta na conversa a transcrição bruta das mensagens de voz.',
      'stt.elevenlabs.languageCode':
        'Código de idioma ISO-639-3 opcional. Em branco, a ElevenLabs detecta automaticamente.',
      'updates.nonInteractiveLocalChanges':
        'Quando o Hermes é atualizado pelo app sem uma pergunta no terminal, define se as alterações locais no código-fonte serão preservadas (stash) ou descartadas (discard). Atualizações pelo terminal sempre perguntam.'
    },
    uninstallSection: {
      dangerZone: 'Zona de perigo',
      checkingInstalled: 'Verificando o que está instalado…',
      uninstallHermes: 'Desinstalar o Hermes',
      chooseHowMuch:
        'Escolha quanto deseja remover. O app será fechado para concluir; abra o instalador novamente quando quiser voltar.',
      confirmUninstall: 'Confirmar desinstalação',
      confirmBody: what => `Isso removerá ${what}. Esta ação não pode ser desfeita.`,
      appLabel: 'App:',
      couldNotStart: 'Não foi possível iniciar a desinstalação.',
      uninstalling: 'Desinstalando…',
      yesUninstall: 'Sim, desinstalar',
      options: {
        gui: {
          title: 'Desinstalar somente a interface de chat',
          description: 'Remove este app para desktop. O agente Hermes, suas configurações e conversas permanecem.',
          consequence: 'a interface de chat para desktop (este app e os dados dele)'
        },
        lite: {
          title: 'Desinstalar interface e agente, manter meus dados',
          description:
            'Remove o app e o agente Hermes, mas mantém configurações, conversas e segredos para uma reinstalação futura.',
          consequence: 'a interface de chat e o agente Hermes (configurações, conversas e segredos serão mantidos)'
        },
        full: {
          title: 'Desinstalar tudo',
          description:
            'Remove o app, o agente e todos os dados do usuário — configurações, conversas, tarefas agendadas, segredos e logs.',
          consequence:
            'TUDO — a interface de chat, o agente Hermes e todas as suas configurações, conversas, segredos e logs'
        }
      }
    },
    poolLimits: {
      warmBotBackendsAria: 'Backends de bots aquecidos',
      warmBotBackendsTitle: 'Backends de bots aquecidos',
      backendIdleTimeoutAria: 'Tempo limite de inatividade do backend em milissegundos',
      backendIdleTimeoutTitle: 'Tempo limite de inatividade do backend'
    },
    customEndpoints: {
      active: 'Ativo',
      apiKeySet: 'Chave de API configurada',
      use: 'Usar',
      editTitle: 'Editar endpoint',
      addTitle: 'Adicionar endpoint',
      fields: {
        name: 'Nome',
        providerId: 'ID do provedor',
        endpointUrl: 'URL do endpoint',
        defaultModel: 'Modelo padrão',
        context: 'Contexto',
        apiKey: 'Credencial da API',
        apiKeyNewPlaceholder: 'Deixe em branco para manter a chave atual',
        apiKeyPlaceholder: 'Opcional',
        useNewChats: 'Usar em novas conversas',
        discoverModels: 'Descobrir modelos'
      },
      test: 'Testar',
      save: 'Salvar',
      newEndpoint: 'Novo endpoint',
      apiMode: 'Modo da API',
      autoDetect: 'Detectar automaticamente',
      couldNotLoad: 'Não foi possível carregar os endpoints personalizados',
      endpointSaved: 'Endpoint personalizado salvo.',
      saveFailed: 'Falha ao salvar',
      endpointReachable: 'O endpoint está acessível.',
      endpointReachableTransport: transport => `O endpoint está acessível (rota ${transport} disponível).`,
      endpointReachableModels: (reachable, count) => `${reachable} ${count} modelos encontrados.`,
      endpointValidationFailed: 'Falha na validação do endpoint.',
      validationFailed: 'Falha na validação',
      activationFailed: 'Falha na ativação',
      deleteConfirm: name => `Excluir ${name}?`,
      deleteFailed: 'Falha ao excluir',
      title: 'Endpoints personalizados',
      deleteEndpoint: 'Excluir endpoint',
      emptyDescription: 'Adicione abaixo um endpoint compatível com a OpenAI.',
      emptyTitle: 'Nenhum endpoint personalizado',
      namePlaceholder: 'Axet Proxy',
      contextPlaceholder: 'Automático'
    },
    computerUse: {
      accessibility: 'Acessibilidade',
      screenRecording: 'Gravação da tela',
      driverHealth: 'Status do driver'
    },
    about: {
      heading: 'Hermes Desktop',
      version: value => `versão ${value}`,
      versionUnavailable: 'Versão indisponível',
      bundleOutOfSync: 'Build do aplicativo desatualizado',
      bundleOutOfSyncDesc:
        'O runtime do Hermes foi atualizado, mas o aplicativo Desktop ainda usa um build antigo — novos recursos da interface, como o Modo Bot, não aparecerão até que ele seja atualizado. Execute a atualização abaixo para reconstruir o aplicativo. Se o aviso continuar, reinstale usando o instalador Desktop mais recente.',
      bundleOutOfSyncAction: 'Obter o instalador',
      bundleSwapPending: 'Reinicie para concluir a atualização',
      bundleSwapPendingDesc:
        'O aplicativo atualizado já está instalado — o Hermes só precisa reiniciar para carregá-lo. Seus chats e configurações não serão alterados.',
      bundleSwapPendingAction: 'Reiniciar Hermes',
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
      updateReadyUnknown: 'Uma nova atualização está pronta.',
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
      minimizeToTrayTitle: 'Minimizar para a bandeja',
      minimizeToTrayDesc:
        'Minimize as janelas ou feche a janela principal para ocultá-las na bandeja do sistema (barra de menus no macOS) e manter o Hermes em execução. Use Sair do Hermes no menu da bandeja ou Cmd+Q para encerrar. Desativado por padrão; aplica-se apenas a este dispositivo.',
      minimizeToTrayUnavailable:
        'A bandeja do sistema não está disponível. As janelas serão minimizadas e fechadas normalmente. Desative e ative novamente esta opção para tentar de novo.',
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
      toolsetsWipeConfirm:
        'Remover todos os conjuntos de ferramentas ativados? Isso desativa memória, terminal, busca na web, delegação e a maioria das outras ferramentas até que você as ative novamente.',
      keepAwakeTitle: 'Manter o computador ativo',
      keepAwakeDesc:
        'Impede que esta máquina entre em suspensão, para que execuções longas ou noturnas continuem. A tela ainda pode escurecer.',
      disableF12Title: 'Desativar DevTools com F12',
      disableF12Desc: 'Impedir que F12 abra as Ferramentas do desenvolvedor. Ctrl+Shift+I continua funcionando.',
      attachmentSizeTitle: 'Tamanho máximo para prévias e carregamento de imagens',
      attachmentSizeDesc:
        'Define o tamanho máximo, em MB, de um arquivo local que o Desktop carrega para prévias e anexos de imagem. O padrão é 16. Anexos remotos que não sejam imagens têm um limite separado de 256 MB. Valores muito altos carregam o arquivo inteiro na memória e podem travar ou encerrar o app.',
      attachmentSizeUnit: 'MB',
      attachmentSizeLabel: 'Tamanho máximo, em MB, para prévias e carregamento de imagens',
      showOptions: 'Mostrar opções'
    },
    hudModifier: {
      title: 'Toque para abrir o HUD',
      description:
        'Pressione e solte ⌘ + Option no Mac ou Ctrl + Alt no Windows/Linux para trazer o HUD para a frente a partir de qualquer aplicativo. Desativado por padrão; aplica-se apenas a este dispositivo.',
      permission:
        'Autorize o Hermes em Ajustes do Sistema → Privacidade e Segurança → Monitoramento de Entrada e tente novamente. Esse gesto não registra as teclas digitadas nem captura sua tela.',
      unavailable:
        'O auxiliar de gestos do HUD não pôde ser iniciado ou parou inesperadamente. Tente novamente ou reinicie o Hermes. O atalho atual do HUD ainda funciona dentro do Hermes.',
      missingHelper:
        'O auxiliar de gestos do HUD está ausente nesta instalação do Hermes. Atualize ou reinstale o Hermes e tente novamente.',
      unsupportedSession:
        'Esta sessão da área de trabalho não oferece suporte a toques globais em teclas modificadoras. No Linux, é necessário usar X11; Wayland não é compatível.'
    },
    screenshot: {
      enabledTitle: 'Atalho de captura de tela',
      enabledDesc:
        'Pressione as duas teclas Command ao mesmo tempo em qualquer aplicativo para capturar a janela em primeiro plano e anexá-la ao seu rascunho atual no Hermes. Nunca é enviada automaticamente. Desativado por padrão; aplica-se apenas a este Mac. O conteúdo da janela pode ser confidencial — revise o anexo antes de enviar.',
      statusTitle: 'Status do atalho de captura de tela',
      checking: 'Verificando o atalho de captura de tela…',
      disabled: 'O atalho de captura de tela está desativado.',
      starting: 'Iniciando o monitor do atalho. Ele ainda não está pronto.',
      ready: 'O atalho está pronto. As capturas de tela são anexadas ao seu rascunho atual sem serem enviadas.',
      inputPermission:
        'A permissão de Monitoramento de Entrada permite que o Hermes detecte as duas teclas Command enquanto outro aplicativo estiver ativo. Autorize o Hermes em Ajustes do Sistema → Privacidade e Segurança → Monitoramento de Entrada, volte aqui e tente novamente.',
      screenPermission:
        'A permissão de Gravação da Tela permite que o Hermes capture a janela do aplicativo em primeiro plano quando você usa este atalho. Autorize o Hermes em Ajustes do Sistema → Privacidade e Segurança → Gravação da Tela, volte aqui e tente novamente. Reinicie o Hermes se o macOS solicitar.',
      openSettings: 'Abrir Ajustes do Sistema',
      retry: 'Tentar novamente',
      unavailable: 'O atalho de captura de tela não está disponível. Tente novamente ou desative-o.',
      errorTitle: 'Erro no atalho de captura de tela',
      loadFailed: 'Não foi possível ler o status do atalho. Tente novamente para verificar a configuração atual.',
      saveFailed:
        'Não foi possível confirmar a alteração do atalho. Tente novamente para verificar a configuração atual.',
      permissionFailed:
        'Não foi possível abrir os Ajustes do Sistema. Abra Privacidade e Segurança manualmente e tente novamente.',
      captureFailed: 'Não foi possível capturar a janela em primeiro plano. Nada foi anexado nem enviado.',
      contextChanged: 'O rascunho atual mudou durante a captura. A captura de tela não foi anexada nem enviada.'
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
      receipt: (id, outcome) => `Recibo ${id} · ${outcome}`,
      receiptVersions: (pre, post) => `${pre} → ${post}`,
      scopesRestored: profiles => `Perfis restaurados: ${profiles}`,
      scopeNotRestored: (profile, error) => `Perfil “${profile}” não restaurado: ${error}`
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
      cloudSavedTitle: 'Gateways da nuvem salvos',
      cloudSavedDesc:
        'Use um gateway salvo sem alterar o padrão. Faça login abaixo para adicionar instâncias. Gerencie os nomes e o login na lista de conexões salvas.',
      cloudUseSaved: 'Usar gateway',
      cloudActive: 'Ativo nesta janela',
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
    search: {
      placeholder: 'Pesquisar em todas as configurações…',
      pill: 'Pesquisar'
    },
    profileScope: {
      appliesTo: 'Aplica-se a',
      editsProfile: profile => `As alterações nesta página aplicam-se ao perfil “${profile}”.`
    },
    mcp: {
      loading: 'Carregando servidores MCP...',
      invalidJson: 'JSON do MCP inválido',
      saveFailed: 'Falha ao salvar',
      removeFailed: 'Falha ao remover',
      reloadFailed: 'Falha ao recarregar MCP',
      savedTitle: 'Servidor MCP salvo',
      savedMessage: name => `${name} passa a valer depois de recarregar o MCP.`,
      disabled: 'desativado',
      name: 'Nome',
      serverJson: 'JSON do servidor',
      remove: 'Remover',
      test: 'Testar conexão',
      catalogLoading: 'Carregando o catálogo do MCP...',
      catalogInstallFailed: name => `Falha ao instalar ${name}`,
      catalogEnvRequired: 'Preencha os valores obrigatórios antes de instalar.',
      capabilitySummary: (tools, prompts, resources) =>
        `${[`${tools} ferramentas`, ...(prompts ? [`${prompts} prompts`] : []), ...(resources ? [`${resources} recursos`] : [])].join(', ')} ativados`,
      costTokens: tokens => `~${tokens} tokens/chamada`,
      usage30d: uses => `${uses} usos/30 dias`,
      statusConnecting: 'Conectando…',
      statusNeedsAuth: 'Requer autenticação',
      statusError: 'Erro',
      statusOff: 'Desativado',
      allServers: 'todos os servidores',
      authenticatedTitle: 'Autenticado',
      authenticatedMessage: (server, count) => `${server}: ${count} ferramentas`,
      authenticate: 'Autenticar',
      noOutput: 'Nenhuma saída ainda.',
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
      deepLinkErrorTooLarge: 'A configuração excede o limite de 32 KB.'
    },
    model: {
      setupProviderFallback: 'provedor',
      setUpProvider: name => `Configurar ${name}`,
      staleAuxBefore: (count, names) =>
        `${count} ${count === 1 ? 'tarefa auxiliar' : 'tarefas auxiliares'} (${names}) ainda ${count === 1 ? 'é executada' : 'são executadas'} em `,
      staleAuxAfter: ', não no seu modelo principal.',
      staleAuxOtherProviders: 'outros provedores',
      moaEnabled: 'Ativado',
      moaSetDefault: 'Definir como padrão',
      moaNewPresetPlaceholder: 'nova predefinição',
      moaAddPreset: 'Adicionar predefinição',
      customModel: 'Modelo personalizado…',
      customModelPlaceholder: 'ID do modelo',
      chooseFromList: 'Escolher na lista',
      moaDefault: 'Padrão:',
      moaReferenceToggle: (enabled, index) => `${enabled ? 'Desativar' : 'Ativar'} referência ${index}`,
      moaReferenceTitle: index => `Referência ${index}`,
      moaAddReference: 'Adicionar modelo de referência',
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
      loadFailed: 'Não foi possível carregar os modelos',
      restartRequired:
        'Este backend está executando código antigo após uma atualização. Reinicie-o para carregar o código novo.',
      restartBackend: 'Reiniciar backend',
      restartingBackend: 'Reiniciando backend…',
      restartFailed: 'Não foi possível reiniciar o backend',
      auxiliaryTitle: 'Modelos auxiliares',
      resetAllToMain: 'Redefinir todos para o principal',
      auxiliaryDesc:
        'Por padrão, as tarefas de apoio rodam no modelo principal. Atribua um modelo dedicado a qualquer tarefa para sobrescrever.',
      setToMain: 'Definir como principal',
      change: 'Alterar',
      autoUseMain: 'automático · usa o modelo principal',
      inheritMainEffort: 'herdar · esforço do modelo principal',
      providerDefault: '(Provedor padrão)',
      fallbackAdd: 'Adicionar fallback',
      fallbackEmpty: 'Nenhum modelo de fallback — o modelo padrão é usado a menos que falhe.',
      notInCatalog: 'não está na lista de modelos deste provedor — as chamadas podem recair em um modelo de reserva.',
      moaTitle: 'Mixture of Agents',
      moaPreset: 'Predefinição',
      moaDescription:
        'Configure predefinições nomeadas que aparecem como modelos no provedor Mixture of Agents. O agregador é o modelo atuante — ele executa cada etapa do loop de ferramentas, e quase todo o custo da execução é cobrado pelo provedor dele. Por padrão, as referências apenas dão orientações uma vez a cada turno do usuário.',
      moaAggregator: 'Agregador',
      moaAggregatorBilled: 'modelo atuante · cobrança da execução',
      moaReferenceHint: 'por padrão, dá orientações uma vez por turno',
      tasks: {
        vision: {
          label: 'Visão',
          hint: 'Análise de imagem'
        },
        compression: {
          label: 'Compactação',
          hint: 'Compactação de contexto'
        },
        skills_hub: {
          label: 'Hub de habilidades',
          hint: 'Busca de habilidades'
        },
        approval: {
          label: 'Aprovação',
          hint: 'Aprovação automática inteligente'
        },
        mcp: {
          label: 'MCP',
          hint: 'Roteamento de ferramentas MCP'
        },
        title_generation: {
          label: 'Geração de título',
          hint: 'Títulos das sessões'
        },
        review: {
          label: 'Revisão',
          hint: '/review subagente revisor'
        },
        triage_specifier: {
          label: 'Especificador de triagem',
          hint: 'Detalhamento de especificações do Kanban'
        },
        kanban_decomposer: {
          label: 'Decompositor do Kanban',
          hint: 'Decomposição de tarefas'
        },
        profile_describer: {
          label: 'Descritor de perfil',
          hint: 'Descrições automáticas de perfil'
        },
        curator: {
          label: 'Curador',
          hint: 'Revisão do uso de habilidades'
        }
      }
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
          'Nenhum modelo alcança velocidade total neste hardware; este chega mais perto enquanto executa inteiramente na memória da GPU.'
      },
      noRecommendationTitle: 'Nenhuma recomendação automática para este computador',
      noRecommendationDetail:
        'A configuração automática exige um modelo selecionado que caiba inteiramente na memória da GPU ou na memória unificada. Você ainda pode escolher um modelo abaixo ou explorar mais modelos.',
      noRecommendationAction: 'Explorar modelos',
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
      ejected: 'Modelo removido da memória — memória da GPU liberada.',
      ejectFailed: 'Não foi possível remover o modelo da memória',
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
    billing: {
      perMonth: amount => `${amount}/mês`,
      creditsPerMonth: amount => `${amount} créditos/mês`,
      usageLabel: label => `Uso de ${label}`,
      freeTier: {
        signIn: 'Entrar',
        title: 'Você está no plano gratuito da Nous',
        message: 'Entre com uma conta Nous para liberar mais modelos e ferramentas.',
        caption:
          'Executa no nous/welcome com conectores incluídos. Ao entrar, seus conectores são mantidos e você recebe as ferramentas que exigem conta e todos os outros modelos.',
        name: 'Nous · plano gratuito',
        footnote:
          'O plano gratuito não tem saldo nem cobrança. Pagamentos e uso aparecem quando você entra com uma conta Nous.',
        plan: 'Plano gratuito',
        model: 'Modelo',
        connectors: 'Conectores',
        included: 'Incluído'
      },
      amountValidation: {
        reloadTo: 'Recarregar até',
        greaterThanThreshold: 'O valor de recarga deve ser maior que o limite.',
        decimal: label => `${label}: informe um valor em dólares com no máximo 2 casas decimais.`,
        positive: label => `${label}: o valor deve ser maior que US$ 0.`,
        minimum: (label, amount) => `${label}: o mínimo é ${amount}.`,
        maximum: (label, amount) => `${label}: o máximo é ${amount}.`
      },
      stepUp: {
        openVerification: 'Abrir página de verificação',
        dismiss: 'Dispensar',
        waiting: 'Aguardando o link de verificação…',
        verify: 'Verificar para continuar',
        deniedTitle: 'A verificação não foi aprovada',
        deniedBody: 'A verificação terminou sem permitir gastos remotos neste terminal.',
        successTitle: 'Verificação concluída',
        successBody: 'Gastos remotos estão permitidos neste terminal.'
      },
      charge: {
        added: amount => (amount ? `US$ ${amount} adicionados.` : 'Créditos adicionados.'),
        failedTitle: 'Falha na cobrança',
        unconfirmedTitle: 'Resultado da cobrança não confirmado',
        unconfirmedBody: message =>
          `${message} O resultado da última cobrança não foi confirmado; verifique seu saldo e histórico antes de tentar novamente.`,
        checkTitle: 'Não foi possível verificar a cobrança',
        checkBody: 'Não foi possível verificar a cobrança.',
        untrackedTitle: 'Não foi possível acompanhar a cobrança',
        untrackedBody: 'O serviço de cobrança aceitou a solicitação, mas não retornou um ID de cobrança.',
        timeoutTitle: 'Ainda em processamento após 5 minutos',
        timeoutBody: 'A cobrança ainda pode ser concluída. Verifique o portal antes de tentar novamente.',
        authenticationRequired:
          'Seu banco exige verificação (3DS). Conclua a verificação no portal para finalizar esta compra.',
        expired: 'Seu cartão expirou. Atualize-o no portal.',
        declined: 'Seu cartão foi recusado. Tente outro cartão no portal.',
        failedBody: reason => `A cobrança não foi concluída (${reason}).`
      },
      title: 'Cobrança',
      preview: 'prévia',
      summary: {
        balance: 'Saldo',
        plan: 'Plano',
        autoRefill: 'Recarga automática'
      },
      sections: {
        invoices: 'Faturas',
        plan: 'Plano',
        paymentAndCredits: 'Pagamento e créditos',
        usage: 'Uso'
      },
      usage: {
        title: 'Uso'
      },
      buyCredits: {
        customAmount: 'Valor personalizado de créditos',
        title: 'Comprar créditos agora',
        buyButton: 'Comprar',
        processing: 'Processando… verificando a conclusão',
        added: amount => `${amount} adicionados. Atualizando o saldo.`,
        retry: 'Tentar novamente',
        openPortal: 'Abrir portal'
      },
      plan: {
        title: 'Planos',
        changePlan: 'Alterar plano',
        viewPlans: 'Ver planos',
        backAria: 'Voltar para cobrança',
        current: 'Plano atual',
        scheduled: 'Agendado',
        empty: 'Não há planos disponíveis para alteração agora.',
        undo: 'Desfazer',
        undoing: 'Desfazendo…',
        downgrade: 'Mudar para plano inferior',
        confirmDowngrade: 'Confirmar mudança para plano inferior',
        tryAgain: 'Tentar novamente',
        checkingChange: 'Verificando esta alteração…',
        cannotChange: 'Essa alteração não pode ser feita aqui.',
        alreadyOn: name => `Você já está no plano ${name}; não há nada para alterar.`,
        notScheduleable: 'Esta alteração não pode ser agendada aqui.',
        scheduling: 'Agendando…',
        cancel: 'Cancelar',
        effectScheduled: (targetName, effectiveAt, creditsDelta) =>
          `Mudança para ${targetName} — entra em vigor em ${effectiveAt}. Nenhuma cobrança agora; você mantém o plano atual até lá.${creditsDelta ? ` Alteração nos créditos mensais: ${creditsDelta}.` : ''}`
      },
      autoReload: {
        threshold: 'Limite',
        thresholdAria: 'Limite da recarga automática',
        reloadTo: 'Recarregar até',
        reloadToAria: 'Valor da recarga automática',
        turnOffConfirm: 'Desativar a recarga automática?',
        turnOff: 'Desativar',
        disable: 'Desativar',
        updated: 'Recarga automática atualizada.',
        turnedOff: 'Recarga automática desativada.',
        manage: 'Gerenciar',
        save: 'Salvar',
        saving: 'Salvando…',
        cancel: 'Cancelar'
      },
      state: {
        notice: {
          loggedOut: {
            title: 'Conecte sua conta Nous',
            message: 'Execute /portal no TUI ou abra o portal da Nous para conectar sua conta.',
            action: 'Abrir portal ↗'
          },
          noCard: {
            title: 'Nenhum método de pagamento cadastrado',
            message:
              'A compra de créditos adicionais e a recarga automática permanecem desativadas até que um cartão seja cadastrado. Adicione um no portal.',
            action: 'Adicionar cartão ↗'
          }
        },
        paymentMethod: {
          title: 'Método de pagamento',
          description: 'Gerencie o cartão usado para recargas e renovações de assinatura.',
          addAction: 'Adicionar método de pagamento',
          updateAction: 'Atualizar',
          provenance: {
            autoRefill: 'cartão da recarga automática',
            customerDefault: 'padrão do cliente',
            subPin: 'cartão da assinatura',
            suffix: label => ` - ${label}`
          }
        },
        buyCredits: {
          description: 'Uma cobrança única no seu cartão, adicionada ao saldo hoje.'
        },
        autoRefill: {
          title: 'Recarregar quando o saldo estiver baixo',
          genericDescription: 'Mantenha seu saldo recarregado quando ele cair abaixo do limite.',
          offPill: 'Desativada',
          enabledPill: 'Ativada',
          notAvailablePill: '—',
          manageCaption: 'Gerencie a recarga automática pelo portal.',
          turnOnCaption: 'Ative a recarga automática pelo portal',
          chargesDescription: (reloadTo, threshold) =>
            `Cobra ${reloadTo} automaticamente quando seu saldo cai abaixo de ${threshold}.`,
          distinctCardCaption: cardLabel => `A recarga automática cobra em ${cardLabel} — reconcilie no portal`,
          distinctCardFallback: 'um cartão diferente',
          reconcileAction: 'Reconciliar ↗'
        },
        usage: {
          subscriptionCredits: {
            title: 'Créditos da assinatura',
            barLabel: 'Créditos restantes da assinatura',
            captionResets: date => `Renova em ${date}`,
            valueOf: (remaining, monthly) => `${remaining} de ${monthly} restantes`,
            valueOver: (remaining, monthly, over) => `${remaining} de ${monthly} restantes · ${over} excedentes`
          },
          topupCredits: {
            title: 'Créditos adicionais',
            caption: 'Não expiram'
          },
          monthlyCap: {
            title: 'Limite mensal de gastos',
            barLabel: 'Limite mensal de gastos utilizado',
            captionDefault: 'Teto padrão',
            captionSpending: 'Gastos remotos mensais',
            valueUsed: (spent, limit) => `${spent} de ${limit} utilizados`
          }
        },
        planCard: {
          freeTier: 'Gratuito',
          chooseAction: 'Escolher ↗',
          adjustPlanAction: 'Ajustar plano ↗',
          unavailableCaption: 'Os detalhes da assinatura estão indisponíveis; ainda é possível abrir o portal.',
          downgradeCaption: (tierName, when) => `Muda para ${tierName} em ${when}.`,
          cancellationCaption: when => `Será cancelado em ${when}.`,
          renewsCaption: date => `Renova em ${date}`,
          noSubscriptionCaption: 'Sem assinatura ativa — modelos pagos consomem créditos adicionais.'
        }
      },
      errors: {
        consentRequired: {
          title: 'É necessário confirmar o cartão',
          message: 'Confirme este cartão para cobranças pelo terminal no portal'
        },
        insufficientScope: {
          title: 'Gastos remotos precisam de aprovação',
          message: 'É necessário permitir gastos remotos. Inicie uma recarga para autorizar e tente novamente.'
        },
        remoteSpendingRevoked: {
          title: 'Os gastos remotos foram interrompidos',
          messageByAdmin: 'Um administrador interrompeu os gastos remotos deste terminal.',
          messageBySelf: 'Você interrompeu os gastos remotos deste terminal.'
        },
        remoteSpendingReconnect: who =>
          `${who} Reconecte em Configurações → Gateway para autorizar este dispositivo novamente.`,
        sessionRevoked: {
          title: 'Sessão desconectada',
          message: 'Sua sessão foi desconectada. Entre novamente em Configurações → Gateway.'
        },
        cliBillingDisabled: {
          title: 'Gastos remotos estão desativados',
          message:
            'Os gastos remotos estão desativados para esta conta — um administrador de cobrança pode ativá-los na página Hermes Agent do portal.'
        },
        roleRequired: {
          title: 'É necessário ter função de administrador',
          message:
            'Adicionar fundos exige administrador ou proprietário da organização. Peça a um administrador ou gerencie pelo portal.'
        },
        idempotencyConflict: {
          title: 'Inicie uma nova recarga',
          message: '🔴 Essa chave de cobrança já foi usada para outro valor. Inicie uma nova recarga.'
        },
        noPaymentMethod: {
          title: 'Nenhum cartão salvo',
          message:
            '💳 Ainda não há cartão salvo para cobranças pelo terminal. Configure um no portal ' +
            '(compras únicas de créditos não salvam um cartão reutilizável).'
        },
        orgAccessDenied: {
          title: 'Acesso à organização negado',
          message: 'Este token não está vinculado a uma organização que você possa gerenciar'
        },
        monthlyCapExceeded: {
          title: 'Limite mensal de gastos atingido',
          messageReached: '🔴 Limite mensal de gastos atingido.',
          messageHeadroom: remaining => `🔴 Limite mensal de gastos atingido — US$ ${remaining} de margem restante.`
        },
        rateLimited: {
          title: 'Muitas cobranças neste momento',
          message: mins =>
            mins > 0
              ? `🟡 Muitas cobranças neste momento (tente novamente em aproximadamente ${mins} min). Isso não é uma falha de pagamento.`
              : '🟡 Muitas cobranças neste momento. Isso não é uma falha de pagamento.'
        },
        stripeUnavailable: {
          title: 'A Stripe está com problemas',
          message: mins =>
            mins > 0
              ? `A Stripe está com problemas — tente novamente em aproximadamente ${mins} min`
              : 'A Stripe está com problemas — tente novamente em instantes'
        },
        upgradeCapExceeded: {
          title: 'Limite diário de alteração de plano atingido',
          message: 'Limite diário de alteração de plano atingido — tente novamente amanhã'
        },
        endpointUnavailable: {
          title: 'Endpoint de cobrança indisponível',
          message:
            'O endpoint de cobrança retornou uma resposta que não é JSON; talvez não esteja disponível nesta implantação.'
        },
        timeout: {
          title: 'A solicitação de cobrança expirou',
          message: 'A solicitação de cobrança expirou.'
        },
        transport: {
          title: 'Falha na conexão de cobrança',
          message: 'A solicitação de cobrança falhou antes de chegar ao gateway.'
        },
        default: {
          title: 'Falha na solicitação de cobrança',
          message: 'Falha na solicitação de cobrança.'
        }
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
      nousAuthFailedMessage: 'Tente novamente.',
      nousAuthTryAgain: 'Tentar novamente',
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
      postSetupErrorMessage: step =>
        `A configuração de ${step} não foi concluída. Abra os logs para entender o motivo e execute a configuração novamente.`,
      postSetupOpenLogs: 'Abrir logs',
      postSetupRunAgain: 'Executar novamente',
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
          'Você pode selecionar este backend agora — os comandos vão falhar até a configuração ser concluída.',
        needsSetupConfirmTitle: backend => `Selecionar ${backend} mesmo assim?`,
        needsSetupConfirmDescription: detail =>
          `${detail} As sessões iniciadas após esta alteração não terão ferramentas de terminal ou de arquivos até que a configuração seja concluída.`,
        needsSetupConfirmDescriptionGeneric:
          'Este backend ainda não está configurado. As sessões iniciadas após esta alteração não terão ferramentas de terminal ou de arquivos até que a configuração seja concluída.',
        needsSetupConfirmAction: 'Selecionar mesmo assim',
        unavailableTitle: 'Os comandos de terminal não estão disponíveis',
        unavailableMessage: backend =>
          `O Hermes não pode executar comandos de shell no momento: ${backend} não está pronto. Mude para Local ou conclua a configuração de ${backend} e tente novamente.`,
        openBackendSettings: 'Abrir configurações do terminal',
        useLocal: 'Usar Local',
        switchedToLocal: 'Agora os comandos de terminal são executados localmente. Aplica-se a novas sessões.'
      },
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
    }
  },
  skills: {
    tabSkills: 'Habilidades',
    tabToolsets: 'Ferramentas',
    configuringProfile: 'Configurando:',
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
    tabPlugins: 'Plugins',
    plugins: {
      agentTitle: 'Plugins do agente',
      agentBlurb:
        'Amplie o agente para o perfil selecionado com ferramentas, hooks e provedores. As alterações entram em vigor após reiniciar o gateway.',
      pageBlurb: 'Um plugin pode ampliar este aplicativo, o agente ou ambos — cada parte tem seu próprio controle.',
      halfDesktop: 'Desktop',
      halfDesktopHint: 'este aplicativo, igual para todos os perfis',
      halfAgent: 'Agente',
      halfAgentIn: profile => `Agente em ${profile}`,
      defaultProfile: 'Hermes (padrão)',
      kindAgent: 'Agente',
      kindDesktop: 'Desktop',
      kindBoth: 'Agente + Desktop',
      installAgentHere: 'Instalar aqui',
      installAgentHereTip: profile =>
        `A parte de desktop está carregada neste aplicativo, mas a parte do agente não está instalada em ${profile}. Instale-a nesse perfil.`,
      installAgentHereNoOrigin:
        'A parte do agente não está instalada neste perfil, e este pacote foi copiado manualmente (sem entrada no catálogo nem repositório remoto do Git), portanto não pode ser instalado por aqui. Copie a pasta dele para o perfil ou reinstale pelo Git.',
      desktopHalfPending: 'copiando…',
      desktopHalfPendingTip:
        'Este pacote inclui uma parte de desktop que ainda não foi copiada para o aplicativo. Use Reexaminar ou reinicie o aplicativo.',
      desktopHalfRemote: 'indisponível (backend remoto)',
      desktopHalfRemoteTip:
        'A parte de desktop deste pacote está no disco do backend remoto, que este aplicativo não consegue ler. Para usá-la aqui, execute Instalar pelo Git com a URL do repositório do pacote e o destino Desktop marcado — isso clonará a parte de desktop neste computador.',
      emptyAll: 'Ainda não há plugins.',
      empty: 'Nenhum plugin do agente instalado para este perfil.',
      emptyHint: 'Explore o catálogo abaixo e instale um plugin revisado com um clique.',
      loadFailed: 'Não foi possível carregar os plugins do agente',
      toggleFailed: name => `Não foi possível alterar o estado de ${name}`,
      legacyBackend:
        'Este backend é anterior aos controles de plugins endereçados por chave — atualize o Hermes para gerenciá-lo aqui.',
      portableBadge: 'portátil',
      serverStates: {
        connected: 'conectado',
        app_not_running: 'aplicativo não está em execução',
        endpoint_unavailable: 'endpoint indisponível',
        no_interactive_session: 'nenhuma sessão interativa',
        version_too_old: 'versão muito antiga',
        missing_app: 'aplicativo ausente',
        unknown: 'status desconhecido'
      },
      catalogTitle: 'Catálogo de plugins',
      catalogBrowse: 'Explorar',
      catalogHide: 'Ocultar o navegador do catálogo',
      catalogHint:
        'Clique em "+ Adicionar a este agente" em qualquer plugin — as entradas revisadas são instaladas no perfil selecionado, fixadas no commit indicado. Plugins que incluem componentes para o agente e para o desktop oferecem as duas partes.',
      alreadyInstalled: name => `${name} já está instalado neste perfil.`,
      catalogProvenance: sha => `Instalado pelo catálogo do Hermes${sha ? ` no commit fixado ${sha}` : ''}.`,
      pinnedProvenance: sha =>
        `Fixado no commit ${sha}. As atualizações serão recusadas até que ele seja reinstalado com um novo commit fixado.`,
      pinnedBadge: sha => `fixado em ${sha}`,
      tierOfficial: 'oficial',
      tierCommunity: 'comunidade',
      updateToPin: sha => `Atualizar para ${sha}`,
      updateFailed: name => `Não foi possível atualizar ${name}`,
      updated: name => `${name} foi atualizado para o commit atual do catálogo. Reinicie o gateway para aplicar.`,
      updateConsentTitle: name => `${name} solicita mais permissões`,
      updateConsentBody: (name, sha) =>
        `O novo commit do catálogo para ${name} (${sha}) adiciona recursos que a versão instalada não tem. Aplique-o somente se você confiar neles:`,
      updateConsentConfirm: 'Aplicar atualização',
      uninstall: 'Desinstalar',
      uninstallTip: (name, profile) => `Desinstalar ${name} de ${profile}`,
      uninstallConfirmTitle: name => `Desinstalar ${name}?`,
      uninstallConfirmBody: (name, profile) =>
        `Isso exclui os arquivos do plugin do perfil ${profile}. Qualquer componente para desktop incluído nele também será removido. Você pode reinstalá-lo pelo catálogo ou pelo Git a qualquer momento.`,
      uninstallFailed: name => `Não foi possível desinstalar ${name}`,
      uninstalled: name => `${name} foi desinstalado. Reinicie o gateway para descarregá-lo.`,
      uninstallDesktopTip: name => `Desinstalar ${name} deste aplicativo`,
      uninstallDesktopConfirmBody: name =>
        `Isso exclui ${name} da pasta desktop-plugins neste computador e o descarrega agora. Você pode reinstalá-lo pelo Git ou recolocar a pasta a qualquer momento.`,
      uninstalledDesktop: name => `${name} foi desinstalado.`,
      deepLinkErrorTitle: 'Link de instalação do plugin rejeitado',
      deepLinkCatalogInvalidName: 'O nome do catálogo no link está ausente ou é inválido.',
      deepLinkCatalogUnknown: name => `“${name}” não está no catálogo de plugins do Hermes. Nada foi instalado.`,
      deepLinkCatalogUnavailable:
        'Não foi possível carregar o catálogo de plugins do Hermes. Verifique sua conexão e abra o link novamente.',
      settingsToggle: name => `Configurações: ${name}`,
      settingsForm: {
        save: 'Salvar configurações',
        saved: name => `As configurações de ${name} foram salvas.`,
        saveFailed: name => `Não foi possível salvar as configurações de ${name}`,
        optional: '(opcional)',
        secretSet: '•••••••• (definido)',
        secretStoredAs: env =>
          `Armazenado no .env do perfil como ${env}, nunca em config.yaml; deixe em branco para manter o valor atual.`
      }
    },
    officialCatalog: 'Disponível para instalar',
    officialPill: 'Oficial',
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
      close: 'Fechar',
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
      installBlockedTitle: name => `Não foi possível instalar ${name}`,
      installBlockedMessage: (findings, unverified) =>
        `A verificação de segurança sinalizou ${findings > 0 ? `${findings} ${findings === 1 ? 'item' : 'itens'}` : 'padrões de risco'} para revisar${unverified ? ' e a skill vem de uma fonte não verificada' : ''}. Leia a verificação antes de decidir se confia no autor.`,
      viewScan: 'Ver verificação',
      openLog: 'Abrir log',
      actionLog: 'Log da ação',
      alreadyInstalled: name => `“${name}” já está instalado`,
      pickerTitle: 'Skills Hub',
      pickerBrowse: 'Explorar o hub completo',
      pickerHide: 'Ocultar navegador do hub',
      pickerHint:
        'Clique em “+ Adicionar a este agente” em qualquer skill — ela será instalada e aparecerá na lista acima.',
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
    extendedTranscript: 'Transcrição estendida',
    transcriptTruncated: 'Exibindo os 16 KiB mais recentes',
    transcriptUnavailable: 'Transcrição ao vivo indisponível',
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
    moreAgents: count => `+${count} agentes adicionais`,
    queued: 'Na fila',
    waitingActivity: 'Aguardando atividade',
    steer: 'Orientar',
    steerPlaceholder: 'Instruções para este subagente',
    steerQueued: 'Na fila para o próximo ponto de verificação',
    stopRequested: 'Parada solicitada',
    requestRejected: 'O subagente não aceitou a solicitação',
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
    sections: {
      maintenance: 'Manutenção',
      sessions: 'Sessões',
      system: 'Sistema',
      usage: 'Uso'
    },
    sectionDescriptions: {
      maintenance: 'Diagnóstico, cópias de segurança, curador e dados de memória',
      sessions: 'Buscar e gerenciar sessões',
      system: 'Status, logs e ações do sistema',
      usage: 'Tokens, custo e atividade das habilidades ao longo do tempo'
    },
    nav: {
      newChat: {
        title: 'Nova sessão',
        detail: 'Iniciar uma sessão nova'
      },
      settings: {
        title: 'Configurações',
        detail: 'Configurar o Hermes Desktop'
      },
      capabilities: {
        title: 'Recursos',
        detail: 'Skills, ferramentas, servidores MCP e plugins'
      },
      messaging: {
        title: 'Mensagens',
        detail: 'Configurar Telegram, Slack, Discord e mais'
      },
      artifacts: {
        title: 'Artefatos',
        detail: 'Explorar saídas geradas'
      }
    },
    sectionEntries: {
      sessions: {
        title: 'Painel de sessões',
        detail: 'Buscar, fixar e gerenciar sessões'
      },
      system: {
        title: 'Painel do sistema',
        detail: 'Status do gateway, logs, reinício/atualização'
      },
      usage: {
        title: 'Painel de uso',
        detail: 'Tokens, custo e atividade das habilidades'
      }
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
    openBrowser: 'Abrir navegador',
    gatewayRestartFailed: 'Falha ao reiniciar o gateway.',
    sharedGatewayRestartTitle: 'Reiniciar o gateway compartilhado?',
    sharedGatewayRestartDescription: bots => `Todos os bots neste dispositivo serão reconectados: ${bots}`,
    sharedGatewayRestartConfirm: 'Reiniciar todos',
    sharedGatewayRestarted: count => `Gateway compartilhado reiniciado (${count} ${count === 1 ? 'bot' : 'bots'})`,
    updateHermes: 'Atualizar o Hermes',
    reloadWindow: 'Recarregar janela',
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
    sharedListenerUrl: 'Disponível no listener do gateway compartilhado em',
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
    appliedLive: 'Aplicado ao gateway em execução.',
    connectingLive: 'O gateway em execução está se conectando com as novas credenciais.',
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
    revokeDesc: name => `${name} perderá o acesso e deixará de ser reconhecido a partir da próxima mensagem.`,
    approvedUser: name => `Acesso aprovado para ${name}`,
    approvedHint: 'Será reconhecido automaticamente na próxima mensagem.',
    revokedUser: name => `Acesso revogado para ${name}`,
    failedApprove: name => `Falha ao aprovar ${name}`,
    failedRevoke: name => `Falha ao revogar ${name}`,
    pairingLockedOut:
      'Muitas aprovações falharam — esta plataforma foi bloqueada temporariamente. Tente novamente mais tarde.',
    waitingSince: minutes => (minutes < 1 ? 'agora mesmo' : `${minutes}min atrás`),
    restartNeeded: 'Salvo. Reinicie o gateway de mensagens para que as novas configurações entrem em vigor.',
    restartNow: 'Reiniciar agora',
    restarting: 'Reiniciando…',
    restartFailedManual: 'O Hermes não conseguiu reiniciar para aplicar suas configurações de mensagens',
    restartFailedManualDetail: 'Tente reiniciar novamente; se ainda falhar, abra os logs e envie os diagnósticos.',
    restartAgain: 'Reiniciar novamente',
    openLogs: 'Abrir logs',
    telegramQr: {
      title: 'Escolha como conectar seu bot do Telegram',
      subtitle:
        'As duas opções conectam um bot que você controla e salvam as credenciais dele somente nesta instalação do Hermes.',
      quickSetup: 'Configuração rápida',
      recommended: 'Recomendado',
      quickHelp:
        'Escaneie um código QR e confirme no Telegram. O Hermes cria o bot e detecta automaticamente seu ID de usuário do Telegram.',
      createWithQr: 'Criar com QR',
      starting: 'Iniciando…',
      replaceWarning:
        'As credenciais do Telegram já estão configuradas. Uma nova configuração por QR ou um novo token de bot substituirá o bot atual quando você salvar.',
      scanHint: 'Escaneie com o aplicativo do Telegram no seu celular ou abra o link neste computador.',
      waiting: 'Aguardando o Telegram…',
      expiresIn: remaining => `Expira em ${remaining}`,
      expired: 'Expirado',
      openTelegram: 'Abrir o Telegram',
      ready: 'Bot criado',
      allowedUsers: 'Usuários permitidos',
      ownerDetected: 'Proprietário detectado',
      addAtLeastOne: 'Adicione pelo menos um ID de usuário do Telegram.',
      userIdPlaceholder: 'ID de usuário do Telegram',
      add: 'Adicionar',
      numericOnly: 'Os IDs de usuário permitidos do Telegram devem ser numéricos.',
      saveAndRestart: 'Salvar e reiniciar',
      applying: 'Salvando…',
      pairingExpired: 'O pareamento com o Telegram expirou. Inicie uma nova configuração por QR para tentar novamente.',
      stillWaiting: detail => `Ainda aguardando o Telegram. Nova tentativa após: ${detail}`,
      savedRestarting: 'Telegram salvo; reiniciando o gateway…',
      savedRestartFailed: detail => `Telegram salvo; falha ao reiniciar o gateway${detail}`
    },
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
      TELEGRAM_PROXY: {
        label: 'URL do proxy',
        help: 'Necessário apenas em redes onde o Telegram está bloqueado.'
      },
      DISCORD_BOT_TOKEN: {
        label: 'Token do bot',
        help: 'Crie um aplicativo no Discord Developer Portal, adicione um bot e cole o token dele.'
      },
      DISCORD_ALLOWED_USERS: {
        label: 'IDs de usuário do Discord permitidos',
        help: 'Recomendado. IDs de usuário do Discord separados por vírgula.'
      },
      DISCORD_REPLY_TO_MODE: {
        label: 'Estilo de resposta',
        help: 'first, all ou off.'
      },
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
      MATTERMOST_ALLOW_ALL_USERS: {
        label: 'Permitir todos os usuários do Mattermost'
      },
      MATTERMOST_HOME_CHANNEL: {
        label: 'Canal principal'
      },
      QQ_ALLOW_ALL_USERS: {
        label: 'Permitir todos os usuários do QQ'
      },
      QQBOT_HOME_CHANNEL: {
        label: 'Canal principal do QQ',
        help: 'Canal ou grupo padrão para entrega do cron.'
      },
      QQBOT_HOME_CHANNEL_NAME: {
        label: 'Nome do canal principal do QQ'
      },
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
      MATTERMOST_URL: {
        label: 'URL do servidor',
        placeholder: 'https://mattermost.example.com'
      },
      MATTERMOST_TOKEN: {
        label: 'Token do bot'
      },
      MATTERMOST_ALLOWED_USERS: {
        label: 'IDs de usuário permitidos',
        help: 'Recomendado. IDs de usuário do Mattermost separados por vírgula.'
      },
      MATRIX_HOMESERVER: {
        label: 'URL do homeserver',
        placeholder: 'https://matrix.org'
      },
      MATRIX_ACCESS_TOKEN: {
        label: 'Token de acesso'
      },
      MATRIX_USER_ID: {
        label: 'ID de usuário do bot',
        placeholder: '@hermes:example.org'
      },
      MATRIX_ALLOWED_USERS: {
        label: 'IDs de usuário do Matrix permitidos',
        help: 'Recomendado. IDs de usuário separados por vírgula, no formato @usuario:servidor.'
      },
      SIGNAL_HTTP_URL: {
        label: 'URL da ponte do Signal',
        placeholder: 'http://127.0.0.1:8080',
        help: 'URL de uma ponte REST do signal-cli em execução.'
      },
      SIGNAL_ACCOUNT: {
        label: 'Número de telefone',
        help: 'O número registrado na sua ponte do signal-cli.'
      },
      SIGNAL_ALLOWED_USERS: {
        label: 'Usuários do Signal permitidos',
        help: 'Recomendado. Identificadores do Signal separados por vírgula.'
      },
      WHATSAPP_ENABLED: {
        label: 'Ativar a ponte do WhatsApp',
        help: 'Definido automaticamente pelo botão abaixo. Não mexa a menos que saiba que precisa.'
      },
      WHATSAPP_MODE: {
        label: 'Modo da ponte'
      },
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
    subscriptions: count => `Assinaturas (${count})`,
    hint: 'As alterações nas assinaturas são recarregadas automaticamente assim que o receptor estiver em execução. Assinaturas desativadas rejeitam eventos recebidos.',
    empty: 'Nenhuma assinatura de webhook ainda.',
    disabledTitle: 'Receptor de webhooks desativado',
    disabledBody:
      'Os webhooks são uma plataforma própria do gateway. Ative-os aqui para aceitar eventos HTTP recebidos; canais de conversa só são necessários quando uma assinatura entrega mensagens no Telegram, Discord, Slack ou outro canal.',
    enable: 'Ativar webhooks',
    enabling: 'Ativando...',
    enabled: name => `Ativado: "${name}"`,
    disabled: name => `Desativado: "${name}"`,
    enableRow: 'Ativar',
    disableRow: 'Desativar',
    delete: 'Excluir',
    deleting: 'Excluindo...',
    deleted: 'Webhook excluído',
    deleteTitle: 'Excluir webhook',
    deleteDescPrefix: 'Isso vai remover permanentemente ',
    deleteDescSuffix: '. Isso não pode ser desfeito.',
    deleteFailed: name => `Falha ao excluir "${name}"`,
    toggleFailed: (name, enabled) => `Falha ao ${enabled ? 'ativar' : 'desativar'} "${name}"`,
    newSubscription: 'Nova assinatura',
    restarting: 'Reiniciando o gateway...',
    restartNeeded: 'Os webhooks estão ativados, mas o gateway ainda precisa reiniciar para o receptor entrar no ar.',
    restartGateway: 'Reiniciar o gateway',
    restartingGateway: 'Reiniciando...',
    restartFailed: detail => `Falha ao reiniciar o gateway${detail}`,
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
    createFailed: detail => `Falha ao criar: ${detail}`,
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
    switchToConnection: name => `Trocar para ${name}`,
    switchConnectionFailed: name => `Não foi possível conectar a ${name}`,
    manageProfiles: 'Gerenciar perfis…',
    connectGateway: 'Gerenciar gateways…',
    fleet: {
      allOnGateway: 'Todos os perfis neste gateway',
      gateway: gateway => `Perfis em ${gateway}`,
      gatewayUnreachable: gateway => `${gateway} · inacessível`,
      onGateway: (name, gateway) => `${name} · ${gateway}`,
      switchTo: (name, gateway) => `Alternar para ${name} em ${gateway}`,
      deleteOn: gateway => ` em ${gateway}`
    },
    remoteOverride: {
      menuItem: 'Conectar a um host remoto…',
      badge: host => `Executa em ${host}`,
      title: profile => `Conectar ${profile} a um host remoto`,
      description:
        'As sessões neste perfil serão executadas no Hermes remoto para o qual você apontar, em vez deste computador.',
      urlLabel: 'Endereço remoto',
      urlPlaceholder: 'https://hermes.example.com',
      urlInvalid: 'Insira um endereço completo começando com http:// ou https://',
      tokenLabel: 'Token de acesso',
      tokenPlaceholder: 'Cole o token de sessão remota',
      tokenSavedHint: 'Um token já está salvo. Deixe em branco para mantê-lo.',
      plainTextOptIn:
        'Este computador não possui armazenamento seguro de chaves, portanto o token será salvo sem criptografia no disco. Salvar mesmo assim.',
      collisionWarning: label =>
        `Um gateway chamado “${label}” já existe nas Configurações. A conexão deste perfil é separada e não irá alterá-lo.`,
      confirmTitle: 'Conectar este perfil a um host remoto?',
      confirmNote: (profile, host) =>
        `Novos chats em ${profile} serão executados em ${host}. Aquele computador executará comandos e lerá arquivos lá, não neste. Conecte-se apenas a um host em que você confia.`,
      confirmBack: 'Voltar',
      connect: 'Conectar',
      connecting: 'Conectando…',
      disconnect: 'Remover conexão remota',
      savedTitle: 'Perfil conectado',
      savedMessage: (profile, host) => `${profile} agora executa em ${host}`,
      removedTitle: 'Conexão remota removida',
      removedMessage: profile => `${profile} agora executa neste computador`,
      removeFailed: 'Não foi possível remover a conexão remota',
      authFailedTitle: 'O host remoto recusou o token salvo',
      authFailedMessage: (profile, host) =>
        `${host} recusou o token salvo para ${profile}. Ele pode ter sido alterado no lado remoto.`,
      updateToken: 'Inserir novo token…'
    },
    actions: 'Ações',
    color: 'Cor…',
    colorFor: 'Cor',
    openInNewWindow: 'Abrir em uma nova janela',
    setAsDefault: 'Definir como padrão',
    defaultProfile: 'Perfil padrão',
    defaultSet: name => `${name} agora é o perfil padrão`,
    defaultDescription:
      'Usado quando o Hermes é aberto e em novos chats. As sessões existentes permanecem em seus perfis.',
    failedSetDefault: 'Não foi possível definir o perfil padrão',
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
    exportMenu: 'Exportar…',
    editSoul: 'Editar SOUL.md…',
    copySetup: 'Copiar configuração',
    copying: 'Copiando...',
    modelLabel: 'Modelo',
    skillsLabel: 'Habilidades',
    notSet: 'Não definido',
    soulDesc: 'O prompt de sistema e as instruções de persona incorporados a este perfil.',
    soulOptional: 'opcional',
    soulPlaceholder: mode => `O prompt de sistema / persona deste perfil.
Deixe em branco para manter o padrão ${mode}.`,
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
    displayNameTitle: 'Nomear este agente',
    displayNameDesc: 'Define um nome exibido em todo o aplicativo. O ID interno do perfil continua sendo “default”.',
    displayNameLabel: 'Nome exibido',
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
    failedRename: 'Falha ao renomear o perfil'
  },
  modelAssignment: {
    saveFailed: 'O Hermes não salvou essa alteração de modelo.',
    confirmTitle: 'Aviso sobre a seleção do modelo',
    confirmDetail: 'Confirme somente se você aceitar essa condição.',
    confirmAction: 'Confirmar',
    declined: 'Alteração de modelo cancelada — você recusou o aviso sobre o nível de treinamento com dados.'
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
    lastRunFailed: 'A última execução falhou:',
    editJob: 'Editar tarefa',
    runAgain: 'Executar novamente',
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
    overdueSince: 'Atrasado desde:',
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
        '*': {
          fields: {
            time: {
              label: 'Que horas?',
              help: 'Hora local em formato 24h, ex.: 08:00'
            },
            deliver: {
              label: 'Onde entregar?'
            },
            recurrence: {
              label: 'Repetir em',
              options: {
                everyday: 'Todos os dias',
                weekdays: 'Dias úteis',
                weekends: 'Fins de semana'
              }
            },
            day: {
              label: 'Qual dia?',
              options: {
                sunday: 'Domingo',
                monday: 'Segunda-feira',
                tuesday: 'Terça-feira',
                wednesday: 'Quarta-feira',
                thursday: 'Quinta-feira',
                friday: 'Sexta-feira',
                saturday: 'Sábado'
              }
            }
          }
        },
        'morning-brief': {
          title: 'Briefing matinal',
          description: 'Um breve resumo diário: agenda de hoje, clima e itens urgentes aguardando você.'
        },
        'important-mail': {
          title: 'Monitor de e-mails importantes',
          description:
            'Verifica sua caixa de entrada periodicamente e notifica APENAS sobre e-mails que realmente precisam de atenção.',
          fields: {
            interval_min: {
              label: 'Com que frequência?',
              help: 'Minutos entre as verificações'
            },
            criteria: {
              label: 'Só me avise se o e-mail…',
              default: 'precisar de resposta hoje, vier do meu gestor ou da minha família, ou mencionar um prazo'
            }
          }
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
          description: 'Um lembrete agendado flexível com seu texto personalizado.',
          fields: {
            what: {
              label: 'Lembrar-me de…',
              default: 'fazer uma pausa e me alongar'
            }
          }
        },
        'evening-winddown': {
          title: 'Desaceleração noturna',
          description: 'Reflita sobre as realizações de hoje e prepare-se para o amanhã.'
        },
        'news-digest': {
          title: 'Resumo de notícias por tópico',
          description: 'Resumo de notícias, pesquisas ou atualizações sobre um tópico de interesse.',
          fields: {
            topic: {
              label: 'Qual tópico?',
              default: 'IA e tecnologia',
              help: 'Um assunto, produto, pessoa ou expressão de busca'
            },
            count: {
              label: 'Quantos itens?'
            }
          }
        },
        'bill-renewal-watch': {
          title: 'Lembrete de contas e renovações',
          description: 'Acompanhe contas a vencer, assinaturas e renovações de serviços.',
          fields: {
            what: {
              label: 'O que está para vencer?',
              default: 'minha assinatura de streaming será renovada em breve'
            }
          }
        },
        'price-watch': {
          title: 'Monitoramento de preço e disponibilidade',
          description: 'Monitore o preço de um produto ou anúncio e alerte quando mudar.',
          fields: {
            item: {
              label: 'O que exatamente deve ser monitorado?',
              default: 'uma URL de produto ou a descrição exata de um voo, hotel ou anúncio',
              help: 'URL ou descrição precisa — variante, datas e vendedor'
            },
            condition: {
              label: 'Avise-me quando…',
              default: 'o preço total ficar abaixo da minha meta',
              help: 'Preço-limite com moeda, disponibilidade ou mudança nos termos'
            },
            interval_h: {
              label: 'Com que frequência?',
              help: 'Horas entre verificações — respeite os limites de requisições'
            }
          }
        },
        'competitor-watch': {
          title: 'Monitoramento de notícias de concorrentes',
          description: 'Acompanhe concorrentes específicos para notícias de produtos ou empresas.',
          fields: {
            companies: {
              label: 'Quais empresas?',
              default: 'dois ou três concorrentes, pelo nome oficial',
              help: 'Nomes oficiais e domínios; aliases ajudam a remover duplicatas'
            },
            categories: {
              label: 'Quais eventos importam?',
              default:
                'lançamentos de produtos, mudanças de preço, captação, parcerias, mudanças executivas e incidentes'
            }
          }
        },
        'habit-checkin': {
          title: 'Acompanhamento de hábitos',
          description: 'Acompanhe o progresso diário em hábitos e metas pessoais.',
          fields: {
            habit: {
              label: 'Qual hábito?',
              default: '20 minutos de leitura'
            }
          }
        },
        'hydration-move': {
          title: 'Lembrete de hidratação e movimento',
          description: 'Lembretes amigáveis para manter-se hidratado e fazer pausas para se mover.',
          fields: {
            interval_hours: {
              label: 'Com que frequência?',
              help: 'Horas entre os lembretes'
            },
            start_hour: {
              label: 'Hora de início',
              help: 'Primeira hora da janela ativa, no formato de 24 horas'
            },
            end_hour: {
              label: 'Hora de término',
              help: 'Última hora da janela ativa, no formato de 24 horas'
            }
          }
        },
        'meal-plan': {
          title: 'Plano alimentar semanal',
          description: 'Gere um plano alimentar semanal e uma lista de compras consolidada.',
          fields: {
            diet: {
              label: 'Dieta?',
              options: {
                'no restrictions': 'Sem restrições',
                vegetarian: 'Vegetariana',
                vegan: 'Vegana',
                'high-protein': 'Rica em proteínas',
                'low-carb': 'Baixo teor de carboidratos'
              }
            },
            meals: {
              label: 'Refeições por dia?',
              options: {
                'dinner only': 'Somente jantar',
                'lunch and dinner': 'Almoço e jantar',
                'all three': 'As três refeições'
              }
            },
            effort: {
              label: 'Nível de preparo?',
              options: {
                quick: 'Rápido',
                medium: 'Médio',
                ambitious: 'Elaborado'
              }
            }
          }
        },
        'learn-daily': {
          title: 'Pílula diária de aprendizado',
          description: 'Receba pequenas pílulas de conhecimento diárias sobre um tópico que você está aprendendo.',
          fields: {
            topic: {
              label: 'Aprender sobre…',
              default: 'vocabulário em espanhol'
            }
          }
        },
        'gratitude-journal': {
          title: 'Prompt de gratidão e reflexão',
          description: 'Prompt diário para gratidão e reflexão consciente.'
        },
        'on-this-day': {
          title: 'Descoberta ‘neste dia na história’',
          description: 'Descubra eventos interessantes ou memórias pessoais deste dia na história.',
          fields: {
            flavor: {
              label: 'Que tipo?',
              options: {
                'on this day in history': 'Neste dia na história',
                'word of the day': 'Palavra do dia',
                'science fact': 'Fato científico',
                'quote of the day': 'Citação do dia'
              }
            }
          }
        }
      },
      emptyTitle: 'Nenhum modelo de automação disponível',
      emptyDesc: 'Nenhum modelo de automação está disponível neste backend.'
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
    kind: {
      code: 'Código',
      html: 'Página interativa',
      svg: 'Gráfico'
    },
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
    filter: {
      grouping: 'Agrupamento',
      ordering: 'Ordenação',
      show: 'Mostrar',
      filters: 'Filtros',
      status: 'Status',
      pullRequest: 'Pull request',
      profile: 'Perfil',
      project: 'Projeto',
      archived: 'Arquivadas',
      resetToDefaults: 'Restaurar padrões',
      expandAll: 'Expandir tudo',
      collapseAll: 'Recolher tudo',
      inboxStyle: 'Estilo caixa de entrada',
      updated: 'Atualização',
      created: 'Criação',
      tokens: 'Tokens',
      cost: 'Custo',
      manual: 'Manual',
      preview: 'Prévia',
      pr: 'PR',
      needsInput: 'Aguardando resposta',
      working: 'Em andamento',
      unread: 'Não lidas',
      draft: 'Rascunho',
      idle: 'Inativa',
      open: 'Aberta',
      merged: 'Mesclada',
      closed: 'Fechada',
      noPR: 'Sem PR'
    },
    gatewayGroups: {
      grouping: 'Gateway e perfil',
      rename: 'Renomear grupo',
      aliasLabel: 'Nome de exibição',
      aliasHint: 'Somente o nome de exibição; os nomes do gateway e do perfil permanecem inalterados.',
      resetName: 'Redefinir nome',
      moveUp: 'Mover para cima',
      moveDown: 'Mover para baixo',
      reorder: 'Reordenar grupo',
      actions: 'Ações do grupo'
    },
    profileRail: 'Barra de perfis',
    nav: {
      'new-session': 'Nova sessão',
      capabilities: 'Recursos',
      messaging: 'Mensagens',
      artifacts: 'Artefatos',
      cron: 'Tarefas agendadas'
    },
    searchAria: 'Buscar sessões',
    searchPlaceholder: 'Buscar sessões…',
    clearSearch: 'Limpar a busca',
    noMatch: query => `Nenhuma sessão corresponde a “${query}”.`,
    results: 'Resultados',
    pinned: 'Fixadas',
    sessions: 'Sessões',
    terminal: 'Terminal',
    files: 'Arquivos',
    review: 'Revisão',
    logs: 'Logs',
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
    projectLoadFailed: 'Não foi possível carregar as sessões',
    noSessions: 'Nenhuma sessão ainda',
    storageCorrupt: {
      title: 'O banco de dados de sessões está danificado',
      body: profiles =>
        `O Hermes não consegue ler todo o histórico de sessões de ${profiles}. Os chats ausentes nesta lista não foram excluídos; o arquivo em que estão armazenados está danificado.`,
      action: 'Encerre o Hermes neste perfil e inspecione o arquivo sem alterá-lo ou restaure um snapshot:',
      guide: 'Guia de recuperação'
    },
    noFilterMatches: 'Nenhuma sessão corresponde a estes filtros',
    projects: {
      showAllSessions: 'Mostrar todas as sessões',
      sectionLabel: 'Projetos',
      home: 'Início',
      autoDiscovered: 'Detectado automaticamente',
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
      worktreeStaleBackend:
        'Atualize o backend Hermes para criar worktrees nesta conexão remota — ele é anterior à API de worktrees do Git.',
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
      showAllCount: count => `Mostrar todas as ${count} sessões`,
      back: 'Todos os projetos'
    },
    newSessionIn: label => `Nova sessão em ${label}`,
    showMoreIn: (count, label) => `Mostrar mais ${count} em ${label}`,
    loading: 'Carregando…',
    loadMore: 'Carregar mais',
    loadCount: step => `Carregar mais ${step}`,
    messageCount: count => `${count} ${count === 1 ? 'mensagem' : 'mensagens'}`,
    toolCallCount: count => `${count} ${count === 1 ? 'chamada de ferramenta' : 'chamadas de ferramenta'}`,
    row: {
      pin: 'Fixar',
      unpin: 'Desafixar',
      markUnread: 'Marcar como não lida',
      markRead: 'Marcar como lida',
      unreadFailed: 'Não foi possível atualizar o estado de leitura',
      copyId: 'Copiar ID',
      export: 'Exportar',
      branchFrom: 'Ramificar',
      rename: 'Renomear',
      archive: 'Arquivar',
      newWindow: 'Nova janela',
      openInTerminal: 'Abrir no terminal',
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
      deleteTitle: 'Excluir sessão?',
      deleteDesc: title => `Isso excluirá permanentemente “${title}”. Não é possível desfazer.`,
      deleting: 'Excluindo…',
      deleted: 'Sessão excluída',
      untitledChat: id => `chat ${id}`,
      messageCount: count => `${count} ${count === 1 ? 'mensagem' : 'mensagens'}`,
      todoProgress: '',
      ageNow: 'agora',
      ageDay: 'd',
      ageHour: 'h',
      ageMin: 'min'
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
    },
    markAllRead: 'Marcar todas como lidas'
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
    voiceControls: 'Voz',
    voiceEngine: 'Mecanismo de chat por voz',
    voiceEngineChained: 'Conversão de fala em texto + voz do Hermes',
    voiceEngineLive: 'GPT-Live (full-duplex, delega para o Hermes)',
    voiceEngineLiveNeedsKey: 'Requer uma chave de API da OpenAI',
    voiceEngineChangeFailed: 'Não foi possível alterar o mecanismo de chat por voz',
    voiceEngineChainedShort: 'conversão de fala em texto',
    voiceEngineLiveShort: 'GPT-Live',
    voiceDictation: 'Ditado por voz',
    speakReplies: 'Ler as respostas em voz alta',
    stopSpeakingReplies: 'Parar de ler as respostas em voz alta',
    wakeWord: phrase => `Palavra de ativação "${phrase}"`,
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
      '/quit': 'sair do Hermes',
      '/start': 'Confirma os sinais de início da plataforma sem responder',
      '/new': 'Inicia um novo chat no desktop',
      '/topic': 'Ativa ou inspeciona sessões por tópico em mensagens diretas do Telegram',
      '/save': 'Salva a transcrição atual em JSON',
      '/retry': 'Tenta novamente a última mensagem (reenvia ao agente)',
      '/prompt': 'Escreve seu próximo prompt no $EDITOR (markdown) e depois o envia',
      '/undo': 'Volta N turnos do usuário e envia um novo prompt (padrão: 1)',
      '/title': 'Renomeia a sessão atual',
      '/handoff': 'Transfere esta sessão para uma plataforma de mensagens',
      '/branch': 'Cria um novo chat a partir da mensagem mais recente',
      '/worktree': 'Exibe, lista, cria ou remove worktrees isoladas do git',
      '/compress': 'Compacta o contexto desta conversa',
      '/rollback':
        'Lista ou restaura checkpoints do sistema de arquivos (a restauração mantém suas edições manuais; --all substitui isso)',
      '/export': 'Exporta um perfil (configurações, skills e tema) para um arquivo compartilhável',
      '/import': 'Importa um arquivo de perfil compartilhado como um novo perfil',
      '/stop': 'Interrompe o turno ativo e os processos em segundo plano',
      '/pause': "Pausa novos trabalhos globalmente (parada de emergência); '/pause off' retoma",
      '/bg': 'Executa um prompt em uma sessão separada em segundo plano',
      '/btw': 'Faz uma pergunta paralela sobre esta conversa sem interrompê-la',
      '/agents': 'Exibe os agentes ativos e as tarefas em execução',
      '/journey': 'Abre o grafo de memória — skills e memórias ao longo do tempo',
      '/queue': 'Coloca um prompt na fila para o próximo turno ou usa list/edit/rm/move/clear nos prompts na fila',
      '/steer': 'Insere uma mensagem após a próxima chamada de ferramenta sem interromper',
      '/goal': 'Define um objetivo contínuo no qual o Hermes trabalha ao longo dos turnos até alcançá-lo',
      '/heartbeat': 'Define um prompt recorrente que retorna a esta sessão quando ela está ociosa',
      '/refine': 'Revisa esta conversa agora e salva as lições na memória/nas skills',
      '/review': 'Inicia um subagente independente para revisar o trabalho recém-discutido (PR, código, documentação)',
      '/loop': 'Executa novamente um prompt em intervalos recorrentes nesta sessão',
      '/plan': 'Escreve um plano de implementação em markdown em .hermes/plans/ sem executar nada',
      '/moa': 'Executa um prompt com a predefinição padrão Mixture of Agents e depois restaura seu modelo',
      '/subgoal': 'Adiciona ou gerencia critérios extras no objetivo ativo',
      '/status': 'Exibe o status atual da sessão',
      '/egress': 'Exibe o status do proxy de saída do Docker',
      '/context':
        'Exibe uma visão detalhada da janela de contexto, com medidor de uso, divisão por categoria, estatísticas de compactação e taxa de transferência',
      '/whoami': 'Exibe seu nível de acesso a comandos de barra (admin / usuário)',
      '/profile': 'Troca o perfil ativo do Hermes',
      '/codex-runtime': 'Ativa ou desativa o runtime do codex app-server para modelos OpenAI/Codex',
      '/personality': 'Define uma personalidade predefinida',
      '/battery': 'Ativa ou desativa um indicador de bateria colorido na barra de status',
      '/timestamps': 'Ativa ou desativa horários [HH:MM] nas mensagens e em /history',
      '/diff': 'Exibe as alterações do git no diretório de trabalho',
      '/focus': 'Ativa ou desativa a visualização de foco — mostra apenas seu prompt e a resposta final',
      '/yolo': 'Ativa ou desativa o YOLO — aprova automaticamente comandos perigosos',
      '/approvals': 'Exibe ou define o modo persistente de aprovação de comandos perigosos',
      '/reasoning': 'Esforço ou exibição de raciocínio [<level> [--global]|show|hide|full|clamp]',
      '/skin': 'Troca o tema do desktop ou avança para o próximo',
      '/wake': 'Controla o detector de palavra de ativação do desktop [on|off|status]',
      '/tools': 'Gerencia ferramentas: /tools [list|disable|enable] [name...]',
      '/memory': 'Revisa gravações de memória pendentes / ativa ou desativa a etapa de aprovação',
      '/bundles': 'Lista pacotes de skills (aliases /<name> para várias skills)',
      '/pet': 'Ativa, desativa ou adota um mascote do petdex (/pet, /pet list, /pet boba)',
      '/hatch': 'Gera um novo pet (abre o gerador de pets)',
      '/learn':
        'Aprende uma skill reutilizável a partir de qualquer coisa que você descrever (dirs, URLs, este chat, anotações)',
      '/init': 'Gera ou atualiza as instruções do projeto em AGENTS.md a partir da análise de um repositório',
      '/suggestions': 'Revisa automações sugeridas (accept/dismiss)',
      '/blueprint': 'Configura uma automação a partir de um modelo de blueprint',
      '/browser': 'Gerencia a conexão CDP do navegador [connect|disconnect|status] (somente gateway local)',
      '/palette': 'Abre a paleta de comandos com busca aproximada (também Ctrl+P)',
      '/usage':
        'Exibe o uso de tokens e os limites de taxa; `reset` resgata uma redefinição acumulada do limite do Codex',
      '/subscription': 'Exibe seu plano da Nous e permite alterá-lo no navegador',
      '/topup': 'Exibe seu saldo da Nous e permite gerenciar a cobrança no portal',
      '/platform': 'Pausa, retoma ou lista uma plataforma de gateway com falha',
      '/version': 'Exibe a versão do Hermes Agent',
      '/debug': 'Envia um relatório de depuração (informações do sistema + logs) e gera links compartilháveis',
      '/model': 'Troca o modelo desta sessão'
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
    urlPlaceholder: 'https://example.com/post',
    urlHintPre: 'Inclua a URL completa, por exemplo ',
    attach: 'Anexar',
    queued: count => `${count} na fila`,
    queuedPaused: count => `${count} na fila — pausados`,
    attachmentOnly: 'Turno só com anexo',
    emptyTurn: 'Turno vazio',
    hiddenQueued: 'Nota de configuração',
    attachments: count => `${count} anexo${count === 1 ? '' : 's'}`,
    editingInComposer: 'Editando no compositor',
    editingQueuedInComposer: 'Editando o turno da fila no compositor',
    restoredDraftNotice: 'Sua mensagem não enviada foi restaurada',
    restoredDraftUndo: 'Desfazer',
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
    githubSuggestions: {
      label: 'Configurar GitHub',
      tip: 'O GitHub funciona por meio das skills da CLI gh — clique para conectar sua conta',
      done: 'Adicionado /github-auth',
      doneTip: 'Envie a mensagem e o agente orientará você no login do GitHub'
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
    hideStack: 'Ocultar painel de status',
    showStack: 'Mostrar painel de status',
    agents: 'agentes',
    background: count => `${count} em segundo plano`,
    goalActive: 'Objetivo ativo',
    goalBlocked: 'Objetivo bloqueado',
    goalDone: 'Objetivo concluído',
    goalPaused: 'Objetivo pausado',
    goalWaiting: 'Objetivo em espera',
    subagents: count => `${count} subagente${count === 1 ? '' : 's'}`,
    todos: (done, total) => `Tarefas ${done}/${total}`,
    running: 'Executando',
    stop: 'Parar',
    dismiss: 'Dispensar',
    exit: code => `saída ${code}`,
    control: {
      goalActiveTurns: (turn, maxTurns) => `Turno ${turn}/${maxTurns}`,
      goalDoneTurns: turns => `${turns} turno${turns === 1 ? '' : 's'}`,
      goalTurn: turn => `Turno ${turn}`,
      goalActions: 'Ações do objetivo',
      viewDetails: 'Ver detalhes',
      addCriterion: 'Adicionar critério',
      addCriterionDialogTitle: 'Adicionar critério',
      addCriterionPlaceholder: 'Digite o texto do critério...',
      criterionLabel: 'Critério',
      pauseGoal: 'Pausar objetivo',
      resumeGoal: 'Retomar objetivo',
      resumeNow: 'Retomar agora',
      clearGoal: 'Limpar objetivo',
      clearGoalConfirmTitle: 'Limpar objetivo?',
      clearGoalConfirmBody: 'Tem certeza de que deseja limpar o objetivo ativo? Esta ação não pode ser desfeita.',
      copyCriterion: index => `Copiar critério ${index}`,
      removeCriterion: index => `Remover critério ${index}`,
      removeCriterionConfirmTitle: index => `Remover critério ${index}?`,
      removeCriterionConfirmBody: index => `Tem certeza de que deseja remover o critério ${index}?`,
      clearCriteria: 'Limpar todos os critérios',
      clearCriteriaConfirmTitle: 'Limpar todos os critérios?',
      clearCriteriaConfirmBody: 'Tem certeza de que deseja remover todos os critérios deste objetivo?',
      criteriaHeader: count => `Critérios · ${count}`,
      noCriteria: 'Nenhum critério',
      goalDetailsTitle: 'Detalhes do objetivo',
      objectiveLabel: 'Objetivo',
      contractOutcome: 'Resultado',
      contractVerification: 'Verificação',
      contractConstraints: 'Restrições',
      contractBoundaries: 'Limites',
      contractStopWhen: 'Parar quando',
      waitBarrierTitle: 'Condição de espera',
      waitUntil: target => `Aguardando até ${target}`,
      waitSession: target => `Aguardando a sessão ${target}`,
      waitPid: pid => `Aguardando o processo ${pid}`,
      qualityGatesTitle: 'Critérios de qualidade',
      gateCommand: 'Comando',
      gateAttempts: (attempts, max) => `${attempts}/${max} tentativas`,
      gateTimeout: seconds => `${seconds}s de tempo limite`,
      gateLastExit: code => (code === null ? 'Pendente' : `Código de saída: ${code}`),
      loopActive: 'Loop ativo',
      loopPaused: 'Loop pausado',
      loopDeferred: 'Loop adiado',
      loopFinished: 'Loop concluído',
      loopRuns: runs => `${runs} rodada${runs === 1 ? '' : 's'}`,
      loopRunCount: (current, total) => `Execução ${current}/${total}`,
      loopNext: time => `próxima ${time}`,
      loopEverySeconds: seconds => `a cada ${seconds}s`,
      loopEveryMinutes: minutes => `a cada ${minutes}min`,
      loopEveryHours: hours => `a cada ${hours}h`,
      loopSelfPaced: 'ritmo próprio',
      loopActions: 'Ações do loop',
      pauseLoop: 'Pausar loop',
      resumeLoop: 'Retomar loop',
      stopLoop: 'Interromper loop',
      stopLoopConfirmTitle: 'Interromper loop?',
      stopLoopConfirmBody: 'Tem certeza de que deseja interromper este loop?',
      dismissLoop: 'Dispensar loop',
      loopPromptLabel: 'Prompt',
      loopCadenceLabel: 'Cadência',
      loopUntilLabel: 'Condição de término',
      loopDeferredNotice: 'Um objetivo ativo controla a sessão no momento.',
      loopAwaitingResponse: 'Aguardando resposta',
      heartbeatActive: 'Heartbeat ativo',
      heartbeatPaused: 'Heartbeat pausado',
      heartbeatEveryMinutes: minutes => `a cada ${minutes} min`,
      heartbeatEveryHours: hours => `a cada ${hours} h`,
      heartbeatEverySeconds: seconds => `a cada ${seconds} s`,
      heartbeatNext: time => `próximo: ${time}`,
      heartbeatDueWaitingForIdle: 'programado — aguardando ficar ocioso',
      heartbeatActions: 'Ações do heartbeat',
      pauseHeartbeat: 'Pausar heartbeat',
      resumeHeartbeat: 'Retomar heartbeat',
      clearHeartbeat: 'Limpar heartbeat',
      clearHeartbeatConfirmTitle: 'Limpar heartbeat?',
      clearHeartbeatConfirmBody: 'Tem certeza de que deseja limpar este heartbeat?',
      heartbeatFiredCount: count => `Disparado ${count} vez${count === 1 ? '' : 'es'}`,
      actionFailed: msg => `Falha na ação: ${msg}`,
      actionSucceeded: 'Ação concluída',
      copySuccess: 'Critério copiado para a área de transferência',
      copyFailure: 'Falha ao copiar o critério para a área de transferência',
      continuationFailed: 'Falha ao enviar a continuação do objetivo',
      continuationQueued: 'Objetivo retomado — continuação na fila até o turno atual terminar',
      continuationBusy: 'Objetivo retomado — sessão ocupada; use /interrupt no turno atual para continuar',
      controlUnavailable: msg => `Controles da sessão indisponíveis: ${msg}`,
      dismissError: 'Dispensar erro',
      add: 'Adicionar'
    },
    coding: {
      title: 'Árvore de trabalho',
      noBranch: 'Nenhuma branch',
      detached: 'desanexado',
      clean: 'Limpo',
      changed: count => `${count} arquivo${count === 1 ? '' : 's'} alterado${count === 1 ? '' : 's'}`,
      ahead: count => `${count} à frente`,
      behind: count => `${count} atrás`,
      review: 'Revisar',
      close: 'Fechar',
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
      agentShipUnavailable: 'O chat que contém essas alterações não está na tela.',
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
    connectionRetry:
      'O Hermes não conseguiu acessar o servidor de atualizações. Verifique sua conexão com a internet e tente novamente. Se você usa um Hermes remoto, confirme que ele está online.',
    gitUnusable: 'O Hermes não conseguiu executar o Git neste computador, então não pôde verificar se há atualizações.',
    connectionSettings: 'Configurações de conexão',
    openDownloadPage: 'Abrir página de download',
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
    technicalDetails: 'Detalhes técnicos',
    notNow: 'Agora não',
    clientAlsoBehindTitle: 'O app de desktop está desatualizado',
    clientAlsoBehindMessage:
      'O backend está atualizado, mas este app de desktop ainda usa uma versão antiga. Atualize-o para obter as correções mais recentes.',
    clientAlsoBehindAction: 'Atualizar o app de desktop',
    everythingDispatched: 'Atualização enviada',
    everythingSkipped: 'Ignorada',
    everythingRowFailed: 'Falha na atualização',
    everythingFanoutFailedTitle: 'Não foi possível atualizar outras instâncias',
    changeLogNew: 'Novidades',
    changeLogFixed: 'Correções',
    changeLogFaster: 'Mais rápido',
    changeLogImproved: 'Melhorias',
    changeLogOther: 'Outras melhorias',
    changeLogFallbackLabel: 'Nesta atualização',
    changeLogFallbackItem: 'Melhorias e correções',
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
  handoffTour: {
    profileTitle: 'Sua primeira tarefa é executada no perfil padrão',
    profileText:
      'Esta barra alterna entre perfis. O perfil destacado agora é o padrão, onde fica a sessão da tarefa. O outro é o perfil de configuração, onde fica a conversa de boas-vindas.',
    sessionsTitle: 'Cada perfil mantém suas próprias sessões',
    sessionsText:
      'Esta lista pertence ao perfil padrão. Nova sessão inicia uma sessão no perfil selecionado. Alterne os perfis na barra e a lista mudará junto.',
    stayTitle: 'O Hermes está a um clique de distância',
    stayText:
      'Alterne para o perfil de configuração e abra Boas-vindas ao Hermes sempre que quiser uma ajuda. Ele continuará lá.'
  },
  guidedGreeting: {
    line: 'Olá, pode entrar. Eu sou o Hermes. Dê-me dois minutos para deixar tudo pronto para você e então vamos trabalhar em algo que você realmente queira fazer.\n\nMas primeiro: como devo chamar você?',
    nameSuggestion: name => `(Também posso chamar você apenas de ${name}, se preferir.)`
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
    remoteUrlPlaceholder: 'https://gateway.example.com/hermes',
    probing: 'Detectando a autenticação do gateway...',
    probeError:
      'O Hermes não consegue acessar esse endereço. Verifique a URL e se o outro computador está executando o Hermes — as opções de login aparecem assim que ele responder.',
    probeErrorDetails: 'Detalhes',
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
    reloadRetry: 'recarregar e tentar novamente',
    openLogs: 'Abrir logs'
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
    localModelsTitle: 'Execute modelos localmente',
    localModelsPitch: 'Não precisa de conta — baixe um modelo e execute-o nesta máquina',
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
      openai: {
        short: 'Modelos da família GPT',
        description: 'Acesso direto aos modelos da OpenAI.'
      },
      gemini: {
        short: 'Modelos Gemini',
        description: 'Acesso direto aos modelos Google Gemini.'
      },
      xai: {
        short: 'Modelos Grok',
        description: 'Acesso direto aos modelos Grok da xAI.'
      },
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
    signInExpired:
      'O login expirou aguardando autorização. Isso normalmente significa que a página de login travou na aba aberta (problema no servidor) — conclua o login nela e tente novamente. Se continuar falhando, use uma chave de API ou a alternativa pela CLI.',
    signInDidNotFinish: provider =>
      `O login com ${provider} não foi concluído. Verifique sua conexão com a internet e tente novamente ou escolha outro provedor.`,
    tryAgain: 'Tentar novamente',
    useApiKeyInstead: 'Usar uma chave de API',
    errorDetails: 'Detalhes',
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
  freeTier: {
    providerRowTitle: 'Nous · plano gratuito',
    providerRowPitch: 'Entre com uma conta Nous para liberar mais modelos e ferramentas.',
    readyTitle: 'O Hermes está pronto.',
    readyCaption: 'Grátis · conectores incluídos',
    begin: 'Começar',
    signInInstead: 'Entrar com uma conta Nous',
    otherProviders: 'Outros provedores',
    stripTitle: 'A inferência gratuita da Nous e os conectores já estão disponíveis.',
    stripBody: 'Abra o seletor de modelos para experimentá-los ou entre com uma conta Nous.',
    openModelPicker: 'Abrir seletor de modelos',
    dismiss: 'Dispensar',
    providerName: 'Nous',
    statusLabel: model => `Nous · ${model}`,
    signIn: 'Entrar',
    signInHeading: 'Entre com uma conta Nous para liberar mais modelos e ferramentas.',
    settingUp: 'Configurando a inferência gratuita…',
    codeBody: 'Digite este código no navegador para concluir o login.',
    copyLink: 'Copiar link',
    doNotShare: 'Não compartilhe este código.',
    waiting: 'Aguardando o login…',
    finishingHeading: 'Concluindo o login…',
    finishingBody: 'Aprovado no navegador. Obtendo os tokens da sua conta.',
    signedInAs: email => `Conectado como ${email}`,
    signedIn: 'Login concluído.',
    completedBody: 'Sua conta agora inclui inferência e ferramentas.',
    defaultModel: 'Modelo padrão',
    change: 'Alterar',
    done: 'Concluído',
    notNow: 'Agora não',
    tryAgain: 'Tentar novamente',
    startAgain: 'Recomeçar',
    didNotComplete: 'O login não foi concluído',
    rejectedBody: 'Sem problema, você ainda está usando o serviço gratuito da Nous. Entre quando estiver pronto.',
    supersededBody: 'Um código de login mais recente substituiu este. Use o código mais recente ou recomece.',
    timedOutHeading: 'Esse link de login expirou',
    timedOutBody: 'Recomece quando estiver pronto. Você ainda está usando o serviço gratuito da Nous.',
    retiredBody:
      'Sua sessão terminou antes da conclusão do login. O Hermes iniciará outra; depois, entre novamente quando estiver pronto.',
    errorBody: 'O login não foi concluído. Tente novamente quando estiver pronto.',
    busyHeading: 'Quase lá',
    busyBody: wait =>
      `O Hermes não conseguiu concluir seu login porque o serviço da Nous está ocupado. Tente novamente em ${wait}. Enquanto isso, sua sessão continuará aqui.`,
    unreachableBody:
      'O Hermes não conseguiu acessar o serviço da Nous para concluir seu login. Verifique sua conexão com a internet e tente novamente. Sua sessão continuará aqui.',
    alreadySignedInHeading: 'Você já está conectado.',
    alreadySignedInBody: 'Este Hermes já está conectado a uma conta Nous.',
    setupFailed: {
      gateClosed:
        'Esta versão do Hermes não pode iniciar sem uma conta Nous. Entre ou crie uma conta: é grátis e leva apenas um minuto.',
      paused:
        'O uso do Hermes sem login está temporariamente pausado. O Hermes continuará verificando. O login é gratuito e permite começar agora mesmo.',
      rateLimited: wait =>
        `Muitas pessoas estão começando agora, então o Hermes tentará novamente em ${wait}. O login é gratuito e evita a espera.`,
      unreachable:
        'O Hermes não conseguiu acessar o serviço da Nous. Verifique sua conexão com a internet e toque em Tentar novamente. Ou conecte outro provedor por enquanto.',
      serverError:
        'O serviço da Nous teve um problema. Toque em Tentar novamente daqui a pouco ou conecte outro provedor por enquanto.',
      powRequired:
        'O servidor da Nous solicitou uma prova de trabalho, mas isso ainda não foi implementado no seu Agent. Entre ou crie uma conta Nous gratuita para continuar.',
      locked: 'Esta sessão não pode continuar sem login. Entre ou crie uma conta Nous gratuita para continuar.',
      generic:
        'O Hermes não conseguiu configurar o acesso gratuito sem login. O login é gratuito; outra opção é conectar outro provedor.',
      signInBelow: 'O login é gratuito. Escolha Nous abaixo.',
      tryAgain: 'Tentar novamente',
      retrying: 'Tentando novamente…'
    }
  },
  modelPicker: {
    title: 'Trocar de modelo',
    current: 'atual:',
    unknown: '(desconhecido)',
    search: 'Filtrar provedores e modelos...',
    noModels: 'Nenhum modelo encontrado.',
    addProvider: 'Adicionar provedor',
    loadFailed: 'Não foi possível carregar modelos',
    loadingIntoMemory: 'Carregando na memória',
    downloading: 'Baixando',
    localDownloadsHeading: 'Local',
    noAuthenticatedProviders: 'Nenhum provedor autenticado.',
    pro: 'Pro',
    proNeedsSubscription: 'Modelos Pro precisam de uma assinatura paga da Nous.',
    free: 'Gratuito',
    freeTier: 'Plano gratuito',
    priceTitle: 'Preço de entrada / saída por milhão de tokens',
    wasPrice: 'era',
    customModel: 'Modelo personalizado',
    addCustomModelAction: 'Adicionar modelo personalizado…',
    customModelPlaceholder: 'Digite um ID de modelo, por exemplo, openai/gpt-5'
  },
  modelVisibility: {
    title: 'Modelos',
    search: 'Pesquisar modelos',
    noAuthenticatedProviders: 'Nenhum provedor autenticado.',
    addProvider: 'Adicionar provedor…',
    addCustomModel: 'Adicionar modelo personalizado',
    removeCustomModel: 'Remover modelo personalizado'
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
      sendsOnRoute: level => `envia ${level} nesta rota`,
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
      resetStatusbar: '',
      toggleApprovalMode: 'Aprovações',
      toggleBackendVersion: 'Versão do backend',
      toggleCacheHitRate: 'Taxa de acerto do cache',
      toggleCommandCenter: 'Central de comandos',
      toggleContextUsage: 'Medidor de contexto',
      toggleRunningTimer: 'Cronômetro do turno',
      toggleSessionTimer: 'Cronômetro da sessão',
      toggleTerminal: 'Terminal',
      toggleTokensPerSecond: 'Tokens por segundo',
      toggleVersion: 'Versão e atualizações',
      toggleFreeTier: 'Plano gratuito',
      toggleWorkspace: 'Espaço de trabalho',
      cacheHitRateTitle:
        'Taxa de acerto do cache de prompts nesta sessão — tokens em cache custam menos, portanto valores maiores são mais econômicos',
      tokensPerSecondTitle: 'Tokens de saída por segundo, calculados pela média das últimas 10 chamadas ao modelo',
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
      systemResources: {
        title: 'Recursos do sistema',
        loading: 'Recursos…',
        gpuUtilization: 'Uso da GPU',
        gpuMemory: 'Memória da GPU',
        ram: 'RAM',
        unifiedNote: 'Memória unificada — a GPU e o sistema compartilham este conjunto.',
        toggle: 'Recursos do sistema'
      },
      contextUsagePanel: {
        categories: {
          conversation: 'conversa',
          mcp: 'MCP',
          memory: 'Memória',
          rules: 'Regras',
          skills: 'Habilidades',
          subagent_definitions: 'Especificações de subagentes',
          system_prompt: 'Prompt de sistema',
          tool_definitions: 'Especificações de ferramentas'
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
      providerModelTitle: (provider, model) => `${provider} · ${model}`
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
    showIgnored: 'Mostrar arquivos ignorados pelo Git',
    hideIgnored: 'Ocultar arquivos ignorados pelo Git',
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
      remoteLoopback:
        'Este endereço aponta para a máquina que executa seu agente, não para esta máquina. O painel de navegador carrega páginas localmente; um servidor remoto precisa de encaminhamento de porta ou de um host acessível.',
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
      goBack: 'Voltar',
      goForward: 'Avançar',
      reload: 'Recarregar página',
      address: 'Endereço',
      addressPlaceholder: 'Digite um endereço',
      blankPageBody: 'Digite um endereço acima para navegar ou peça ao Hermes para abrir uma página.',
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
      fallbackTitle: 'Pré-visualizar',
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
  interfaceMode: {
    title: 'Modo da interface',
    hint: 'Altera o que é exibido, não o que o Hermes pode fazer.',
    sessionNote:
      'Definido pelo modo Simples. Uma alteração aqui dura apenas nesta sessão; mude para Avançado para torná-la permanente.',
    simple: {
      label: 'Simples',
      description: 'Para conversar com o Hermes. Barra lateral e chat, sem painéis de terminal, arquivos ou diffs.'
    },
    advanced: {
      label: 'Avançado',
      description:
        'Para desenvolvedores. Terminal, arquivos, diffs, barra de status e layouts, do jeito que você configurar.'
    }
  },
  zones: {
    showTabStrip: 'Mostrar abas',
    hideTabStrip: 'Ocultar abas',
    showStripTab: title => `Mostrar ${title}`,
    hideStripTab: title => `Ocultar ${title}`,
    lastTabKeptTitle: 'A última aba permanece',
    lastTabKeptBody:
      'Esta zona precisa de pelo menos uma aba visível. Mostre outra aba primeiro ou recolha a barra lateral inteira.',
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
      loadingSession: 'Carregando a sessão',
      showEarlier: 'Mostrar mensagens anteriores',
      loadingResponse: 'O Hermes está carregando uma resposta',
      loadingLocalModel: model => `Carregando ${model} na memória`,
      processingPrompt: 'Processando prompt',
      resumeWhenBackgroundDone: count =>
        count === 1
          ? 'Vai retomar quando a tarefa em segundo plano terminar'
          : `Vai retomar quando ${count} tarefas em segundo plano terminarem`,
      thinking: 'Pensando',
      thought: 'Pensou',
      thoughtBriefly: 'Pensou brevemente',
      thoughtFor: duration => `Pensou por ${duration}`,
      turnDuration: duration => `Esta rodada levou ${duration}`,
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
      errorLayerBodies: {
        auth: 'O serviço de IA rejeitou seu login. Verifique as credenciais deste provedor e envie sua mensagem novamente.',
        billing:
          'Sua conta não tem mais créditos neste provedor. Adicione créditos ou troque de provedor e envie novamente.',
        disk: 'Seu disco está cheio, então o Hermes não conseguiu salvar esta conversa. Libere espaço e tente novamente.',
        endpoint:
          'O Hermes não consegue acessar o servidor do seu modelo personalizado. Verifique se ele está em execução e envie sua mensagem novamente.',
        gateway:
          'O Hermes encontrou um problema interno ao iniciar esta resposta. Envie sua mensagem novamente; se continuar acontecendo, envie os diagnósticos.',
        generic:
          'Algo deu errado enquanto o Hermes respondia. Tente novamente ou copie os detalhes se continuar acontecendo.',
        provider:
          'O serviço de IA não conseguiu concluir esta solicitação. Tente novamente daqui a pouco ou troque de provedor.',
        runtime:
          'O Hermes encontrou um problema interno ao iniciar esta resposta. Envie sua mensagem novamente; se continuar acontecendo, envie os diagnósticos.',
        streaming: 'A conexão caiu antes de a resposta terminar. Tente novamente para reenviá-la.'
      },
      errorCodes: {
        auth: {
          title: provider => `${provider} rejeitou seu login`,
          body: provider =>
            `As credenciais salvas para ${provider} não foram aceitas. Corrija-as em Configurações ou troque de provedor e envie sua mensagem novamente.`
        },
        auth_permanent: {
          title: provider => `${provider} rejeitou seu login`,
          body: provider =>
            `As credenciais salvas para ${provider} são inválidas ou foram revogadas. Atualize-as ou troque de provedor e envie sua mensagem novamente.`
        },
        billing: {
          title: 'Sem créditos',
          body: provider =>
            `Sua conta ${provider} não tem mais créditos. Adicione créditos ou troque de provedor e envie novamente.`
        },
        rate_limit: {
          title: 'O serviço de IA está ocupado',
          body: provider =>
            `${provider} está limitando as solicitações no momento. Aguarde um minuto e tente novamente.`
        },
        upstream_rate_limit: {
          title: 'O serviço de IA está ocupado',
          body: provider =>
            `${provider} está limitando as solicitações no momento. Aguarde um minuto e tente novamente.`
        },
        overloaded: {
          title: 'O serviço de IA está sobrecarregado',
          body: provider =>
            `${provider} está com problemas no momento. Tente novamente daqui a pouco ou troque de provedor.`
        },
        server_error: {
          title: 'O serviço de IA teve um problema',
          body: provider =>
            `${provider} retornou um erro de servidor. Tente novamente daqui a pouco ou troque de provedor.`
        },
        timeout: {
          title: 'A resposta excedeu o tempo limite',
          body: provider => `${provider} não respondeu a tempo. Tente novamente para reenviar.`
        },
        stream_drop: {
          title: 'A resposta foi interrompida',
          body: 'A conexão caiu antes de a resposta terminar. Tente novamente para reenviá-la.'
        },
        upstream_blocked: {
          title: 'Um firewall bloqueou a solicitação',
          body: provider =>
            `Um firewall ou CDN na frente de ${provider} bloqueou a solicitação antes que ela chegasse ao modelo — sua chave provavelmente está correta. Defina um cabeçalho User-Agent por meio de extra_headers do provedor em Configurações ou troque de provedor e envie sua mensagem novamente.`
        },
        ssl_cert_verification: {
          title: 'Falha na conexão segura',
          body: provider =>
            `O Hermes não conseguiu verificar a conexão segura com ${provider}. Verifique as configurações de rede ou proxy, ou troque de provedor, e envie sua mensagem novamente.`
        },
        context_overflow: {
          title: 'Esta conversa é longa demais',
          body: 'A conversa não cabe mais no modelo. Compacte-a ou inicie um novo chat e envie novamente.'
        },
        payload_too_large: {
          title: 'Esta mensagem é grande demais',
          body: 'A solicitação era grande demais para o modelo. Compacte a conversa ou inicie um novo chat e envie novamente.'
        },
        model_not_found: {
          title: 'Este modelo não está disponível',
          body: provider =>
            `${provider} não oferece este modelo na sua conta. Escolha outro modelo e envie sua mensagem novamente.`
        },
        provider_policy_blocked: {
          title: 'Este modelo está bloqueado pelas configurações da sua conta',
          body: provider =>
            `${provider} não encaminharia esta solicitação com as configurações de dados ou privacidade da sua conta. Escolha outro modelo ou troque de provedor.`
        },
        content_policy_blocked: {
          title: 'O serviço de IA recusou esta solicitação',
          body: provider => `${provider} não respondeu a esta mensagem. Edite-a e envie novamente.`
        },
        format_error: {
          title: 'O serviço de IA rejeitou a solicitação',
          body: provider =>
            `${provider} não aceitou a forma como esta solicitação foi criada. Troque de provedor ou envie os diagnósticos para que possamos investigar.`
        },
        truncated: {
          title: 'A resposta foi interrompida',
          body: 'O modelo parou antes de terminar. Tente novamente para receber uma resposta completa.'
        },
        invalid_response: {
          title: 'O serviço de IA enviou uma resposta ilegível',
          body: provider => `${provider} retornou algo que o Hermes não conseguiu ler. Tente novamente em instantes.`
        },
        empty_response: {
          title: 'O serviço de IA enviou uma resposta vazia',
          body: provider => `${provider} não retornou nada para esta mensagem. Tente novamente em instantes.`
        },
        loop_error: {
          title: 'O Hermes ficou preso em um loop',
          body: 'A resposta continuou repetindo as mesmas etapas, então o Hermes a interrompeu. Tente novamente ou inicie um novo chat se isso acontecer de novo.'
        },
        SESSION_NOT_OWNED: {
          title: 'Este chat está aberto em outro lugar',
          body: 'Este chat está aberto em outra janela ou terminal do Hermes. Feche-o lá e envie sua mensagem novamente, ou inicie um novo chat aqui.'
        },
        disk_full: {
          title: 'Disco cheio',
          body: 'Seu disco está cheio, então o Hermes não conseguiu salvar esta conversa. Libere espaço e tente novamente.'
        },
        free_tier_disabled: {
          title: 'O uso do Hermes sem entrar em uma conta está desativado no momento',
          body: 'Entre com uma conta Nous para continuar conversando. É grátis.'
        },
        free_tier_rate_limited: {
          title: 'Você esgotou o limite de conversas sem entrar em uma conta',
          body: 'O limite será renovado em breve. Entre com uma conta Nous para ter um limite maior. É grátis.'
        },
        free_tier_at_capacity: {
          title: 'As conversas sem entrar em uma conta estão muito ocupadas no momento',
          body: 'Entre para pular a fila — é grátis — ou tente novamente daqui a pouco.'
        },
        free_tier_model_not_free: {
          title: 'Esse modelo não está disponível sem entrar em uma conta',
          body: 'Por enquanto, o Hermes usa o modelo gratuito. Entre com uma conta Nous para acessar mais modelos. É grátis.'
        },
        free_tier_route: {
          title: 'O Hermes não conseguiu acessar o modelo gratuito por esta rota',
          body: 'Entre com uma conta Nous — é grátis — ou verifique a configuração NOUS_INFERENCE_BASE_URL.'
        },
        free_tier_outage: {
          title: 'O modelo gratuito está com dificuldades para responder no momento',
          body: 'Tente enviar sua mensagem novamente em um minuto.'
        },
        free_tier_refused: {
          title: 'O Hermes não conseguiu enviar isso sem entrar em uma conta',
          body: 'Entrar com uma conta Nous é grátis.'
        }
      },
      errorAuthKinds: {
        api_key: {
          title: provider => `${provider} rejeitou sua chave de API`,
          body: provider => `A chave salva para ${provider} é inválida ou foi revogada. Atualize-a e tente novamente.`
        },
        oauth: {
          title: provider => `Seu login no ${provider} expirou`
        }
      },
      errorDetails: 'Detalhes',
      errorGenericProvider: 'O serviço de IA',
      errorToastTitle: 'O Hermes não conseguiu concluir a resposta',
      errorRetry: 'Tentar novamente',
      errorLimitResets: time => `O limite será redefinido às ${time}`,
      errorRetryAtReset: time => `Tente novamente quando o limite for redefinido (${time})`,
      errorRetryScheduled: (time, wait) => `Nova tentativa às ${time} — em ${wait}`,
      errorRetryScheduledCancel: 'Cancelar',
      errorStartNewSession: 'Iniciar nova sessão',
      errorSwitchProvider: 'Trocar provedor',
      errorChooseModel: 'Escolher um modelo',
      errorCompressConversation: 'Compactar conversa',
      errorCompressFailed: 'Não foi possível compactar a conversa',
      errorOpenHermesFolder: 'Abrir pasta do Hermes',
      errorOpenHermesFolderFailed: 'Não foi possível abrir a pasta do Hermes',
      errorUpdateApiKey: 'Atualizar chave de API',
      errorSignInAgain: provider => `Entrar novamente no ${provider}`,
      errorSignInFreeTier: 'Entrar com uma conta Nous',
      errorOauthExpired: provider =>
        `Seu login no ${provider} expirou ou foi revogado. Entre novamente para continuar conversando.`,
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
      gatewayDisconnected:
        'O Hermes está offline agora. O comando continua aguardando sua resposta (até o limite de tempo da aprovação). Reconecte e envie-a novamente.',
      sendFailed: 'Não foi possível enviar a resposta de aprovação',
      reconnect: 'Reconectar',
      timedOutSystemLine:
        'O tempo para aprovação se esgotou — o comando não foi executado. Peça ao Hermes para tentar novamente ou aumente o limite em Configurações → Segurança → Tempo limite de aprovação.',
      openSafetySettings: 'Abrir configurações de Segurança',
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
      confirmAndContinueLabel: 'Confirmar e continuar',
      answeredBadge: 'Respondida',
      questionProgress: (answered, total) => `${answered} de ${total} respondidas`,
      lateAnswer: (question, choice) => `Sobre "${question}" — minha resposta: ${choice}`,
      lateAnswerTip: 'Rascunhar esta resposta como mensagem de complemento',
      lateAnswerHint:
        'Este prompt não está mais aguardando. Escolha uma opção para rascunhá-la como mensagem de complemento.'
    },
    catalogInstall: {
      preparing: 'Preparando a instalação…',
      install: 'Instalar',
      advanced: 'Avançado',
      skip: 'Pular',
      installing: 'Instalando…',
      installed: 'Instalado',
      notInstalled: 'Não instalado',
      failed: 'Falhou',
      showNames: 'mostrar nomes',
      hideNames: 'ocultar nomes',
      skill: name => `habilidade ${name}`,
      kind: {
        plugin: 'plugin',
        skill: 'habilidade'
      },
      tier: {
        official: 'oficial',
        community: 'comunidade'
      },
      targetProfile: profile => `Instala no seu perfil ${profile}`,
      sendFailed: 'Não foi possível enviar sua resposta. Tente novamente.',
      commitLabel: 'Commit',
      subdirLabel: 'Pasta',
      securityHeading: 'Segurança',
      scan: {
        passed: 'Verificação aprovada',
        warnings: 'A verificação encontrou avisos',
        failed: 'Falha na verificação'
      },
      requirementsLabel: 'Requer',
      credentialsHeading: 'Credenciais'
    },
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
      failedCalls: count => `Falha em ${count} chamada${count === 1 ? '' : 's'} de ferramenta`,
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
      failedOne: '1 etapa falhou',
      failedMany: count => `${count} etapas falharam`,
      statusRunning: 'Executando',
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
        browser_navigate: {
          done: 'Abriu a página',
          pending: 'Abrindo a página',
          pendingAction: 'Abrindo'
        },
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
        browser_type: {
          done: 'Digitou na página',
          pending: 'Digitando na página',
          pendingAction: 'Digitando'
        },
        clarify: {
          done: 'Fez uma pergunta',
          pending: 'Fazendo uma pergunta',
          pendingAction: 'Perguntando'
        },
        cronjob: {
          done: 'Tarefa de cron',
          pending: 'Agendando a tarefa de cron',
          pendingAction: 'Agendando'
        },
        edit_file: {
          done: 'Editou o arquivo',
          pending: 'Editando o arquivo',
          pendingAction: 'Editando'
        },
        execute_code: {
          done: 'Executou o código',
          pending: 'Executando o código',
          pendingAction: 'Executando'
        },
        image_generate: {
          done: 'Gerou a imagem',
          pending: 'Gerando a imagem',
          pendingAction: 'Gerando'
        },
        list_files: {
          done: 'Listou os arquivos',
          pending: 'Listando os arquivos',
          pendingAction: 'Listando'
        },
        memory: {
          done: 'Salvou na memória',
          pending: 'Salvando na memória',
          pendingAction: 'Salvando'
        },
        patch: {
          done: 'Aplicou patch no arquivo',
          pending: 'Aplicando patch no arquivo',
          pendingAction: 'Aplicando patch'
        },
        read_file: {
          done: 'Leu o arquivo',
          pending: 'Lendo o arquivo',
          pendingAction: 'Lendo'
        },
        search_files: {
          done: 'Buscou nos arquivos',
          pending: 'Buscando nos arquivos',
          pendingAction: 'Buscando'
        },
        session_search_recall: {
          done: 'Buscou no histórico da sessão',
          pending: 'Pesquisando o histórico da sessão',
          pendingAction: 'Pesquisando'
        },
        terminal: {
          done: 'Executou o comando',
          pending: 'Executando o comando',
          pendingAction: 'Executando'
        },
        todo: {
          done: 'Atualizou as tarefas',
          pending: 'Atualizando as tarefas',
          pendingAction: 'Atualizando'
        },
        vision_analyze: {
          done: 'Analisou a imagem',
          pending: 'Analisando a imagem',
          pendingAction: 'Analisando'
        },
        web_extract: {
          done: 'Leu a página web',
          pending: 'Lendo a página web',
          pendingAction: 'Lendo'
        },
        web_search: {
          done: 'Buscou na web',
          pending: 'Buscando na web',
          pendingAction: 'Buscando'
        },
        write_file: {
          done: 'Editou o arquivo',
          pending: 'Editando o arquivo',
          pendingAction: 'Editando'
        }
      }
    }
  },
  prompts: {
    gatewayDisconnected: 'O gateway do Hermes não está conectado',
    reconnect: 'Reconectar',
    sudoSendFailed: 'Não foi possível enviar a senha do sudo',
    secretSendFailed: 'Não foi possível enviar o segredo',
    sudoTitle: 'Senha de administrador',
    sudoDesc:
      'O Hermes precisa da sua senha do sudo para executar um comando privilegiado. Ela é enviada apenas ao seu agente local.',
    sudoCommandUnavailable: 'Este agente não forneceu o comando. Cancele se não conseguir verificá-lo na conversa.',
    sudoInstallDesc:
      'O Hermes precisa da sua senha sudo para instalar os pacotes do Bot Screen (TigerVNC + Xfce) no host do gateway. Ela é enviada somente para esse host.',
    sudoPlaceholder: 'senha do sudo',
    secretTitle: 'Segredo necessário',
    secretDesc: 'O Hermes precisa de uma credencial para continuar.',
    secretPlaceholder: 'valor do segredo',
    vaultUnlockSendFailed: 'Não foi possível enviar a senha mestra',
    vaultUnlockTitle: name => `Desbloquear ${name}`,
    vaultUnlockDesc: name =>
      `O agente quer entrar em um site usando um login salvo em ${name}. Digite sua senha mestra para desbloqueá-lo nesta sessão — ela vai diretamente para ${name} nesta máquina e nunca é armazenada nem exibida ao agente.`,
    vaultUnlockPlaceholder: 'Senha mestra',
    vaultUnlockKeepLocked: 'Manter bloqueado',
    vaultUnlockConfirm: 'Desbloquear',
    vaultSaveSendFailed: 'Não foi possível salvar o login',
    vaultSaveTitle: site => `Salvar seu login do ${site}?`,
    vaultSaveDesc: origin =>
      `O Hermes chegou a uma página de login em ${origin} e não tem um login salvo para ela. Insira-o uma vez aqui; ele será criptografado nesta máquina e preenchido na página sem que o modelo veja a senha.`,
    vaultSaveIdentifierLabel: 'E-mail ou nome de usuário',
    vaultSaveIdentifierPlaceholder: 'you@example.com',
    vaultSavePasswordPlaceholder: 'Senha',
    vaultSaveFootnote: 'Gerencie os logins salvos em Configurações → Senhas e logins.',
    vaultSaveDecline: 'Não salvar',
    vaultSaveConfirm: 'Salvar e entrar',
    vaultCodeSendFailed: 'Não foi possível enviar o código',
    vaultCodeTitle: site => `Código de verificação para ${site}`,
    vaultCodeDesc: site =>
      `${site} está solicitando um código de uso único (mensagem de texto, e-mail ou aplicativo autenticador). Digite-o aqui e o Hermes o inserirá na página; o modelo nunca o verá.`,
    vaultCodeLabel: 'Código',
    vaultCodeFootnote:
      'Dica: salve a chave do autenticador com este login em Configurações → Senhas e logins, e o Hermes inserirá os códigos para você.',
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
    editTurnUnavailable: 'Esta rodada não está mais no histórico do servidor — talvez tenha sido compactada.',
    resumeFailed: 'Falha ao retomar',
    readOnlyTranscriptTitle: 'Aberto somente para leitura',
    readOnlyTranscriptBody:
      'Nenhum backend conectado assumiu este chat mais antigo ainda, por isso ele foi aberto como uma transcrição somente para leitura. O histórico está intacto; o envio fica desativado até que um backend o assuma.',
    readOnlyTranscriptSendBlocked:
      'Este chat está aberto como uma transcrição somente para leitura — o envio está desativado.',
    resumeStrandedTitle: 'Não foi possível carregar esta sessão',
    resumeStrandedBody:
      'A conexão com esta sessão falhou e as tentativas automáticas desistiram. Verifique se o gateway está rodando e tente de novo.',
    poolSlotTimeoutBody:
      'Há bots demais em execução ao mesmo tempo para o limite deste computador. Aumente o limite em Configurações → Avançado ou espere um deles terminar e tente novamente.',
    poolSlotTimeoutOpenSettings: 'Abrir configurações avançadas',
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
    modelSwitchConfirmBody: 'Esta troca de modelo precisa de confirmação.',
    modelSwitchConfirmLabel: 'Trocar mesmo assim',
    modelSwitchConfirmTitle: model => `Trocar para ${model}?`,
    modelSwitchConfirmTitleFallback: 'Trocar de modelo?',
    modelSwitchFailed: 'Falha ao trocar de modelo',
    modelSwitchKeepLabel: 'Manter o modelo atual',
    modelSwitchStaleNotice: 'A seleção mudou — a troca de modelo não foi aplicada.',
    hydrationSyncing: profile => `Sincronizando ${profile}…`,
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
    pastedContent: 'Conteúdo colado',
    pasteAttachFailed: 'Não foi possível anexar o texto colado',
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
      timedOut: 'Tempo esgotado aguardando o gateway. O `hermes gateway` está rodando?',
      startMessaging: 'Começar a enviar mensagens'
    }
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
      'local-runtime-update': {
        title: 'Uma atualização do mecanismo local está disponível',
        text: 'Atualize o mecanismo que executa seus modelos locais. Solicitações locais ativas podem ser interrompidas.',
        action: 'Atualizar agora'
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
  },
  errors: {
    genericFailure: 'Algo deu errado',
    boundaryTitle: 'Algo quebrou na interface',
    boundaryDesc: 'A tela encontrou um erro inesperado. Suas conversas e configurações estão seguras.',
    boundaryDetails: 'Detalhes',
    sendDiagnostics: 'Enviar diagnósticos',
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
      toggle: open => `${open ? 'Mostrar' : 'Ocultar'} barra lateral`
    }
  }
}

export const ptBrOverrides = ptBr
