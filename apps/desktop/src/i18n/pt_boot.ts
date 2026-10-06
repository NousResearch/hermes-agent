import type { TranslationOverrides } from './define-locale'

export const ptBoot = {
  boot: {
    ready: 'Hermes Desktop está pronto',
    desktopBootFailedWithMessage: message => `A inicializa\xE7\xE3o do Desktop falhou: ${message}`,
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
      restartHermes: 'Reiniciar o Hermes',
      openLogs: 'Abrir logs',
      desktopBootFailed: 'Falha na inicialização do Desktop',
      gatewayConnectionLost: 'Conexão com o gateway perdida',
      gatewayConnectionLostDetail:
        'Ainda tentando reconectar em segundo plano. Você pode continuar lendo e escrevendo — abra Configurações do gateway se isso persistir.',
      reconnectNow: 'Reconectar agora',
      connectionSettings: 'Configurações de conexão',
      gatewaySignInRequired: 'Login no gateway necessário',
      gatewaySignInRequiredDetail: 'Faça login novamente para reconectar. Seus chats e configurações estão seguros.',
      signInAgain: 'Fazer login novamente',
      ipcBridgeUnavailable: 'A ponte IPC do Desktop está indisponível.'
    },
    causes: {
      exitedEarly: 'O serviço em segundo plano do Hermes parou logo após iniciar.',
      timedOut: 'O serviço em segundo plano do Hermes não respondeu a tempo.',
      permission: 'O Hermes não conseguiu gravar na própria pasta de dados (problema de permissão).',
      diskFull: 'O disco está cheio, então o Hermes não conseguiu iniciar.',
      portInUse: 'Outro programa está usando a porta de rede de que o Hermes precisa.',
      installMissing: 'Falta uma parte da instalação do Hermes. Escolha Reparar instalação para restaurá-la.'
    },
    failure: {
      title: 'O Hermes não pôde ser iniciado',
      description:
        'O gateway em segundo plano não iniciou. Tente uma das etapas de recuperação abaixo. Nenhuma das opções exclui seus chats ou configurações.',
      details: 'Detalhes',
      remoteTitle: 'Login remoto no gateway necessário',
      remoteDescription:
        'Sua sessão no gateway remoto expirou. Faça login novamente. Nenhuma das opções exclui seus chats ou configurações.',
      retry: 'Tentar novamente',
      repairInstall: 'Reparar instalação',
      useLocalGateway: 'Usar gateway local',
      gatewaySettings: 'Configurações do gateway',
      back: 'Voltar',
      openLogs: 'Abrir logs',
      repairHint: 'Reparar executa o instalador novamente e pode levar alguns minutos.',
      bundledReinstallHint:
        'Esta instalação embutida não consegue se reparar por dentro do app. Reinstale o app para restaurar o backend.',
      reinstallApp: 'Reinstalar o app',
      remoteSignInHint: signInLabel =>
        `Encerra a sess\xE3o remota salva e abre ${signInLabel}. Use Usar gateway local para alternar para o backend embutido.`,
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
  remoteDisplayBanner: {
    message: reason =>
      `Renderiza\xE7\xE3o por software ativa \u2014 monitor remoto detectado (${reason}). A acelera\xE7\xE3o de GPU est\xE1 desativada para evitar cintila\xE7\xE3o.`
  },
  updates: {
    discontinuedTitle: 'Esta versão do Hermes não tem mais suporte',
    discontinuedBody:
      'Esta versão do Hermes não tem mais suporte e pode parar de funcionar: desinstale-a. Seus dados permanecem no disco.',
    channels: {
      stable: 'Estável',
      canary: 'Canary'
    },
    bundleSwapPending: 'Reinicie para concluir a atualização',
    bundleSwapPendingDesc:
      'O app atualizado já está instalado; o Hermes só precisa reiniciar para carregá-lo. Chats e configurações não são afetados.',
    bundleSwapPendingAction: 'Reiniciar o Hermes',
    stages: {
      idle: 'Preparando…',
      prepare: 'Preparando…',
      fetch: 'Baixando…',
      pull: 'Quase lá…',
      pydeps: 'Finalizando…',
      update: 'Atualizando o Hermes…',
      rebuild: 'Recompilando o app desktop…',
      restart: 'Reiniciando o Hermes…',
      done: 'Atualização concluída',
      manual: 'Atualize pelo terminal',
      guiSkew: 'Atualize o app desktop',
      error: 'Atualização pausada'
    },
    checking: 'Verificando…',
    checkFailedTitle: 'Não foi possível verificar atualizações',
    tryAgain: 'Tentar novamente',
    notAvailableTitle: 'Atualização indisponível',
    unsupportedMessage: 'Esta versão do Hermes não consegue se atualizar por dentro do app.',
    connectionRetry:
      'O Hermes não conseguiu acessar o servidor de atualizações. Verifique sua conexão com a internet e tente novamente. Se você usa um Hermes remoto, verifique se ele está online.',
    gitUnusable: 'O Hermes não conseguiu executar o Git neste computador, então não pôde verificar atualizações.',
    connectionSettings: 'Configurações de conexão',
    openDownloadPage: 'Abrir a página de download',
    latestBody: 'Você está usando a versão mais recente.',
    latestBodyBackend: 'O backend está na versão mais recente.',
    allSetTitle: 'Tudo certo',
    availableTitle: 'Nova atualização disponível',
    availableBody: 'Uma nova versão do Hermes está pronta para instalar.',
    availableTitleBackend: 'Atualização do backend disponível',
    availableBodyBackend: 'Uma versão mais nova do backend do Hermes conectado está pronta para instalar.',
    availableBodyNoChangelog:
      'Uma versão mais nova está pronta. As notas de versão não estão disponíveis para este tipo de instalação.',
    availableBodyAppInstaller:
      'Uma nova versão do Hermes está pronta. O Hermes vai fechar, o Windows conclui a atualização e o Hermes reabre sozinho.',
    updateNow: 'Atualizar agora',
    maybeLater: 'Talvez depois',
    moreChanges: count => `+ ${count} alteraç${count === 1 ? 'ão' : 'ões'} incluída${count === 1 ? '' : 's'}.`,
    copyFullLog: 'Copiar o changelog completo',
    manualTitle: 'Atualize pelo terminal',
    manualUnavailableTitle: 'Não é possível atualizar daqui',
    manualBody:
      'Você instalou o Hermes pela linha de comando, então as atualizações também são feitas lá. Cole isto no seu terminal:',
    manualBodyBackend: 'O backend do Hermes é gerenciado fora deste app. Execute isto no servidor que o hospeda:',
    manualPickedUp: 'O Hermes passa a usar a nova versão na próxima vez que você o iniciar.',
    manualPickedUpBackend: 'O backend passa a usar a nova versão depois que a atualização terminar.',
    guiSkewTitle: 'Atualize o app desktop',
    guiSkewBody:
      'O backend foi atualizado, mas o pacote deste app desktop não mudou. Atualize ou reinstale o app desktop do Hermes (seu AppImage / .deb / .rpm) para acompanhar.',
    copy: 'Copiar',
    copied: 'Copiado',
    done: 'Concluído',
    applyingBody:
      'O atualizador do Hermes assume em uma janela própria e reabre o Hermes automaticamente ao terminar. Não reabra o Hermes por conta própria durante a atualização.',
    applyingBodyBackend:
      'O backend remoto está aplicando a atualização e vai reiniciar. O Hermes reconecta automaticamente quando ele voltar.',
    applyingClose: 'Esta janela vai fechar enquanto a atualização roda; depois o Hermes reabre sozinho.',
    applyingBodyAppInstaller:
      'O Hermes vai fechar e o Windows conclui a atualização. O Hermes reabre quando terminar; você não precisa fazer nada.',
    applyingCloseAppInstaller: 'Esta janela vai fechar, o Windows conclui a atualização e o Hermes reabre sozinho.',
    checkUnknownTitleAppInstaller: 'Não foi possível verificar atualizações',
    checkUnknownBodyAppInstaller:
      'O Windows não conseguiu verificar atualizações agora. As atualizações também são instaladas automaticamente quando você reinicia o Hermes.',
    errorTitle: 'A atualização não terminou',
    errorBody: 'Fique tranquilo: nada foi perdido. Você pode tentar de novo agora.',
    blockerTitle: 'Fechar as prévias locais para atualizar o Hermes?',
    blockerBody:
      'O Hermes precisa encerrar estas prévias locais antes de atualizar. Isso não modifica nem exclui os seus arquivos.',
    foreignBlockerTitle: 'Feche outros processos para atualizar o Hermes',
    foreignBlockerBody:
      'O Hermes não consegue encerrar estes processos com segurança de forma automática. Feche o app, o terminal ou o serviço dono de cada um e tente atualizar de novo.',
    mixedBlockerBody:
      'O Hermes pode fechar as prévias locais listadas abaixo. Os outros processos precisam ser fechados manualmente antes de a atualização continuar.',
    closePreviewsAndUpdate: 'Fechar as prévias e atualizar',
    closePreviewsAndCheckAgain: 'Fechar as prévias e verificar de novo',
    localPreview: 'Prévia local',
    portLabel: port => `Porta ${port}`,
    pidLabel: pid => `PID ${pid}`,
    technicalDetails: 'Detalhes técnicos',
    notNow: 'Agora não',
    clientAlsoBehindTitle: 'O app desktop está desatualizado',
    clientAlsoBehindMessage:
      'O backend está atualizado, mas este app desktop ainda está em uma versão mais antiga. Atualize-o para receber as correções mais recentes.',
    clientAlsoBehindAction: 'Atualizar o app desktop',
    everythingDispatched: 'Atualização disparada',
    everythingSkipped: 'Ignorada',
    everythingRowFailed: 'Falha na atualização',
    everythingFanoutFailedTitle: 'Não foi possível atualizar as outras instâncias',
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
      restarting: 'Backend reiniciando para carregar a atualização…',
      notAvailable: 'Atualização indisponível para este backend.',
      failed: 'Falha na atualização do backend.',
      noReturn:
        'O backend não voltou a ficar online. A atualização pode não ter sido concluída: verifique o host do backend.'
    },
    appName: 'Hermes',
    version: value => `Versão ${value}`,
    versionUnavailable: 'Versão indisponível',
    checkNow: 'Verificar agora',
    seeWhatsNew: 'Ver o que há de novo',
    releaseNotes: 'Notas de lançamento',
    onLatest: 'Você está na versão mais recente.',
    installing: 'Uma atualização está sendo instalada.',
    cantReach: 'Não conseguimos alcançar o servidor de atualizações.',
    tapCheck: 'Toque em "Verificar agora" para buscar atualizações.',
    updateReady: count =>
      `Uma nova atualiza\xE7\xE3o est\xE1 pronta (${count} ${count === 1 ? 'mudan\xE7a inclu\xEDda' : 'mudan\xE7as inclu\xEDdas'}).`,
    updateReadyUnknown: 'Uma nova atualização está pronta.',
    availableBodyRelease: tag => `A versão ${tag} está pronta para instalar.`,
    lastChecked: age => `Última verificação ${age}`,
    never: 'nunca',
    justNow: 'agora há pouco',
    minAgo: count => `há ${count} min`,
    hoursAgo: count => `há ${count} h`,
    daysAgo: count => `há ${count} d`,
    justNowSuffix: ' · agora há pouco',
    bundleOutOfSync: 'Build do app desatualizado',
    bundleOutOfSyncDesc:
      'O runtime do Hermes foi atualizado, mas o app desktop ainda é uma versão antiga — novas funcionalidades (como o Bot Mode) não aparecerão até que seja atualizado. Rode a atualização abaixo para recompilar o app. Se o aviso continuar, reinstale a partir do instalador mais recente.',
    bundleOutOfSyncAction: 'Baixar instalador',
    checkingShort: 'Verificando…',
    releaseAvailable: tag => `A versão ${tag} está disponível.`,
    versionDetailsTitle: 'Detalhes da versão',
    versionDetailsBody: 'Esta instalação é gerenciada fora do app. Atualize-a da mesma forma que você a instalou.',
    versionDetailsVersion: 'Versão',
    versionDetailsCommit: 'Commit',
    versionDetailsBuildOrigin: 'Origem do build',
    versionDetailsDistribution: 'Distribuição',
    versionDetailsDistributionDesktop: 'App desktop',
    versionDetailsDistributionDesktopMsix: 'App desktop (MSIX)',
    versionDetailsDistributionDesktopInstaller: 'App desktop (instalador)',
    versionDetailsDistributionSourceInstaller: 'Código-fonte (script de instalação)',
    versionDetailsDistributionSourceInstallerDesktop: 'Código-fonte (script de instalação) + hermes desktop',
    versionDetailsDistributionSource: 'Código-fonte',
    versionDetailsDistributionSourceDesktop: 'Código-fonte + hermes desktop',
    versionDetailsDistributionStore: 'Microsoft Store',
    versionDetailsRuntime: 'Runtime',
    versionDetailsRuntimeEmbedded: 'Runtime embutido',
    versionDetailsRuntimeExternal: 'Externo (usa o runtime da máquina)',
    versionDetailsInstallId: 'ID da instalação',
    versionDetailsUncommittedChanges: 'alterações não commitadas'
  },
  handoffTour: {
    profileTitle: 'Sua primeira tarefa roda no perfil padrão',
    profileText:
      'Esta barra alterna entre perfis. O que está aceso agora é o padrão, onde fica a sessão da tarefa. O outro é o perfil de configuração, onde fica o chat de boas-vindas.',
    sessionsTitle: 'Cada perfil guarda as próprias sessões',
    sessionsText:
      'Esta lista pertence ao perfil padrão. Nova sessão cria uma no perfil selecionado. Troque de perfil na barra e a lista muda junto.',
    stayTitle: 'O Hermes está a um clique',
    stayText:
      'Mude para o perfil de configuração e abra “Welcome to Hermes” sempre que quiser uma ajuda. Ele fica por lá.'
  },
  guidedGreeting: {
    line: 'Ei, pode entrar. Eu sou o Hermes. Me dê dois minutinhos para arrumar o lugar ao seu redor, e depois vamos pôr em prática algo que você realmente queira resolver.\n\nMas, antes, como devo chamar você?',
    nameSuggestion: name => `(Se preferir, também posso chamar você de ${name}.)`
  },
  install: {
    stageStates: {
      pending: 'Pendente',
      running: 'Instalando',
      succeeded: 'Concluído',
      skipped: 'Ignorado',
      failed: 'Falha'
    },
    oneTimeTitle: 'O Hermes precisa de uma instalação única',
    unsupportedDesc: platform =>
      `A instalação automática na primeira execução ainda não está disponível no ${platform}. Abra o Terminal e execute o comando abaixo; depois, reabra este app. As próximas execuções pulam esta etapa.`,
    installCommand: 'Comando de instalação',
    copyCommand: 'Copiar comando',
    viewDocs: 'Ver a documentação de instalação',
    installTo: 'Será instalado em',
    retryAfterRun: 'Já executei: tentar de novo',
    setupChoiceTitle: 'Configurar o Hermes Desktop',
    setupChoiceDesc:
      'Conecte este app a um gateway do Hermes que você já mantém, ou instale o Hermes localmente neste computador.',
    setupChoiceDescLocal: 'Instale o Hermes neste computador ou conecte-se a um gateway do Hermes que você já mantém.',
    connectExistingTitle: 'Conectar a um Hermes existente',
    connectExistingShort: 'Conectar a existente',
    connectExistingDesc:
      'Use um backend remoto com token de sessão ou login pelo navegador. Nenhuma instalação local será iniciada.',
    installLocalTitle: 'Instalar o Hermes localmente',
    installLocalDesc: 'Baixa o Hermes, cria o ambiente Python dele e executa o backend neste computador.',
    useLocalTitle: 'Usar o Hermes neste computador',
    useLocalDesc: 'Um runtime do Hermes já está instalado aqui: inicie-o com um clique. Nada é baixado.',
    bundledLocalDesc: 'Use o runtime do Hermes incluído neste app: o backend embutido é a instalação local.',
    localStartUnavailable: 'Não foi possível iniciar a instalação local. Reinicie o Hermes Desktop e tente novamente.',
    remoteSetupTitle: 'Conectar a um Hermes existente',
    remoteSetupDesc:
      'Informe a URL do seu gateway. O Hermes Desktop detecta se ele precisa de token ou de login pelo navegador.',
    remoteUrlTitle: 'URL do gateway',
    remoteUrlDesc: 'Use a URL base do gateway do Hermes, incluindo https:// quando for remoto.',
    remoteUrlPlaceholder: 'https://gateway.exemplo.com.br/hermes',
    probing: 'Detectando a autenticação do gateway…',
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
    enterUrlFirst: 'Informe primeiro a URL do gateway.',
    signInIncomplete: 'A janela de login foi fechada antes de a autenticação ser concluída.',
    tokenTitle: 'Token de sessão',
    tokenDesc: 'Cole o token de sessão do arquivo .env do gateway remoto.',
    pasteSessionToken: 'Cole o token de sessão',
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
      'Uma das etapas da instalação não foi concluída. Isso pode acontecer quando outra cópia do Hermes está em execução, a conexão com a internet caiu ou o antivírus bloqueou o instalador. Feche outras janelas do Hermes, escolha Recarregar e tente novamente. Se falhar de novo, abra os logs e envie-os ao suporte.',
    activeDesc:
      'Esta é uma configuração única. O instalador do Hermes está baixando dependências e configurando a sua máquina. As próximas inicializações pulam esta etapa.',
    progress: (completed, total) => `${completed} de ${total} etapas concluídas`,
    currentStage: stage => ` -- agora: ${stage}`,
    fetchingManifest: 'Obtendo o manifesto do instalador…',
    error: 'Erro',
    hideOutput: 'Ocultar a saída do instalador',
    showOutput: 'Mostrar a saída do instalador',
    lines: count => `${count} linha${count === 1 ? '' : 's'}`,
    noOutput: 'Ainda sem saída.',
    cancelling: 'Cancelando…',
    cancelInstall: 'Cancelar instalação',
    transcriptSaved: 'Transcrição completa salva em',
    copiedOutput: 'Copiado!',
    copyOutput: 'Copiar saída',
    reloadRetry: 'Recarregar e tentar de novo',
    openLogs: 'Abrir logs'
  },
  onboarding: {
    headerTitle: 'Vamos configurar o Hermes Agent para você',
    headerDesc: 'Conecte um provedor de modelos para começar a conversar. A maioria das opções leva um clique.',
    preparingInstall:
      'O Hermes está concluindo a instalação. Na primeira execução, isso costuma levar menos de um minuto.',
    starting: 'Iniciando o Hermes…',
    lookingUpProviders: 'Procurando provedores…',
    collapse: 'Recolher',
    otherProviders: 'Outros provedores',
    haveApiKey: 'Tenho uma chave de API',
    chooseLater: 'Escolho um provedor depois',
    recommended: 'Recomendado',
    connected: 'Conectado',
    featuredPitch: 'Uma assinatura, mais de 300 modelos de ponta: a forma recomendada de usar o Hermes',
    fireworksPitch: 'API direta de modelos: modelos de ponta hospedados na Fireworks',
    localModelsTitle: 'Rode modelos localmente',
    localModelsPitch: 'Sem precisar de conta: baixe um modelo e rode nesta máquina',
    openRouterPitch: 'Uma chave, centenas de modelos: um bom padrão',
    apiKeyOptions: {
      fireworks: {
        short: 'API direta de modelos',
        description: 'Acesso direto aos modelos hospedados pela Fireworks AI.'
      },
      openrouter: {
        short: 'uma chave, vários modelos',
        description: 'Hospeda centenas de modelos por trás de uma única chave. Bom padrão para novas instalações.'
      },
      openai: {
        short: 'modelos da classe GPT',
        description: 'Acesso direto aos modelos da OpenAI.'
      },
      gemini: {
        short: 'modelos Gemini',
        description: 'Acesso direto aos modelos Gemini do Google.'
      },
      xai: {
        short: 'modelos Grok',
        description: 'Acesso direto aos modelos Grok da xAI.'
      },
      local: {
        short: 'autohospedado',
        description:
          'Aponte o Hermes para um endpoint local ou autohospedado compatível com a OpenAI (vLLM, llama.cpp, Ollama etc.).'
      }
    },
    backToSignIn: 'Voltar ao login',
    getKey: 'Obter uma chave',
    replaceCurrent: 'Substituir o valor atual',
    pasteApiKey: 'Cole a chave de API',
    localApiKeyPlaceholder: 'Chave de API (opcional, só se o seu endpoint exigir uma)',
    localModelNamePlaceholder: 'Nome do modelo (ex.: command-a-plus-05-2026)',
    couldNotSave: 'Não foi possível salvar a credencial.',
    connecting: 'Conectando',
    update: 'Atualizar',
    flowSubtitles: {
      pkce: 'Abre o navegador para fazer login e depois continua aqui',
      device_code: 'Abre uma página de verificação no navegador, e o Hermes conecta automaticamente',
      external: 'Entre uma vez no seu terminal e volte para conversar'
    },
    startingSignIn: provider => `Iniciando o login em ${provider}...`,
    verifyingCode: provider => `Verificando o seu código com ${provider}...`,
    connectedProvider: provider => `${provider} conectado`,
    connectedPicking: provider => `${provider} conectado. Escolhendo um modelo padrão...`,
    signInFailed: 'Falha no login. Tente novamente.',
    signInExpired:
      'A página de login expirou antes de você concluir. Tente novamente e complete a etapa do navegador em alguns minutos, ou use uma chave de API.',
    signInDidNotFinish: provider =>
      `O login com ${provider} não foi concluído. Verifique sua conexão com a internet e tente de novo, ou escolha outro provedor.`,
    tryAgain: 'Tentar novamente',
    useApiKeyInstead: 'Usar uma chave de API',
    errorDetails: 'Detalhes',
    pickDifferentProvider: 'Escolher outro provedor',
    signInWith: provider => `Entrar com ${provider}`,
    openedBrowser: provider => `Abrimos ${provider} no seu navegador.`,
    authorizeThere: 'Autorize o Hermes por lá.',
    copyAuthCode: 'Copie o código de autorização e cole abaixo.',
    pasteAuthCode: 'Cole o código de autorização',
    reopenAuthPage: 'Reabrir a página de autorização',
    autoBrowser: provider =>
      `Abrimos ${provider} no seu navegador. Autorize o Hermes por lá e você será conectado automaticamente: não há nada para copiar ou colar.`,
    reopenSignInPage: 'Reabrir a página de login',
    waitingAuthorize: 'Aguardando você autorizar…',
    externalPending: provider =>
      `${provider} faz o login pela própria CLI. Execute este comando em um terminal, depois volte e escolha “Já fiz login”:`,
    signedIn: 'Já fiz login',
    deviceCodeOpened: provider => `Abrimos ${provider} no seu navegador. Digite este código por lá:`,
    reopenVerification: 'Reabrir a página de verificação',
    copy: 'Copiar',
    defaultModel: 'Modelo padrão',
    freeTier: 'Plano gratuito',
    pro: 'Pro',
    free: 'Gratuito',
    price: (input, output) => `${input} de entrada / ${output} de saída por Mtok`,
    change: 'Alterar',
    startChatting: 'Começar',
    docs: provider => `Documentação de ${provider}`
  },
  freeTier: {
    providerRowTitle: 'Nous · plano gratuito',
    providerRowPitch: 'Faça login com uma conta Nous para liberar mais modelos e ferramentas.',
    readyTitle: 'O Hermes está pronto.',
    readyCaption: 'Gratuito · conectores incluídos',
    begin: 'Começar',
    signInInstead: 'Fazer login com uma conta Nous',
    otherProviders: 'Outros provedores',
    stripTitle: 'A inferência gratuita da Nous e os conectores já estão disponíveis.',
    stripBody: 'Abra o seletor de modelo para experimentá-los, ou faça login com uma conta Nous.',
    openModelPicker: 'Abrir o seletor de modelo',
    dismiss: 'Dispensar',
    providerName: 'Nous',
    statusLabel: model => `Nous · ${model}`,
    signIn: 'Fazer login',
    signInHeading: 'Faça login com uma conta Nous para liberar mais modelos e ferramentas.',
    settingUp: 'Configurando a inferência gratuita…',
    codeBody: 'Digite este código no navegador para concluir o login.',
    copyLink: 'Copiar link',
    doNotShare: 'Não compartilhe este código.',
    waiting: 'Aguardando o login…',
    finishingHeading: 'Concluindo o login…',
    finishingBody: 'Aprovado no navegador. Coletando os tokens da sua conta.',
    signedInAs: email => `Login feito como ${email}`,
    signedIn: 'Login feito.',
    completedBody: 'Sua conta agora inclui inferência e ferramentas.',
    defaultModel: 'Modelo padrão',
    change: 'Alterar',
    done: 'Concluir',
    notNow: 'Agora não',
    tryAgain: 'Tentar novamente',
    startAgain: 'Recomeçar',
    didNotComplete: 'O login não foi concluído',
    rejectedBody: 'Sem problema, você continua no serviço gratuito da Nous. Faça login quando quiser.',
    supersededBody: 'Um código de login mais novo substituiu este. Use o mais recente ou recomece.',
    timedOutHeading: 'Esse link de login expirou',
    timedOutBody: 'Recomece quando quiser. Você continua no serviço gratuito da Nous.',
    retiredBody:
      'A sua sessão terminou antes de o login ser concluído. O Hermes vai iniciar uma nova; depois, faça login de novo quando quiser.',
    errorBody: 'O login não foi concluído. Tente novamente quando quiser.',
    busyHeading: 'Quase lá',
    busyBody: wait =>
      `O Hermes não conseguiu concluir o seu login porque o serviço da Nous está ocupado. Tente de novo em ${wait}. A sua sessão continua aqui enquanto isso.`,
    unreachableBody:
      'O Hermes não conseguiu acessar o serviço da Nous para concluir o seu login. Verifique sua conexão com a internet e tente novamente. A sua sessão continua aqui.',
    alreadySignedInHeading: 'Você já está conectado.',
    alreadySignedInBody: 'Este Hermes já está conectado a uma conta Nous.',
    setupFailed: {
      gateClosed:
        'Esta versão do Hermes não consegue iniciar sem uma conta Nous. Faça login ou crie uma: é grátis e leva só um minuto.',
      paused:
        'O uso do Hermes sem login está pausado por um instante. O Hermes continuará verificando. Fazer login é grátis e deixa você começar agora mesmo.',
      rateLimited: wait =>
        `Muita gente está começando agora, então o Hermes tentará de novo em ${wait}. Fazer login é grátis e dispensa a espera.`,
      unreachable:
        'O Hermes não conseguiu acessar o serviço da Nous. Verifique sua conexão com a internet e toque em Tentar novamente. Ou conecte outro provedor por enquanto.',
      serverError:
        'O serviço da Nous teve um soluço. Toque em Tentar novamente daqui a pouco, ou conecte outro provedor por enquanto.',
      powRequired:
        'O servidor da Nous pediu uma prova de trabalho, mas isso ainda não está implementado no seu Agent. Faça login ou crie uma conta Nous gratuita para continuar.',
      locked:
        'Esta sessão não pode continuar sem login. Faça login ou crie uma conta Nous gratuita para seguir em frente.',
      generic:
        'O Hermes não conseguiu configurar o acesso gratuito sem login. Fazer login é grátis; ou conecte outro provedor.',
      signInBelow: 'Fazer login é grátis. Escolha a Nous abaixo.',
      tryAgain: 'Tentar novamente',
      retrying: 'Tentando novamente…'
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
        text: 'Skills são pastas de instruções que o Hermes carrega quando o trabalho exige.'
      },
      messaging: {
        title: 'Hermes longe da sua mesa',
        text: 'Conecte Telegram, Discord, Slack e mais: o mesmo agente, a mesma memória.'
      },
      artifacts: {
        title: 'Tudo o que o Hermes criou',
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
        title: 'Perfis são separados',
        text: 'Cada um é um Hermes próprio: chaves, memória e sessões separadas.'
      },
      'composer-mentions': {
        title: 'Anexar e comandar',
        text: 'Digite @ para trazer um arquivo para a conversa, / para executar um comando.'
      },
      'local-runtime-update': {
        title: 'Há uma atualização do mecanismo local',
        text: 'Atualize o mecanismo que executa os seus modelos locais. As solicitações locais ativas podem ser interrompidas.',
        action: 'Atualizar agora'
      },
      'local-setup': {
        title: 'Esta máquina pode executar modelos localmente',
        text: 'Seu hardware pode servir um modelo local. Os chats ficam no seu computador e não custam nada.',
        action: 'Configurar'
      },
      'right-pane': {
        title: 'O painel de trabalho',
        text: 'Arquivos, terminal, revisão e o navegador integrado compartilham o lado direito.'
      }
    }
  }
} satisfies Pick<
  TranslationOverrides,
  | 'boot'
  | 'remoteDisplayBanner'
  | 'updates'
  | 'handoffTour'
  | 'guidedGreeting'
  | 'install'
  | 'onboarding'
  | 'freeTier'
  | 'tips'
>
