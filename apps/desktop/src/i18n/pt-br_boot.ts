import type { Translations } from './types'

export const ptBrBoot = {
  boot: {
    ready: 'O Hermes Desktop está pronto',
    desktopBootFailedWithMessage: message => `Falha ao iniciar o Desktop: ${message}`,
    steps: {
      connectingGateway: 'Conectando ao gateway ativo do Desktop',
      loadingSettings: 'Carregando configurações do Hermes',
      loadingSessions: 'Carregando sessões recentes',
      retryingRemoteBackend: 'Reconectando ao backend remoto do Hermes…',
      startingDesktopConnection: 'Iniciando conexão do Desktop',
      startingHermesDesktop: 'Iniciando o Hermes Desktop…'
    },
    errors: {
      backgroundExited:
        'O serviço que executa suas conversas foi encerrado inesperadamente. Reinicie-o para continuar — suas conversas e configurações estão seguras.',
      backgroundExitedDuringStartup: 'O Hermes parou logo após iniciar.',
      backendStopped: 'O Hermes parou de funcionar em segundo plano',
      restartHermes: 'Reiniciar Hermes',
      openLogs: 'Abrir logs',
      desktopBootFailed: 'Não foi possível iniciar o Hermes',
      gatewayConnectionLost: 'O Hermes perdeu a conexão',
      gatewayConnectionLostDetail:
        'Ainda estamos tentando reconectar. Você pode continuar lendo e escrevendo rascunhos. Se isso persistir, reconecte agora ou verifique suas configurações de conexão.',
      reconnectNow: 'Reconectar agora',
      connectionSettings: 'Configurações de conexão',
      gatewaySignInRequired: 'Sua sessão no Hermes remoto foi encerrada',
      gatewaySignInRequiredDetail: 'Entre novamente para reconectar. Suas conversas e configurações estão seguras.',
      signInAgain: 'Entrar novamente',
      ipcBridgeUnavailable:
        'O Hermes Desktop não conseguiu se comunicar com sua própria camada de segundo plano. Reinicie o aplicativo.'
    },
    causes: {
      exitedEarly: 'O serviço em segundo plano do Hermes parou logo após iniciar.',
      timedOut: 'O serviço em segundo plano do Hermes não respondeu a tempo.',
      permission: 'O Hermes não conseguiu gravar na sua pasta de dados (problema de permissão).',
      diskFull: 'O disco está cheio, por isso o Hermes não pôde iniciar.',
      portInUse: 'Outro programa está usando a porta de rede de que o Hermes precisa.',
      installMissing: 'Parte da instalação do Hermes está faltando. Escolha Reparar instalação para restaurá-la.'
    },
    failure: {
      title: 'Não foi possível iniciar o Hermes',
      description:
        'O serviço em segundo plano do Hermes não iniciou. Tente uma das opções de recuperação abaixo. Nenhuma delas exclui suas conversas ou configurações.',
      details: 'Detalhes',
      remoteTitle: 'É necessário entrar no gateway remoto',
      remoteDescription:
        'Sua sessão no gateway remoto expirou. Entre novamente para reconectar. Nada aqui exclui suas conversas ou configurações.',
      webappTitle: 'Entre no Hermes',
      webappDescription: 'Execute hermes webapp neste computador para entrar neste navegador.',
      retry: 'Tentar novamente',
      repairInstall: 'Reparar instalação',
      useLocalGateway: 'Usar gateway local',
      gatewaySettings: 'Configurações do gateway',
      back: 'Voltar',
      openLogs: 'Abrir logs',
      repairHint: 'A reparação executa o instalador novamente e pode levar alguns minutos em um computador novo.',
      bundledReinstallHint:
        'Esta instalação com backend incluído não pode se reparar pelo aplicativo — reinstale o aplicativo para restaurar o backend.',
      reinstallApp: 'Reinstalar o aplicativo',
      remoteSignInHint: signInLabel =>
        `Encerra a sessão remota salva no navegador e abre ${signInLabel}. Use Usar gateway local para mudar para o backend incluído.`,
      signOutAndSignIn: 'Sair e entrar novamente',
      remoteFailureHint:
        'Verifique a URL do gateway e o login em Configurações do gateway, ou mude para o gateway local.',
      cloudDownTitle: 'O agente do Nous Cloud está fora do ar',
      cloudDownDescription:
        'O agente na nuvem gerenciado pela Nous ao qual este gateway se conecta está retornando um erro de servidor. Não é possível reiniciá-lo por aqui — verifique seu status, mude para o gateway local ou procure suporte.',
      cloudDownHint:
        'Os botões abaixo abrem o Nous Portal (status e controles da instância) e nosso Discord para suporte.',
      cloudDownCheckPortal: 'Verificar status no Portal',
      cloudDownDiscord: 'Obter ajuda no Discord',
      hideRecentLogs: 'Ocultar logs recentes',
      showRecentLogs: 'Mostrar logs recentes',
      signedInTitle: 'Login concluído',
      signedInMessage: 'Reconectando ao gateway remoto…',
      signInIncompleteTitle: 'Login incompleto',
      signInIncompleteMessage: 'A janela de login foi fechada antes da conclusão da autenticação.',
      signInFailed: 'Falha ao entrar',
      signInToRemoteGateway: 'Entrar no gateway remoto',
      signInWithProvider: provider => `Entrar com ${provider}`,
      identityProvider: 'seu provedor de identidade'
    },
    updateHold: {
      title: 'Uma atualização anterior ainda bloqueia o Hermes',
      titleUnverified: 'O Hermes não consegue confirmar se a última atualização terminou',
      description:
        'O Hermes está aguardando para não carregar arquivos que uma atualização ainda pode estar alterando. Ele inicia automaticamente assim que o bloqueio termina.',
      heldByProcess: pid =>
        `A atualização (processo ${pid}) foi encerrada, mas um processo iniciado por ela ainda bloqueia a instalação do Hermes.`,
      heldUnknown:
        'Uma atualização foi encerrada, mas um processo iniciado por ela ainda bloqueia a instalação do Hermes.',
      unverified:
        'O auxiliar de atualização não conseguiu verificar quem controla a instalação do Hermes agora. O Hermes continua verificando.',
      since: time => `Aguardando desde ${time}`,
      lastChecked: time => `Última verificação: ${time}`,
      recoveryHint:
        'Isso normalmente se resolve em alguns minutos. Caso contrário: saia do Hermes, encerre os processos git ou hermes restantes (ou reinicie o computador) e abra o Hermes novamente.',
      checkAgain: 'Verificar novamente',
      quit: 'Sair do Hermes',
      openLogs: 'Abrir logs',
      startAnyway: 'Iniciar mesmo assim…',
      confirmTitle: 'Iniciar o Hermes enquanto a atualização ainda o bloqueia?',
      confirmBody:
        'O processo de atualização restante ainda pode estar alterando os arquivos do Hermes. Iniciar agora pode carregar uma instalação parcialmente atualizada, que talvez não funcione até você executar a atualização novamente. O Hermes registra esta escolha no log e mantém o marcador de atualização.',
      confirmKeepWaiting: 'Continuar aguardando',
      confirmStart: 'Iniciar mesmo assim',
      startAnywayRefused:
        'O bloqueio da instalação mudou antes que o Hermes pudesse iniciar. Confira e tente novamente.'
    }
  }
} satisfies Pick<Translations, 'boot'>
