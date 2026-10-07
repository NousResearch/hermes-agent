import type { TranslationOverrides } from './define-locale'

export const ptSettingsB = {
  managedUpdates: {
    title: 'Atualizações gerenciadas',
    intro:
      'Atualize instalações SSH gerenciadas pelo Desktop de forma transacional: as sessões são drenadas, o checkout remoto é atualizado e todos os perfis são restaurados com um recibo correlacionado.',
    sshConnection: 'Instalação SSH gerenciada pelo Desktop',
    update: 'Atualizar',
    updating: 'Atualizando…',
    progress: 'Drenando as sessões, atualizando a instalação remota e restaurando os perfis…',
    updated: 'Atualizado',
    partial: 'Atualizado, mas a restauração falhou',
    refused: 'Recusado',
    failed: 'Falha na atualização',
    alreadyRunning: 'Já há uma atualização em andamento',
    receipt: (id, outcome) => `Recibo ${id} · ${outcome}`,
    receiptVersions: (pre, post) => `${pre} → ${post}`,
    scopesRestored: profiles => `Perfis restaurados: ${profiles}`,
    scopeNotRestored: (profile, error) => `Perfil “${profile}” não restaurado: ${error}`
  },
  gateway: {
    loading: 'Carregando as configurações do gateway…',
    unavailableTitle: 'Configurações do gateway indisponíveis',
    unavailableDesc:
      'As configurações de conexão só podem ser alteradas no app Hermes Desktop do computador que o executa.',
    title: 'Conexão do gateway',
    envOverride: 'substituído por variável de ambiente',
    intro:
      'Local por padrão. Use remoto quando este app deve controlar um backend do Hermes em outro lugar. As conexões de gateway são por máquina; os perfis são descobertos a partir dos gateways que você conecta.',
    envOverrideTitle: 'Esta conexão foi fixada pelo modo como o Hermes foi iniciado.',
    envOverrideDesc:
      'Uma configuração de inicialização fora do app escolheu esta conexão, então as opções abaixo são somente leitura. Reinicie o Hermes sem essa configuração — ou peça a quem a definiu — para alterá-la aqui.',
    modeTitle: 'Modo de conexão',
    localTitle: 'Gateway local',
    localDesc: 'Inicia um backend privado do Hermes no localhost. É o padrão e funciona offline.',
    remoteTitle: 'Gateway remoto',
    remoteDesc: 'Conecta este shell desktop a um backend remoto do Hermes.',
    remoteAuthHint:
      'Gateways hospedados usam OAuth ou usuário e senha; os autohospedados podem usar um token de sessão.',
    cloudTitle: 'Hermes Cloud',
    cloudDesc: 'Entre uma vez no Hermes Cloud e escolha entre os agentes da sua conta — nenhuma URL para colar.',
    cloudSignInTitle: 'Hermes Cloud',
    cloudSignIn: 'Entrar no Hermes Cloud',
    cloudSignedIn: 'Conectado ao Hermes Cloud',
    cloudNeedsSignIn: 'Entre no Hermes Cloud para descobrir os agentes da sua conta.',
    cloudSignedInDesc: 'Você está conectado. Escolha um agente abaixo; a sessão é atualizada automaticamente.',
    cloudAgentsTitle: 'Seus agentes',
    cloudOrgPickerTitle: 'Escolha uma organização',
    cloudOrgSelect: 'Selecionar',
    cloudOrgChange: 'Trocar organização',
    cloudOrgRole: role => `Função: ${role}`,
    cloudLoadingAgents: 'Carregando seus agentes…',
    cloudNoAgents: {
      before: 'Nenhum agente encontrado nesta conta. Crie um no ',
      linkText: 'portal da Nous',
      after: ', depois atualize.'
    },
    cloudRefresh: 'Atualizar',
    cloudConnect: 'Conectar',
    cloudSavedTitle: 'Gateways Cloud salvos',
    cloudSavedDesc:
      'Use um gateway salvo sem alterar o seu padrão. Faça login abaixo para adicionar instâncias. Gerencie nomes e logins na lista de conexões salvas.',
    cloudUseSaved: 'Usar gateway',
    cloudActive: 'Ativo nesta janela',
    cloudConnecting: 'Conectando…',
    cloudDiscoverFailed: 'Não foi possível carregar os seus agentes do Hermes Cloud',
    cloudConnectFailed: 'Não foi possível conectar a esse agente',
    cloudSignInFailed: 'Falha no login do Hermes Cloud',
    cloudSignedOutTitle: 'Sessão do Hermes Cloud encerrada',
    cloudSignedOutMessage: 'A sessão do Hermes Cloud foi limpa.',
    cloudConnectedTitle: 'Conectado',
    cloudConnectedPill: 'Conectado',
    cloudConnectedTo: name => `Conectado a ${name}.`,
    cloudAgentProvisioning: 'Provisionando…',
    cloudStatusLabel: status => `Status: ${status}`,
    remoteUrlTitle: 'URL remota',
    remoteUrlDesc: 'URL base do backend remoto do painel. Prefixos de caminho são aceitos, por exemplo /hermes.',
    probing: 'Verificando como este gateway autentica…',
    probeError:
      'O Hermes não consegue acessar esse endereço. Verifique a URL e se o outro computador está executando o Hermes — as opções de login aparecem assim que ele responder.',
    signedIn: 'Conectado',
    signIn: 'Entrar',
    signOut: 'Sair',
    signInWith: provider => `Entrar com ${provider}`,
    authTitle: 'Autenticação',
    authSignedInPassword:
      'Este gateway usa usuário e senha. Você está conectado; a sessão é atualizada automaticamente.',
    authSignedInOauth: 'Este gateway usa OAuth. Você está conectado; a sessão é atualizada automaticamente.',
    authNeedsPassword: 'Este gateway usa nome de usuário e senha. Entre para autorizar este app desktop.',
    authNeedsOauth: provider => `Este gateway usa OAuth. Entre com ${provider} para autorizar este app desktop.`,
    tokenTitle: 'Token de sessão',
    tokenDesc:
      'O token de sessão do painel usado para acesso REST e WebSocket. Deixe em branco para manter o token salvo.',
    existingToken: value => `Token existente ${value}`,
    savedToken: 'salvo',
    pasteSessionToken: 'Cole o token de sessão',
    plainTextConfirmTitle: 'Armazenar o token do gateway em texto simples?',
    plainTextConfirmDesc:
      'Nenhum serviço de chaveiro do sistema operacional foi encontrado nesta máquina, então o token seria salvo sem criptografia no arquivo de configurações de conexão do app, legível por qualquer processo executado como este usuário. Instale ou ative o chaveiro do sistema (GNOME Keyring ou KWallet no Linux) para ter armazenamento criptografado.',
    plainTextConfirmAction: 'Salvar em texto simples',
    plainTextStoredTitle: 'Token armazenado em texto simples',
    plainTextStoredDesc:
      'O armazenamento seguro não está disponível, então o token salvo fica sem criptografia no arquivo de configurações de conexão do app nesta máquina. Instale ou ative o chaveiro do sistema (GNOME Keyring ou KWallet no Linux) para criptografá-lo.',
    keychainEncryptionTitle: 'Criptografar segredos salvos com o chaveiro do sistema',
    keychainEncryptionDesc:
      'Desativado por padrão. Quando ativado, tokens de gateway e credenciais de login são criptografados com o chaveiro do sistema (Acesso às Chaves, GNOME Keyring ou DPAPI do Windows); o sistema pode pedir permissão ou uma senha. Quando desativado, são armazenados como arquivos simples legíveis apenas pela sua conta de usuário.',
    keychainEncryptionFailed: 'Não foi possível alterar a criptografia dos segredos',
    testRemote: 'Testar remoto',
    saveForRestart: 'Salvar para a próxima reinicialização',
    saveAndReconnect: 'Salvar e reconectar',
    diagnostics: 'Diagnóstico',
    diagnosticsDesc: 'Mostra o desktop.log no gerenciador de arquivos, útil quando o gateway não consegue iniciar.',
    openLogs: 'Abrir logs',
    incompleteTitle: 'Gateway remoto incompleto',
    incompleteSignIn: 'Informe uma URL remota e faça login antes de alternar para o remoto.',
    incompleteToken: 'Informe uma URL remota e um token de sessão antes de alternar para o remoto.',
    incompleteSignInTest: 'Informe uma URL remota e faça login antes de testar.',
    incompleteTokenTest: 'Informe uma URL remota e um token de sessão antes de testar.',
    enterUrlFirst: 'Informe uma URL remota primeiro.',
    restartingTitle: 'Conexão do gateway reiniciando',
    savedTitle: 'Configurações do gateway salvas',
    restartingMessage: 'O Hermes Desktop vai se reconectar com as configurações salvas; o shell continua aberto.',
    savedMessage: 'Salvo para a próxima reinicialização.',
    connectedTo: (baseUrl, version) => `Conectado a ${baseUrl}${version ? ` · Hermes ${version}` : ''}`,
    reachableTitle: 'Gateway remoto acessível',
    signedOutTitle: 'Sessão encerrada',
    signedOutMessage: 'A sessão do gateway remoto foi limpa.',
    failedLoad: 'Falha ao carregar as configurações do gateway',
    signInFailed: 'Falha no login',
    signOutFailed: 'Falha ao sair',
    testFailed: 'O teste do gateway remoto falhou',
    applyFailed: 'Não foi possível aplicar as configurações do gateway',
    saveFailed: 'Não foi possível salvar as configurações do gateway',
    sshTitle: 'Conectar via SSH',
    sshDesc:
      'O Hermes é iniciado no host remoto por SSH e tunelado até este app; você não precisa iniciar nem expor nada. Exige acesso SSH por chave funcionando para o host.',
    sshTrustHint:
      'A primeira chave de host apresentada é considerada confiável e fixada; mudanças posteriores são bloqueadas.',
    sshHostTitle: 'Host',
    sshHostDesc: 'usuário@host ou um alias de Host do ~/.ssh/config.',
    sshHostPick: 'Selecione um host…',
    sshHostPickTitle: 'Host',
    sshHostPickDesc: 'Um alias de Host do ~/.ssh/config, ou Personalizado para digitar um.',
    sshHostCustom: 'Personalizado (digitar manualmente)…',
    sshUserTitle: 'Usuário',
    sshUserDesc: 'Em branco = ~/.ssh/config ou o seu usuário atual.',
    sshUserPlaceholder: 'do ~/.ssh/config',
    sshPortTitle: 'Porta',
    sshPortDesc: 'Em branco = 22 ou a porta do ~/.ssh/config.',
    sshKeyTitle: 'Arquivo de identidade',
    sshKeyDesc: 'Caminho da chave privada. Em branco = ssh-agent ou ~/.ssh/config.',
    sshHermesPathTitle: 'Caminho do Hermes (opcional)',
    sshHermesPathDesc: 'Caminho completo do binário hermes remoto. Em branco = detecção automática.',
    sshHermesPathPlaceholder: 'detecção automática',
    sshTestConnection: 'Testar SSH',
    sshConnect: 'Conectar',
    sshButtonsHint: 'Salvar aplica na próxima inicialização. Conectar reconecta agora.',
    sshReachable: (host, platform) => `Acessível: ${host} (${platform}) — Hermes encontrado`,
    sshIncompleteHost: 'Informe um host SSH antes de conectar.',
    sshErrUnreachable: 'Não foi possível alcançar esse host por SSH. Verifique o host, a porta e a sua rede.',
    sshErrAuth:
      'Falha na autenticação SSH. Carregue sua chave no ssh-agent (ssh-add) ou defina um IdentityFile no ~/.ssh/config; o Hermes executa o ssh de forma não interativa.',
    sshErrHostKey:
      'A chave do host MUDOU desde a última conexão. Confirme que isso é esperado, depois execute ssh-keygen -R <host> e reconecte.',
    sshErrNotInstalled:
      'O Hermes não está instalado no host remoto. Instale-o lá (curl -fsSL https://hermes-agent.nousresearch.com/install.sh | sh) ou defina o caminho do Hermes.',
    sshErrPlatform:
      'Plataforma remota não compatível. O modo SSH do Hermes Desktop oferece suporte a hosts remotos Linux, macOS e Windows.',
    sshErrTimeout: 'A conexão SSH expirou. O host pode estar inacessível ou em repouso.',
    sshErrUpdateRequired: 'Atualize o Hermes no host remoto antes de conectar com o Desktop SSH.',
    sshErrInteractiveAuth:
      'O Tailscale SSH exige uma verificação interativa no navegador. No Terminal, execute `ssh <host> true`, conclua a verificação e tente de novo: o Hermes executa o SSH de forma não interativa.',
    sshErrUnknown: 'A conexão SSH falhou.'
  },
  keys: {
    loading: 'Carregando chaves de API e credenciais…',
    failedLoad: 'Falha ao carregar as chaves de API',
    empty: 'Nada configurado nesta categoria ainda.'
  },
  search: {
    placeholder: 'Buscar em todas as configurações…',
    pill: 'Buscar'
  },
  profileScope: {
    appliesTo: 'Aplica-se a',
    editsProfile: profile => `As alterações desta página valem para o perfil “${profile}”.`
  },
  mcp: {
    loading: 'Carregando servidores MCP…',
    invalidJson: 'JSON de MCP inválido',
    saveFailed: 'Falha ao salvar',
    removeFailed: 'Falha ao remover',
    reloadFailed: 'Falha ao recarregar o MCP',
    savedTitle: 'Servidor MCP salvo',
    savedMessage: name => `${name} é aplicado depois de recarregar o MCP.`,
    disabled: 'desativado',
    name: 'Nome',
    serverJson: 'JSON do servidor',
    remove: 'Remover',
    test: 'Testar conexão',
    catalogLoading: 'Carregando o catálogo MCP…',
    catalogInstallFailed: name => `Falha ao instalar ${name}`,
    catalogEnvRequired: 'Preencha os valores obrigatórios antes de instalar.',
    capabilitySummary: (tools, prompts, resources) =>
      `${[`${tools} ferramentas`, ...(prompts ? [`${prompts} prompts`] : []), ...(resources ? [`${resources} recursos`] : [])].join(', ')} ativados`,
    costTokens: tokens => `~${tokens} tokens/chamada`,
    usage30d: uses => `${uses} usos/30d`,
    statusConnecting: 'Conectando…',
    statusNeedsAuth: 'Requer autenticação',
    statusError: 'Erro',
    statusOff: 'Desativado',
    allServers: 'Todos os servidores',
    authenticatedTitle: 'Autenticado',
    authenticatedMessage: (server, count) => `${server}: ${count} ferramentas`,
    authenticate: 'Autenticar',
    noOutput: 'Ainda sem saída.',
    deepLinkTitle: 'Adicionar servidor MCP?',
    deepLinkDescription:
      'Um link pediu para adicionar este servidor MCP ao Hermes. Revise a configuração exata abaixo: ela vem do link, não do Hermes.',
    deepLinkStdioWarning:
      'Este servidor executa um processo local na sua máquina com o comando exibido abaixo. Só continue se confiar na origem.',
    deepLinkConfirm: 'Adicionar servidor',
    deepLinkNameInvalid: 'Os nomes usam de 1 a 64 letras, dígitos, pontos, hifens ou sublinhados.',
    deepLinkNameConflict: name => `Já existe um servidor chamado ${name}: escolha outro nome ou cancele.`,
    deepLinkErrorTitle: 'Link de instalação de MCP rejeitado',
    deepLinkErrorName: 'O nome do servidor no link está ausente ou é inválido.',
    deepLinkErrorConfig: 'A configuração do link não é um JSON válido codificado em base64.',
    deepLinkErrorShape: 'A configuração deve ser um objeto JSON com um campo `url` ou `command` do tipo texto.',
    deepLinkErrorUrl: 'Somente URLs de servidor http:// e https:// são permitidas.',
    deepLinkErrorTooLarge: 'O conteúdo da configuração excede o limite de 32 KB.'
  },
  model: {
    setupProviderFallback: 'provedor',
    setUpProvider: name => `Configurar ${name}`,
    staleAuxBefore: (count, names) =>
      `${count} tarefa${count === 1 ? '' : 's'} auxiliar${count === 1 ? '' : 'es'} (${names}) ainda ${count === 1 ? 'roda' : 'rodam'} em `,
    staleAuxAfter: ', e não o seu modelo principal.',
    staleAuxOtherProviders: 'outros provedores',
    moaEnabled: 'Ativado',
    moaSetDefault: 'Definir como padrão',
    moaNewPresetPlaceholder: 'novo preset',
    moaAddPreset: 'Adicionar preset',
    customModel: 'Modelo personalizado…',
    customModelPlaceholder: 'ID do modelo',
    chooseFromList: 'Escolher da lista',
    moaDefault: 'Padrão:',
    moaReferenceToggle: (enabled, index) => `${enabled ? 'Desativar' : 'Ativar'} a referência ${index}`,
    moaReferenceTitle: index => `Referência ${index}`,
    moaAddReference: 'Adicionar modelo de referência',
    loading: 'Carregando a configuração de modelos…',
    appliesDesc: 'Aplica-se a novas sessões. Use o seletor de modelo no chat para trocar o modelo atual.',
    provider: 'Provedor',
    model: 'Modelo',
    applying: 'Aplicando…',
    mainAppliedTitle: 'Modelo principal atualizado',
    mainAppliedMessage: model => `As novas sessões usarão ${model}.`,
    defaultsLabel: 'Padrões',
    reasoning: 'Raciocínio',
    reasoningOff: 'Desativado',
    speed: 'Velocidade',
    speedStandard: 'Padrão',
    defaultsFailed: 'Falha ao salvar os padrões de modelo',
    loadFailed: 'Não foi possível carregar os modelos',
    restartRequired:
      'Este backend está executando código antigo após uma atualização. Reinicie-o para carregar o novo código.',
    restartBackend: 'Reiniciar o backend',
    restartingBackend: 'Reiniciando o backend…',
    restartFailed: 'Não foi possível reiniciar o backend',
    auxiliaryTitle: 'Modelos auxiliares',
    resetAllToMain: 'Redefinir para o principal',
    staleAuxDismiss: 'Não mostrar novamente',
    auxiliaryDesc:
      'Tarefas auxiliares rodam no modelo principal por padrão. Atribua um modelo dedicado a qualquer tarefa para sobrescrever.',
    setToMain: 'Definir para principal',
    change: 'Alterar',
    autoUseMain: 'auto · usar modelo principal',
    inheritMainEffort: 'herdar · esforço do modelo principal',
    inheritsFrom: task => `herda de ${task}`,
    followTask: task => `Seguir ${task}`,
    providerDefault: '(padrão do provedor)',
    fallbackAdd: 'Adicionar fallback',
    fallbackEmpty: 'Nenhum modelo de fallback: o modelo padrão é usado, a menos que falhe.',
    notInCatalog: 'não está na lista de modelos deste provedor; as chamadas podem recorrer a um reserva.',
    moaTitle: 'Mistura de agentes',
    moaPreset: 'Preset',
    moaDescription:
      'Configure presets nomeados que aparecem como modelos no provedor Mistura de agentes. O agregador é o modelo que age: ele executa todas as etapas do loop de ferramentas, e quase todo o custo da execução é cobrado do provedor dele. Por padrão, as referências só aconselham uma vez por turno do usuário.',
    moaAggregator: 'Agregador',
    moaAggregatorBilled: 'modelo que age · cobrado pela execução',
    moaReferenceHint: 'aconselha uma vez por turno, por padrão',
    tasks: {
      vision: {
        label: 'Visão',
        hint: 'Análise de imagens'
      },
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
        label: 'MCP',
        hint: 'Roteamento de ferramentas MCP'
      },
      title_generation: {
        label: 'Geração de títulos',
        hint: 'Títulos das sessões'
      },
      review: {
        label: 'Revisão',
        hint: 'Subagente revisor do /review'
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
        hint: 'Revisão de uso de skills'
      }
    }
  },
  localModels: {
    connectionChanged: 'A conexão dos modelos locais mudou',
    title: 'Modelos locais',
    runtimeTitle: 'Runtime local',
    runtimeReady: backend => `Pronto · ${backend}`,
    serverRunning: 'Em execução',
    runtimeInstalled: 'Runtime llama.cpp instalado',
    runtimeInstalledDetail: (tag, backend) =>
      `Build ${tag}, backend ${backend}. O Hermes inicia e gerencia o servidor para você.`,
    installTitle: 'Instalar o runtime local',
    installDetail:
      'Baixa o mecanismo de inferência llama.cpp (algumas centenas de MB). Os modelos que você baixa rodam inteiramente nesta máquina: sem conta, e nada sai do seu computador.',
    installAction: 'Instalar runtime',
    installing: 'Instalando o runtime…',
    installFailed: 'Falha na instalação do runtime',
    hardwareTitle: 'Esta máquina',
    hardwareLoading: 'Verificando o seu hardware…',
    vram: label => `${label} de memória da GPU`,
    ram: label => `${label} de RAM`,
    unifiedMemory: 'Memória unificada',
    modelsTitle: 'Modelos',
    recommended: 'Recomendado',
    recommendedReason: {
      'product-default': 'O modelo padrão para esta máquina, escolhido pelo fabricante dela.',
      'best-quality-resident':
        'O modelo de maior qualidade que roda inteiramente na sua GPU em velocidade total. As sugestões equilibram qualidade e velocidade prevista neste hardware.',
      'speed-gated-quality':
        'Um modelo de maior qualidade cabe nesta máquina, mas responderia devagar demais pela largura de banda da memória. Este é o melhor modelo que continua rápido.',
      'fastest-resident':
        'Nenhum modelo atinge a velocidade total neste hardware; este é o que mais se aproxima, rodando inteiramente na memória da GPU.'
    },
    noRecommendationTitle: 'Sem recomendação automática para esta máquina',
    noRecommendationDetail:
      'A configuração automática exige um modelo curado que caiba inteiramente na memória da GPU ou unificada. Você ainda pode escolher um modelo abaixo ou explorar mais modelos.',
    noRecommendationAction: 'Explorar modelos',
    downloaded: 'Baixado',
    downloadAction: size => `Baixar · ${size}`,
    downloadProgress: (done, total) => `${done} de ${total}`,
    downloadStatusRunning: 'Baixando',
    downloadSpeed: rate => `${rate}`,
    downloadEta: time => `~${time} restantes`,
    downloadEtaSeconds: count => `${count} s`,
    downloadEtaMinutes: count => `${count} min`,
    downloadEtaHours: (hours, minutes) => (minutes ? `${hours} h ${minutes} min` : `${hours} h`),
    downloadPausedLabel: 'Pausado',
    downloadPauseAction: 'Pausar',
    downloadResumeAction: 'Retomar',
    downloadDoneToast: model => `${model} está pronto.`,
    installDoneToast: 'Runtime local instalado e pronto.',
    quickstartTitle: 'Rode um modelo nesta máquina',
    quickstartDetail: (model, size) =>
      `Um clique configura tudo: o mecanismo local, ${model} (download de ${size}) e o seu padrão para novos chats. Nada sai deste computador.`,
    quickstartDetailReady: model => `Um clique torna ${model} o seu padrão para novos chats. Tudo roda nesta máquina.`,
    quickstartAction: 'Configurar para mim',
    quickstartConfigure: 'Configure…',
    quickstartDoneToast: model => `${model} está configurado: os novos chats rodam nesta máquina.`,
    quickstartFailed: 'Falha na configuração do modelo local',
    quickstartStageEngine: 'Mecanismo',
    quickstartStageModel: 'Modelo',
    quickstartStageFinish: 'Concluir',
    useAction: 'Usar',
    activePill: 'Padrão',
    updateTitle: 'Atualização do mecanismo disponível',
    updateDetail: (next, current) =>
      `Há um build mais novo do llama.cpp (${next}) pronto para instalar; você está no ${current}. Os modelos continuam funcionando durante o download.`,
    updateAction: 'Atualizar mecanismo',
    updating: 'Atualizando o mecanismo…',
    upToDateTitle: 'Mecanismo atualizado',
    upToDateDetail: (tag, backend) => `Executando llama.cpp ${tag} (${backend}) — a build configurada.`,
    activeDetail: 'Novos chats usam este modelo, que é carregado quando você envia a primeira mensagem',
    activeNotLoaded: 'Carrega na sua primeira mensagem',
    loadedPill: 'Na memória',
    placementResident: 'tudo na GPU',
    placementSpilled: 'parcialmente na RAM',
    placementResidentTip: 'Roda inteiramente na memória da GPU com esta janela de contexto, em velocidade total.',
    placementSpilledTip:
      'Parte deste modelo roda a partir da RAM do sistema: funciona, mas mais devagar. Uma versão mais compacta ou um contexto menor caberia por inteiro.',
    loadingPill: 'Carregando…',
    ejectTip: 'Liberar a memória da GPU (carrega de novo na próxima mensagem)',
    ejected: 'Modelo descarregado. Memória da GPU liberada.',
    ejectFailed: 'Não foi possível descarregar o modelo',
    stopServer: 'Desativar',
    startServer: 'Ativar',
    runtimeRunningDetail:
      'O servidor local está em execução. Desativá-lo libera toda a memória da GPU e impede que novos chats usem modelos locais até você reativá-lo.',
    serverStopped: 'Servidor local parado. Memória da GPU liberada.',
    serverStarted: 'Servidor local em execução.',
    serverStopFailed: 'Não foi possível parar o servidor local',
    serverStartFailed: 'Não foi possível iniciar o servidor local',
    activating: 'Iniciando…',
    activateFailed: model => `Não foi possível trocar para ${model}`,
    activateDoneToast: model => `Os novos chats usam ${model}.`,
    downloadFailed: model => `Falha no download de ${model}`,
    downloadPauseFailed: model => `Não foi possível pausar o download de ${model}`,
    downloadResumeFailed: model => `Não foi possível retomar o download de ${model}`,
    pillFitsGpu: 'Cabe na sua GPU',
    pillUsesRam: 'Usa a RAM do sistema',
    pillTooBig: 'Grande demais para esta máquina',
    browseTitle: 'Encontrar mais modelos',
    browseHint:
      'Pesquise em todo o Hugging Face. Os modelos que você baixa aqui são dimensionados automaticamente para a sua máquina, mas não são testados por nós.',
    browsePlaceholder: 'Buscar modelos por nome ou autor…',
    browseSearching: 'Pesquisando no Hugging Face',
    browseListing: 'Lendo os arquivos do modelo',
    browseShowFiles: 'Mostrar arquivos',
    browseRefresh: 'Atualizar',
    browseDownloads: 'downloads',
    browseLikes: 'curtidas',
    browseGated: 'exige login no Hugging Face',
    browseNoGguf: 'Nenhum arquivo de modelo compatível encontrado.',
    browseFitUnknown: 'Compatibilidade desconhecida',
    browseAlreadyDownloaded: 'Já baixado.',
    addedByYou: 'Adicionado por você',
    browseDownloadStarted: 'Baixando {name}',
    browseDownloadAria: 'Baixar {name}',
    sideloadButton: 'Adicionar arquivo de modelo',
    sideloadTitle: 'Escolha um arquivo de modelo GGUF',
    sideloadDone: '{name} adicionado.',
    sideloadAlreadyPresent: 'Já está na sua biblioteca.',
    pillFullContext: max => `Contexto completo de ${max}`,
    pillFullContextTip: 'Roda com a janela de contexto completa do modelo desde o início',
    pillUpTo: max => `Contexto de até ${max}`,
    pillGrowsTip: 'Cresce automaticamente conforme a conversa precisa de mais espaço',
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
      signIn: 'Fazer login',
      title: 'Você está no plano gratuito da Nous',
      message: 'Faça login com uma conta Nous para liberar mais modelos e ferramentas.',
      caption:
        'Roda em nous/welcome, com conectores incluídos. Fazer login mantém seus conectores e adiciona as ferramentas que exigem conta e todos os outros modelos.',
      name: 'Nous · plano gratuito',
      footnote:
        'O plano gratuito não tem saldo nem nada a pagar. Pagamento e uso aparecem quando você faz login com uma conta Nous.',
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
      openVerification: 'Abrir a página de verificação',
      dismiss: 'Dispensar',
      waiting: 'Aguardando o link de verificação…',
      verify: 'Verifique para continuar',
      deniedTitle: 'A verificação não foi aprovada',
      deniedBody: 'A verificação terminou sem permitir Gastos Remotos para este terminal.',
      successTitle: 'Verificação concluída',
      successBody: 'Gastos Remotos estão permitidos para este terminal.'
    },
    charge: {
      added: amount => (amount ? `US$ ${amount} adicionados.` : 'Créditos adicionados.'),
      failedTitle: 'Falha na cobrança',
      unconfirmedTitle: 'Resultado da cobrança não confirmado',
      unconfirmedBody: message =>
        `${message} O resultado da sua última cobrança não foi confirmado: confira o saldo e o histórico antes de tentar de novo.`,
      checkTitle: 'Não foi possível verificar a cobrança',
      checkBody: 'Não foi possível verificar a cobrança.',
      untrackedTitle: 'Não foi possível acompanhar a cobrança',
      untrackedBody: 'O serviço de cobrança aceitou a solicitação, mas não retornou um ID de cobrança.',
      timeoutTitle: 'Ainda processando após 5 minutos',
      timeoutBody: 'A cobrança ainda pode ser concluída. Verifique o portal antes de tentar de novo.',
      authenticationRequired: 'O seu banco exige verificação (3DS). Conclua no portal para finalizar esta compra.',
      expired: 'O seu cartão expirou. Atualize-o no portal.',
      declined: 'O seu cartão foi recusado. Tente outro cartão no portal.',
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
      customAmount: 'Valor de crédito personalizado',
      title: 'Comprar créditos agora',
      buyButton: 'Comprar',
      processing: 'Processando… verificando a liquidação',
      added: amount => `${amount} adicionados. O saldo está sendo atualizado.`,
      retry: 'Tentar novamente',
      openPortal: 'Abrir portal'
    },
    plan: {
      title: 'Planos',
      changePlan: 'Mudar de plano',
      viewPlans: 'Ver planos',
      backAria: 'Voltar para a cobrança',
      current: 'Plano atual',
      scheduled: 'Agendado',
      empty: 'Nenhum plano disponível para troca no momento.',
      undo: 'Desfazer',
      undoing: 'Desfazendo…',
      downgrade: 'Reduzir plano',
      confirmDowngrade: 'Confirmar a redução',
      tryAgain: 'Tentar novamente',
      checkingChange: 'Verificando esta alteração…',
      cannotChange: 'Essa alteração não pode ser feita aqui.',
      alreadyOn: name => `Você já está no plano ${name}: nada a alterar.`,
      notScheduleable: 'Esta alteração não pode ser agendada aqui.',
      scheduling: 'Agendando…',
      cancel: 'Cancelar',
      effectScheduled: (targetName, effectiveAt, creditsDelta) =>
        `Mudança para ${targetName}, com efeito em ${effectiveAt}. Sem cobrança agora; você mantém o plano atual até lá.${creditsDelta ? ` Alteração dos créditos mensais: ${creditsDelta}.` : ''}`
    },
    autoReload: {
      threshold: 'Limite',
      thresholdAria: 'Limite da recarga automática',
      reloadTo: 'Recarregar até',
      reloadToAria: 'Valor de recarga automática',
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
          title: 'Conecte a sua conta Nous',
          message: 'Faça login com a sua conta Nous para ver aqui o seu saldo, plano e uso.',
          action: 'Fazer login'
        },
        openPortal: 'Abrir portal ↗',
        noCard: {
          title: 'Nenhuma forma de pagamento cadastrada',
          message:
            'A compra de créditos de recarga e a recarga automática ficam desativadas até haver um cartão cadastrado. Adicione um no portal.',
          action: 'Adicionar cartão ↗'
        }
      },
      paymentMethod: {
        title: 'Forma de pagamento',
        description: 'Gerencie o cartão usado para recargas e renovações de assinatura.',
        addAction: 'Adicionar forma de pagamento',
        updateAction: 'Atualizar',
        provenance: {
          autoRefill: 'cartão da recarga automática',
          customerDefault: 'padrão do cliente',
          subPin: 'cartão da assinatura',
          suffix: label => ` - ${label}`
        }
      },
      buyCredits: {
        description: 'Uma única cobrança no seu cartão, adicionada ao seu saldo hoje.'
      },
      autoRefill: {
        title: 'Recarregar quando estiver baixo',
        genericDescription: 'Mantém o saldo abastecido quando cair abaixo do seu limite.',
        offPill: 'Desativada',
        enabledPill: 'Ativada',
        notAvailablePill: '—',
        manageCaption: 'Gerencie a recarga automática pelo portal.',
        turnOnCaption: 'Ative a recarga automática pelo portal',
        chargesDescription: (reloadTo, threshold) =>
          `Cobra ${reloadTo} automaticamente quando o saldo ficar abaixo de ${threshold}.`,
        distinctCardCaption: cardLabel => `A recarga automática cobra ${cardLabel}: concilie no portal`,
        distinctCardFallback: 'outro cartão',
        reconcileAction: 'Conciliar ↗'
      },
      usage: {
        subscriptionCredits: {
          title: 'Créditos da assinatura',
          barLabel: 'Créditos da assinatura restantes',
          captionResets: date => `Renova em ${date}`,
          valueOf: (remaining, monthly) => `${remaining} de ${monthly} restantes`,
          valueOver: (remaining, monthly, over) => `${remaining} de ${monthly} restantes · ${over} excedentes`
        },
        topupCredits: {
          title: 'Créditos de recarga',
          caption: 'Não expiram'
        },
        monthlyCap: {
          title: 'Limite mensal de gastos',
          barLabel: 'Limite mensal de gastos utilizado',
          captionDefault: 'Teto padrão',
          captionSpending: 'Gastos remotos do mês',
          valueUsed: (spent, limit) => `${spent} de ${limit} usados`
        }
      },
      planCard: {
        freeTier: 'Gratuito',
        chooseAction: 'Escolher ↗',
        adjustPlanAction: 'Ajustar plano ↗',
        unavailableCaption: 'Os detalhes da assinatura estão indisponíveis; abrir o portal continua disponível.',
        downgradeCaption: (tierName, when) => `Muda para ${tierName} em ${when}.`,
        cancellationCaption: when => `Cancela em ${when}.`,
        renewsCaption: date => `Renova em ${date}`,
        noSubscriptionCaption: 'Nenhuma assinatura ativa: os modelos pagos consomem os créditos de recarga.'
      }
    },
    errors: {
      consentRequired: {
        title: 'Confirmação do cartão necessária',
        message: 'Confirme este cartão para cobranças do terminal no portal'
      },
      insufficientScope: {
        title: 'Os Gastos Remotos precisam de aprovação',
        message: 'É preciso permitir Gastos Remotos. Inicie uma recarga para permiti-los e tente novamente.'
      },
      remoteSpendingRevoked: {
        title: 'Os gastos remotos foram interrompidos',
        messageByAdmin: 'Um administrador interrompeu os gastos remotos para este terminal.',
        messageBySelf: 'Você interrompeu os gastos remotos para este terminal.'
      },
      remoteSpendingReconnect: who => `${who} Reconecte em Configurações → Gateway para reautorizar este dispositivo.`,
      sessionRevoked: {
        title: 'Sessão encerrada',
        message: 'Sua sessão foi encerrada. Faça login novamente em Configurações → Gateway.'
      },
      cliBillingDisabled: {
        title: 'Os gastos remotos estão desativados',
        message:
          'Os gastos remotos estão desativados para esta conta. Um administrador de cobrança pode ativá-los na página do Hermes Agent no portal.'
      },
      roleRequired: {
        title: 'Função de administrador necessária',
        message:
          'Adicionar fundos exige um administrador ou proprietário da organização. Peça a um administrador ou gerencie pelo portal.'
      },
      idempotencyConflict: {
        title: 'Inicie uma nova recarga',
        message: '🔴 Essa chave de cobrança já foi usada para um valor diferente. Inicie uma nova recarga.'
      },
      noPaymentMethod: {
        title: 'Nenhum cartão salvo',
        message:
          '💳 Ainda não há cartão salvo para cobranças do terminal. Cadastre um no portal (compras únicas de crédito não salvam um cartão reutilizável).'
      },
      orgAccessDenied: {
        title: 'Acesso à organização negado',
        message: 'Este token não está vinculado a uma organização que você possa gerenciar'
      },
      monthlyCapExceeded: {
        title: 'Limite mensal de gastos atingido',
        messageReached: '🔴 Limite mensal de gastos atingido.',
        messageHeadroom: remaining => `\u{1F534} Limite mensal de gastos atingido: restam US$ ${remaining} de margem.`
      },
      rateLimited: {
        title: 'Cobranças demais no momento',
        message: mins =>
          mins > 0
            ? `\u{1F7E1} Cobranças demais no momento (tente de novo em ~${mins} min). Isto não é uma falha de pagamento.`
            : '\u{1F7E1} Cobranças demais no momento. Isto não é uma falha de pagamento.'
      },
      stripeUnavailable: {
        title: 'A Stripe está com problemas',
        message: mins =>
          mins > 0
            ? `A Stripe está com problemas: tente de novo em ~${mins} min`
            : 'A Stripe está com problemas: tente de novo em instantes'
      },
      upgradeCapExceeded: {
        title: 'Limite diário de trocas de plano atingido',
        message: 'Limite diário de trocas de plano atingido. Tente novamente amanhã'
      },
      endpointUnavailable: {
        title: 'Endpoint de cobrança indisponível',
        message:
          'O endpoint de cobrança retornou uma resposta que não é JSON (ele pode não estar disponível nesta implantação).'
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
        title: 'A solicitação de cobrança falhou',
        message: 'A solicitação de cobrança falhou.'
      }
    }
  },
  providers: {
    connectAccount: 'Conectar uma conta',
    haveApiKey: 'Tem uma chave de API?',
    intro:
      'Entre com uma assinatura — nenhuma chave de API para copiar. O Hermes executa o login pelo navegador para você, aqui mesmo no app.',
    connected: 'Conectado',
    collapse: 'Recolher',
    connectAnother: 'Conectar outro provedor',
    otherProviders: 'Outros provedores',
    disconnect: 'Desconectar',
    disconnectInTerminal: 'Desconectar (executa o comando de remoção no terminal)',
    removeConfirm: provider => `Remover ${provider}?`,
    removeExternalGeneric: provider => `${provider} é gerenciado pela própria CLI: remova-o por lá.`,
    removeKeyManaged: provider => `${provider} está configurado por uma chave de API. Remova-o em Chaves de API.`,
    removeTerminalConfirm: (provider, command) =>
      `Desconectar ${provider}? Isto executa “${command}” no terminal para limpar a credencial.`,
    removeTerminalRunning: provider => `Executando a desconexão de ${provider} no terminal…`,
    removedTitle: 'Conta removida',
    removedMessage: provider => `${provider} foi removido.`,
    failedRemove: provider => `Não foi possível remover ${provider}`,
    noProviderKeys: 'Nenhuma chave de API de provedor disponível.',
    searchKeys: 'Buscar provedores…',
    noKeysMatch: 'Nenhum provedor corresponde à sua busca.',
    localEndpoint: {
      title: 'Endpoint local / personalizado',
      description:
        'Aponte o Hermes para qualquer endpoint compatível com a OpenAI (Zyphra, vLLM, llama.cpp, Ollama etc.).'
    },
    loading: 'Carregando provedores…'
  },
  sessions: {
    loading: 'Carregando sessões arquivadas…',
    archivedTitle: 'Sessões arquivadas',
    archivedIntro:
      'Chats arquivados ficam ocultos na barra lateral, mas guardam todas as mensagens. Segure Ctrl/⌘ e clique num chat na barra para arquivá-lo.',
    emptyArchivedTitle: 'Nada arquivado',
    emptyArchivedDesc: 'Arquive um chat para ocultá-lo aqui.',
    unarchive: 'Desarquivar',
    deletePermanently: 'Excluir permanentemente',
    messages: count => `${count} ${count === 1 ? 'mensagem' : 'mensagens'}`,
    restored: 'Restaurada',
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
    removeConfirm: key => `Remover ${key} do .env?`,
    set: 'Definida',
    notSet: 'Não definida',
    selectedTitle: 'Provedor selecionado',
    selectedMessage: provider => `${provider} agora está ativo.`,
    failedSelect: provider => `Falha ao selecionar ${provider}`,
    failedLoad: 'Falha ao carregar a configuração de ferramentas',
    noProviderOptions: 'Este toolset não tem opções de provedor: ative-o e ele funciona com a sua configuração atual.',
    noProviders: 'Nenhum provedor disponível para este toolset no momento.',
    ready: 'Pronto',
    needsSignIn: 'Requer login',
    needsSetup: 'Requer configuração',
    activeBackend: 'Ativo',
    activeBackendHint: 'Este é o seu backend ativo',
    useBackend: 'Usar este backend',
    nousIncluded: 'Incluído em uma assinatura Nous — entre com sua conta Nous para ativar.',
    nousAuthNeededTitle: 'Entre com sua conta Nous',
    nousAuthNeededMessage: provider =>
      `${provider} está salvo, mas só funcionará depois que você entrar com sua conta Nous.`,
    nousAuthSignIn: 'Entrar',
    nousAuthDoneTitle: 'Conta Nous conectada',
    nousAuthDoneMessage: 'Os backends da sua assinatura agora estão ativos.',
    nousAuthFailed: 'O login na Nous não foi concluído',
    nousAuthFailedMessage: 'Tente novamente.',
    nousAuthTryAgain: 'Tentar novamente',
    noApiKeyRequired: 'Nenhuma chave de API necessária.',
    postSetupHint: step =>
      `Este backend precisa de uma instalação única (${step}). Roda nesta máquina e pode levar alguns minutos.`,
    postSetupInstalledHint: 'Instalado. Execute a configuração de novo apenas se algo estiver quebrado.',
    postSetupRun: 'Executar configuração',
    postSetupRerun: 'Executar a configuração novamente',
    postSetupInstalled: 'Instalado',
    postSetupRunning: 'Instalando…',
    postSetupStarting: 'Iniciando…',
    postSetupCompleteTitle: 'Configuração concluída',
    postSetupCompleteMessage: step => `${step} instalado.`,
    postSetupErrorTitle: 'Configuração concluída com erros',
    postSetupErrorMessage: step => `Verifique o log de ${step}.`,
    postSetupOpenLogs: 'Abrir logs',
    postSetupRunAgain: 'Executar novamente',
    postSetupFailed: step => `Falha ao executar a configuração de ${step}`,
    webSearchActive: backend => `Busca: ${backend}`,
    webExtractActive: backend => `Extração: ${backend}`,
    webCapabilityUnset: 'não definido',
    webUseForSearch: 'Usar para Busca',
    webUseForExtract: 'Usar para Extração',
    webUsedForSearch: 'Backend de busca',
    webUsedForExtract: 'Backend de extração',
    webCapabilitySelectedMessage: (provider, capability) => `${provider} agora cuida de ${capability} na web.`,
    failedSelectCapability: provider => `Falha ao definir ${provider}`,
    loadingModels: 'Carregando o catálogo de modelos…',
    modelSectionTitle: 'Modelo',
    modelCount: count => `${count} modelo${count === 1 ? '' : 's'}`,
    modelInUse: 'Em uso',
    modelDefault: 'padrão',
    modelInactiveHint: 'Selecione este backend primeiro para alterar o modelo dele.',
    modelSelectedTitle: 'Modelo selecionado',
    modelSelectedMessage: model => `${model} vale para as novas sessões.`,
    failedSelectModel: model => `Falha ao selecionar ${model}`,
    terminalBackend: {
      sectionTitle: 'Backend de execução',
      loading: 'Verificando os backends de execução…',
      failedLoad: 'Não foi possível carregar os backends de terminal',
      ready: 'Pronto',
      needsSetup: 'Requer configuração',
      unavailable: 'Indisponível',
      inUse: 'Em uso',
      selectedTitle: 'Backend selecionado',
      selectedMessage: backend => `Os comandos de terminal agora rodam via ${backend}. Vale para as novas sessões.`,
      failedSelect: backend => `Falha ao selecionar ${backend}`,
      needsSetupHint:
        'Você pode selecionar esta opção agora — os comandos falharão até que a configuração seja concluída.',
      needsSetupConfirmTitle: backend => `Selecionar ${backend} mesmo assim?`,
      needsSetupConfirmDescription: detail =>
        `${detail} As sessões iniciadas depois desta alteração ficarão sem as ferramentas de terminal e de arquivos até a configuração ser concluída.`,
      needsSetupConfirmDescriptionGeneric:
        'Este backend ainda não foi configurado. As sessões iniciadas depois desta alteração ficarão sem ferramentas de terminal e de arquivos até a configuração ser concluída.',
      needsSetupConfirmAction: 'Selecionar mesmo assim',
      unavailableTitle: 'Os comandos de terminal estão indisponíveis',
      unavailableMessage: backend =>
        `O Hermes não consegue executar comandos de shell agora: ${backend} não está pronto. Volte para Local ou conclua a configuração de ${backend} e tente de novo.`,
      openBackendSettings: 'Abrir configurações do terminal',
      useLocal: 'Usar o local',
      switchedToLocal: 'Os comandos de terminal agora rodam localmente. Vale para novas sessões.'
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
  }
} satisfies TranslationOverrides['settings']
