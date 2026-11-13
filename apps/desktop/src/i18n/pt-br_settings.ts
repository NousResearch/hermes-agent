import { defineFieldCopy } from '@/app/settings/field-copy'

import type { TranslationOverrides } from './define-locale'
import { ptBrGatewayHost } from './pt-br_gateway_host'
import { ptBrLocalModels } from './pt-br_local_models'
import { ptBrMemoryLayers } from './pt-br_memory_layers'
import { ptBrModelSettings } from './pt-br_model_settings'
import { ptBrNotifications } from './pt-br_notifications'
import { ptBrUninstallSection } from './pt-br_uninstall_section'
import { ptBrUpdateChannel } from './pt-br_update_channel'

export const ptBrSettingsOverrides = {
  settings: {
    subpages: {
      appearanceTheme: 'Tema',
      appearanceTypography: 'Tipografia',
      appearanceWindowLayout: 'Janela e layout',
      appearanceChatDisplay: 'Exibição da conversa',
      appearancePet: 'Mascote',
      appearanceGeneral: 'Geral',
      modelMain: 'Modelo principal',
      modelAuxiliary: 'Modelos auxiliares',
      modelMoa: 'Mistura de agentes',
      modelFallbacks: 'Modelos de reserva',
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
      voiceTranscription: 'Fala para texto',
      voiceSpeech: 'Texto para fala',
      advancedRuntime: 'Limites do agente',
      advancedTools: 'Acesso a ferramentas',
      advancedTerminal: 'Backend do terminal',
      advancedOutput: 'Limites de saída',
      advancedDelegation: 'Subagentes',
      advancedDesktop: 'Desktop e inicialização',
      gatewayConnection: 'Gateway principal',
      gatewayDevices: 'Conexões salvas',
      gatewayManagedUpdates: 'Atualizações remotas',
      gatewayManagedUpdatesUnavailable:
        'As atualizações remotas exigem uma versão do Desktop com suporte a atualizações SSH gerenciadas.',
      gatewayManagedUpdatesEmpty: 'Adicione uma conexão SSH em Conexões salvas para gerenciar suas atualizações aqui.',
      keyboardShortcuts: 'Atalhos de teclado',
      hudGesture: 'Gesto do HUD',
      screenCapture: 'Captura de tela',
      notificationAlerts: 'Alertas na área de trabalho',
      notificationSounds: 'Sons',
      archivedSessions: 'Arquivamento e retenção',
      defaultDirectory: 'Pasta padrão dos projetos',
      vaultCredentials: 'Credenciais salvas',
      vaultSources: 'Gerenciadores de senhas',
      appUpdates: 'Versão e atualizações',
      uninstall: 'Desinstalar',
      billingOverview: 'Visão geral',
      billingPlans: 'Planos'
    },
    closeSettings: 'Fechar Configurações',
    exportConfig: 'Exportar configurações',
    importConfig: 'Importar configurações',
    resetToDefaults: 'Restaurar padrões',
    resetConfirm: 'Restaurar todas as configurações para os padrões do Hermes?',
    exportFailed: 'Falha ao exportar',
    resetFailed: 'Falha ao restaurar',
    pluginPages: {
      blurb:
        'Opções adicionadas pelos plugins instalados. Cada plugin tem sua própria página, e alguns adicionam subpáginas.',
      empty: 'Nenhum plugin tem configurações ainda.',
      manage: 'Gerenciar plugins',
      agentSettings: 'Configurações do agente',
      pageCount: (n: number) => (n === 1 ? '1 página' : `${n} páginas`),
      missing: 'Esse plugin não tem uma página de configurações. Ele pode estar desativado ou desinstalado.'
    },
    nav: {
      providers: 'Provedores',
      providerAccounts: 'Contas',
      providerApiKeys: 'Chaves de API',
      providerCustomEndpoints: 'Endpoints personalizados',
      providerLocalModels: 'Modelos locais',
      gateway: 'Gateways',
      apiKeys: 'Ferramentas e chaves',
      keybinds: 'Atalhos de teclado',
      keysTools: 'Ferramentas',
      keysSettings: 'Configurações',
      mcp: 'MCP',
      archivedChats: 'Conversas arquivadas',
      sessions: 'Sessões',
      about: 'Sobre',
      billing: 'Faturamento',
      notifications: 'Notificações',
      vault: 'Senhas e logins',
      plugins: 'Plugins'
    },
    plugins: {
      title: 'Plugins do Desktop',
      openFolder: 'Abrir pasta de plugins do Desktop',
      rescan: 'Verificar novamente',
      reveal: 'Mostrar no gerenciador de arquivos',
      failed: 'falhou',
      kinds: { bundled: 'incluído', disk: 'no disco', runtime: 'ambiente de execução' },
      installModal: {
        installFromGit: 'Instalar pelo Git',
        reviewRepository: 'Revisar repositório',
        repoPlaceholder: 'https://github.com/owner/repo',
        title: 'Instalar plugin',
        description: 'Revise o conteúdo deste repositório antes de instalar qualquer coisa.',
        repoLabel: 'Repositório',
        verified: 'Verificado',
        unverified: 'Não verificado',
        reviewedByNous: 'Revisado pela Nous.',
        installsAndEnables: profile => `Instala para ${profile} e ativa imediatamente.`,
        installsOnly: profile => `Instala para ${profile}.`,
        advanced: 'Avançado',
        versionLabel: 'Versão',
        reviewedPin: sha => (sha ? `Revisão fixada ${sha}` : 'Revisão fixada'),
        latestCommit: 'Commit mais recente',
        sourceMoved: sha =>
          sha
            ? `O repositório mudou desde a sua revisão. Revise a nova versão (commit ${sha}) antes de instalar.`
            : 'O repositório mudou desde a sua revisão. Lendo a nova versão…',
        legacyBackendConfirm:
          'Este backend é mais antigo que o aplicativo e não pode garantir que a versão revisada seja a instalada: se o repositório mudou desde então, a versão mais nova será instalada. Atualize o backend para ter essa garantia.',
        installAnyway: 'Instalar mesmo assim',
        partialInstall: sha =>
          sha
            ? `Apenas parte deste plugin está instalada: a parte do agente está no commit ${sha}, mas a parte do Desktop não pôde acompanhar porque o repositório mudou desde a sua revisão. Ative Forçar reinstalação em Avançado para substituir ambas as partes pela versão exibida, ou Cancele para manter apenas a parte do agente.`
            : 'Apenas parte deste plugin está instalada: a parte do agente está instalada, mas a parte do Desktop não pôde acompanhar porque o repositório mudou desde a sua revisão. Ative Forçar reinstalação em Avançado para substituir ambas as partes pela versão exibida, ou Cancele para manter apenas a parte do agente.',
        nothingToInstall: 'Este pacote não tem nada para instalar aqui.',
        includesHeading: 'Este pacote inclui',
        agentLabel: 'Plugin do agente',
        desktopLabel: 'Interface do Desktop',
        profileLabel: 'Instalar para o perfil',
        agentTargetLocal: (profile, dir) => `Instala no backend de ${profile} (${dir})`,
        agentTargetRemote: profile => `Instala no backend conectado de ${profile}`,
        toolsConnected: n => (n === 1 ? '1 ferramenta conectada' : `${n} ferramentas conectadas`),
        skillsReady: names =>
          names.length === 1 ? `habilidade ${names[0]} pronta` : `${names.length} habilidades prontas`,
        nextChat: 'mais ferramentas disponíveis na sua próxima conversa',
        serverNotConnected: (server, reason) =>
          `O servidor MCP ${server} não está conectado${reason ? `: ${reason}` : '.'}`,
        missingEnvAction: 'Configurar',
        alreadyInstalled: (name: string) => `${name} já está instalado.`,
        desktopTarget: 'Instala na pasta local desktop-plugins deste aplicativo',
        desktopTargetFromPackage: 'Carregado neste aplicativo a partir do pacote acima — igual para todos os perfis',
        desktopOnlyNote: 'Pacotes exclusivos do Desktop não instalam um plugin de agente no backend.',
        insecureWarning:
          'Esta URL usa um esquema inseguro ou local. Prefira https:// ou git@ para instalações em produção.',
        securityIntro:
          'Instale apenas de fontes em que você confia — revise o repositório abaixo para ver o que será adicionado.',
        sourceHeading: 'Código-fonte',
        viewRepository: 'Ver repositório',
        viewPluginFiles: 'Ver arquivos do plugin',
        enableAgent: 'Ativar plugin do agente após a instalação',
        forceReinstall: 'Forçar reinstalação (substituir se já estiver instalado)',
        pinToCommit: 'Fixar em um commit (opcional)',
        pinToCommitPlaceholder: 'SHA completo do commit com 40 caracteres',
        pinToCommitHint:
          'Todos que instalarem este SHA receberão o mesmo código; o plugin recusará atualizações até ser fixado em outra revisão. Deixe vazio para usar o commit mais recente.',
        pinToCommitInvalid: 'Deve ser um SHA completo de commit com 40 caracteres (branches e tags não são aceitas).',
        install: 'Instalar',
        installing: 'Instalando…',
        probing: 'Inspecionando repositório…',
        probeUnavailable: 'A inspeção de plugins não está disponível neste ambiente.',
        desktopUnavailable: 'A instalação de plugins do Desktop não está disponível neste ambiente.',
        selectComponent: 'Selecione pelo menos um componente para instalar.',
        agentSuccess: name => `Plugin do agente ${name} instalado`,
        desktopSuccess: name => `Plugin do Desktop ${name} instalado`,
        agentFailed: 'Falha ao instalar plugin do agente',
        installUncertain:
          'O Hermes parou de aguardar o resultado da instalação, mas o plugin ainda pode estar sendo instalado. Feche esta janela e use Verificar novamente em Plugins antes de tentar Instalar de novo.',
        desktopFailed: 'Falha ao instalar plugin do Desktop',
        missingEnv: (name, vars) =>
          `${name} está instalado, mas precisa de uma chave para funcionar: ${vars}. Adicione agora, ou as ferramentas do plugin falharão.`
      }
    },
    vault: {
      title: 'Senhas e logins',
      blurb:
        'Diga "entre no GitHub" e o agente faz o login por você. Na primeira vez que encontrar uma página de login, ele pedirá suas credenciais ali mesmo; depois disso, funciona automaticamente. As senhas são criptografadas nesta máquina e preenchidas diretamente na página — o modelo nunca as vê.',
      count: n => `${n} salvos`,
      loadFailed: 'Não foi possível carregar os itens do cofre',
      empty: 'Nada salvo ainda',
      emptyDesc:
        'Você não precisa adicionar nada aqui. Peça ao agente para entrar em um site e ele pedirá as credenciais uma única vez, na hora. Use Adicionar se preferir cadastrá-las antes.',
      add: 'Adicionar',
      addTitle: 'Adicionar login, cartão ou endereço',
      addDescription: 'Armazenado com criptografia nesta máquina. O agente nunca vê a senha.',
      added: 'Salvo.',
      adding: 'Salvando…',
      addConfirm: 'Salvar',
      kindField: 'Tipo',
      kinds: { login: 'Login', payment: 'Cartão de pagamento', address: 'Endereço' },
      labelField: 'Nome',
      labelPlaceholder: 'ex.: conta de trabalho do GitHub',
      labelRequired: 'É necessário informar um nome.',
      originField: 'Origem do site',
      originPlaceholder: 'https://github.com',
      originPlaceholderCheckout: 'https://shop.example.com',
      originInvalid: 'Insira uma URL válida, como https://example.com.',
      identifierTypeField: 'Tipo de identificador',
      identifierTypes: { email: 'E-mail', phone: 'Telefone', username: 'Nome de usuário' },
      identifierField: 'Identificador',
      identifierShown: identifier => identifier,
      passwordField: 'Senha',
      loginFieldsRequired: 'Identificador e senha são obrigatórios.',
      cardNumberField: 'Número do cartão',
      cardNameField: 'Nome no cartão',
      expMonthField: 'Mês de validade',
      expYearField: 'Ano de validade',
      cvcField: 'CVC',
      postalField: 'CEP',
      addressLine1Field: 'Endereço — linha 1',
      addressLine2Field: 'Endereço — linha 2',
      cityField: 'Cidade',
      stateField: 'Estado / região',
      countryField: 'País',
      optional: '(opcional)',
      createdOn: date => `Adicionado em ${date}`,
      deleteAction: 'Remover item salvo',
      otpField: 'Chave do autenticador',
      otpPlaceholder: 'Segredo Base32 ou link otpauth://',
      otpHint:
        'A "chave de configuração" que o site mostra ao ativar a autenticação de dois fatores. Com ela salva, o próprio Hermes gera os códigos.',
      twoFactorBadge: '2FA automático',
      deleteTitle: 'Excluir este item?',
      deleteDescription: label => `"${label}" será removido. Essa ação não pode ser desfeita.`,
      deleteConfirm: 'Excluir',
      sources: {
        title: 'Gerenciadores de senhas',
        blurb:
          'Gerenciadores de senhas instalados são detectados automaticamente. O agente pede para você desbloquear um deles na primeira vez que precisar de um login (uma vez por sessão); apenas um token de sessão fica na memória, e o agente nunca vê sua senha mestra nem qualquer login.',
        toggleFailed: 'Não foi possível atualizar o gerenciador de senhas',
        notInstalled: name =>
          `Não detectado. Instale a ferramenta de linha de comando do ${name} e entre nela; o Hermes a detectará automaticamente.`,
        disabledDesc: 'Detectado, mas desativado para o Hermes.',
        lockedDesc:
          'Detectado. O agente pedirá o desbloqueio quando precisar de um login, ou você pode desbloquear agora.',
        unlockedDesc:
          'Desbloqueado nesta sessão. Bloqueia automaticamente após 30 minutos de inatividade ou ao fechar o Hermes.',
        statusLocked: 'Bloqueado',
        statusNotDetected: 'Não detectado',
        statusOff: 'Desativado',
        statusUnlocked: 'Desbloqueado',
        unlock: 'Desbloquear',
        unlocking: 'Desbloqueando…',
        lock: 'Bloquear',
        unlocked: name => `${name} desbloqueado nesta sessão.`,
        unlockTitle: name => `Desbloquear ${name}`,
        unlockDescription:
          'Insira sua senha mestra. Ela é entregue ao gerenciador de senhas nesta máquina e descartada — nunca é armazenada, registrada em logs ou mostrada ao agente.',
        masterPasswordPlaceholder: 'Senha mestra'
      }
    },
    notifications: ptBrNotifications,
    sections: {
      model: 'Modelo',
      chat: 'Conversa',
      appearance: 'Aparência',
      workspace: 'Área de trabalho',
      safety: 'Segurança',
      memory: 'Memória e contexto',
      voice: 'Voz',
      advanced: 'Avançado'
    },
    searchPlaceholder: {
      about: 'Sobre o Hermes Desktop',
      config: 'Buscar configurações...',
      gateway: 'Conexão com o gateway...',
      keys: 'Buscar chaves de API...',
      mcp: 'Buscar servidores MCP...',
      sessions: 'Buscar sessões arquivadas...'
    },
    modeOptions: {
      light: { label: 'Claro', description: 'Interface clara' },
      dark: { label: 'Escuro', description: 'Área de trabalho com menos brilho' },
      system: { label: 'Sistema', description: 'Seguir a aparência do sistema' }
    },
    appearance: {
      chatTextScaleTitle: 'Tamanho do texto da conversa',
      chatTextScaleDesc:
        'Ajusta o texto da conversa e o editor de mensagens em relação à escala da interface. Barras laterais e controles mantêm o mesmo tamanho.',
      title: 'Aparência',
      intro: 'Somente no Desktop. O modo define o brilho; o tema define a paleta e os elementos visuais da conversa.',
      colorMode: 'Modo de cor',
      colorModeDesc: 'Escolha um modo fixo ou deixe o Hermes seguir a configuração do sistema.',
      toolViewTitle: 'Exibição das chamadas de ferramentas',
      toolViewDesc: 'Produto oculta os dados brutos das ferramentas; Técnico mostra a entrada e a saída completas.',
      hideCodeDiffsTitle: 'Ocultar diferenças de código',
      hideCodeDiffsDesc:
        'Mostrar edições de arquivos como linhas de ferramentas com contagens de linhas adicionadas e removidas, sem o código.',
      hideThreadTimelineTitle: 'Ocultar barras da linha do tempo',
      hideThreadTimelineDesc: 'Ocultar as barras de navegação na borda direita de cada conversa.',
      reasoningCollapsedTitle: 'Recolher o raciocínio por padrão',
      reasoningCollapsedDesc: 'Manter o raciocínio transmitido disponível, sem expandi-lo até você abrir.',
      uiScaleTitle: 'Escala da interface',
      uiScaleDesc: (percent: number) =>
        `Ajusta o texto e os controles em todo o aplicativo. Cmd/Ctrl com +, - e 0 também funciona. Atual: ${percent}%.`,
      sessionDensityTitle: 'Densidade da lista de sessões',
      sessionDensityDesc: 'Escolha quanto contexto aparece abaixo dos títulos das sessões na barra lateral.',
      sessionDensityCompact: 'Compacta',
      sessionDensityComfortable: 'Confortável',
      sessionDensityDetailed: 'Detalhada',
      tabStripTitle: 'Barra de abas',
      tabStripDesc:
        'Mostrar abas acima de uma área. Automático oculta as abas quando há um único painel, a menos que outra conversa ou área de blocos esteja aberta.',
      tabStripAuto: 'Automático',
      tabStripAlways: 'Sempre',
      tabStripNever: 'Nunca',
      appActionsTitle: 'Ações do aplicativo',
      appActionsDesc:
        'Posição de Configurações, Layout e HUD na barra de título. À direita deixa espaço para as abas à esquerda.',
      appActionsLeft: 'À esquerda',
      appActionsRight: 'À direita',
      terminalFontTitle: 'Fonte do terminal',
      terminalFontDesc:
        'Escolha uma fonte instalada para os terminais do Desktop. Nerd Fonts exibem Powerlevel10k e ícones do shell; deixe vazio para usar a JetBrains Mono incluída.',
      terminalFontPlaceholder: 'MesloLGS NF ou uma lista de fontes CSS',
      terminalFontPreview: 'Prévia dos glifos',
      terminalFontReset: 'Usar padrão',
      chatFontTitle: 'Fonte da conversa',
      chatFontDesc:
        'Escolha uma fonte instalada para a conversa e o restante do aplicativo. Útil para fontes que facilitam a leitura, como OpenDyslexic; deixe vazio para usar a fonte do tema.',
      chatFontPlaceholder: 'OpenDyslexic ou uma lista de fontes CSS',
      chatFontPreview: 'Prévia',
      chatFontSample: 'A rápida raposa marrom salta sobre o cão preguiçoso. 0123456789',
      chatFontReset: 'Usar fonte do tema',
      translucencyTitle: 'Translucidez da janela',
      translucencyDesc:
        'Veja a área de trabalho através de toda a janela, inclusive do texto. Ajustada separadamente para os modos claro e escuro.',
      translucencyGlassDesc:
        'Vidro fosco: a área de trabalho aparece com um desfoque suave, enquanto o texto permanece nítido. Ajustado separadamente para os modos claro e escuro.',
      translucencyModeClear: 'Transparente',
      translucencyModeGlass: 'Vidro',
      translucencyTintTitle: 'Tonalidade',
      translucencyFadeTitle: 'Transparência',
      translucencyFrostTitle: 'Efeito fosco',
      translucencyFrost: {
        'under-window': 'Profundo',
        popover: 'Suave',
        titlebar: 'Claro',
        header: 'Brilho intenso'
      },
      translucencyScopeTitle: 'Área',
      translucencyScope: { window: 'Janela inteira', sidebar: 'Somente barra lateral' },
      backdropTitle: 'Fundo da conversa',
      backdropDesc: 'A imagem sutil da estátua atrás da conversa.',
      userBubbleTitle: 'Balão de mensagem',
      userBubbleDesc: 'O nível de transparência das suas mensagens. Sólido em 0; apenas o contorno permanece em 100.',
      textDirectionTitle: 'Direção do texto',
      textDirectionDesc:
        'Como as mensagens e o editor escolhem a direção do texto. Automático segue a primeira letra de cada parágrafo; escolha uma direção quando textos mistos ficarem desalinhados. O código sempre permanece da esquerda para a direita.',
      textDirection: { auto: 'Automático', rtl: 'Da direita para a esquerda', ltr: 'Da esquerda para a direita' },
      introSplashTitle: 'Tela de introdução',
      introSplashDesc: 'O logotipo e a sugestão exibidos em uma conversa vazia.',
      modelPricingTitle: 'Preços dos modelos',
      modelPricingDesc:
        'Mostrar preços de entrada, saída e leitura de cache por milhão de tokens no seletor de modelos.',
      reactionsTitle: 'Reações a mensagens',
      reactionsDesc: 'Reações com emojis no estilo iMessage — reaja às mensagens, e o Hermes pode reagir às suas.',
      tipsTitle: 'Dicas no aplicativo',
      tipsDesc:
        'Dicas ocasionais do aplicativo e do Hermes. Cada dica aparece uma vez. Desativa automaticamente após seus primeiros 30 dias; você pode reativar.',
      tipsReset: (count: number) => `Mostrar ${count} ${count === 1 ? 'dica' : 'dicas'} novamente`,
      toursTitle: 'Tours guiados',
      toursDesc:
        'Permitir que o Hermes destaque cada etapa enquanto guia você pelo aplicativo. Desativa automaticamente após seus primeiros 30 dias; você pode reativar.',
      composerPopoutTitle: 'Editor flutuante',
      composerPopoutDesc:
        'Permitir arrastar o editor para fora de sua posição fixa. Quando desativado, ele permanece fixo na parte inferior.',
      fileBrowserTitle: 'Explorador de arquivos',
      fileBrowserDesc:
        'Mostrar o explorador de arquivos ao lado da conversa quando uma área de trabalho estiver aberta. O botão na barra de título também altera esta opção.',
      vibeHeartsTitle: 'Corações de carinho',
      vibeHeartsDesc:
        'Corações flutuantes quando você agradece, diz "te amo", "bom bot" ou envia um coração. Independente das Reações a mensagens acima.',
      embedsTitle: 'Conteúdo incorporado',
      embedsDesc:
        'Prévias interativas são carregadas de sites de terceiros (YouTube, X, …). Perguntar mostra um espaço reservado até você permitir cada serviço; Sempre carrega automaticamente; Desativado mantém links simples.',
      embedsAsk: 'Perguntar',
      embedsAlways: 'Sempre',
      embedsOff: 'Desativado',
      embedsReset: (count: number) => `Redefinir ${count} ${count === 1 ? 'serviço permitido' : 'serviços permitidos'}`,
      resumeLastSessionTitle: 'Reabrir a última conversa ao iniciar',
      resumeLastSessionDesc:
        'Quando ativado, o aplicativo reabre sua conversa mais recente ao iniciar. Desative para sempre começar com uma nova conversa.',
      product: 'Produto',
      productDesc: 'Atividade das ferramentas em linguagem acessível, com resumos concisos.',
      technical: 'Técnico',
      technicalDesc: 'Incluir argumentos e resultados brutos das ferramentas e detalhes de baixo nível.',
      themeTitle: 'Tema',
      themeDesc: 'Paletas exclusivas do Desktop. O modo selecionado é aplicado por cima.',
      themeSearchPlaceholder: 'Buscar nos seus temas ou no Marketplace do VS Code…',
      themeProfileNote: profile => `Salvo para o perfil ${profile} — cada perfil mantém seu próprio tema.`,
      installTitle: 'Instalar do VS Code',
      installDesc:
        'Cole o ID de uma extensão do Marketplace (ex.: dracula-theme.theme-dracula) para converter seu tema de cores em uma paleta do Desktop.',
      installPlaceholder: 'publisher.extension',
      installButton: 'Instalar',
      installing: 'Instalando…',
      installError: 'Não foi possível instalar esse tema.',
      installed: name => `“${name}” instalado.`,
      removeTheme: 'Remover tema',
      importedBadge: 'Importado',
      pet: {
        title: 'Mascote',
        intro:
          'Adote um mascote animado do petdex que flutua sobre o aplicativo e reage ao que o Hermes faz — corre enquanto as ferramentas executam, comemora os sucessos e fica triste com os erros.',
        restartHint:
          'Os mascotes precisam de uma reinicialização rápida — o aplicativo em execução foi iniciado antes da adição deste recurso. Encerre e reabra o Hermes, depois volte aqui.',
        scaleTitle: 'Tamanho',
        scaleDesc: 'Redimensionar o mascote flutuante. Aplica em todos os lugares instantaneamente.',
        roamTitle: 'Passear',
        roamDesc: 'Deixar o mascote passear pela janela por conta própria quando estiver ocioso.',
        chooseTitle: 'Escolha um mascote',
        chooseDesc: 'Ao escolher um mascote, ele é instalado (se necessário) e ativado.',
        searchPlaceholder: 'Buscar mascotes…',
        unreachable: 'Não foi possível acessar a galeria petdex. Verifique sua conexão e reabra esta página.',
        noMatch: query => `Nenhum mascote corresponde a "${query}".`,
        installedTag: 'instalado',
        generatedTag: 'Gerado',
        countCapped: (cap, total) => `Exibindo ${cap} de ${total} — digite para filtrar.`,
        count: n => `${n} mascote${n === 1 ? '' : 's'}.`,
        uninstall: name => `Desinstalar ${name}`,
        delete: name => `Excluir ${name}`,
        deleteTitle: name => `Excluir ${name}?`,
        deleteBody: 'Isso exclui permanentemente o mascote — ele não poderá ser reinstalado.',
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
        turnOnFailed: 'Não foi possível ativar o mascote.',
        turnOffFailed: 'Não foi possível desativar o mascote.'
      }
    },
    fieldLabels: defineFieldCopy({
      model: 'Modelo padrão',
      modelContextLength: 'Janela de contexto do modelo principal (substituição)',
      fallbackProviders: 'Modelos de reserva',
      toolsets: 'Conjuntos de ferramentas ativados',
      timezone: 'Fuso horário',
      display: { personality: 'Personalidade', showReasoning: 'Blocos de raciocínio' },
      desktop: {
        repoScanEnabled: 'Descoberta automática de repositórios',
        repoScanRoots: 'Pastas de busca de repositórios',
        repoScanExcludePaths: 'Caminhos de repositórios excluídos'
      },
      agent: {
        maxTurns: 'Máximo de etapas do agente',
        imageInputMode: 'Anexos de imagem',
        apiMaxRetries: 'Novas tentativas de API',
        serviceTier: 'Nível de serviço',
        toolUseEnforcement: 'Obrigatoriedade do uso de ferramentas'
      },
      terminal: {
        cwd: 'Pasta de trabalho',
        backend: 'Backend de execução',
        timeout: 'Tempo limite dos comandos',
        persistentShell: 'Shell persistente',
        envPassthrough: 'Repasse de variáveis de ambiente',
        dockerImage: 'Imagem do Docker',
        singularityImage: 'Imagem do Singularity',
        modalImage: 'Imagem do Modal',
        daytonaImage: 'Imagem do Daytona'
      },
      fileReadMaxChars: 'Limite de leitura de arquivos',
      toolOutput: {
        maxBytes: 'Limite de saída do terminal',
        maxLines: 'Limite de linhas por página de arquivo',
        maxLineLength: 'Limite de comprimento das linhas'
      },
      codeExecution: { mode: 'Modo de execução de código' },
      approvals: {
        mode: 'Modo de aprovação',
        timeout: 'Tempo limite de aprovação',
        mcpReloadConfirm: 'Confirmar recargas do MCP'
      },
      commandAllowlist: 'Lista de comandos permitidos',
      security: { redactSecrets: 'Ocultar segredos', allowPrivateUrls: 'Permitir URLs privadas' },
      browser: {
        allowPrivateUrls: 'URLs privadas no navegador',
        autoLocalForPrivateUrls: 'Navegador local para URLs privadas',
        useRealProfile: 'Usar meu perfil real do navegador'
      },
      checkpoints: { enabled: 'Pontos de restauração de arquivos', maxSnapshots: 'Limite de pontos de restauração' },
      voice: {
        maxRecordingSeconds: 'Duração máxima da gravação',
        autoTts: 'Ler respostas em voz alta',
        voiceChatMode: 'Modo de conversa por voz',
        gptLive: { voice: 'Voz do GPT-Live', instructions: 'Personalidade do GPT-Live' }
      },
      stt: {
        enabled: 'Fala para texto',
        echoTranscripts: 'Exibir transcrições na conversa',
        provider: 'Provedor de fala para texto',
        streaming: 'Transcrição ao vivo',
        local: { model: 'Modelo local de transcrição', language: 'Idioma da transcrição' },
        openai: { model: 'Modelo STT da OpenAI', streamingModel: 'Modelo de transcrição ao vivo da OpenAI' },
        groq: { model: 'Modelo STT da Groq' },
        mistral: { model: 'Modelo STT da Mistral' },
        xai: { model: 'Modelo STT da xAI' },
        deepinfra: { model: 'Modelo STT da DeepInfra' },
        elevenlabs: {
          modelId: 'Modelo STT da ElevenLabs',
          languageCode: 'Idioma da ElevenLabs',
          tagAudioEvents: 'Marcar eventos de áudio',
          diarize: 'Identificação de falantes'
        }
      },
      tts: {
        provider: 'Provedor de texto para fala',
        edge: { voice: 'Voz do Edge' },
        openai: { model: 'Modelo TTS da OpenAI', voice: 'Voz da OpenAI' },
        elevenlabs: { voiceId: 'Voz da ElevenLabs', modelId: 'Modelo da ElevenLabs' },
        xai: {
          voiceId: 'Voz da xAI (Grok)',
          language: 'Idioma da xAI',
          speed: 'Velocidade de reprodução da xAI',
          autoSpeechTags: 'Marcações automáticas de fala da xAI',
          optimizeStreamingLatency: 'Otimização de latência de transmissão da xAI',
          sampleRate: 'Taxa de amostragem da xAI',
          bitRate: 'Taxa de bits da xAI'
        },
        minimax: { model: 'Modelo TTS da MiniMax', voiceId: 'Voz da MiniMax' },
        mistral: { model: 'Modelo TTS da Mistral', voiceId: 'Voz da Mistral' },
        gemini: { model: 'Modelo TTS do Gemini', voice: 'Voz do Gemini' },
        neutts: { model: 'Modelo do NeuTTS', device: 'Dispositivo do NeuTTS' },
        kittentts: { model: 'Modelo do KittenTTS', voice: 'Voz do KittenTTS' },
        piper: { voice: 'Voz do Piper' },
        deepinfra: { model: 'Modelo TTS da DeepInfra', voice: 'Voz da DeepInfra' }
      },
      memory: {
        memoryEnabled: 'Memória persistente',
        userProfileEnabled: 'Perfil do usuário',
        memoryCharLimit: 'Limite de memória',
        userCharLimit: 'Limite do perfil',
        provider: 'Provedor de memória'
      },
      context: { engine: 'Mecanismo de contexto' },
      compression: {
        enabled: 'Compactação automática',
        threshold: 'Limiar de compactação',
        codexGpt55Autoraise: 'Aumento automático do limiar de compactação do Codex',
        targetRatio: 'Meta de compactação',
        protectLastN: 'Mensagens recentes protegidas'
      },
      auxiliary: { compression: { timeout: 'Tempo limite do modelo de compactação (s)' } },
      delegation: {
        model: 'Modelo do subagente',
        provider: 'Provedor do subagente',
        maxIterations: 'Limite de turnos do subagente',
        maxConcurrentChildren: 'Subagentes em paralelo',
        childTimeoutSeconds: 'Tempo limite do subagente',
        reasoningEffort: 'Esforço de raciocínio do subagente'
      },
      updates: { nonInteractiveLocalChanges: 'Alterações locais em atualizações pelo aplicativo' }
    }),
    fieldDescriptions: defineFieldCopy({
      model: 'Usado para novas conversas, a menos que você escolha outro modelo no editor.',
      modelContextLength:
        'Substitui a janela de contexto detectada apenas do modelo PRINCIPAL da conversa (em tokens). Deixe em 0 para usar o valor detectado do modelo selecionado. Não afeta modelos auxiliares/MoA.',
      fallbackProviders: 'Entradas de reserva no formato provider:model para tentar se o modelo padrão falhar.',
      display: {
        personality: 'Estilo padrão do assistente para novas sessões.',
        showReasoning: 'Mostrar seções de raciocínio quando o backend as fornecer.'
      },
      desktop: {
        repoScanEnabled: 'Buscar repositórios Git nas pastas locais para exibir em Projetos.',
        repoScanRoots: 'Pastas onde buscar. Deixe vazio para buscar na sua pasta pessoal.',
        repoScanExcludePaths: 'Pastas e subpastas a ignorar durante a descoberta de repositórios.'
      },
      timezone: 'Identificador de fuso horário IANA. Vazio usa o fuso horário do sistema.',
      browser: {
        useRealProfile:
          'A navegação local usa seus logins reais. O Hermes copia o perfil do seu navegador padrão (cookies, logins e preferências) para uma cópia gerenciada e a controla com o Chromium incluído — seu perfil em uso nunca é aberto diretamente, e a cópia é atualizada a cada execução. Também permite que o agente abra uma sessão local com seu perfil real quando solicitado, mesmo com um backend de navegador na nuvem configurado. Somente navegadores Chromium (Chrome, Edge, Brave, Brave Origin, Chromium) são compatíveis; um navegador padrão que não seja Chromium gera uma mensagem clara de erro. Desativado por padrão.'
      },
      agent: {
        imageInputMode: 'Controla como os anexos de imagem são enviados ao modelo.',
        maxTurns: 'Limite de turnos com chamadas de ferramentas antes de o Hermes interromper uma execução.'
      },
      terminal: {
        cwd: 'Pasta padrão do projeto para o trabalho com ferramentas e terminal.',
        persistentShell:
          'Backend SSH: manter variáveis exportadas entre comandos. Desativado, cada comando começa no ambiente de login; a pasta de trabalho sempre é mantida.',
        envPassthrough: 'Variáveis de ambiente a repassar à execução de ferramentas.',
        dockerImage: 'Imagem de contêiner usada quando o backend de execução é Docker.',
        singularityImage: 'Imagem usada quando o backend de execução é Singularity.',
        modalImage: 'Imagem usada quando o backend de execução é Modal.',
        daytonaImage: 'Imagem usada quando o backend de execução é Daytona.'
      },
      codeExecution: { mode: 'O rigor com que a execução de código é restrita ao projeto atual.' },
      fileReadMaxChars: 'Máximo de caracteres que o Hermes pode ler em uma única solicitação de arquivo.',
      approvals: {
        mode: 'Como o Hermes lida com comandos que precisam de aprovação explícita.',
        timeout: 'Quanto tempo as solicitações de aprovação aguardam antes de expirar.'
      },
      security: { redactSecrets: 'Ocultar segredos detectados do conteúdo visível ao modelo sempre que possível.' },
      checkpoints: { enabled: 'Criar cópias de restauração antes de editar arquivos.' },
      memory: {
        memoryEnabled: 'Salvar memórias duradouras que possam ajudar em sessões futuras.',
        userProfileEnabled: 'Manter um perfil compacto das preferências do usuário.'
      },
      context: { engine: 'Estratégia para gerenciar conversas longas próximas do limite de contexto.' },
      compression: {
        enabled: 'Resumir o contexto antigo quando as conversas ficarem grandes.',
        codexGpt55Autoraise:
          'Aumentar o limiar de compactação para 85% nos modelos compatíveis com OAuth do ChatGPT Codex.'
      },
      auxiliary: {
        compression: {
          timeout:
            'Segundos de espera por chamada ao modelo auxiliar de compactação (padrão: 120). Aumente para modelos locais lentos.'
        }
      },
      voice: {
        autoTts: 'Ler automaticamente as respostas do assistente em voz alta.',
        voiceChatMode:
          'chained: fala para texto → Hermes → texto para fala com os provedores abaixo. gpt-live: um único modelo de voz full-duplex da OpenAI (gpt-live-1) ouve e fala, e encaminha cada solicitação real ao Hermes — qualquer modelo selecionado responde com o conjunto completo de ferramentas. Exige uma chave de API da OpenAI; a camada de voz custa US$ 0,05 por minuto.',
        gptLive: {
          voice: 'Voz do modo GPT-Live. IDs de voz personalizados são aceitos.',
          instructions:
            'Frases extras para a personalidade da voz ao vivo (tom, ritmo e idioma). O Hermes mantém seu próprio prompt de sistema.'
        }
      },
      tts: {
        xai: {
          voiceId: 'ID de voz da xAI (ex.: eve) ou um ID de voz personalizado.',
          language: 'Código do idioma falado (ex.: en, pt-BR) ou "auto" para detecção automática.',
          speed: 'Velocidade de reprodução. 0.7 = mais lenta, 1.0 = normal, 1.5 = mais rápida.',
          autoSpeechTags:
            'Permitir que um LLM insira marcações expressivas de áudio ([laughing], [sighs]) no roteiro antes da síntese.',
          optimizeStreamingLatency: 'Equilíbrio entre latência e qualidade. 0 = melhor qualidade, 2 = menor latência.',
          sampleRate: 'Taxa de amostragem de áudio em Hz. Maior = melhor qualidade e arquivos maiores.',
          bitRate: 'Taxa de bits de MP3 em bps. Aplica-se apenas quando o codec é mp3.'
        },
        neutts: { device: 'Dispositivo de inferência local para o NeuTTS.' }
      },
      stt: {
        enabled: 'Ativar a transcrição de fala local ou por um provedor.',
        echoTranscripts: 'Publicar a transcrição bruta 🎙️ das mensagens de voz na conversa.',
        streaming: 'Mostrar o texto enquanto você fala (OpenAI, xAI, ElevenLabs). Usa a gravação em caso de falha.',
        elevenlabs: {
          languageCode: 'Código de idioma ISO-639-3 opcional. Vazio permite a detecção automática pela ElevenLabs.'
        }
      },
      updates: {
        nonInteractiveLocalChanges:
          'Quando o Hermes se atualizar pelo aplicativo (sem pergunta no terminal), manter as edições locais do código-fonte (stash) ou descartá-las (discard). Atualizações pelo terminal sempre perguntam.'
      }
    }),
    uninstallSection: ptBrUninstallSection,
    memoryLayers: ptBrMemoryLayers,
    customEndpoints: {
      active: 'Ativo',
      apiKeySet: 'Chave de API definida',
      use: 'Usar',
      editTitle: 'Editar endpoint',
      addTitle: 'Adicionar endpoint',
      fields: {
        name: 'Nome',
        providerId: 'ID do provedor',
        endpointUrl: 'URL do endpoint',
        defaultModel: 'Modelo padrão',
        context: 'Contexto',
        apiKey: 'Chave de API',
        apiKeyNewPlaceholder: 'Deixe vazio para manter a chave atual',
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
      endpointReachableTransport: transport => `O endpoint está acessível (rota ${transport} atendida).`,
      endpointReachableModels: (reachable, count) => `${reachable} ${count} modelos encontrados.`,
      endpointValidationFailed: 'Falha na validação do endpoint.',
      validationFailed: 'Falha na validação',
      activationFailed: 'Falha na ativação',
      deleteConfirm: name => `Excluir ${name}?`,
      deleteFailed: 'Falha ao excluir',
      title: 'Endpoints personalizados',
      deleteEndpoint: 'Excluir endpoint',
      emptyDescription: 'Adicione abaixo um endpoint compatível com OpenAI.',
      emptyTitle: 'Nenhum endpoint personalizado',
      namePlaceholder: 'Axet Proxy',
      contextPlaceholder: 'Automático'
    },
    computerUse: {
      accessibility: 'Acessibilidade',
      screenRecording: 'Gravação de tela',
      driverHealth: 'Integridade do driver'
    },
    about: { updates: 'Atualizações', channel: ptBrUpdateChannel },
    config: {
      minimizeToTrayTitle: 'Minimizar para a bandeja',
      minimizeToTrayDesc:
        'Minimizar janelas ou fechar a janela principal para ocultá-las na bandeja do sistema (barra de menus no macOS) e manter o Hermes em execução. Use Encerrar Hermes no menu da bandeja ou Cmd+Q para sair. Desativado por padrão; aplica-se apenas a este dispositivo.',
      minimizeToTrayUnavailable:
        'A bandeja do sistema não está disponível. As janelas serão minimizadas e fechadas normalmente. Desative e ative esta opção para tentar novamente.',
      none: 'Nenhum',
      noneParen: '(nenhum)',
      builtinOnly: 'Somente integrado',
      notSet: 'Não definido',
      commaSeparated: 'valores separados por vírgula',
      searchPlaceholder: 'Buscar…',
      noResults: 'Nenhum resultado encontrado',
      systemDefault: 'Padrão do sistema',
      loading: 'Carregando configurações do Hermes...',
      emptyTitle: 'Nada para configurar',
      emptyDesc: 'Esta seção não tem configurações ajustáveis.',
      failedLoad: 'Falha ao carregar as configurações',
      autosaveFailed: 'Falha ao salvar automaticamente',
      imported: 'Configurações importadas',
      invalidJson: 'JSON de configurações inválido',
      toolsetsWipeConfirm:
        'Remover todos os conjuntos de ferramentas ativados? Isso desativa memória, terminal, busca na web, delegação e a maioria das outras ferramentas até você reativá-las.',
      keepAwakeTitle: 'Manter o computador ativo',
      keepAwakeDesc:
        'Impedir que esta máquina entre em suspensão. "Durante o trabalho" só a mantém ativa enquanto um turno está em andamento, para que execuções noturnas continuem sem manter o notebook acordado a semana toda. A tela ainda pode escurecer.',
      keepAwakeOff: 'Desativado',
      keepAwakeWhileWorking: 'Durante o trabalho',
      keepAwakeAlways: 'Sempre',
      disableF12Title: 'Desativar DevTools com F12',
      disableF12Desc:
        'Impedir que F12 abra as Ferramentas do desenvolvedor. Ctrl+Shift+I (ou Cmd+Opt+I no Mac) continua funcionando.',
      alwaysExternalLinksTitle: 'Sempre abrir links no navegador externo',
      alwaysExternalLinksDesc:
        'Abrir todos os links clicados no navegador do sistema em vez do navegador integrado. "Abrir no navegador integrado" no menu de contexto continua funcionando.',
      attachmentSizeTitle: 'Tamanho máximo de carregamento de prévias / imagens',
      attachmentSizeDesc:
        'Tamanho máximo, em MB, de um arquivo local que o Desktop carrega para prévias e anexos de imagem. O padrão é 16. Anexos remotos que não sejam imagens têm um limite separado de 256 MB. Um valor muito alto carrega o arquivo inteiro na memória e pode travar ou encerrar o aplicativo.',
      attachmentSizeUnit: 'MB',
      attachmentSizeLabel: 'Tamanho máximo de carregamento de prévias / imagens em megabytes',
      voiceShortcutHintTitle: 'Atalho de gravação de voz',
      voiceShortcutHintDesc:
        'Defina o atalho de gravação de voz em Configurações → Atalhos de teclado ("Iniciar / parar conversa por voz"). O valor voice.record_key só se aplica à CLI e à TUI.',
      showOptions: 'Mostrar opções'
    },
    hudModifier: {
      title: 'Toque para abrir o HUD',
      description:
        'Pressione e solte ⌘ + Option no Mac, ou Ctrl + Alt no Windows/Linux, para trazer o HUD à frente em qualquer aplicativo. Desativado por padrão; aplica-se apenas a este dispositivo.',
      permission:
        'Permita o Hermes em Ajustes do Sistema → Privacidade e Segurança → Monitoramento de Entrada e tente novamente. Este gesto não registra teclas nem captura sua tela.',
      unavailable:
        'O auxiliar do gesto do HUD não pôde iniciar ou parou inesperadamente. Tente novamente ou reinicie o Hermes. O atalho existente do HUD continua funcionando dentro do Hermes.',
      missingHelper:
        'Esta instalação do Hermes não inclui o auxiliar do gesto do HUD. Atualize ou reinstale o Hermes e tente novamente.',
      unsupportedSession:
        'Esta sessão de desktop não permite toques globais nas teclas modificadoras. O Linux exige X11; Wayland não é compatível.'
    },
    screenshot: {
      enabledTitle: 'Atalho de captura de tela',
      enabledDesc:
        'Pressione as duas teclas Command juntas em qualquer aplicativo para capturar a janela em primeiro plano e anexá-la ao rascunho atual do Hermes. Nunca envia automaticamente. Desativado por padrão; aplica-se apenas a este Mac. O conteúdo da janela pode ser sensível — revise o anexo antes de enviar.',
      statusTitle: 'Status do atalho de captura de tela',
      checking: 'Verificando atalho de captura de tela…',
      disabled: 'O atalho de captura de tela está desativado.',
      starting: 'Iniciando o monitor do atalho. Ele ainda não está pronto.',
      ready: 'O atalho está pronto. As capturas de tela são anexadas ao rascunho atual sem enviar.',
      inputPermission:
        'A permissão de Monitoramento de Entrada permite ao Hermes detectar as duas teclas Command enquanto outro aplicativo está ativo. Permita o Hermes em Ajustes do Sistema → Privacidade e Segurança → Monitoramento de Entrada, volte aqui e tente novamente.',
      screenPermission:
        'A permissão de Gravação de Tela permite ao Hermes capturar a janela em primeiro plano ao usar este atalho. Permita o Hermes em Ajustes do Sistema → Privacidade e Segurança → Gravação de Tela, volte aqui e tente novamente. Reinicie o Hermes se o macOS solicitar.',
      openSettings: 'Abrir Ajustes do Sistema',
      retry: 'Tentar novamente',
      unavailable: 'O atalho de captura de tela não está disponível. Tente novamente ou desative-o.',
      errorTitle: 'Erro no atalho de captura de tela',
      loadFailed: 'Não foi possível ler o status do atalho. Tente novamente para verificar a configuração atual.',
      saveFailed:
        'Não foi possível confirmar a alteração do atalho. Tente novamente para verificar a configuração atual.',
      permissionFailed:
        'Não foi possível abrir os Ajustes do Sistema. Abra Privacidade e Segurança manualmente e tente novamente.',
      captureFailed: 'Não foi possível capturar a janela em primeiro plano. Nada foi anexado ou enviado.',
      contextChanged: 'O rascunho atual mudou durante a captura. A captura de tela não foi anexada nem enviada.'
    },
    quickEntry: {
      enabledTitle: 'Entrada rápida',
      enabledDesc:
        'Abra um pequeno editor de qualquer lugar com um atalho global e envie uma solicitação sem abrir o Hermes.',
      shortcutTitle: 'Atalho de entrada rápida',
      shortcutDesc: 'Exige pelo menos uma tecla modificadora, ex.: CommandOrControl+Shift+Space.',
      active: 'O atalho está ativo.',
      takenBy: 'Outro aplicativo já usa este atalho — escolha outro.',
      invalidShortcut: 'Atalho inválido. Inclua pelo menos uma tecla modificadora.'
    },
    credentials: {
      pasteKey: 'Colar chave',
      pasteLabelKey: label => `Colar chave de ${label}`,
      optional: 'Opcional',
      enterValueFirst: 'Insira um valor primeiro.',
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
      revealValue: 'Revelar valor',
      replace: 'Substituir',
      set: 'Definir',
      clear: 'Limpar'
    },
    connections: {
      title: 'Gateways cadastrados',
      intro:
        'Gerencie este dispositivo e todos os gateways Hermes que ele pode acessar por conexões remotas, SSH ou Cloud.',
      stagedNote:
        'Todos os gateways permanecem conectados. Perfis, conversas, mensagens e tarefas agendadas permanecem no respectivo gateway, e Sessões pode mostrar todos ao mesmo tempo.',
      launchModeTitle: 'Ao iniciar, voltar a Sessões no último gateway usado',
      launchModeDesc: 'Quando desativado, Sessões abre no gateway principal.',
      searchPlaceholder: 'Buscar gateways…',
      noSearchResults: 'Nenhum gateway corresponde à sua busca.',
      loadFailed: 'Não foi possível carregar as conexões',
      currentPill: 'Atual',
      primaryPill: 'Principal',
      managedPill: 'Gerenciado pelo aplicativo',
      addConnection: 'Adicionar conexão',
      editConnection: 'Editar',
      removeConnection: 'Remover',
      removeConfirmTitle: 'Remover esta conexão?',
      removeConfirmDesc: (label: string) =>
        `“${label}” será removida deste aplicativo. A instância em si não será alterada — você pode adicioná-la novamente a qualquer momento.`,
      makePrimary: 'Tornar principal',
      testConnection: 'Testar',
      testOk: 'Acessível',
      testFailed: 'Falha no teste de conexão',
      saveFailed: 'Não foi possível salvar a conexão',
      removeFailed: 'Não foi possível remover a conexão',
      updateAll: 'Atualizar todas as instâncias',
      updateAllRunning: 'Atualizando todas as instâncias…',
      updateAllDone: 'Atualizações enviadas',
      updateAllFailed: 'Falha ao distribuir as atualizações',
      updateSkippedCloud: 'Gerenciado pelo Hermes Cloud',
      kindLocal: 'Local',
      kindRemote: 'Gateway remoto',
      kindCloud: 'Hermes Cloud',
      kindSsh: 'SSH',
      kindLocalDesc: 'O ambiente de execução do Hermes gerenciado por este aplicativo.',
      kindRemoteDesc: 'Um gateway Hermes acessível por HTTP(S) — rede local, Tailscale ou internet.',
      kindCloudDesc: 'Uma instância hospedada descoberta pela sua conta do Hermes Cloud.',
      kindSshDesc: 'Uma instalação do Hermes acessada por SSH.',
      labelTitle: 'Nome',
      labelDesc:
        'Obrigatório. Exibido em todos os lugares onde esta instância aparece; deve ser único (ex.: “Laboratório”, “Notebook de trabalho”).',
      labelPlaceholder: 'Laboratório',
      urlTitle: 'URL do gateway',
      sshHostTitle: 'Host SSH',
      headersTitle: 'Cabeçalhos extras do gateway',
      headersDesc:
        'Enviados com cada solicitação HTTP e WebSocket a este gateway — para proxies de acesso como Cloudflare Access (CF-Access-Client-Id / CF-Access-Client-Secret). Os valores são armazenados com criptografia. Cabeçalhos gerenciados pelo Hermes (Authorization, Cookie, Host…) são ignorados.',
      headerValuePlaceholder: 'Valor',
      headerValueSaved: 'Salvo — deixe vazio para manter',
      headerAdd: 'Adicionar cabeçalho',
      headerRemove: 'Remover',
      duplicateLocal: 'Este aplicativo já gerencia uma conexão local — só pode haver uma.',
      duplicateUrl: (label: string) => `Já existe uma conexão com esta URL de gateway (“${label}”).`,
      duplicateSsh: (label: string) => `Já existe uma conexão com este host SSH (“${label}”).`,
      sameBackendHint: (label: string) => `Mesmo backend de “${label}”`,
      localAddHint: 'Local não está disponível: a conexão local gerenciada já existe (só pode haver uma).',
      cloudAddHint:
        'Dica: entrar no Hermes Cloud acima descobre seus agentes automaticamente — use este formulário apenas para cadastrar manualmente a URL de uma instância conhecida.',
      save: 'Salvar conexão',
      saving: 'Salvando…',
      cancel: 'Cancelar',
      empty: 'Nenhuma conexão cadastrada ainda.'
    },
    managedUpdates: {
      title: 'Atualizações gerenciadas',
      intro:
        'Atualize instalações SSH gerenciadas pelo Desktop de forma transacional: as sessões são finalizadas, o checkout remoto é atualizado e todos os perfis são restaurados com um comprovante correlacionado.',
      sshConnection: 'Instalação SSH gerenciada pelo Desktop',
      update: 'Atualizar',
      updating: 'Atualizando…',
      progress: 'Finalizando sessões, atualizando a instalação remota e restaurando perfis…',
      updated: 'Atualizado',
      partial: 'Atualizado — falha na restauração',
      refused: 'Recusado',
      failed: 'Falha na atualização',
      alreadyRunning: 'Atualização já em andamento',
      receipt: (id: string, outcome: string) => `Comprovante ${id} · ${outcome}`,
      receiptVersions: (pre: string, post: string) => `${pre} → ${post}`,
      scopesRestored: (profiles: string) => `Perfis restaurados: ${profiles}`,
      scopeNotRestored: (profile: string, error: string) => `Perfil “${profile}” não restaurado: ${error}`
    },
    gateway: {
      loading: 'Carregando configurações do gateway...',
      ...ptBrGatewayHost,
      title: 'Conexão com o gateway',
      envOverride: 'substituição por variável de ambiente',
      intro:
        'Local por padrão. Use remoto quando este aplicativo precisar controlar um backend Hermes em outro lugar. As conexões com gateways são definidas por máquina; os perfis são descobertos nos gateways aos quais você se conecta.',
      envOverrideTitle: 'Esta conexão foi fixada pela forma como o Hermes foi iniciado.',
      envOverrideDesc:
        'Uma configuração de inicialização externa ao aplicativo escolheu esta conexão, então as opções abaixo são somente leitura. Reinicie o Hermes sem essa configuração — ou peça a quem a definiu — para alterá-la aqui.',
      modeTitle: 'Modo de conexão',
      localTitle: 'Gateway local',
      localDesc: 'Iniciar um backend Hermes privado no localhost. Este é o padrão e funciona sem internet.',
      remoteTitle: 'Gateway remoto',
      remoteDesc: 'Conectar esta interface Desktop a um backend Hermes remoto.',
      remoteAuthHint:
        'Gateways hospedados usam OAuth ou nome de usuário e senha; os auto-hospedados podem usar um token de sessão.',
      cloudTitle: 'Hermes Cloud',
      cloudDesc: 'Entre uma vez no Hermes Cloud e escolha entre os agentes da sua conta — sem precisar colar uma URL.',
      cloudSignInTitle: 'Hermes Cloud',
      cloudSignIn: 'Entrar no Hermes Cloud',
      cloudSignedIn: 'Conectado ao Hermes Cloud',
      cloudNeedsSignIn: 'Entre no Hermes Cloud para descobrir os agentes da sua conta.',
      cloudSignedInDesc: 'Você está conectado. Escolha um agente abaixo; a sessão é renovada automaticamente.',
      cloudAgentsTitle: 'Seus agentes',
      cloudOrgPickerTitle: 'Escolha uma organização',
      cloudOrgSelect: 'Selecionar',
      cloudOrgChange: 'Trocar organização',
      cloudOrgRole: role => `Função: ${role}`,
      cloudLoadingAgents: 'Carregando seus agentes…',
      cloudNoAgents: {
        before: 'Nenhum agente encontrado nesta conta. Crie um no ',
        linkText: 'portal da Nous',
        after: ' e atualize a lista.'
      },
      cloudRefresh: 'Atualizar',
      cloudConnect: 'Conectar',
      cloudSavedTitle: 'Gateways Cloud salvos',
      cloudSavedDesc:
        'Use um gateway salvo sem alterar seu padrão. Entre abaixo para adicionar instâncias. Gerencie nomes e logins na lista de conexões salvas.',
      cloudUseSaved: 'Usar gateway',
      cloudActive: 'Ativo nesta janela',
      cloudConnecting: 'Conectando…',
      cloudDiscoverFailed: 'Não foi possível carregar seus agentes do Hermes Cloud',
      cloudConnectFailed: 'Não foi possível conectar a esse agente',
      cloudSignInFailed: 'Falha ao entrar no Hermes Cloud',
      cloudSignedOutTitle: 'Você saiu do Hermes Cloud',
      cloudSignedOutMessage: 'A sessão do Hermes Cloud foi removida.',
      cloudConnectedTitle: 'Conectado',
      cloudConnectedPill: 'Conectado',
      cloudConnectedTo: name => `Conectado a ${name}.`,
      cloudAgentProvisioning: 'Provisionando…',
      cloudStatusLabel: status => `Status: ${status}`,
      remoteUrlTitle: 'URL remota',
      remoteUrlDesc: 'URL base do backend do painel remoto. Prefixos de caminho são aceitos, por exemplo /hermes.',
      probing: 'Verificando a autenticação deste gateway…',
      probeError:
        'O Hermes não consegue acessar esse endereço. Verifique a URL e se o outro computador está executando o Hermes — as opções de login aparecem quando ele responder.',
      signedIn: 'Conectado',
      signIn: 'Entrar',
      signOut: 'Sair',
      signInWith: provider => `Entrar com ${provider}`,
      authTitle: 'Autenticação',
      authSignedInPassword:
        'Este gateway usa nome de usuário e senha. Você está conectado; a sessão é renovada automaticamente.',
      authSignedInOauth: 'Este gateway usa OAuth. Você está conectado; a sessão é renovada automaticamente.',
      authNeedsPassword: 'Este gateway usa nome de usuário e senha. Entre para autorizar este aplicativo Desktop.',
      authNeedsOauth: provider =>
        `Este gateway usa OAuth. Entre com ${provider} para autorizar este aplicativo Desktop.`,
      tokenTitle: 'Token de sessão',
      tokenDesc:
        'O token de sessão do painel usado para acesso REST e WebSocket. Deixe vazio para manter o token salvo.',
      existingToken: value => `Token existente ${value}`,
      savedToken: 'salvo',
      pasteSessionToken: 'Colar token de sessão',
      plainTextConfirmTitle: 'Armazenar o token do gateway em texto simples?',
      plainTextConfirmDesc:
        'Nenhum serviço de chaveiro do sistema foi encontrado nesta máquina. O token seria salvo sem criptografia no arquivo de configurações de conexão do aplicativo, legível por qualquer processo executado por este usuário. Instale ou ative o chaveiro do sistema (GNOME Keyring ou KWallet no Linux) para armazenamento criptografado.',
      plainTextConfirmAction: 'Salvar em texto simples',
      plainTextStoredTitle: 'Token armazenado em texto simples',
      plainTextStoredDesc:
        'O armazenamento seguro não está disponível, então o token salvo está sem criptografia no arquivo de configurações de conexão do aplicativo nesta máquina. Instale ou ative o chaveiro do sistema (GNOME Keyring ou KWallet no Linux) para criptografá-lo.',
      keychainEncryptionTitle: 'Criptografar segredos salvos com o chaveiro do sistema',
      keychainEncryptionDesc:
        'Desativado por padrão. Quando ativado, tokens de gateway e credenciais de login são criptografados com o chaveiro do sistema (Acesso às Chaves, GNOME Keyring ou Windows DPAPI) — o sistema pode pedir permissão ou senha. Quando desativado, são armazenados como arquivos simples legíveis apenas pela sua conta de usuário.',
      keychainEncryptionFailed: 'Não foi possível alterar a criptografia dos segredos',
      testRemote: 'Testar conexão remota',
      saveForRestart: 'Salvar para a próxima reinicialização',
      saveAndReconnect: 'Salvar e reconectar',
      diagnostics: 'Diagnóstico',
      diagnosticsDesc: 'Mostrar desktop.log no gerenciador de arquivos — útil quando o gateway não inicia.',
      openLogs: 'Abrir logs',
      incompleteTitle: 'Gateway remoto incompleto',
      incompleteSignIn: 'Insira uma URL remota e entre antes de mudar para o modo remoto.',
      incompleteToken: 'Insira uma URL remota e um token de sessão antes de mudar para o modo remoto.',
      incompleteSignInTest: 'Insira uma URL remota e entre antes de testar.',
      incompleteTokenTest: 'Insira uma URL remota e um token de sessão antes de testar.',
      enterUrlFirst: 'Insira uma URL remota primeiro.',
      restartingTitle: 'Reiniciando conexão com o gateway',
      savedTitle: 'Configurações do gateway salvas',
      restartingMessage:
        'O Hermes Desktop se reconectará usando as configurações salvas — a interface permanecerá aberta.',
      savedMessage: 'Salvo para a próxima reinicialização.',
      connectedTo: (baseUrl, version) => `Conectado a ${baseUrl}${version ? ` · Hermes ${version}` : ''}`,
      reachableTitle: 'Gateway remoto acessível',
      signedOutTitle: 'Você saiu',
      signedOutMessage: 'A sessão do gateway remoto foi removida.',
      failedLoad: 'Falha ao carregar configurações do gateway',
      signInFailed: 'Falha ao entrar',
      signOutFailed: 'Falha ao sair',
      testFailed: 'Falha no teste do gateway remoto',
      applyFailed: 'Não foi possível aplicar as configurações do gateway',
      saveFailed: 'Não foi possível salvar as configurações do gateway',
      sshTitle: 'Conectar via SSH',
      sshDesc:
        'O Hermes é iniciado no host remoto por SSH e conectado por um túnel a este aplicativo — você não precisa iniciar nem expor nada. Exige acesso SSH funcional por chave ao host.',
      sshTrustHint:
        'A primeira chave de host apresentada é aceita e fixada; alterações posteriores bloqueiam a conexão por segurança.',
      sshHostTitle: 'Host',
      sshHostDesc: 'user@host ou um alias Host de ~/.ssh/config.',
      sshHostPick: 'Selecione um host…',
      sshHostPickTitle: 'Host',
      sshHostPickDesc: 'Um alias Host de ~/.ssh/config, ou Personalizado para digitar um.',
      sshHostCustom: 'Personalizado (inserir manualmente)…',
      sshUserTitle: 'Usuário',
      sshUserDesc: 'Vazio = ~/.ssh/config ou seu usuário atual.',
      sshUserPlaceholder: 'de ~/.ssh/config',
      sshPortTitle: 'Porta',
      sshPortDesc: 'Vazio = 22 ou a porta de ~/.ssh/config.',
      sshKeyTitle: 'Arquivo de identidade',
      sshKeyDesc: 'Caminho da chave privada. Vazio = ssh-agent ou ~/.ssh/config.',
      sshHermesPathTitle: 'Caminho do Hermes (opcional)',
      sshHermesPathDesc: 'Caminho completo do executável hermes remoto. Vazio = detecção automática.',
      sshHermesPathPlaceholder: 'detecção automática',
      sshTestConnection: 'Testar SSH',
      sshConnect: 'Conectar',
      sshButtonsHint: 'Salvar aplica na próxima inicialização. Conectar reconecta agora.',
      sshReachable: (host, platform) => `Acessível: ${host} (${platform}) — Hermes encontrado`,
      sshIncompleteHost: 'Insira um host SSH antes de conectar.',
      sshErrUnreachable: 'Não foi possível acessar esse host por SSH. Verifique o host, a porta e sua rede.',
      sshErrAuth:
        'Falha na autenticação SSH. Carregue sua chave no ssh-agent (ssh-add) ou defina IdentityFile em ~/.ssh/config — o Hermes executa ssh de forma não interativa.',
      sshErrHostKey:
        'A chave do host MUDOU desde a última conexão. Verifique se isso era esperado, execute ssh-keygen -R <host> e reconecte.',
      sshErrNotInstalled:
        'O Hermes não está instalado no host remoto. Instale-o lá (curl -fsSL https://hermes-agent.nousresearch.com/install.sh | sh) ou defina o caminho do Hermes.',
      sshErrPlatform:
        'Plataforma remota incompatível. O modo SSH do Hermes Desktop aceita hosts remotos Linux, macOS e Windows.',
      sshErrTimeout: 'O tempo limite da conexão SSH expirou. O host pode estar inacessível ou em suspensão.',
      sshErrUpdateRequired: 'Atualize o Hermes no host remoto antes de conectar com o Desktop SSH.',
      sshErrInteractiveAuth:
        'O Tailscale SSH exige uma verificação interativa no navegador. No Terminal, execute `ssh <host> true`, conclua a verificação e tente novamente — o Hermes executa SSH de forma não interativa.',
      sshErrUnknown: 'Falha na conexão SSH.'
    },
    keys: {
      loading: 'Carregando chaves de API e credenciais...',
      failedLoad: 'Falha ao carregar chaves de API',
      empty: 'Nada configurado nesta categoria ainda.'
    },
    search: { placeholder: 'Buscar em todas as configurações…', pill: 'Buscar' },
    profileScope: {
      appliesTo: 'Aplica-se a',
      editsProfile: profile => `As alterações nesta página se aplicam ao perfil “${profile}”.`
    },
    mcp: {
      loading: 'Carregando servidores MCP...',
      invalidJson: 'JSON do MCP inválido',
      saveFailed: 'Falha ao salvar',
      removeFailed: 'Falha ao remover',
      reloadFailed: 'Falha ao recarregar o MCP',
      savedTitle: 'Servidor MCP salvo',
      savedMessage: name => `${name} será aplicado após recarregar o MCP.`,
      disabled: 'desativado',
      name: 'Nome',
      serverJson: 'JSON do servidor',
      remove: 'Remover',
      test: 'Testar conexão',
      catalogLoading: 'Carregando catálogo MCP...',
      catalogInstallFailed: name => `Falha ao instalar ${name}`,
      catalogEnvRequired: 'Preencha os valores obrigatórios antes de instalar.',
      capabilitySummary: (tools, prompts, resources) =>
        `${[`${tools} ferramentas`, ...(prompts ? [`${prompts} prompts`] : []), ...(resources ? [`${resources} recursos`] : [])].join(', ')} ativados`,
      costTokens: tokens => `~${tokens} tokens/chamada`,
      usage30d: uses => `${uses} usos/30 dias`,
      statusConnecting: 'Conectando…',
      statusNeedsAuth: 'Autenticação necessária',
      statusError: 'Erro',
      statusOff: 'Desativado',
      allServers: 'Todos os servidores',
      authenticatedTitle: 'Autenticado',
      authenticatedMessage: (server, count) => `${server}: ${count} ferramentas`,
      authenticate: 'Autenticar',
      noOutput: 'Nenhuma saída ainda.',
      deepLinkTitle: 'Adicionar servidor MCP?',
      deepLinkDescription:
        'Um link solicitou a adição deste servidor MCP ao Hermes. Revise a configuração exata abaixo — ela vem do link, não do Hermes.',
      deepLinkStdioWarning:
        'Este servidor executa um processo local na sua máquina com o comando abaixo. Continue apenas se você confiar na origem.',
      deepLinkConfirm: 'Adicionar servidor',
      deepLinkNameInvalid: 'Os nomes usam de 1 a 64 letras, dígitos, pontos, hífens ou sublinhados.',
      deepLinkNameConflict: name => `Já existe um servidor chamado ${name} — escolha outro nome ou cancele.`,
      deepLinkErrorTitle: 'Link de instalação MCP rejeitado',
      deepLinkErrorName: 'O nome do servidor no link está ausente ou é inválido.',
      deepLinkErrorConfig: 'A configuração do link não é um JSON válido codificado em base64.',
      deepLinkErrorShape: 'A configuração deve ser um objeto JSON com um campo `url` ou `command` do tipo string.',
      deepLinkErrorUrl: 'Somente URLs de servidor http:// e https:// são permitidas.',
      deepLinkErrorTooLarge: 'Os dados de configuração excedem o limite de 32 KB.'
    },
    model: ptBrModelSettings,
    localModels: ptBrLocalModels,
    billing: {
      perMonth: amount => `${amount}/mês`,
      creditsPerMonth: amount => `${amount} créditos/mês`,
      usageLabel: label => `Uso de ${label}`,
      freeTier: {
        signIn: 'Entrar',
        title: 'Você está no nível gratuito da Nous',
        message: 'Entre com uma conta Nous para liberar mais modelos e ferramentas.',
        caption:
          'Usa nous/welcome com conectores incluídos. Ao entrar, você mantém seus conectores e ganha as ferramentas que exigem uma conta e todos os outros modelos.',
        name: 'Nous · nível gratuito',
        footnote:
          'O nível gratuito não tem saldo nem nada a pagar. Pagamentos e uso aparecem quando você entra com uma conta Nous.',
        plan: 'Nível gratuito',
        model: 'Modelo',
        connectors: 'Conectores',
        included: 'Incluídos'
      },
      amountValidation: {
        reloadTo: 'Saldo após recarga',
        greaterThanThreshold: 'O saldo após recarga deve ser maior que o limiar.',
        decimal: label => `${label}: insira um valor em dólares com no máximo 2 casas decimais.`,
        positive: label => `${label}: o valor deve ser maior que US$ 0.`,
        minimum: (label, amount) => `${label}: o mínimo é ${amount}.`,
        maximum: (label, amount) => `${label}: o máximo é ${amount}.`
      },
      stepUp: {
        openVerification: 'Abrir página de verificação',
        dismiss: 'Dispensar',
        waiting: 'Aguardando link de verificação…',
        verify: 'Verificar para continuar',
        deniedTitle: 'A verificação não foi aprovada',
        deniedBody: 'A verificação foi concluída sem permitir gastos remotos neste terminal.',
        successTitle: 'Verificação concluída',
        successBody: 'Gastos remotos estão permitidos neste terminal.'
      },
      charge: {
        added: amount => (amount ? `US$ ${amount} adicionados.` : 'Créditos adicionados.'),
        failedTitle: 'Falha na cobrança',
        unconfirmedTitle: 'Resultado da cobrança não confirmado',
        unconfirmedBody: message =>
          `${message} O resultado da sua última cobrança não foi confirmado — verifique seu saldo e histórico antes de tentar novamente.`,
        checkTitle: 'Não foi possível verificar a cobrança',
        checkBody: 'Não foi possível verificar a cobrança.',
        untrackedTitle: 'Não foi possível acompanhar a cobrança',
        untrackedBody: 'O serviço de faturamento aceitou a solicitação, mas não retornou um ID de cobrança.',
        timeoutTitle: 'Ainda processando após 5 minutos',
        timeoutBody: 'A cobrança ainda pode ser efetivada. Verifique o portal antes de tentar novamente.',
        authenticationRequired: 'Seu banco exige verificação (3DS). Conclua-a no portal para finalizar esta compra.',
        expired: 'Seu cartão venceu. Atualize-o no portal.',
        declined: 'Seu cartão foi recusado. Tente outro cartão no portal.',
        failedBody: reason => `A cobrança não foi concluída (${reason}).`
      },
      title: 'Faturamento',
      preview: 'prévia',
      summary: { balance: 'Saldo', plan: 'Plano', autoRefill: 'Recarga automática' },
      sections: { invoices: 'Faturas', plan: 'Plano', paymentAndCredits: 'Pagamento e créditos', usage: 'Uso' },
      usage: { title: 'Uso' },
      buyCredits: {
        customAmount: 'Valor personalizado de créditos',
        title: 'Comprar créditos agora',
        buyButton: 'Comprar',
        processing: 'Processando… verificando efetivação',
        added: amount => `${amount} adicionados. O saldo está sendo atualizado.`,
        retry: 'Tentar novamente',
        openPortal: 'Abrir portal'
      },
      plan: {
        title: 'Planos',
        changePlan: 'Alterar plano',
        viewPlans: 'Ver planos',
        backAria: 'Voltar ao faturamento',
        current: 'Plano atual',
        scheduled: 'Agendado',
        empty: 'Nenhum plano disponível para troca no momento.',
        undo: 'Desfazer',
        undoing: 'Desfazendo…',
        downgrade: 'Mudar para um plano inferior',
        confirmDowngrade: 'Confirmar mudança para plano inferior',
        tryAgain: 'Tentar novamente',
        checkingChange: 'Verificando esta alteração…',
        cannotChange: 'Essa alteração não pode ser feita aqui.',
        alreadyOn: name => `Você já está no ${name} — não há nada a alterar.`,
        notScheduleable: 'Esta alteração não pode ser agendada aqui.',
        scheduling: 'Agendando…',
        cancel: 'Cancelar',
        effectScheduled: (targetName, effectiveAt, creditsDelta) =>
          `Mudança para ${targetName} — entra em vigor em ${effectiveAt}. Sem cobrança agora; você mantém seu plano atual até lá.${creditsDelta ? ` Alteração nos créditos mensais: ${creditsDelta}.` : ''}`
      },
      keepGoingOnCredits: {
        title: 'Continuar usando créditos',
        description:
          'Usar seus créditos quando o uso incluído acabar. Desativado: o Hermes aguarda a renovação da franquia.',
        approveInBrowser:
          'Aprove os gastos remotos no navegador para alterar esta opção. Ela se aplica assim que você aprovar.',
        openAgain: 'Abrir novamente',
        cancel: 'Cancelar',
        approvedRetry: 'Os gastos remotos estão aprovados, mas a alteração não foi salva.',
        tryAgain: 'Tentar novamente'
      },
      autoReload: {
        threshold: 'Limiar',
        thresholdAria: 'Limiar da recarga automática',
        reloadTo: 'Recarregar até',
        reloadToAria: 'Saldo desejado após recarga automática',
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
            message: 'Entre com sua conta Nous para ver seu saldo, plano e uso aqui.',
            action: 'Entrar'
          },
          openPortal: 'Abrir portal ↗',
          noCard: {
            title: 'Nenhuma forma de pagamento cadastrada',
            message:
              'A compra de créditos avulsos e a recarga automática ficam desativadas até que um cartão seja cadastrado. Adicione um no portal.',
            action: 'Adicionar cartão ↗'
          }
        },
        paymentMethod: {
          title: 'Forma de pagamento',
          description: 'Gerencie o cartão usado para recargas e renovações da assinatura.',
          addAction: 'Adicionar forma de pagamento',
          updateAction: 'Atualizar',
          provenance: {
            autoRefill: 'cartão de recarga automática',
            customerDefault: 'padrão do cliente',
            subPin: 'cartão da assinatura',
            suffix: label => ` - ${label}`
          }
        },
        buyCredits: { description: 'Uma cobrança única no seu cartão, adicionada ao seu saldo hoje.' },
        autoRefill: {
          title: 'Recarregar quando o saldo baixar',
          genericDescription: 'Mantenha seu saldo abastecido quando ele cair abaixo do limiar definido.',
          offPill: 'Desativada',
          enabledPill: 'Ativada',
          notAvailablePill: '—',
          manageCaption: 'Gerencie a recarga automática no portal.',
          turnOnCaption: 'Ative a recarga automática no portal',
          chargesDescription: (reloadTo, threshold) =>
            `Cobra ${reloadTo} automaticamente quando seu saldo fica abaixo de ${threshold}.`,
          distinctCardCaption: cardLabel => `A recarga automática cobra em ${cardLabel} — confira no portal`,
          distinctCardFallback: 'outro cartão',
          reconcileAction: 'Conferir ↗'
        },
        usage: {
          subscriptionCredits: {
            title: 'Créditos da assinatura',
            barLabel: 'Créditos restantes da assinatura',
            captionResets: date => `Renova em ${date}`,
            valueOf: (remaining, monthly) => `${remaining} de ${monthly} restantes`,
            valueOver: (remaining, monthly, over) => `${remaining} de ${monthly} restantes · ${over} excedentes`
          },
          topupCredits: { title: 'Créditos avulsos', caption: 'Não expiram' },
          monthlyCap: {
            title: 'Limite mensal de gastos',
            barLabel: 'Limite mensal de gastos utilizado',
            captionDefault: 'Teto padrão',
            captionSpending: 'Gastos remotos mensais',
            valueUsed: (spent, limit) => `${spent} de ${limit} utilizados`
          },
          window: {
            today: 'Hoje',
            week: 'Esta semana',
            month: 'Este mês',
            topTier: 'Modelos grandes (mensal)',
            barLabel: title => `Uso incluído consumido de ${title}`,
            captionPaused: 'Pausado: falha no pagamento da assinatura',
            captionPausedOther: 'Pausado: consulte Faturamento no portal',
            captionTopTierUsedUp: 'Franquia mensal esgotada, os créditos cobrem o restante',
            captionTopTierUsedUpKeepGoingOff:
              'Franquia mensal esgotada, os outros modelos incluídos continuam funcionando',
            captionResetsIn: duration => `Renova em ${duration}`,
            captionIncluded: 'Uso incluído',
            valueUsed: percent => `${percent} utilizado`
          }
        },
        planCard: {
          freeTier: 'Gratuito',
          chooseAction: 'Escolher ↗',
          adjustPlanAction: 'Ajustar plano ↗',
          unavailableCaption: 'Os detalhes da assinatura não estão disponíveis; você ainda pode abrir o portal.',
          downgradeCaption: (tierName, when) => `Muda para ${tierName} em ${when}.`,
          cancellationCaption: when => `Cancela em ${when}.`,
          renewsCaption: date => `Renova em ${date}`,
          noSubscriptionCaption: 'Nenhuma assinatura ativa — os modelos pagos consomem créditos avulsos.'
        }
      },
      errors: {
        consentRequired: {
          title: 'Confirmação de cartão necessária',
          message: 'Confirme este cartão no portal para cobranças pelo terminal'
        },
        insufficientScope: {
          title: 'Gastos remotos precisam de aprovação',
          message: 'Esta ação exige permissão para gastos remotos. Inicie uma recarga para permitir e tente novamente.'
        },
        remoteSpendingRevoked: {
          title: 'Os gastos remotos foram interrompidos',
          messageByAdmin: 'Um administrador interrompeu os gastos remotos deste terminal.',
          messageBySelf: 'Você interrompeu os gastos remotos deste terminal.'
        },
        remoteSpendingReconnect: who =>
          `${who} Reconecte em Configurações -> Gateway para autorizar este dispositivo novamente.`,
        sessionRevoked: {
          title: 'Sessão encerrada',
          message: 'Sua sessão foi encerrada. Entre novamente em Configurações → Gateway.'
        },
        cliBillingDisabled: {
          title: 'Gastos remotos desativados',
          message:
            'Os gastos remotos estão desativados para esta conta — um administrador de faturamento pode ativá-los na página Hermes Agent do portal.'
        },
        roleRequired: {
          title: 'Permissão de administrador necessária',
          message:
            'Adicionar fundos exige um administrador ou proprietário da organização. Peça a um administrador ou gerencie no portal.'
        },
        idempotencyConflict: {
          title: 'Inicie uma nova recarga',
          message: '🔴 Essa chave de cobrança já foi usada para outro valor. Inicie uma nova recarga.'
        },
        noPaymentMethod: {
          title: 'Nenhum cartão salvo',
          message:
            '💳 Ainda não há cartão salvo para cobranças pelo terminal. Cadastre um no portal ' +
            '(compras avulsas de créditos não salvam um cartão reutilizável).'
        },
        orgAccessDenied: {
          title: 'Acesso à organização negado',
          message: 'Este token não está vinculado a uma organização que você pode gerenciar'
        },
        monthlyCapExceeded: {
          title: 'Limite mensal de gastos atingido',
          messageReached: '🔴 Limite mensal de gastos atingido.',
          messageHeadroom: remaining => `🔴 Limite mensal de gastos atingido — restam US$ ${remaining} de margem.`
        },
        rateLimited: {
          title: 'Muitas cobranças no momento',
          message: mins =>
            mins > 0
              ? `🟡 Muitas cobranças no momento (tente novamente em ~${mins} min). Isso não é uma falha no pagamento.`
              : '🟡 Muitas cobranças no momento. Isso não é uma falha no pagamento.'
        },
        stripeUnavailable: {
          title: 'A Stripe está com problemas',
          message: mins =>
            mins > 0
              ? `A Stripe está com problemas — tente novamente em ~${mins} min`
              : 'A Stripe está com problemas — tente novamente em breve'
        },
        upgradeCapExceeded: {
          title: 'Limite diário de alterações de plano atingido',
          message: 'Limite diário de alterações de plano atingido — tente novamente amanhã'
        },
        endpointUnavailable: {
          title: 'Endpoint de faturamento indisponível',
          message:
            'O endpoint de faturamento retornou uma resposta que não é JSON (ele pode não estar disponível nesta implantação).'
        },
        timeout: {
          title: 'Tempo limite da solicitação de faturamento esgotado',
          message: 'O tempo limite da solicitação de faturamento esgotou.'
        },
        transport: {
          title: 'Falha na conexão de faturamento',
          message: 'A solicitação de faturamento falhou antes de chegar ao gateway.'
        },
        default: { title: 'Falha na solicitação de faturamento', message: 'A solicitação de faturamento falhou.' }
      }
    },
    providers: {
      connectAccount: 'Conectar uma conta',
      haveApiKey: 'Prefere usar uma chave de API?',
      intro:
        'Entre com uma assinatura — sem precisar copiar uma chave de API. O Hermes faz o login pelo navegador para você, aqui no aplicativo.',
      connected: 'Conectado',
      collapse: 'Recolher',
      connectAnother: 'Conectar outro provedor',
      otherProviders: 'Outros provedores',
      disconnect: 'Desconectar',
      disconnectInTerminal: 'Desconectar (executa o comando de remoção no terminal)',
      removeConfirm: provider => `Remover ${provider}?`,
      removeExternalGeneric: provider => `${provider} é gerenciado pela própria CLI — remova por lá.`,
      removeKeyManaged: provider => `${provider} está configurado por uma chave de API. Remova em Chaves de API.`,
      removeTerminalConfirm: (provider, command) =>
        `Desconectar ${provider}? Isso executa "${command}" no terminal para limpar a credencial.`,
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
          'Direcione o Hermes para qualquer endpoint compatível com OpenAI (Zyphra, vLLM, llama.cpp, Ollama etc.).'
      },
      loading: 'Carregando provedores...'
    },
    sessions: {
      loading: 'Carregando sessões arquivadas…',
      archivedTitle: 'Sessões arquivadas',
      archivedIntro:
        'Conversas arquivadas ficam ocultas na barra lateral, mas mantêm todas as mensagens. Use Alt/⌥+Shift+clique em uma conversa na barra lateral para arquivá-la.',
      emptyArchivedTitle: 'Nada arquivado',
      emptyArchivedDesc: 'Arquive uma conversa para ocultá-la aqui.',
      unarchive: 'Desarquivar',
      deletePermanently: 'Excluir permanentemente',
      messages: count => `${count} ${count === 1 ? 'mensagem' : 'mensagens'}`,
      restored: 'Restaurada',
      deleteConfirm: title => `Excluir permanentemente "${title}"? Essa ação não pode ser desfeita.`,
      autoArchiveTitle: 'Arquivar automaticamente conversas inativas',
      autoArchiveDesc:
        'Arquivar automaticamente conversas que você não usa há algum tempo. Conversas fixadas nunca são arquivadas, e nada é excluído — as conversas arquivadas apenas vêm para cá.',
      autoArchiveDaysLabel: 'Arquivar após',
      autoArchiveDaysUnit: 'dias de inatividade',
      autoArchiveFailed: 'Não foi possível atualizar o arquivamento automático',
      defaultDirTitle: 'Pasta padrão dos projetos',
      defaultDirDesc:
        'Novas sessões começam nesta pasta, a menos que você escolha outra. Deixe sem definir para usar sua pasta pessoal.',
      defaultDirUpdated: shortcut =>
        `Pasta padrão dos projetos atualizada — inicie uma nova conversa${shortcut ? ` (${shortcut})` : ''} para a alteração ter efeito`,
      defaultsTo: label => `O padrão é ${label}.`,
      change: 'Alterar',
      choose: 'Escolher',
      clear: 'Limpar',
      notSet: 'Não definido',
      failedLoad: 'Não foi possível carregar as sessões arquivadas',
      unarchiveFailed: 'Falha ao desarquivar',
      deleteFailed: 'Falha ao excluir',
      updateDirFailed: 'Não foi possível atualizar a pasta padrão',
      clearDirFailed: 'Não foi possível limpar a pasta padrão'
    },
    toolsets: {
      loadingConfig: 'Carregando configurações',
      savedTitle: 'Credencial salva',
      savedMessage: key => `${key} atualizado.`,
      removedTitle: 'Credencial removida',
      removedMessage: key => `${key} removido.`,
      failedSave: key => `Falha ao salvar ${key}`,
      failedRemove: key => `Falha ao remover ${key}`,
      failedReveal: key => `Falha ao revelar ${key}`,
      removeConfirm: key => `Remover ${key} do .env?`,
      set: 'Definido',
      notSet: 'Não definido',
      selectedTitle: 'Provedor selecionado',
      selectedMessage: provider => `${provider} está ativo agora.`,
      failedSelect: provider => `Falha ao selecionar ${provider}`,
      failedLoad: 'Falha ao carregar a configuração das ferramentas',
      noProviderOptions:
        'Este conjunto de ferramentas não tem opções de provedor — ative-o e ele funcionará com sua configuração atual.',
      noProviders: 'Nenhum provedor disponível para este conjunto de ferramentas no momento.',
      ready: 'Pronto',
      needsSignIn: 'Login necessário',
      needsSetup: 'Configuração necessária',
      activeBackend: 'Ativo',
      activeBackendHint: 'Este é seu backend ativo',
      useBackend: 'Usar este backend',
      nousIncluded: 'Incluído na assinatura Nous — entre com sua conta Nous para ativar.',
      nousAuthNeededTitle: 'Entre com sua conta Nous',
      nousAuthNeededMessage: provider =>
        `${provider} está salvo, mas só funcionará depois que você entrar com sua conta Nous.`,
      nousAuthSignIn: 'Entrar',
      nousAuthDoneTitle: 'Conta Nous conectada',
      nousAuthDoneMessage: 'Os backends da sua assinatura estão ativos agora.',
      nousAuthFailed: 'O login na Nous não foi concluído',
      nousAuthFailedMessage: 'Tente novamente.',
      nousAuthTryAgain: 'Tentar novamente',
      noApiKeyRequired: 'Não exige chave de API.',
      postSetupHint: step =>
        `Este backend precisa de uma instalação única (${step}). Executa nesta máquina — pode levar alguns minutos.`,
      postSetupInstalledHint: 'Instalado. Execute a configuração novamente apenas se algo não estiver funcionando.',
      postSetupRun: 'Executar configuração',
      postSetupRerun: 'Executar configuração novamente',
      postSetupInstalled: 'Instalado',
      postSetupRunning: 'Instalando…',
      postSetupStarting: 'Iniciando…',
      postSetupCompleteTitle: 'Configuração concluída',
      postSetupCompleteMessage: step => `${step} instalado.`,
      postSetupErrorTitle: 'Configuração concluída com erros',
      postSetupErrorMessage: step =>
        `A configuração de ${step} não foi concluída. Abra os logs para ver o motivo e execute a configuração novamente.`,
      postSetupOpenLogs: 'Abrir logs',
      postSetupRunAgain: 'Executar novamente',
      postSetupFailed: step => `Falha ao executar a configuração de ${step}`,
      webSearchActive: backend => `Busca: ${backend}`,
      webExtractActive: backend => `Extração: ${backend}`,
      webCapabilityUnset: 'não definido',
      webUseForSearch: 'Usar para busca',
      webUseForExtract: 'Usar para extração',
      webUsedForSearch: 'Backend de busca',
      webUsedForExtract: 'Backend de extração',
      webCapabilitySelectedMessage: (provider, capability) => `${provider} agora cuida de ${capability} na web.`,
      failedSelectCapability: provider => `Falha ao definir ${provider}`,
      loadingModels: 'Carregando catálogo de modelos...',
      modelSectionTitle: 'Modelo',
      modelCount: count => `${count} modelo${count === 1 ? '' : 's'}`,
      modelInUse: 'Em uso',
      modelDefault: 'padrão',
      modelInactiveHint: 'Selecione este backend primeiro para alterar seu modelo.',
      modelSelectedTitle: 'Modelo selecionado',
      modelSelectedMessage: model => `${model} será aplicado às novas sessões.`,
      failedSelectModel: model => `Falha ao selecionar ${model}`,
      terminalBackend: {
        sectionTitle: 'Backend de execução',
        loading: 'Verificando backends de execução…',
        failedLoad: 'Não foi possível carregar os backends do terminal',
        ready: 'Pronto',
        needsSetup: 'Configuração necessária',
        unavailable: 'Indisponível',
        inUse: 'Em uso',
        selectedTitle: 'Backend selecionado',
        selectedMessage: backend => `Os comandos do terminal agora executam via ${backend}. Aplica-se a novas sessões.`,
        failedSelect: backend => `Falha ao selecionar ${backend}`,
        needsSetupHint:
          'Este backend está selecionado, mas sua configuração está incompleta — os comandos falharão até que a configuração seja concluída.',
        needsSetupConfirmTitle: backend => `Selecionar ${backend} mesmo assim?`,
        needsSetupConfirmDescription: detail =>
          `${detail} As sessões iniciadas após esta alteração não terão ferramentas de terminal nem de arquivos até a configuração ser concluída.`,
        needsSetupConfirmDescriptionGeneric:
          'Este backend ainda não está configurado. As sessões iniciadas após esta alteração não terão ferramentas de terminal nem de arquivos até a configuração ser concluída.',
        needsSetupConfirmAction: 'Selecionar mesmo assim',
        unavailableTitle: 'Os comandos do terminal não estão disponíveis',
        unavailableMessage: backend =>
          `O Hermes não consegue executar comandos de shell agora: ${backend} não está pronto. Mude para Local, ou termine de configurar ${backend} e tente novamente.`,
        openBackendSettings: 'Abrir configurações do terminal',
        useLocal: 'Usar Local',
        switchedToLocal: 'Os comandos do terminal agora executam localmente. Aplica-se a novas sessões.'
      },
      browserRealProfile: {
        label: 'Usar meu perfil real do navegador',
        description:
          'Copia os logins e cookies do seu navegador padrão para uma cópia gerenciada usada pelo agente, e também conecta a aba Navegador do aplicativo com esses dados. Seu perfil em uso nunca é aberto diretamente. Aplica-se a novas sessões.',
        enabledTitle: 'Navegação com perfil real ativada',
        enabledMessage: 'Novas sessões navegarão com uma cópia do perfil do seu navegador padrão.',
        disabledTitle: 'Navegação com perfil real desativada',
        disabledMessage:
          'A cópia do perfil será excluída e os logins correspondentes serão encerrados no navegador integrado; novas sessões usarão um navegador limpo.',
        failedSave: 'Não foi possível salvar a configuração de perfil real',
        reimportSignins: 'Reimportar logins',
        signinsImportedTitle: 'Logins do navegador integrado',
        signinsImported: count =>
          `O navegador integrado foi conectado com ${count} cookie${count === 1 ? '' : 's'} do perfil do seu navegador.`,
        signinsRemote:
          'Este backend executa em outra máquina, então seu perfil de navegador não é importado para o navegador integrado.',
        signinsFailed: 'Não foi possível importar logins para o navegador integrado',
        prompt: {
          title: 'Continue conectado aos seus sites',
          body: 'Permita que o Hermes navegue com uma cópia do perfil do seu navegador padrão, para que os sites abram já conectados, tanto para o agente quanto na aba Navegador do aplicativo.',
          bulletSnapshot: 'Cookies e logins são copiados para uma cópia gerenciada.',
          bulletLiveProfile: 'Seu perfil de navegador em uso nunca é aberto diretamente.',
          bulletLocal: 'Nada sai deste computador.',
          dontShowAgain: 'Não mostrar novamente',
          notNow: 'Agora não',
          enable: 'Usar meu perfil'
        }
      }
    }
  }
} satisfies TranslationOverrides
