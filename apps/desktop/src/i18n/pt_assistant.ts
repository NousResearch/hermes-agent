import type { TranslationOverrides } from './define-locale'

export const ptAssistant = {
  assistant: {
    thread: {
      loadingSession: 'Carregando a sessão',
      showEarlier: 'Mostrar mensagens anteriores',
      loadingResponse: 'O Hermes está carregando uma resposta',
      loadingLocalModel: model => `Carregando ${model} na memória`,
      processingPrompt: 'Processando o prompt',
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
      branchNewChat: 'Ramificar em novo chat',
      react: 'Reagir',
      dismissError: 'Dispensar o erro',
      responseStopped: 'Resposta interrompida',
      errorLayers: {
        auth: 'Erro de autenticação',
        billing: 'Sem créditos',
        disk: 'Disco cheio',
        endpoint: 'Erro de endpoint personalizado',
        gateway: 'Erro do gateway',
        generic: 'O turno falhou',
        provider: 'Erro do provedor',
        runtime: 'Erro do runtime local',
        streaming: 'Erro de conexão de streaming'
      },
      errorLayerBodies: {
        auth: 'O serviço de IA rejeitou o seu login. Verifique as credenciais deste provedor e envie a mensagem de novo.',
        billing: 'A sua conta não tem mais créditos neste provedor. Recarregue ou troque de provedor e envie de novo.',
        disk: 'O seu disco está cheio, então o Hermes não conseguiu salvar esta conversa. Libere espaço e tente de novo.',
        endpoint:
          'O Hermes não consegue acessar o seu servidor de modelo personalizado. Verifique se ele está em execução e envie a mensagem de novo.',
        gateway:
          'O Hermes teve um problema interno ao iniciar esta resposta. Envie a mensagem de novo; se continuar acontecendo, envie o diagnóstico.',
        generic:
          'Algo deu errado enquanto o Hermes respondia. Tente de novo ou copie os detalhes se o problema persistir.',
        provider:
          'O serviço de IA não conseguiu concluir esta solicitação. Tente de novo em instantes ou troque de provedor.',
        runtime:
          'O Hermes teve um problema interno ao iniciar esta resposta. Envie a mensagem de novo; se continuar acontecendo, envie o diagnóstico.',
        streaming: 'A conexão caiu antes de a resposta terminar. Tente de novo para reenviá-la.'
      },
      errorCodes: {
        auth: {
          title: provider => `${provider} rejeitou o seu login`,
          body: provider =>
            `As credenciais salvas para ${provider} não foram aceitas. Corrija-as em Configurações ou troque de provedor e envie a mensagem de novo.`
        },
        auth_permanent: {
          title: provider => `${provider} rejeitou o seu login`,
          body: provider =>
            `As credenciais salvas para ${provider} são inválidas ou foram revogadas. Atualize-as ou troque de provedor e envie a mensagem de novo.`
        },
        billing: {
          title: 'Sem créditos',
          body: provider =>
            `A sua conta em ${provider} não tem mais créditos. Recarregue ou troque de provedor e envie de novo.`
        },
        rate_limit: {
          title: 'O serviço de IA está ocupado',
          body: provider => `${provider} está limitando as solicitações agora. Espere um minuto e tente de novo.`
        },
        upstream_rate_limit: {
          title: 'O serviço de IA está ocupado',
          body: provider => `${provider} está limitando as solicitações agora. Espere um minuto e tente de novo.`
        },
        overloaded: {
          title: 'O serviço de IA está sobrecarregado',
          body: provider => `${provider} está com problemas agora. Tente de novo em instantes ou troque de provedor.`
        },
        server_error: {
          title: 'O serviço de IA teve um problema',
          body: provider =>
            `${provider} retornou um erro de servidor. Tente de novo em instantes ou troque de provedor.`
        },
        timeout: {
          title: 'Não foi possível acessar o serviço de IA',
          body: provider =>
            `Não foi possível acessar ${provider}, ou ele não respondeu a tempo. Verifique sua conexão com a internet e tente de novo.`
        },
        stream_drop: {
          title: 'A resposta foi interrompida',
          body: 'A conexão caiu antes de a resposta terminar. Tente de novo para reenviá-la.'
        },
        no_reply: {
          title: 'A resposta não terminou',
          body: 'O Hermes encerrou este turno sem uma resposta. Tente de novo para reenviá-la.'
        },
        upstream_blocked: {
          title: 'Um firewall bloqueou a solicitação',
          body: provider =>
            `Um firewall ou CDN na frente de ${provider} bloqueou a solicitação antes de ela chegar ao modelo; provavelmente a sua chave está certa. Defina um cabeçalho User-Agent pelo extra_headers do provedor em Configurações ou troque de provedor e envie a mensagem de novo.`
        },
        ssl_cert_verification: {
          title: 'Falha na conexão segura',
          body: provider =>
            `O Hermes não conseguiu verificar a conexão segura com ${provider}. Verifique as configurações de rede ou de proxy, ou troque de provedor, e envie a mensagem de novo.`
        },
        context_overflow: {
          title: 'Esta conversa ficou longa demais',
          body: 'A conversa não cabe mais no modelo. Compacte-a ou inicie um novo chat e envie de novo.'
        },
        payload_too_large: {
          title: 'Esta mensagem é grande demais',
          body: 'A solicitação ficou grande demais para o modelo. Compacte a conversa ou inicie um novo chat e envie de novo.'
        },
        model_not_found: {
          title: 'Este modelo não está disponível',
          body: provider =>
            `${provider} não oferece este modelo na sua conta. Escolha outro modelo e envie a mensagem de novo.`
        },
        provider_policy_blocked: {
          title: 'Este modelo está bloqueado pelas configurações da sua conta',
          body: provider =>
            `${provider} não encaminharia esta solicitação com as configurações de dados ou de privacidade da sua conta. Escolha outro modelo ou troque de provedor.`
        },
        content_policy_blocked: {
          title: 'O serviço de IA recusou esta solicitação',
          body: provider => `${provider} não responderia a esta mensagem. Edite-a e envie de novo.`
        },
        format_error: {
          title: 'O serviço de IA rejeitou a solicitação',
          body: provider =>
            `${provider} não aceitou a forma como esta solicitação foi montada. Troque de provedor ou envie o diagnóstico para investigarmos.`
        },
        truncated: {
          title: 'A resposta foi cortada',
          body: 'O modelo parou antes de terminar. Tente de novo para obter uma resposta completa.'
        },
        invalid_response: {
          title: 'O serviço de IA enviou uma resposta ilegível',
          body: provider => `${provider} retornou algo que o Hermes não conseguiu ler. Tente de novo em instantes.`
        },
        empty_response: {
          title: 'O serviço de IA enviou uma resposta vazia',
          body: provider => `${provider} não retornou nada para esta mensagem. Tente de novo em instantes.`
        },
        loop_error: {
          title: 'O Hermes ficou preso em um loop',
          body: 'A resposta ficou repetindo os mesmos passos, então o Hermes a interrompeu. Tente de novo ou inicie um novo chat se acontecer outra vez.'
        },
        SESSION_NOT_OWNED: {
          title: 'Este chat está aberto em outro lugar',
          body: 'Este chat está aberto em outra janela ou terminal do Hermes. Feche-o lá e envie a mensagem de novo, ou inicie um novo chat aqui.'
        },
        disk_full: {
          title: 'Disco cheio',
          body: 'O seu disco está cheio, então o Hermes não conseguiu salvar esta conversa. Libere espaço e tente de novo.'
        },
        free_tier_disabled: {
          title: 'O uso do Hermes sem login está desativado no momento',
          body: 'Faça login com uma conta Nous para continuar conversando. É grátis.'
        },
        free_tier_rate_limited: {
          title: 'Você usou toda a cota de conversas sem login',
          body: 'Ela se renova em breve. Faça login com uma conta Nous para ter uma cota maior. É grátis.'
        },
        free_tier_at_capacity: {
          title: 'O chat sem login está muito movimentado agora',
          body: 'Faça login para furar a fila (é grátis) ou tente de novo daqui a pouco.'
        },
        free_tier_model_not_free: {
          title: 'Esse modelo não está disponível sem login',
          body: 'Por enquanto, o Hermes usa o modelo gratuito. Faça login com uma conta Nous para ter mais modelos. É grátis.'
        },
        free_tier_route: {
          title: 'O Hermes não conseguiu acessar o modelo gratuito por esta rota',
          body: 'Faça login com uma conta Nous (é grátis) ou verifique a configuração NOUS_INFERENCE_BASE_URL.'
        },
        free_tier_outage: {
          title: 'O modelo gratuito está com dificuldade para responder agora',
          body: 'Tente enviar a mensagem de novo em um minuto.'
        },
        free_tier_refused: {
          title: 'O Hermes não conseguiu enviar isso sem login',
          body: 'Fazer login com uma conta Nous é grátis.'
        }
      },
      errorAuthKinds: {
        api_key: {
          title: provider => `${provider} rejeitou a sua chave de API`,
          body: provider => `A chave salva para ${provider} é inválida ou foi revogada. Atualize-a e tente de novo.`
        },
        oauth: {
          title: provider => `O seu login em ${provider} expirou`
        }
      },
      errorDetails: 'Detalhes',
      errorGenericProvider: 'O serviço de IA',
      errorToastTitle: 'O Hermes não conseguiu concluir a resposta',
      errorRetry: 'Tentar novamente',
      errorLimitResets: time => `O limite é renovado às ${time}`,
      errorRetryAtReset: time => `Tentar de novo quando o limite for renovado (${time})`,
      errorRetryScheduled: (time, wait) => `Nova tentativa às ${time}, em ${wait}`,
      errorRetryScheduledCancel: 'Cancelar',
      errorStartNewSession: 'Iniciar nova sessão',
      errorSwitchProvider: 'Mudar provedor',
      errorChooseModel: 'Escolher um modelo',
      errorCompressConversation: 'Compactar a conversa',
      errorCompressFailed: 'Não foi possível compactar a conversa',
      errorOpenHermesFolder: 'Abrir a pasta do Hermes',
      errorOpenHermesFolderFailed: 'Não foi possível abrir a pasta do Hermes',
      errorUpdateApiKey: 'Atualizar a chave de API',
      errorSignInAgain: provider => `Fazer login em ${provider} de novo`,
      errorSignInFreeTier: 'Fazer login com uma conta Nous',
      errorOauthExpired: provider =>
        `O seu login em ${provider} expirou ou foi revogado. Faça login de novo para continuar conversando.`,
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
      copyFullResponse: 'Copiar a resposta completa',
      readAloudFullResponseHint: 'Shift+clique: ler a resposta completa',
      editMessage: 'Editar mensagem',
      expandMessage: 'Expandir mensagem',
      scrollToBottom: 'Rolar para o fim',
      stop: 'Parar',
      restorePrevious: 'Restaurar o checkpoint anterior',
      restoreCheckpoint: 'Restaurar checkpoint',
      restoreFromHere: 'Restaurar checkpoint: reexecutar a partir deste prompt',
      restoreTitle: 'Restaurar para este checkpoint?',
      restoreBody:
        'Tudo o que vem depois deste prompt é removido da conversa, e o prompt é executado de novo a partir daqui.',
      restoreConfirm: 'Restaurar e reexecutar',
      restoreNext: 'Restaurar o próximo checkpoint',
      goForward: 'Avançar',
      sendEdited: 'Enviar mensagem editada',
      attachingFile: 'Anexando…'
    },
    approval: {
      gatewayDisconnected: 'O gateway do Hermes não está conectado',
      sendFailed: 'Não foi possível enviar resposta de aprovação',
      reconnect: 'Reconectar',
      timedOutSystemLine:
        'A aprovação expirou: o comando não foi executado. Peça ao Hermes para tentar de novo ou aumente o limite em Configurações → Segurança → Tempo limite de aprovação.',
      openSafetySettings: 'Abrir as configurações de Segurança',
      run: 'Executar',
      command: 'Comando',
      commandDetails: 'Detalhes do comando',
      moreOptions: 'Mais opções de aprovação',
      allowSession: 'Permitir esta sessão',
      alwaysAllowMenu: 'Sempre permitir…',
      jumpToApproval: 'Aprovação necessária',
      reject: 'Rejeitar',
      alwaysTitle: 'Sempre permitir este comando?',
      alwaysDescription: pattern =>
        `Isto adiciona o padrão “${pattern}” à sua lista de permissões permanente (~/.hermes/config.yaml). O Hermes não perguntará de novo sobre comandos como este, nesta sessão nem em sessões futuras.`,
      alwaysAllow: 'Sempre permitir'
    },
    clarify: {
      notReady: 'A solicitação de esclarecimento ainda não está pronta',
      gatewayDisconnected: 'O gateway do Hermes não está conectado',
      sendFailed: 'Não foi possível enviar resposta de esclarecimento',
      loadingQuestion: 'Carregando a pergunta…',
      other: 'Outro (digite a sua resposta)',
      placeholder: 'Digite a sua resposta…',
      skip: 'Pular',
      skipped: 'Ignorada',
      noAnswer: 'Sem resposta',
      confirmAndContinueLabel: 'Confirmar e continuar',
      singleSelectHint: 'Escolha uma',
      multiSelectHint: 'Selecione todas as que se aplicam',
      questionProgress: (answered, total) => `${answered} de ${total} respondidas`,
      notDelivered:
        'Esta pergunta não chegou ao app, então não pode ser respondida aqui. Pressione Parar para encerrar o turno e responda no chat.'
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
      skill: name => `skill ${name}`,
      kind: {
        plugin: 'plugin',
        skill: 'skill'
      },
      tier: {
        official: 'oficial',
        community: 'da comunidade'
      },
      targetProfile: profile => `Instala no seu perfil ${profile}`,
      sendFailed: 'Não foi possível enviar a sua resposta. Tente novamente.',
      commitLabel: 'Commit',
      subdirLabel: 'Pasta',
      securityHeading: 'Segurança',
      scan: {
        passed: 'Varredura aprovada',
        warnings: 'A varredura encontrou avisos',
        failed: 'A varredura falhou'
      },
      requirementsLabel: 'Requer',
      requiresHermes: range => `Hermes ${range}`,
      envVar: name => `variável de ambiente ${name}`,
      credentialsHeading: 'Credenciais',
      phase: {
        downloading: 'Baixando…',
        python_packages: 'Instalando pacotes Python…',
        loading_tools: 'Carregando as ferramentas…'
      },
      serverNotConnected: (server, reason) => `Servidor MCP ${server} não conectado${reason ? `: ${reason}` : ''}`,
      notEnabled: 'Instalado, mas não ativado',
      missingEnv: names => `Defina ${names} para concluir a configuração`,
      alreadyInstalled: 'Já instalado; mantido como está'
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
      envRequired: 'Preencha as credenciais necessárias primeiro',
      sendFailed: 'Não foi possível enviar resposta de configuração MCP',
      reloadFailed:
        'O servidor foi salvo, mas falhou ao recarregar as ferramentas MCP: elas carregam na próxima sessão',
      gatewayDisconnected: 'O gateway do Hermes não está conectado'
    },
    tool: {
      copyCode: 'Copiar código',
      renderingImage: 'Renderizando a imagem',
      copyOutput: 'Copiar saída',
      copyCommand: 'Copiar comando',
      copyContent: 'Copiar conteúdo',
      copyUrl: 'Copiar URL',
      copyResults: 'Copiar resultados',
      copyQuery: 'Copiar consulta',
      copyFile: 'Copiar arquivo',
      copyPath: 'Copiar caminho',
      failedCalls: count => `${count} chamada${count === 1 ? '' : 's'} de ferramenta com falha`,
      skillActivity: {
        loading: 'Carregando a skill',
        loaded: 'Skill carregada',
        loadFailed: 'Falha ao carregar a skill',
        readingResource: 'Lendo o recurso da skill',
        readResource: 'Recurso da skill lido',
        resourceFailed: 'Falha ao ler o recurso da skill',
        listing: 'Listando as skills',
        listed: 'Skills listadas',
        listFailed: 'Falha ao listar as skills',
        unavailable: 'Resultado da skill indisponível'
      },
      outputAlt: 'Saída da ferramenta',
      rawResponse: 'Resposta bruta',
      copyActivity: 'Copiar atividade',
      recoveredOne: 'Recuperado após 1 etapa com falha',
      recoveredMany: count => `Recuperado após ${count} etapas com falha`,
      failedOne: '1 etapa falhou',
      failedMany: count => `${count} etapas falharam`,
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
        prefixedDone: (prefix, action) => `${prefix}: ${action}`,
        runningPrefixedTool: (prefix, action) => `Executando ${prefix.toLowerCase()}: ${action.toLowerCase()}`,
        runningTool: action => `Executando ${action.toLowerCase()}`
      },
      titles: {
        browser_click: {
          done: 'Clicou em um elemento da página',
          pending: 'Clicando em um elemento da página',
          pendingAction: 'Clicando'
        },
        browser_fill: {
          done: 'Preencheu um campo do formulário',
          pending: 'Preenchendo um campo do formulário',
          pendingAction: 'Preenchendo'
        },
        browser_navigate: {
          done: 'Abriu a página',
          pending: 'Abrindo a página',
          pendingAction: 'Abrindo'
        },
        browser_snapshot: {
          done: 'Capturou um snapshot da página',
          pending: 'Capturando um snapshot da página',
          pendingAction: 'Capturando'
        },
        browser_take_screenshot: {
          done: 'Capturou uma captura de tela',
          pending: 'Capturando uma captura de tela',
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
          done: 'Tarefa cron',
          pending: 'Agendando uma tarefa cron',
          pendingAction: 'Agendando'
        },
        edit_file: {
          done: 'Editou o arquivo',
          pending: 'Editando o arquivo',
          pendingAction: 'Editando'
        },
        execute_code: {
          done: 'Executou código',
          pending: 'Executando script',
          pendingAction: 'Executando script'
        },
        image_generate: {
          done: 'Gerou uma imagem',
          pending: 'Gerando uma imagem',
          pendingAction: 'Gerando'
        },
        list_files: {
          done: 'Listou arquivos',
          pending: 'Listando arquivos',
          pendingAction: 'Listando'
        },
        memory: {
          done: 'Salvou na memória',
          pending: 'Salvando na memória',
          pendingAction: 'Salvando'
        },
        patch: {
          done: 'Aplicou um patch no arquivo',
          pending: 'Aplicando um patch no arquivo',
          pendingAction: 'Aplicando patch'
        },
        read_file: {
          done: 'Leu o arquivo',
          pending: 'Lendo o arquivo',
          pendingAction: 'Lendo'
        },
        search_files: {
          done: 'Pesquisou arquivos',
          pending: 'Pesquisando arquivos',
          pendingAction: 'Pesquisando'
        },
        session_search_recall: {
          done: 'Pesquisou o histórico de sessões',
          pending: 'Pesquisando o histórico de sessões',
          pendingAction: 'Pesquisando'
        },
        terminal: {
          done: 'Executou um comando',
          pending: 'Executando um comando',
          pendingAction: 'Executando'
        },
        todo: {
          done: 'Atualizou as tarefas',
          pending: 'Atualizando as tarefas',
          pendingAction: 'Atualizando'
        },
        vision_analyze: {
          done: 'Analisou uma imagem',
          pending: 'Analisando uma imagem',
          pendingAction: 'Analisando'
        },
        web_extract: {
          done: 'Leu uma página web',
          pending: 'Lendo uma página web',
          pendingAction: 'Lendo'
        },
        web_search: {
          done: 'Pesquisou na web',
          pending: 'Pesquisando na web',
          pendingAction: 'Pesquisando'
        },
        write_file: {
          done: 'Editou o arquivo',
          pending: 'Editando o arquivo',
          pendingAction: 'Editando'
        }
      }
    }
  }
} satisfies Pick<TranslationOverrides, 'assistant'>
