import { defineLocale } from './define-locale'

export const pt = defineLocale({
  common: {
    apply: 'Aplicar',
    back: 'Voltar',
    save: 'Guardar',
    saving: 'A guardar…',
    cancel: 'Cancelar',
    change: 'Alterar',
    choose: 'Escolher',
    clear: 'Limpar',
    close: 'Fechar',
    collapse: 'Recolher',
    confirm: 'Confirmar',
    connect: 'Ligar',
    connecting: 'A ligar',
    continue: 'Continuar',
    copied: 'Copiado',
    copy: 'Copiar',
    copyFailed: 'Falha ao copiar',
    delete: 'Eliminar',
    docs: 'Documentação',
    done: 'Concluído',
    error: 'Erro',
    expand: 'Expandir',
    failed: 'Falhou',
    formatJson: 'Formatar JSON',
    free: 'Grátis',
    loading: 'A carregar…',
    notSet: 'Não definido',
    refresh: 'Atualizar',
    remove: 'Remover',
    replace: 'Substituir',
    retry: 'Tentar novamente',
    run: 'Executar',
    send: 'Enviar',
    set: 'Definir',
    skip: 'Ignorar',
    update: 'Atualizar',
    tryHint: term => `Experimenta “${term}”`,
    on: 'Ligado',
    off: 'Desligado'
  },

  boot: {
    ready: 'Hermes Desktop está pronto'
  },

  notifications: {
    region: 'Notificações',
    hide: 'Ocultar',
    show: 'Mostrar',
    clearAll: 'Limpar tudo',
    dismiss: 'Fechar notificação',
    details: 'Detalhes',
    copyDetail: 'Copiar detalhe',
    copyDetailFailed: 'Não foi possível copiar o detalhe da notificação',
    backendOutOfDateTitle: 'Backend desatualizado',
    backendOutOfDateMessage:
      'O teu backend Hermes é mais antigo do que esta versão desktop e pode não funcionar corretamente. Atualiza para alinhar as versões.',
    updateHermes: 'Atualizar Hermes',
    updateReadyTitle: 'Atualização pronta',
    seeWhatsNew: 'Ver novidades'
  },

  titlebar: {
    hideSidebar: 'Ocultar barra lateral',
    showSidebar: 'Mostrar barra lateral',
    search: 'Pesquisar',
    searchTitle: 'Pesquisar sessões, vistas e ações',
    swapSidebarSides: 'Trocar lados das barras laterais',
    swapSidebarSidesTitle: 'Trocar os lados das sessões e do explorador de ficheiros',
    hideRightSidebar: 'Ocultar barra lateral direita',
    showRightSidebar: 'Mostrar barra lateral direita',
    muteHaptics: 'Desativar feedback háptico',
    unmuteHaptics: 'Ativar feedback háptico',
    openSettings: 'Abrir definições',
    openStarmap: 'Abrir mapa de memória',
    openKeybinds: 'Atalhos de teclado'
  },

  language: {
    label: 'Idioma',
    description: 'Escolhe o idioma da interface desktop.',
    saving: 'A guardar idioma…',
    saveError: 'Falha ao atualizar o idioma',
    switchTo: 'Mudar idioma',
    searchPlaceholder: 'Pesquisar idiomas…',
    noResults: 'Nenhum idioma encontrado'
  },

  settings: {
    closeSettings: 'Fechar definições',
    exportConfig: 'Exportar configuração',
    importConfig: 'Importar configuração',
    resetToDefaults: 'Repor predefinições',
    resetConfirm: 'Repor todas as definições para os valores predefinidos do Hermes?',
    exportFailed: 'Falha ao exportar',
    resetFailed: 'Falha ao repor',
    nav: {
      providers: 'Fornecedores',
      providerAccounts: 'Contas',
      providerApiKeys: 'Chaves de API',
      gateway: 'Gateway',
      apiKeys: 'Ferramentas e chaves',
      keysTools: 'Ferramentas',
      keysSettings: 'Definições',
      mcp: 'MCP',
      archivedChats: 'Conversas arquivadas',
      about: 'Sobre',
      notifications: 'Notificações'
    },
    appearance: {
      title: 'Aspeto'
    }
  },

  cron: {
    promptPlaceholder: 'O que deve o agente fazer em cada execução?'
  },

  composer: {
    lookupNoMatches: 'Sem resultados.'
  },

  assistant: {
    tool: {
      statusRecovered: 'Recuperado'
    }
  }
})
