import type { CatalogInstallTranslations } from './types_catalog_install'

export const ptBrCatalogInstall: CatalogInstallTranslations = {
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
  kind: { plugin: 'plugin', skill: 'habilidade' },
  tier: { official: 'oficial', verified: 'verificado', community: 'comunidade' },
  targetProfile: profile => `Instala no seu perfil ${profile}`,
  sendFailed: 'Não foi possível enviar sua resposta. Tente novamente.',
  commitLabel: 'Commit',
  subdirLabel: 'Pasta',
  securityHeading: 'Segurança',
  scan: { passed: 'Verificação aprovada', warnings: 'A verificação encontrou avisos', failed: 'A verificação falhou' },
  requirementsLabel: 'Requer',
  requiresHermes: range => `Hermes ${range}`,
  envVar: name => `Variável de ambiente ${name}`,
  credentialsHeading: 'Credenciais',
  phase: {
    downloading: 'Baixando…',
    python_packages: 'Instalando pacotes Python…',
    loading_tools: 'Carregando suas ferramentas…'
  },
  serverNotConnected: (server, reason) => `Servidor MCP ${server} não conectado${reason ? `: ${reason}` : ''}`,
  notEnabled: 'Instalado, mas não ativado',
  missingEnv: names => `Defina ${names} para concluir a configuração`,
  alreadyInstalled: 'Já instalado; mantido como está'
}
