import type { DelegationRoutingTranslations } from './types_delegation_routing'

export const ptBrDelegationRouting: DelegationRoutingTranslations = {
  title: 'Roteamento',
  settingsTitle: 'Padrões dos subagentes',
  intro: 'Como cada subagente recebe seu modelo.',
  modes: {
    default: { label: 'Padrão', description: 'Todos os subagentes usam o modelo de subagente abaixo.' },
    model_map: {
      label: 'Rotas de modelos',
      description: 'Defina alguns modelos. O agente escolhe um para cada subagente.'
    },
    profile_routes: {
      label: 'Rotas de perfis',
      description: 'Os subagentes usam um dos seus perfis, com seu próprio modelo, habilidades e chaves.'
    }
  },
  inUse: 'Em uso',
  plugin: 'Plugin',
  pluginMissing: 'Rotas de modelos e Rotas de perfis são fornecidas pelo plugin Advanced Delegation.',
  pluginDisabled: 'O Advanced Delegation está instalado, mas desativado para este perfil.',
  pluginNotLoaded:
    'O Advanced Delegation está ativado, mas o Hermes ainda não o carregou. Reinicie o Hermes para concluir.',
  installPlugin: 'Instalar plugin',
  installUnavailable: 'O Advanced Delegation ainda não está no catálogo de plugins. Instale-o em Recursos › Plugins.',
  enablePlugin: 'Ativar',
  enableFailed: 'Não foi possível ativar o Advanced Delegation',
  modeFailed: 'Não foi possível alterar o roteamento dos subagentes',
  loadFailed: 'Não foi possível carregar o roteamento dos subagentes',
  blockedPill: 'Não está em execução',
  blockedDetail:
    'Este modo está salvo, então o Hermes recusa todos os subagentes em vez de usar o modo Padrão sem avisar.',
  blockedMissing: 'Os subagentes não iniciarão até que o Advanced Delegation seja instalado.',
  blockedDisabled: 'Os subagentes não iniciarão até que o Advanced Delegation seja ativado para este perfil.',
  blockedNotLoaded:
    'Os subagentes não iniciarão até que o Hermes carregue o Advanced Delegation. Reinicie o Hermes para concluir.',
  blockedUnavailable: label =>
    `Os subagentes não iniciarão até que ${label} possa ser executado aqui. Seu plugin informa que isso não é possível.`,
  blockedUnregistered: name =>
    `Os subagentes não iniciarão até que um plugin que forneça “${name}” seja instalado e ativado.`,
  thirdPartyDescription: 'Fornecido por um plugin. As configurações ficam nesse plugin.',

  modelRoutesTitle: 'Rotas de modelos',
  modelRoutesEmpty: 'Nenhuma rota ainda. Até você adicionar uma, os subagentes usam o modelo do agente principal.',
  addRoute: 'Adicionar rota',
  editRoute: 'Editar',
  makeDefault: 'Definir como padrão',
  deleteRoute: 'Excluir',
  defaultBadge: 'Padrão',
  routeActions: name => `Ações para ${name}`,
  deleteConfirmTitle: name => `Excluir a rota “${name}”?`,
  deleteConfirmBody: 'Os subagentes não poderão mais escolhê-la.',
  deleteConfirmNextDefault: name => `Esta é a rota padrão, então “${name}” passará a ser o padrão.`,
  deleteConfirmAction: 'Excluir rota',
  saveFailed: 'Não foi possível salvar as rotas',
  baseUrlRoute: url => `endpoint ${url}`,

  dialogNew: 'Nova rota de modelo',
  dialogEdit: 'Editar rota de modelo',
  dialogDesc: 'O agente lê o nome e a descrição para decidir qual rota um subagente usa.',
  nameLabel: 'Nome',
  namePlaceholder: 'ex.: rapido',
  nameHint: 'Curto, sem espaços. O agente escolhe as rotas por este nome.',
  descriptionLabel: 'Quando usar',
  descriptionPlaceholder: 'ex.: Barato e rápido: buscas, resumos, edições simples.',
  descriptionHint: 'Uma linha que o agente lê antes de escolher.',
  providerLabel: 'Provedor',
  providerPlaceholder: 'Escolha um provedor',
  modelLabel: 'Modelo',
  effortLabel: 'Esforço de raciocínio',
  effortInherit: 'Padrão dos subagentes',
  tierLabel: 'Velocidade',
  tierNormal: 'Normal',
  tierFast: 'Rápida',
  tierUltrafast: 'Ultrarrápida',
  tierHint: 'Apenas nos modelos que oferecem esta opção.',
  daybreakLabel: 'Daybreak',
  daybreakOff: 'Desativado',
  daybreakOn: 'Ativado',
  daybreakHint: 'Acesso confiável da OpenAI com restrições de segurança reduzidas, apenas para esta rota.',
  daybreakUnlisted:
    'Este modelo não está listado para Daybreak na sua conta. Desative a opção ou escolha outro modelo.',
  daybreakNeedsPlugin:
    'Seu plugin Advanced Delegation ainda não executa rotas Daybreak. Atualize-o ou desative esta opção.',
  optional: 'opcional',
  errors: { required: 'Obrigatório', spaces: 'Sem espaços', duplicate: 'Outra rota já usa este nome' },
  saveRoute: 'Salvar rota',
  createRoute: 'Adicionar rota',

  profileRoutesTitle: 'Perfis que os subagentes podem usar',
  profileRoutesHint: 'Cada perfil selecionado traz seu próprio modelo, habilidades, persona e chaves.',
  profileDescriptionPlaceholder: 'Quando usar (opcional)',
  profileDefault: 'Padrão',
  profileEmpty: 'Nenhum outro perfil ainda. Crie um na barra de perfis.',
  profileMissing: name => `${name} (o perfil não existe mais)`,

  inactiveTitle: 'Não usado neste modo',
  inactiveModelMap: route =>
    `As rotas de modelos escolhem o modelo de cada subagente. Os subagentes que não escolhem uma rota usam “${route}”.`,
  inactiveProfiles: profile =>
    `As rotas de perfis escolhem o perfil de cada subagente. Os subagentes que não escolhem um perfil usam “${profile}”.`,
  inactiveModelMapNoDefault:
    'As rotas de modelos escolhem o modelo de cada subagente. Os subagentes que não escolhem uma rota usam o modelo do agente principal.',
  inactiveProfilesNoDefault:
    'As rotas de perfis escolhem o perfil de cada subagente. Os subagentes que não escolhem um perfil usam o modelo do agente principal.',
  inactivePlugin: label => `${label} decide o modelo de cada subagente, então esta configuração não se aplica.`,
  useDefault: 'Mudar para Padrão'
}
