import { ptBrAuxTasks } from './pt-br_aux_tasks'
import { ptBrDelegationRouting } from './pt-br_delegation_routing'
import type { ModelSettingsTranslations } from './types_model_settings'

export const ptBrModelSettings: ModelSettingsTranslations = {
  setupProviderFallback: 'provedor',
  setUpProvider: name => `Configurar ${name}`,
  staleAuxBefore: (count, names) =>
    `${count} tarefa${count === 1 ? '' : 's'} auxiliar${count === 1 ? '' : 'es'} (${names}) ainda ${count === 1 ? 'usa' : 'usam'} `,
  staleAuxAfter: ', e não o seu modelo principal.',
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
  loading: 'Carregando a configuração dos modelos...',
  appliesDesc:
    'Aplica-se a novas sessões. Use o seletor de modelos na caixa de mensagem para trocar o modelo da conversa ativa.',
  provider: 'Provedor',
  model: 'Modelo',
  applying: 'Aplicando...',
  mainAppliedTitle: 'Modelo principal atualizado',
  mainAppliedMessage: model => `Novas sessões usarão ${model}.`,
  defaultsLabel: 'Padrões',
  reasoning: 'Raciocínio',
  reasoningOff: 'Desativado',
  speed: 'Velocidade',
  speedStandard: 'Padrão',
  defaultsFailed: 'Falha ao salvar os padrões do modelo',
  loadFailed: 'Não foi possível carregar os modelos',
  restartRequired:
    'Este backend está executando código antigo após uma atualização. Reinicie-o para carregar o novo código.',
  restartBackend: 'Reiniciar backend',
  restartingBackend: 'Reiniciando o backend...',
  restartFailed: 'Não foi possível reiniciar o backend',
  auxiliaryTitle: 'Modelos auxiliares',
  resetAllToMain: 'Redefinir todos para o principal',
  staleAuxDismiss: 'Não mostrar novamente',
  auxiliaryDesc:
    'As tarefas auxiliares usam o modelo principal por padrão. Atribua um modelo específico a qualquer tarefa para alterar esse comportamento.',
  setToMain: 'Usar o principal',
  change: 'Alterar',
  autoUseMain: 'automático · usar modelo principal',
  inheritMainEffort: 'herdar · esforço do modelo principal',
  inheritsFrom: task => `herda de ${task}`,
  followTask: task => `Seguir ${task}`,
  providerDefault: '(padrão do provedor)',
  fallbackAdd: 'Adicionar alternativa',
  fallbackEmpty: 'Nenhum modelo alternativo — o modelo padrão é usado, a menos que falhe.',
  notInCatalog: 'não está na lista de modelos deste provedor — as chamadas podem usar um modelo de reserva.',
  moaTitle: 'Mistura de agentes',
  moaPreset: 'Predefinição',
  moaDescription:
    'Configure predefinições com nome que aparecem como modelos no provedor Mixture of Agents. O agregador é o modelo executor — ele executa cada etapa do ciclo de ferramentas, e quase todo o custo da execução é cobrado pelo provedor dele. Por padrão, os modelos de referência apenas aconselham uma vez por turno do usuário.',
  moaAggregator: 'Agregador',
  moaAggregatorBilled: 'modelo executor · cobrado pela execução',
  moaReferenceHint: 'aconselha uma vez por turno por padrão',
  tasks: ptBrAuxTasks,
  subagents: {
    inherit: 'Herdar do agente principal',
    modelOnly: 'Modelo personalizado (usar credenciais do agente principal)',
    providerLabel: 'Provedor do subagente',
    modelLabel: 'Modelo do subagente',
    direct: 'Substituição por endpoint direto',
    customProvider: 'Provedor personalizado...',
    customProviderLabel: 'Provedor personalizado do subagente',
    parentModel: 'Usar modelo do agente principal',
    parentModelWarning:
      'O provedor escolhido precisa oferecer suporte ao modelo do agente principal. Uma rota que define apenas o provedor pode falhar se o modelo for incompatível. Se esse provedor definir seu próprio modelo padrão, os subagentes usarão esse modelo.',
    directActive:
      'Um endpoint direto está ativo e tem prioridade sobre o provedor salvo. O endpoint e suas credenciais são mantidos enquanto este modo estiver selecionado.',
    directClear:
      'Aplicar esta rota remove o endpoint direto e sua chave armazenada. As outras configurações de requisição são mantidas.',
    catalogFailed: 'Catálogo de modelos indisponível. Insira um provedor e um modelo manualmente ou tente novamente.',
    draftHint: 'As alterações ficam apenas locais até serem aplicadas.',
    routing: ptBrDelegationRouting
  }
}
