import type { MemoryLayersTranslations } from './types_memory_layers'

// Configurações > Memória e contexto e Central de comandos > Manutenção.
export const ptBrMemoryLayers: MemoryLayersTranslations = {
  activeProvider: 'Provedor de memória ativo',
  selectedProvider: 'Provedor de memória selecionado',
  selectedDesc: current => `Ainda não está ativo: as sessões continuam usando ${current} até você salvar esta escolha.`,
  builtinMemory: 'somente memória integrada',
  noProvider: 'Nenhum — somente memória integrada',
  noProviderDesc:
    'O Hermes usa apenas a camada integrada abaixo. Escolha um provedor externo em Configurações › Memória e contexto para adicionar um.',
  providerDesc: name => `${name} adiciona sua própria recuperação de memórias a cada sessão, além da camada integrada.`,
  providerNotReadyDesc: name =>
    `${name} é o provedor salvo, mas não recupera memórias até estar pronto. A camada integrada funciona de qualquer forma.`,
  providerUnknownDesc: name => `${name} é o provedor salvo; o Hermes não conseguiu verificar se ele está pronto.`,
  statusUnavailable: 'Status da memória indisponível',
  statusUnavailableDesc: 'Não foi possível verificar qual provedor de memória está ativo ou se ele está pronto.',
  retry: 'Tentar novamente',
  status: {
    ready: 'pronto',
    needs_config: 'precisa de configuração',
    unavailable: 'indisponível',
    missing: 'não instalado'
  },
  builtinLayer: 'Camada integrada',
  onByDefault: 'Ativada por padrão',
  on: 'Ativada',
  off: 'Desativada',
  onlyFile: file => `Somente ${file}`,
  builtinDesc:
    'MEMORY.md e USER.md são a memória do próprio Hermes, separada de qualquer provedor externo. Ambos são carregados nas sessões por padrão; as opções de memória e perfil do usuário em Configurações › Memória e contexto desativam cada um deles. Um provedor complementa essa camada; ele nunca a substitui.'
}
