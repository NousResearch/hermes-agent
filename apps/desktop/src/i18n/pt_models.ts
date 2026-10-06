import type { TranslationOverrides } from './define-locale'

export const ptModels = {
  modelAssignment: {
    saveFailed: 'O Hermes não salvou essa alteração de modelo.',
    confirmTitle: 'Aviso sobre a seleção de modelo',
    confirmDetail: 'Confirme apenas se você aceita essa troca.',
    confirmAction: 'Confirmar',
    declined: 'Alteração de modelo cancelada: você recusou o aviso sobre a faixa de treinamento com dados.'
  },
  modelPicker: {
    title: 'Mudar modelo',
    current: 'atual:',
    unknown: '(desconhecido)',
    search: 'Filtrar provedores e modelos…',
    noModels: 'Nenhum modelo encontrado.',
    addProvider: 'Adicionar provedor',
    loadFailed: 'Não foi possível carregar os modelos',
    loadingIntoMemory: 'Carregando na memória',
    downloading: 'Baixando',
    localDownloadsHeading: 'Local',
    noAuthenticatedProviders: 'Nenhum provedor autenticado.',
    pro: 'Pro',
    proNeedsSubscription: 'Os modelos Pro exigem uma assinatura paga da Nous.',
    free: 'Gratuito',
    freeTier: 'Plano gratuito',
    priceTitle: 'Preço de entrada / saída por milhão de tokens',
    wasPrice: 'era',
    customModel: 'Modelo personalizado',
    addCustomModelAction: 'Adicionar modelo personalizado…',
    customModelPlaceholder: 'Digite um ID de modelo, ex.: openai/gpt-5'
  },
  modelVisibility: {
    title: 'Modelos',
    search: 'Buscar modelos',
    noAuthenticatedProviders: 'Nenhum provedor autenticado.',
    addProvider: 'Adicionar provedor…',
    addCustomModel: 'Adicionar modelo personalizado',
    removeCustomModel: 'Remover modelo personalizado',
    resetToDefaults: 'Restaurar padrões',
    resetConfirm: 'Restaurar a visibilidade dos modelos ao padrão?',
    resetDescription:
      'As suas escolhas de modelos exibidos e ocultos são apagadas e a lista padrão de cada provedor volta. Os modelos personalizados que você adicionou são mantidos e exibidos.',
    resetAction: 'Restaurar'
  },
  interfaceMode: {
    title: 'Modo da interface',
    hint: 'Muda o que é exibido, não o que o Hermes pode fazer.',
    sessionNote:
      'Definido pelo modo Simples. Uma alteração aqui vale só para esta sessão; mude para Avançado para torná-la permanente.',
    simple: {
      label: 'Simples',
      description: 'Para conversar com o Hermes. Barra lateral e chat, sem painéis de terminal, arquivos ou diff.'
    },
    advanced: {
      label: 'Avançado',
      description:
        'Para desenvolvedores. Terminal, arquivos, diffs, barra de status e layouts, do jeito que você configurar.'
    }
  }
} satisfies Pick<TranslationOverrides, 'modelAssignment' | 'modelPicker' | 'modelVisibility' | 'interfaceMode'>
