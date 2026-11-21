import type { Translations } from './types'

export const ptBrModelMenu: Translations['shell']['modelMenu'] = {
  pickDeferred: (model: string) => `${model} será usado a partir da sua próxima mensagem`,
  search: 'Buscar modelos',
  noModels: 'Nenhum modelo encontrado',
  editModels: 'Editar modelos…',
  followDefault: 'Usar o padrão das Configurações',
  refreshModels: 'Atualizar modelos',
  favorites: 'Favoritos',
  addFavorite: 'Adicionar aos favoritos',
  removeFavorite: 'Remover dos favoritos',
  favoriteShortcut: '⇧ Clique',
  fast: 'Rápido',
  free: 'grátis',
  cacheRead: 'leitura do cache',
  priceTitle: (input: string, output: string, cache: string) =>
    `Entrada ${input}/Mtok · Saída ${output}/Mtok` + (cache ? ` · Leitura do cache ${cache}/Mtok` : ''),
  localSetup: {
    title: 'Executar localmente · grátis e privado',
    text: (model: string, size: string) => `${model} é compatível com este computador · download de ${size}`,
    action: 'Configurar'
  },
  limited: 'Limite atingido',
  limitedUntil: (time: string) => `Limite atingido até ${time}`,
  limitedTip: (provider: string, time: null | string) =>
    time
      ? `${provider} atingiu seu limite de uso. O limite será renovado às ${time}; você ainda pode escolher um modelo para usar depois.`
      : `${provider} atingiu seu limite de uso. Você ainda pode escolher um modelo para usar quando o limite for renovado.`,
  modelResets: (time: string) => `renova às ${time}`,
  modelLimitedTip: (time: string) =>
    `Este modelo atingiu seu próprio limite, que será renovado às ${time}. Os outros modelos aqui continuam funcionando.`,
  usageLeft: (percent: number, time: null | string) =>
    time ? `${percent}% restantes · renova às ${time}` : `${percent}% restantes`,
  poolAccounts: (count: number) => `${count} ${count === 1 ? 'conta' : 'contas'}`,
  poolLimited: (limited: number, total: number) => `${limited}/${total} contas com limite atingido`,
  poolAccount: (number: number) => `Conta ${number}`,
  poolUnknown: 'Uso indisponível',
  poolUnavailable: 'Entrar novamente',
  usageTip: (provider: string) => `${provider} está perto do limite de uso.`,
  usageWindow: (label: string, percent: number, time: null | string) =>
    time ? `${label}: ${percent}% restantes, renova às ${time}` : `${label}: ${percent}% restantes`
}
