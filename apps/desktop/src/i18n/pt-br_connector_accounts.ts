import type { Translations } from './types'

export const ptBrConnectorAccounts: Translations['connectorsPage']['accounts'] = {
  heading: 'Contas',
  count: (count: number) => `${count} conta${count === 1 ? '' : 's'}`,
  addAnother: 'Adicionar outra conta',
  aliasLabel: 'Nome da conta',
  aliasHint: 'Até 32 letras minúsculas, números e hífens. Por exemplo, trabalho ou casa.',
  aliasInvalid: 'Use letras minúsculas, números e hífens, começando com uma letra ou número.',
  aliasTaken: 'Esse nome já está em uso.',
  add: 'Conectar',
  rename: 'Renomear',
  renameLabel: (name: string) => `Novo nome para ${name}`,
  renameGone: 'Essa conta não está mais disponível. Ela foi removida ou substituída por uma reconexão.',
  remove: 'Remover',
  removeTitle: (app: string, name: string) => `Remover a conta ${name} do ${app}?`,
  removeBody: 'O Hermes deixa de agir por meio desta conta. Suas outras contas continuam conectadas.',
  retiredCount: (count: number) => `${count} substituída${count === 1 ? '' : 's'}`,
  status: {
    expired: 'Acesso expirado',
    failed: 'Não foi possível conectar',
    inactive: 'Inativa',
    pending: 'Aguardando login',
    retired: 'Substituída por uma reconexão',
    revoked: 'Acesso revogado'
  }
}
