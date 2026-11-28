import type { UpdateChannelCopy } from './types_update_channel'

export const ptBrUpdateChannel: UpdateChannelCopy = {
  title: 'Canal de atualização',
  description:
    'O canal estável instala apenas versões publicadas. A opção Cada commit acompanha a branch main conforme as alterações chegam.',
  stable: 'Versões estáveis',
  main: 'Cada commit',
  stableConfirmTitle: 'Mudar para versões estáveis?',
  stableConfirmBody:
    'O Hermes passará para a versão publicada mais recente, que pode ser mais antiga que o código em execução agora. Dados gravados por código mais novo podem não ser lidos corretamente, então faça um backup primeiro.',
  stableConfirm: 'Mudar para o estável',
  failed: 'Não foi possível alterar o canal de atualização',
  detailsLabel: 'Canal',
  branch: (b: string) => `Branch: ${b}`,
  change: 'Alterar',
  latestRelease: (v: string) => `Você está na versão estável mais recente (${v}).`,
  rollbackTitle: (v: string) => `Voltar para a versão estável ${v}`,
  rollbackBody: (v: string) =>
    `Esta compilação é mais recente que a versão estável ${v}. Instalá-la faz o Hermes voltar para essa versão.`,
  heldBody: (v: string) =>
    `Esta compilação é mais recente que a versão estável ${v}. Suas sessões já usam uma versão do gateway que ${v} não consegue abrir, então o Hermes permanecerá nesta compilação até uma nova versão estável ser publicada.`,
  rollbackAction: (v: string) => `Reverter para ${v}`,
  allChanges: 'Ver todas as alterações'
}
