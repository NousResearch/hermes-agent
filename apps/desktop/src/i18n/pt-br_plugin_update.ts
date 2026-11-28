import type { PluginUpdateTranslations } from './types_plugin_update'

/** Textos para atualizar plugins instalados: autorização da revisão do catálogo e da análise de segurança. */
export const ptBrPluginUpdate: PluginUpdateTranslations = {
  folderAccess: {
    title: 'Acesso a pastas',
    hint: (name: string) => `${name} pode ler e alterar arquivos nestas pastas. A revogação tem efeito imediato.`,
    revoke: 'Revogar',
    revokeLabel: (path: string) => `Revogar acesso a ${path}`,
    revoked: (name: string, path: string) => `${name} não pode mais usar ${path}.`,
    revokeFailed: (name: string) => `Não foi possível revogar o acesso de ${name} às pastas`,
    pickerTitle: (name: string) => `Escolha uma pasta para ${name}`,
    pickerButton: 'Permitir acesso',
    refusedProtected: (name: string) =>
      `${name} não pode usar essa pasta: ela contém configurações, chaves ou dados de plugins do Hermes. Escolha uma pasta que não contenha esses dados.`,
    loadFailed: (name: string) => `Não foi possível carregar as pastas compartilhadas com ${name}.`
  },
  updateToPin: (sha: string) => `Atualizar para ${sha}`,
  updateFailed: (name: string) => `Não foi possível atualizar ${name}`,
  updated: (name: string) =>
    `${name} foi atualizado para a revisão atual fixada no catálogo. Reinicie o gateway para aplicar.`,
  updateConsentTitle: (name: string) => `${name} solicita mais permissões`,
  updateConsentBody: (name: string, sha: string) =>
    `A nova revisão de ${name} fixada no catálogo (${sha}) adiciona interfaces que a versão instalada não tem. Aplique somente se você confiar nelas:`,
  updateConsentConfirm: 'Aplicar atualização',
  updateScanTitle: (name: string) => `A análise de segurança sinalizou ${name}`,
  updateScanBody: (name: string, count: number) =>
    `A atualização de ${name} recebeu um parecer de cautela (${count} ${count === 1 ? 'ocorrência' : 'ocorrências'}). Toda atualização é analisada. Aplique somente se você confiar na fonte:`,
  updateScanMore: (count: number) => `+ ${count} ${count === 1 ? 'outra' : 'outras'}`,
  updateScanConfirm: 'Atualizar mesmo assim',
  updateScanBlocked: (name: string) =>
    `A análise de segurança bloqueou a atualização de ${name} (parecer de perigo). Não é possível ignorar esse bloqueio.`,
  updatedGit: (name: string) => `${name} foi atualizado para o commit mais recente.`
}
