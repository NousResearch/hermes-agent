import type { ExternalOpenFailedCopy } from './types_external_open'

export const ptBrExternalOpenFailed: ExternalOpenFailedCopy = {
  title: 'Não foi possível abrir este link',
  message: 'Nenhum navegador está registrado para abrir este endereço. Copie o link e abra-o manualmente.',
  copyUrl: 'Copiar link',
  close: 'Fechar',
  missing: {
    title: 'Arquivo não encontrado',
    message: 'Este arquivo não existe — ele pode ter sido excluído ou movido, ou estar em outro computador.'
  }
}
