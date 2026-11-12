import type { UninstallSectionTranslations } from './types_uninstall_section'

// Configurações > Sobre > Desinstalar, reunido em pt-br.ts como `settings.uninstallSection`.
export const ptBrUninstallSection: UninstallSectionTranslations = {
  dangerZone: 'Zona de perigo',
  checkingInstalled: 'Verificando o que está instalado…',
  uninstallHermes: 'Desinstalar o Hermes',
  managedBody: 'Esta instalação é gerenciada pelo sistema, então o Hermes não pode remover a si mesmo.',
  dataKept: path => `Suas configurações, conversas e segredos ficam em ${path}. Remover o aplicativo não os exclui.`,
  openAppsSettings: 'Abrir configurações de Aplicativos',
  chooseHowMuch:
    'Escolha o que remover. O aplicativo será fechado para concluir; reabra o instalador a qualquer momento para voltar.',
  confirmUninstall: 'Confirmar desinstalação',
  confirmBody: what => `Isso remove ${what}. Não é possível desfazer.`,
  appLabel: 'Aplicativo:',
  couldNotStart: 'Não foi possível iniciar a desinstalação.',
  uninstalling: 'Desinstalando…',
  yesUninstall: 'Sim, desinstalar',
  options: {
    gui: {
      title: 'Desinstalar apenas a interface de conversa',
      description: 'Remova este aplicativo. O agente Hermes, suas configurações e conversas permanecem.',
      consequence: 'a interface de conversa para desktop (este aplicativo e seus dados)'
    },
    lite: {
      title: 'Desinstalar interface e agente, manter meus dados',
      description:
        'Remova o aplicativo e o agente Hermes, mas mantenha configurações, conversas e segredos para uma futura reinstalação.',
      consequence: 'a interface de conversa e o agente Hermes (configurações, conversas e segredos são mantidos)'
    },
    full: {
      title: 'Desinstalar tudo',
      description:
        'Remova o aplicativo, o agente e todos os dados do usuário — configurações, conversas, tarefas agendadas, segredos e logs.',
      consequence:
        'TUDO — a interface de conversa, o agente Hermes e todas as suas configurações, conversas, segredos e logs'
    }
  }
}
