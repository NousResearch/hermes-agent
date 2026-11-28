import type { DesktopCommandNotices } from './types_desktop_commands'

export const ptBrDesktopCommands: DesktopCommandNotices = {
  error: message => `erro: ${message}`,
  rollbackUsage: 'uso: /rollback [list] | diff <n|hash> | <n|hash> [file] [--all]',
  checkpointsDisabled:
    'Os pontos de restauração não estão ativados para esta sessão (defina checkpoints.enabled em config.yaml).',
  checkpointsEmpty: 'Ainda não há pontos de restauração para esta pasta de trabalho.',
  checkpointsHeader: 'Pontos de restauração (mais recentes primeiro):',
  checkpointsHint:
    'Restaure com /rollback <n> [file] (adicione --all para também sobrescrever suas próprias edições); visualize antes com /rollback diff <n>.',
  noDiff: 'Nenhuma alteração desde esse ponto de restauração.',
  rollbackFailed: error => `Falha ao restaurar: ${error}`,
  unknownError: 'erro desconhecido',
  restored: hash => (hash ? `Ponto de restauração ${hash} restaurado.` : 'Ponto de restauração restaurado.'),
  rewound: count => `${count} ${count === '1' ? 'mensagem desfeita' : 'mensagens desfeitas'} nesta conversa.`,
  reopenForRewind: 'Reabra esta conversa para ver o histórico após desfazer as mensagens.',
  agentsEmpty: 'Nenhum subagente ou processo em segundo plano ativo nesta conversa.',
  subagentsHeader: count => `Subagentes (${count}):`,
  subagentFallback: 'subagente',
  statusRunning: 'em execução',
  toolCalls: count => `${count} ${count === '1' ? 'chamada de ferramenta' : 'chamadas de ferramentas'}`,
  processesHeader: count => `Processos em segundo plano (${count}):`,
  configHeader: 'Configuração (exibição, aprovações, colagem e voz; as demais configurações não são mostradas):',
  configDefaults: '(padrões)',
  configFooter: 'Altere em Configurações ou edite config.yaml com `hermes config edit` no computador do gateway.',
  historyHeader: count => `Histórico da conversa (${count} ${count === '1' ? 'mensagem' : 'mensagens'}):`,
  historyEmpty: 'Ainda não há mensagens.',
  historyYou: 'Você',
  historyHermes: 'Hermes',
  saveUsage: 'uso: /save [json|md|html] [filename] [redact]',
  saveLegacyRefused:
    'Não foi salvo: este gateway não pode exportar uma cópia com dados sensíveis ocultados, com nome definido ou em formato diferente de JSON. Atualize o gateway ou execute /save sem opções para gravar o JSON bruto no computador do gateway.',
  savedOnHost: path => `Salvo no computador do gateway: ${path}`,
  unknownPath: '(caminho desconhecido)',
  saveUnsupported: 'erro: esta versão do Desktop não pode salvar arquivos',
  saveDialogTitle: 'Salvar conversa',
  savedTo: path => `Conversa salva em ${path}`,
  saveCancelled: 'Não foi salvo (cancelado).',
  exportRemote:
    '/export empacota um perfil do computador do gateway, que não é este computador. Execute `hermes profile export` naquele computador.',
  importRemote:
    '/import instala um perfil no computador do gateway, que não é este computador. Execute `hermes profile import <archive>` naquele computador.',
  timestampsUsage: 'uso: /timestamps [on|off|status]',
  timestampsOn: 'Horários: ativados (nesta janela; defina display.timestamps em config.yaml para manter)',
  timestampsOff: 'Horários: desativados (nesta janela; defina display.timestamps em config.yaml para manter)'
}
