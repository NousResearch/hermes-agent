import type { DesktopCommandNotices } from './types_desktop_commands'

// Righe desktop di /rollback /agents /config /history /save /export /import /timestamps.
export const itDesktopCommands: DesktopCommandNotices = {
  agentsEmpty: 'Nessun subagente attivo o processo in background in questa chat.',
  checkpointsDisabled: 'I checkpoint non sono attivati per questa sessione (imposta checkpoints.enabled in config.yaml).',
  checkpointsEmpty: 'Ancora nessun checkpoint per questa directory di lavoro.',
  checkpointsHeader: 'Checkpoint (prima i più recenti):',
  checkpointsHint:
    'Ripristina con /rollback <n> [file] (aggiungi --all per sovrascrivere anche le tue modifiche); anteprima con /rollback diff <n>.',
  configDefaults: '(predefiniti)',
  configFooter:
    'Cambia le impostazioni in Impostazioni, oppure modifica config.yaml con `hermes config edit` sull’host del gateway.',
  configHeader:
    'Configurazione (visualizzazione, approvazioni, impostazioni incolla e voce; tutto il resto non è mostrato):',
  error: message => `errore: ${message}`,
  exportRemote:
    '/export prepara un profilo dall’host del gateway, che non è questo computer. Esegui `hermes profile export` su quell’host.',
  historyEmpty: 'Ancora nessun messaggio.',
  historyHeader: count => `Cronologia della conversazione (${count} messaggi):`,
  historyHermes: 'Hermes',
  historyYou: 'Tu',
  importRemote:
    '/import installa un profilo sull’host del gateway, che non è questo computer. Esegui `hermes profile import <archivio>` su quell’host.',
  noDiff: 'Nessuna modifica da quel checkpoint.',
  processesHeader: count => `Processi in background (${count}):`,
  reopenForRewind: 'Riapri questa chat per vedere la conversazione riavvolta.',
  restored: hash => (hash ? `Checkpoint ${hash} ripristinato.` : 'Checkpoint ripristinato.'),
  rewound: count => `${count} messaggi riavvolti da questa conversazione.`,
  rollbackFailed: error => `Rollback non riuscito: ${error}`,
  rollbackUsage: 'uso: /rollback [list] | diff <n|hash> | <n|hash> [file] [--all]',
  saveCancelled: 'Non salvato (annullato).',
  saveDialogTitle: 'Salva conversazione',
  saveLegacyRefused:
    'Non salvato: questo gateway non può esportare una copia oscurata, con nome o non JSON. Aggiorna il gateway, oppure esegui /save senza opzioni per scrivere il JSON grezzo sull’host del gateway.',
  saveUnsupported: 'errore: questa build del desktop non può salvare file',
  saveUsage: 'uso: /save [json|md|html] [nomefile] [redact]',
  savedOnHost: path => `Salvato sull’host del gateway: ${path}`,
  savedTo: path => `Conversazione salvata in ${path}`,
  statusRunning: 'in esecuzione',
  subagentFallback: 'subagente',
  subagentsHeader: count => `Subagenti (${count}):`,
  timestampsOff: 'Timestamp: disattivati (questa finestra; imposta display.timestamps in config.yaml per mantenerlo)',
  timestampsOn: 'Timestamp: attivati (questa finestra; imposta display.timestamps in config.yaml per mantenerlo)',
  timestampsUsage: 'uso: /timestamps [on|off|status]',
  toolCalls: count => `${count} chiamate a strumenti`,
  unknownError: 'errore sconosciuto',
  unknownPath: '(percorso sconosciuto)'
}
