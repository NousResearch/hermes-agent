import type { DesktopCommandNotices } from './types_desktop_commands'

// Svensk text för skrivbordets /rollback, /agents, /config, /history, /save,
// /export, /import och /timestamps. Kommandon och konfigurationsnycklar bevaras.
export const svDesktopCommands: DesktopCommandNotices = {
  error: message => `fel: ${message}`,
  rollbackUsage: 'användning: /rollback [list] | diff <n|hash> | <n|hash> [fil] [--all]',
  checkpointsDisabled: 'Återställningspunkter är inte aktiverade för den här sessionen (ange checkpoints.enabled i config.yaml).',
  checkpointsEmpty: 'Det finns ännu inga återställningspunkter för den här arbetskatalogen.',
  checkpointsHeader: 'Återställningspunkter (nyast först):',
  checkpointsHint:
    'Återställ med /rollback <n> [fil] (lägg till --all för att även skriva över dina egna ändringar). Förhandsgranska med /rollback diff <n>.',
  noDiff: 'Inga ändringar sedan den återställningspunkten.',
  rollbackFailed: error => `Återställningen misslyckades: ${error}`,
  unknownError: 'okänt fel',
  restored: hash => (hash ? `Återställde återställningspunkt ${hash}.` : 'Återställde återställningspunkt.'),
  rewound: count =>
    `Spolade tillbaka ${count} meddelande${count === '1' ? '' : 'n'} i den här konversationen.`,
  reopenForRewind: 'Öppna chatten igen för att se konversationen som spolats tillbaka.',
  agentsEmpty: 'Det finns inga aktiva delagenter eller bakgrundsprocesser i den här chatten.',
  subagentsHeader: count => `Delagenter (${count}):`,
  subagentFallback: 'delagent',
  statusRunning: 'körs',
  toolCalls: count => `${count} verktygsanrop`,
  processesHeader: count => `Bakgrundsprocesser (${count}):`,
  configHeader: 'Konfiguration (visning, godkännanden, inklistring och röstinställningar; övriga inställningar visas inte):',
  configDefaults: '(standardvärden)',
  configFooter: 'Ändra inställningar i Inställningar eller redigera config.yaml med `hermes config edit` på gatewayvärden.',
  historyHeader: count => `Konversationshistorik (${count} meddelanden):`,
  historyEmpty: 'Inga meddelanden ännu.',
  historyYou: 'Du',
  historyHermes: 'Hermes',
  saveUsage: 'användning: /save [json|md|html] [filnamn] [redact]',
  saveLegacyRefused:
    'Sparades inte: den här gatewayen kan inte exportera en maskerad, namngiven kopia eller en kopia i annat format än JSON. Uppdatera gatewayen eller kör /save utan alternativ för att skriva rå JSON på gatewayvärden.',
  savedOnHost: path => `Sparad på gatewayvärden: ${path}`,
  unknownPath: '(okänd sökväg)',
  saveUnsupported: 'fel: det här skrivbordsbygget kan inte spara filer',
  saveDialogTitle: 'Spara konversation',
  savedTo: path => `Sparade konversationen till ${path}`,
  saveCancelled: 'Sparades inte (avbröts).',
  exportRemote:
    '/export paketerar en profil från gatewayvärden, som inte är den här datorn. Kör `hermes profile export` på den datorn.',
  importRemote:
    '/import installerar en profil på gatewayvärden, som inte är den här datorn. Kör `hermes profile import <arkiv>` på den datorn.',
  timestampsUsage: 'användning: /timestamps [on|off|status]',
  timestampsOn: 'Tidsstämplar: på (i det här fönstret; ange display.timestamps i config.yaml för att behålla inställningen)',
  timestampsOff: 'Tidsstämplar: av (i det här fönstret; ange display.timestamps i config.yaml för att behålla inställningen)'
}
