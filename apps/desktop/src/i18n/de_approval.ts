import type { ApprovalCopy } from './types_approval'

export const deApproval: Partial<ApprovalCopy> = {
  gatewayDisconnected: 'Hermes-Gateway ist nicht verbunden',
  sendFailed: 'Genehmigungsantwort konnte nicht gesendet werden',
  reconnect: 'Neu verbinden',
  timedOutSystemLine:
    'Die Freigabe ist abgelaufen — der Befehl wurde nicht ausgeführt. Bitten Sie Hermes, es erneut zu versuchen, oder erhöhen Sie das Limit unter Einstellungen → Sicherheit → Freigabe-Timeout.',
  openSafetySettings: 'Sicherheitseinstellungen öffnen',
  run: 'Ausführen',
  command: 'Befehl',
  moreOptions: 'Weitere Genehmigungsoptionen',
  allowSession: 'Diese Session erlauben',
  alwaysAllowMenu: 'Immer erlauben…',
  jumpToApproval: 'Genehmigung erforderlich',
  reject: 'Ablehnen',
  alwaysTitle: 'Diesen Befehl immer erlauben?',
  alwaysDescription: pattern =>
    `Dies fügt das Muster „${pattern}“ Ihrer dauerhaften Zulassungsliste hinzu (~/.hermes/config.yaml). Hermes fragt bei solchen Befehlen nicht mehr nach – weder in dieser noch in zukünftigen Sessions.`,
  alwaysAllow: 'Immer erlauben',
  deny: 'Ablehnen',
  allow: 'Erlauben',
  allowHint: 'Bis die Sandbox ausgeschaltet wird',
  denyReason: 'Grund für die Ablehnung',
  denyReasonPlaceholder: 'Hermes den Grund nennen (optional)',
  access: {
    title: 'Ordnerzugriff',
    readOnly: 'Nur lesen',
    readWrite: 'Lesen und schreiben',
    alwaysTitle: 'Diesen Ordner immer erlauben?',
    alwaysDescription:
      'Der Ordner wird der Sandbox dieses Profils hinzugefügt. Hermes fragt in keinem Chat mehr danach.'
  },
  network: {
    title: 'Website-Zugriff',
    alwaysTitle: 'Diese Website immer erlauben?',
    alwaysDescription:
      'Die Website wird den erlaubten Websites dieses Profils hinzugefügt. Hermes fragt in keinem Chat mehr danach.'
  },
  schedule: {
    title: 'Diesen Job planen?',
    allowSession: 'Für diesen Chat erlauben',
    alwaysAllowMenu: 'Neue Jobs immer erlauben…',
    alwaysTitle: 'Neue geplante Jobs immer erlauben?',
    alwaysDescription:
      'Hermes erstellt geplante Jobs, ohne vorher zu fragen. Du kannst die Nachfrage wieder einschalten, indem du cron_create aus command_allowlist in config.yaml entfernst.',
    timedOutSystemLine:
      'Zeit für die Freigabe abgelaufen — der Job wurde nicht erstellt. Bitte Hermes, es erneut zu versuchen, oder erhöhe das Limit unter Einstellungen → Sicherheit → Freigabe-Zeitlimit.',
    editTitle: 'Diesen Job ändern?',
    editTimedOutSystemLine:
      'Zeit für die Freigabe abgelaufen — der Job wurde nicht geändert. Bitte Hermes, es erneut zu versuchen, oder erhöhe das Limit unter Einstellungen → Sicherheit → Freigabe-Zeitlimit.'
  }
}
