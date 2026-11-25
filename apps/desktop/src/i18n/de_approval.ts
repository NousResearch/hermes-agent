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
    timedOut: folder =>
      `Die Anfrage ist abgelaufen — der Zugriff auf den Ordner ${folder} wurde nicht gewährt. Bitten Sie Hermes, es erneut zu versuchen.`,
    alwaysTitle: 'Diesen Ordner immer erlauben?',
    alwaysDescription:
      'Der Ordner wird der Sandbox dieses Profils hinzugefügt. Hermes fragt in keinem Chat mehr danach.'
  },
  network: {
    title: 'Website-Zugriff',
    timedOut: site =>
      `Die Anfrage ist abgelaufen — der Zugriff auf ${site} wurde nicht gewährt. Bitten Sie Hermes, es erneut zu versuchen.`,
    alwaysTitle: 'Diese Website immer erlauben?',
    alwaysDescription:
      'Die Website wird den erlaubten Websites dieses Profils hinzugefügt. Hermes fragt in keinem Chat mehr danach.'
  }
}
