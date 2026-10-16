// Typed /undo /retry /steer /queue lines (en_session_commands.ts is the source).
export const deSessionCommands = {
  undoUsage: 'Verwendung: /undo [N]',
  undoNothing: 'nichts zum Rückgängigmachen',
  undidOne: (count: string) => `${count} Nachricht rückgängig gemacht — die entfernte Nachricht steht im Eingabefeld`,
  undidOther: (count: string) =>
    `${count} Nachrichten rückgängig gemacht — die entfernte Nachricht steht im Eingabefeld`,
  retryNothing: 'nichts zum Wiederholen',
  retryUnsupported: '/retry: Medien oder unbekannte Inhaltsteile können nicht wiederholt werden',
  busy: (command: string) =>
    `Sitzung beschäftigt — Hermes antwortet noch. Beende zuerst die aktuelle Antwort (Stopp-Taste oder Strg+C im Terminal), dann führe /${command} aus.`,
  steerUsage: 'Verwendung: /steer <Prompt>',
  steerQueued: (preview: string) => `Steuerung eingereiht — kommt nach dem nächsten Tool-Aufruf an: „${preview}“`,
  steerNoActiveTurn: (preview: string) => `kein aktiver Turn — für den nächsten eingereiht: „${preview}“`,
  steerRejected: 'Steuerung abgelehnt — kein aktiver Turn, für den nächsten Turn eingereiht',
  queued: (preview: string) => `eingereiht: „${preview}“`,
  queueFailed: 'Die Nachricht konnte nicht eingereiht werden',
  countOne: (count: string) => `${count} eingereihte Nachricht`,
  countOther: (count: string) => `${count} eingereihte Nachrichten`,
  queueEditKeptTitle: 'Eingereihte Nachricht nicht geändert',
  queueEditKept:
    'Sie wurde gestartet oder abgelehnt, bevor deine Änderung gespeichert war. Dein bearbeiteter Text steht im Eingabefeld.',
  queueEditUnconfirmedTitle: 'Änderung nicht bestätigt',
  queueEditUnconfirmed:
    'Hermes konnte deine Änderung nicht bestätigen. Speichere erneut, um dieselbe Änderung zu wiederholen; sie wird nie doppelt eingereiht.'
}
