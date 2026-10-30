import type { Translations } from './types'

type ThreadCopy = Translations['assistant']['thread']

// The error card's per-code title and body (lib/error-surface-copy.ts), keyed by error_surface.code.
export const deErrorCodes: Partial<ThreadCopy['errorCodes']> = {
  cron_session_read_only: {
    title: 'Dieser Chat gehört zu einer geplanten Aufgabe',
    body: 'Er führt nur diese Aufgabe aus, daher wurde Ihre Nachricht nicht ausgeführt. Starten Sie einen neuen Chat, um sie zu senden.'
  },
  session_read_only: {
    title: 'Dieser Chat ist schreibgeschützt',
    body: 'Er ist gespeicherter Verlauf, den Hermes anzeigen, aber nicht fortsetzen kann, daher wurde Ihre Nachricht nicht ausgeführt. Starten Sie einen neuen Chat, um sie zu senden.'
  },
  workspace_unavailable: {
    title: 'Der Ordner dieses Chats fehlt',
    body: 'Der Ordner, in dem dieser Chat arbeitet, wurde verschoben oder gelöscht, daher wurde Ihre Nachricht nicht ausgeführt. Legen Sie den Ordner neu an oder geben Sie dem Chat mit „In Projekt verschieben“ einen anderen, und versuchen Sie es dann erneut.'
  },
  launch_credentials_unavailable: {
    title: 'Der API-Schlüssel dieses Chats hat sich geändert',
    body: 'Dieser Chat wurde mit einem API-Schlüssel gestartet, der inzwischen geändert oder entfernt wurde, daher wurde Ihre Nachricht nicht gesendet. Ein offener Chat wechselt nie von selbst zu einem anderen Schlüssel. Wählen Sie „Neue Session starten“, um Ihren aktuellen Schlüssel zu verwenden.'
  }
}
