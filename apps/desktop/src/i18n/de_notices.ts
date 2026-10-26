import { deDesktopCommands } from './de_desktop_commands'
import { deSessionCommands } from './de_session_commands'
import type { TranslationOverrides } from './define-locale'

// Shell notices (remote-display toast, butterbar, keep-awake offer), spread into de.ts.
export const deNotices = {
  remoteDisplayBanner: {
    message: reason =>
      `Software-Rendering aktiv — Remote-Display erkannt (${reason}). GPU-Beschleunigung ist deaktiviert, um Flackern zu verhindern.`
  },
  butterbar: {
    goTo: (index, total) => `Hinweis ${index} von ${total} anzeigen`,
    legal: {
      before: 'Die Nutzung von Hermes Agent unterliegt unseren ',
      terms: 'Nutzungsbedingungen',
      between: ' und unserer ',
      privacy: 'Datenschutzerklärung',
      after: '.'
    }
  },
  promptNotices: {
    legacySendUnconfirmed:
      'Dieser Server konnte das frühere Senden dieser Nachricht nicht bestätigen; sie wurde möglicherweise bereits ausgeführt. Prüfe die Unterhaltung, bevor du sie erneut sendest.',
    mcpReloadDeferred:
      'Dieser Chat ist beschäftigt: Die MCP-Änderung gilt, sobald sein Durchlauf endet und MCP erneut geladen wird, oder wenn du den Chat wieder öffnest',
    mcpReloadUnavailable: servers => `Keine Verbindung: ${servers}`,
    compactedTurnUnavailable:
      'Diese Nachricht wurde beim Komprimieren des Chats zusammengefasst und kann nicht mehr wiederhergestellt oder bearbeitet werden.',
    staleTurnUnavailable: 'Diese Nachricht ist nicht mehr im Verlauf dieses Chats.',
    sessionCommands: deSessionCommands,
    desktopCommands: deDesktopCommands
  },
  keepAwakeOffer: {
    title: 'Ihr Computer ist eingeschlafen, während Hermes gearbeitet hat',
    message:
      'Hermes kann den Computer wach halten, solange eine Aufgabe läuft. Der Bildschirm schaltet sich weiterhin aus, und im Akkubetrieb kann der Computer trotzdem einschlafen.',
    action: 'Während der Arbeit wach halten'
  },
  billingBlock: {
    titleNous: 'Keine Nous-Credits mehr',
    titleProvider: provider => `Keine Credits mehr — ${provider}`,
    fallbackMessage: 'Auf Ihrem Konto sind keine Credits mehr übrig. Fügen Sie Credits hinzu, um fortzufahren.',
    openBilling: 'Billing öffnen',
    addCredits: 'Credits hinzufügen',
    dismiss: 'Schließen'
  }
} satisfies Pick<
  TranslationOverrides,
  'remoteDisplayBanner' | 'butterbar' | 'promptNotices' | 'keepAwakeOffer' | 'billingBlock'
>
