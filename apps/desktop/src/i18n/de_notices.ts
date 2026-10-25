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
  },
  sendDiagnostics: {
    title: 'Diagnosedaten an Nous senden',
    privacyNotice:
      'Damit laden Sie ein Debug-Paket in den internen Nous-Speicher hoch (kein öffentliches Paste). Es enthält Systeminfos (Betriebssystem, Versionen, Provider, welche API-Keys konfiguriert sind – niemals die Keys selbst) sowie vollständige Agent-, Gateway- und Desktop-Logs (bis zu 512 KB je Datei), die sehr wahrscheinlich Gesprächsinhalte, Tool-Ausgaben und Dateipfade enthalten. Geheimnisse werden vor dem Upload geschwärzt. Das Paket ist nur für Nous-Mitarbeitende und freigeschaltete Discord-Moderatoren einsehbar und wird nach 14 Tagen automatisch gelöscht.',
    upload: 'Hochladen',
    uploading: 'Wird hochgeladen…',
    cancel: 'Abbrechen',
    close: 'Schließen',
    copyLink: 'Link kopieren',
    uploadIdFallback: id => `Kein Link zurückgegeben — zitiere die Upload-ID ${id} im Support`,
    doneTitle: 'Diagnosedaten gesendet',
    doneDescription:
      'Ihr Paket wurde privat hochgeladen. Teilen Sie den Link unten in Ihrem Support-Thread, damit das Team Ihre Logs sehen kann.',
    failedTitle: 'Hochladen fehlgeschlagen',
    failedHint:
      'Sie können auch `hermes debug share --nous` im Terminal ausführen oder `hermes debug share --local`, um den Bericht ohne Hochladen auszugeben.',
    handoffLead: 'Diskussion hier fortsetzen:',
    links: {
      github: 'GitHub Issues',
      portal: 'Nous-Portal-Support',
      discord: 'Discord'
    }
  }
} satisfies Pick<
  TranslationOverrides,
  'remoteDisplayBanner' | 'butterbar' | 'promptNotices' | 'keepAwakeOffer' | 'billingBlock' | 'sendDiagnostics'
>
