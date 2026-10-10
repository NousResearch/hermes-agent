import type { ScreenshotTranslations } from './types_screenshot'

export const deScreenshot: ScreenshotTranslations = {
  leftCommand: 'Links ⌘',
  rightCommand: 'Rechts ⌘',
  enabledTitle: 'Screenshot-Kurzbefehl',
  enabledDesc:
    'Drücken Sie in einer beliebigen App die linke und rechte Command-Taste (⌘) gleichzeitig, um deren vorderstes Fenster aufzunehmen und an Ihren aktuellen Hermes-Entwurf anzuhängen. Es wird nie automatisch gesendet. Standardmäßig aus; gilt nur für diesen Mac. Fensterinhalte können vertraulich sein – prüfen Sie den Anhang vor dem Senden.',
  statusTitle: 'Status des Screenshot-Kurzbefehls',
  checking: 'Screenshot-Kurzbefehl wird geprüft…',
  disabled: 'Der Screenshot-Kurzbefehl ist aus.',
  starting: 'Der Kurzbefehl-Listener wird gestartet. Er ist noch nicht bereit.',
  ready: 'Der Kurzbefehl ist bereit. Screenshots werden an Ihren aktuellen Entwurf angehängt, ohne gesendet zu werden.',
  inputPermission:
    'Mit der Berechtigung „Eingabeüberwachung“ kann Hermes die linke und rechte Command-Taste (⌘) erkennen, während eine andere App aktiv ist. Erlauben Sie Hermes unter Systemeinstellungen → Datenschutz & Sicherheit → Eingabeüberwachung, kehren Sie dann hierher zurück und versuchen Sie es erneut.',
  screenPermission:
    'Mit der Berechtigung „Bildschirmaufnahme“ kann Hermes das vorderste App-Fenster aufnehmen, wenn Sie diesen Kurzbefehl verwenden. Erlauben Sie Hermes unter Systemeinstellungen → Datenschutz & Sicherheit → Bildschirmaufnahme, kehren Sie dann hierher zurück und versuchen Sie es erneut. Starten Sie Hermes neu, wenn macOS dazu auffordert.',
  openSettings: 'Systemeinstellungen öffnen',
  retry: 'Erneut versuchen',
  unavailable: 'Der Screenshot-Kurzbefehl ist nicht verfügbar. Versuchen Sie es erneut oder schalten Sie ihn aus.',
  errorTitle: 'Fehler beim Screenshot-Kurzbefehl',
  loadFailed:
    'Der Status des Kurzbefehls konnte nicht gelesen werden. Versuchen Sie es erneut, um die aktuelle Einstellung zu prüfen.',
  saveFailed:
    'Die Änderung am Kurzbefehl konnte nicht bestätigt werden. Versuchen Sie es erneut, um die aktuelle Einstellung zu prüfen.',
  permissionFailed:
    'Die Systemeinstellungen konnten nicht geöffnet werden. Öffnen Sie „Datenschutz & Sicherheit“ manuell und versuchen Sie es erneut.',
  captureFailed: 'Das vorderste Fenster konnte nicht aufgenommen werden. Es wurde nichts angehängt oder gesendet.',
  contextChanged:
    'Der aktuelle Entwurf hat sich während der Aufnahme geändert. Der Screenshot wurde weder angehängt noch gesendet.'
}
