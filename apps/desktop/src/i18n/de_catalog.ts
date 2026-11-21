import type { TranslationOverrides } from './define-locale'

// Provider-source copy for Capabilities > Skills, wired into de.ts as `catalog`.
export const deCatalog = {
  provider: {
    review: (action: string) => `${action} prüfen`,
    confirm: (action: string) => `${action} bestätigen`,
    actions: { install: 'Installation', update: 'Aktualisierung', uninstall: 'Deinstallation' },
    managed: 'Von dieser Quelle verwaltet. Änderungen hier prüfen, statt lokale Dateien zu bearbeiten oder zu löschen.',
    loading: 'Quelle wird geladen…',
    empty: 'Diese Quelle enthält keine passenden Skills.',
    signIn: 'Melden Sie sich über das bereitstellende Plugin an, um diese Quelle zu durchsuchen.',
    unavailable: 'Diese Quelle ist nicht verfügbar. Ihre lokalen Skills bleiben unverändert.',
    incomplete: 'Die Prüfung ist unvollständig oder die Anbieter-Richtlinie erlaubt keine Bestätigung.',
    changed: 'Diese Prüfung ist nicht mehr aktuell. Schließen Sie sie und fordern Sie eine neue Prüfung an.',
    unknown: 'Das Ergebnis ist noch nicht bekannt. Stellen Sie die ursprüngliche Bestätigung wieder her, bevor Sie eine weitere Aktion starten.',
    recover: 'Ursprünglichen Vorgang wiederherstellen',
    finalizing: 'Lokaler Status wird abgeschlossen…',
    setup: 'Nur Einrichtungshinweise. Die Bestätigung führt diese Anweisungen nicht aus.',
    more: (source: string) => `Mehr laden: ${source}`,
    preview: 'Skill-Vorschau',
    status: {
      not_installed: 'Nicht installiert', installed: 'Installiert', removed: 'Außerhalb dieser Quelle entfernt',
      unmanaged_local: 'Nicht verwaltete lokale Kopie', pending: 'Vorgang ausstehend', unavailable: 'Quelle nicht verfügbar',
      unknown: 'Ergebnis unbekannt', completed: 'Abgeschlossen', failed: 'Vorgang abgelehnt'
    }
  }
} satisfies TranslationOverrides['catalog']
