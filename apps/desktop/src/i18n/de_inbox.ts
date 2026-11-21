import type { InboxCopy } from './types_inbox'

export const deInbox: InboxCopy = {
  title: 'Posteingang',
  unreadCount: (count: number) => `${count} ungelesen`,
  loading: 'Posteingang wird geladen…',
  emptyTitle: 'Noch keine Ergebnisse',
  emptyDesc: 'Wenn ein geplanter Job fertig ist, landet sein Ergebnis hier.',
  running: 'Läuft',
  unread: 'Ungelesen',
  noPreview: 'Ohne Antwort beendet.',
  refresh: 'Aktualisieren',
  refreshing: 'Wird aktualisiert…',
  markReadFailed: 'Ergebnis konnte nicht als gelesen markiert werden',
  newResultsTitle: (count: number) => `${count} neue Ergebnisse geplanter Jobs`,
  newResultsBody: 'Öffne den Posteingang, um sie zu lesen.'
}
