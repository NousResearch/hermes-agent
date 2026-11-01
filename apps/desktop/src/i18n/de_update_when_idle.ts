import type { UpdateWhenIdleCopy } from './types_update_when_idle'

export const deUpdateWhenIdle: UpdateWhenIdleCopy = {
  busyTitle: 'Hermes arbeitet noch',
  busyBody: 'Ein Update startet Hermes neu und beendet dabei:',
  chats: (n: number) => (n === 1 ? '1 Chat läuft' : `${n} Chats laufen`),
  cron: (jobs: string) => `Geplanter Job läuft (${jobs})`,
  delegation: 'Hintergrund-Subagenten laufen',
  input: 'Eine Rückfrage wartet auf deine Antwort',
  unknown: 'Hermes konnte nicht bestätigen, dass nichts läuft',
  updateNow: (n: number) => `Jetzt aktualisieren (unterbricht ${n})`,
  updateWhenIdle: 'Aktualisieren, sobald frei',
  waitingTitle: 'Update wartet, bis Hermes fertig ist',
  waitingBody:
    'Das Update startet von selbst, sobald nichts mehr läuft. Arbeite ruhig weiter; Hermes bleibt bis dahin nutzbar.',
  waitingFor: 'Wartet auf:',
  waitingIdle: 'Nichts läuft mehr. Update startet…',
  waitingUntil: (time: string) => `Ist Hermes um ${time} noch beschäftigt, wird das Update übersprungen.`,
  cancel: 'Update abbrechen',
  timedOut:
    'Hermes war weiter beschäftigt, daher lief das Update nicht. Nichts wurde geändert; versuche es später erneut.'
}
