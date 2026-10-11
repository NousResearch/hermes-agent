import type { Translations } from './types'

export const deModelOptions: Translations['shell']['modelOptions'] = {
  noOptions: 'Keine Optionen für dieses Modell',
  options: 'Optionen',
  thinking: 'Denken',
  fast: 'Schnell',
  ultrafast: 'Ultrafast',
  useStandardSpeed: 'Standardgeschwindigkeit verwenden',
  auto: 'Automatisch',
  cold: 'Kalt',
  speedPolicy: 'Geschwindigkeitsrichtlinie',
  effort: 'Aufwand',
  minimal: 'Minimal',
  low: 'Niedrig',
  medium: 'Mittel',
  high: 'Hoch',
  xhigh: 'Extra hoch',
  max: 'Max',
  ultra: 'Ultra',
  sendsOnRoute: (level: string) => `sendet ${level} auf dieser Route`,
  updateFailed: 'Aktualisierung der Modelloptie schlug fehl',
  fastFailed: 'Aktualisierung des Schnell-Modus schlug fehl'
}
