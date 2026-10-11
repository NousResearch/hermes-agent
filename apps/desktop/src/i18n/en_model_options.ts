import type { Translations } from './types'

export const enModelOptions: Translations['shell']['modelOptions'] = {
  noOptions: 'No options for this model',
  options: 'Options',
  thinking: 'Thinking',
  fast: 'Fast',
  ultrafast: 'Ultrafast',
  useStandardSpeed: 'Use standard speed',
  auto: 'Auto',
  cold: 'Cold',
  effort: 'Effort',
  minimal: 'Minimal',
  low: 'Low',
  medium: 'Medium',
  high: 'High',
  xhigh: 'Extra High',
  max: 'Max',
  ultra: 'Ultra',
  sendsOnRoute: (level: string) => `sends ${level} on this route`,
  updateFailed: 'Model option update failed',
  speedPolicy: 'Speed policy',
  fastFailed: 'Fast mode update failed'
}
