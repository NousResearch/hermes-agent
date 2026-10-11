import type { Translations } from './types'

export const esModelOptions: Translations['shell']['modelOptions'] = {
  noOptions: 'No hay opciones para este modelo',
  options: 'Opciones',
  thinking: 'Razonamiento',
  fast: 'Rápido',
  ultrafast: 'Ultrafast',
  useStandardSpeed: 'Usar velocidad estándar',
  auto: 'Automática',
  cold: 'Fría',
  speedPolicy: 'Política de velocidad',
  effort: 'Esfuerzo',
  minimal: 'Mínimo',
  low: 'Bajo',
  medium: 'Medio',
  high: 'Alto',
  xhigh: 'Extra alto',
  max: 'Máximo',
  ultra: 'Ultra',
  sendsOnRoute: (level: string) => `envía ${level} en esta ruta`,
  updateFailed: 'No se pudo actualizar la opción del modelo',
  fastFailed: 'No se pudo actualizar el modo rápido'
}
