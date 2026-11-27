import type { MemoryLayersTranslations } from './types_memory_layers'

export const esMemoryLayers: MemoryLayersTranslations = {
  activeProvider: 'Proveedor de memoria activo',
  selectedProvider: 'Proveedor de memoria seleccionado',
  selectedDesc: current =>
    `Aún no está activo: las sesiones siguen usando ${current} hasta que se guarde esta elección.`,
  builtinMemory: 'solo la memoria integrada',
  noProvider: 'Ninguno — solo memoria integrada',
  noProviderDesc:
    'Hermes usa solo la capa integrada de abajo. Elige un proveedor externo en Ajustes › Memoria y contexto para añadir uno.',
  providerDesc: name => `${name} añade su propio recuerdo a cada sesión, además de la capa integrada.`,
  providerNotReadyDesc: name =>
    `${name} es el proveedor guardado, pero no aporta recuerdos hasta que esté listo. La capa integrada funciona de todos modos.`,
  providerUnknownDesc: name => `${name} es el proveedor guardado; Hermes no pudo leer si está listo.`,
  statusUnavailable: 'Estado de la memoria no disponible',
  statusUnavailableDesc: 'No se pudo leer qué proveedor de memoria está activo ni si está listo.',
  retry: 'Reintentar',
  status: {
    ready: 'listo',
    needs_config: 'requiere configuración',
    unavailable: 'no disponible',
    missing: 'no instalado'
  },
  builtinLayer: 'Capa integrada',
  onByDefault: 'Activada por defecto',
  on: 'Activada',
  off: 'Desactivada',
  onlyFile: file => `solo ${file}`,
  builtinDesc:
    'MEMORY.md y USER.md son la memoria propia de Hermes, independiente de cualquier proveedor externo. Ambos se cargan en las sesiones por defecto; los interruptores de memoria y perfil de usuario en Ajustes › Memoria y contexto desactivan cada uno. Un proveedor se suma a esta capa; nunca la reemplaza.'
}
