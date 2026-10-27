import type { MemoryLayersTranslations } from './types_memory_layers'

export const esMemoryLayers: MemoryLayersTranslations = {
  activeProvider: 'Proveedor de memoria activo',
  noProvider: 'Ninguno — solo memoria integrada',
  noProviderDesc:
    'Hermes usa solo la capa integrada de abajo. Elige un proveedor externo en Proveedor de memoria para añadir uno.',
  providerDesc: name => `${name} añade su propio recuerdo a cada sesión, además de la capa integrada.`,
  status: {
    ready: 'listo',
    needs_config: 'requiere configuración',
    unavailable: 'no disponible',
    missing: 'no instalado'
  },
  builtinLayer: 'Capa integrada',
  alwaysOn: 'Siempre activa',
  builtinDesc:
    'MEMORY.md y USER.md se cargan en cada sesión, con o sin proveedor externo. Un proveedor se suma a esta capa; nunca la reemplaza.'
}
