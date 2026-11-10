import type { TranslationOverride } from '@hermes/shared/i18n'

import { esAuxTasks } from './es_aux_tasks'
import { esDelegationRouting } from './es_delegation_routing'
import type { ModelSettingsTranslations } from './types_model_settings'

export const esModelSettings: TranslationOverride<ModelSettingsTranslations> = {
  setupProviderFallback: 'proveedor',
  setUpProvider: name => `Configurar ${name}`,
  staleAuxBefore: (count, names) =>
    count === 1
      ? `${count} tarea auxiliar (${names}) todavía se ejecuta en `
      : `${count} tareas auxiliares (${names}) todavía se ejecutan en `,
  staleAuxAfter: ', no en tu modelo principal.',
  staleAuxOtherProviders: 'otros proveedores',
  moaEnabled: 'Activado',
  moaSetDefault: 'Establecer como predeterminado',
  moaNewPresetPlaceholder: 'nuevo preajuste',
  moaAddPreset: 'Añadir preajuste',
  customModel: 'Modelo personalizado…',
  customModelPlaceholder: 'ID del modelo',
  chooseFromList: 'Elegir de la lista',
  moaDefault: 'Predeterminado:',
  moaReferenceToggle: (enabled, index) => `${enabled ? 'Desactivar' : 'Activar'} la referencia ${index}`,
  moaReferenceTitle: index => `Referencia ${index}`,
  moaAddReference: 'Añadir modelo de referencia',
  loading: 'Cargando configuración de modelo...',
  appliesDesc: 'Se aplica a sesiones nuevas. Usa el selector de modelo en el compositor para cambiar el chat activo.',
  provider: 'Proveedor',
  model: 'Modelo',
  applying: 'Aplicando...',
  mainAppliedTitle: 'Modelo principal actualizado',
  mainAppliedMessage: model => `Las sesiones nuevas usarán ${model}.`,
  defaultsLabel: 'Valores predeterminados',
  reasoning: 'Razonamiento',
  reasoningOff: 'Desactivado',
  speed: 'Velocidad',
  speedStandard: 'Estándar',
  defaultsFailed: 'No se pudieron guardar los valores predeterminados del modelo',
  loadFailed: 'No se pudieron cargar los modelos',
  restartRequired:
    'Este backend ejecuta código antiguo tras una actualización. Reinícialo para cargar el código nuevo.',
  restartBackend: 'Reiniciar backend',
  restartingBackend: 'Reiniciando backend...',
  restartFailed: 'No se pudo reiniciar el backend',
  auxiliaryTitle: 'Modelos auxiliares',
  resetAllToMain: 'Restablecer todos al principal',
  staleAuxDismiss: 'No volver a mostrar',
  auxiliaryDesc:
    'Las tareas auxiliares usan el modelo principal de forma predeterminada. Asigna un modelo dedicado a cualquier tarea para anularlo.',
  setToMain: 'Usar principal',
  change: 'Cambiar',
  autoUseMain: 'auto · usar modelo principal',
  inheritMainEffort: 'heredar · esfuerzo del modelo principal',
  inheritsFrom: task => `hereda de ${task}`,
  followTask: task => `Seguir ${task}`,
  providerDefault: '(predeterminado del proveedor)',
  fallbackAdd: 'Añadir respaldo',
  fallbackEmpty: 'No hay modelos de respaldo; se usa el modelo predeterminado salvo que falle.',
  notInCatalog: 'no está en la lista de modelos de este proveedor; las llamadas pueden recurrir a un respaldo.',
  moaTitle: 'Mixture of Agents',
  moaPreset: 'Preajuste',
  moaDescription:
    'Configura preajustes con nombre que aparecen como modelos del proveedor Mixture of Agents. El agregador es el modelo que actúa: ejecuta cada paso del bucle de herramientas, y casi todo el coste de la ejecución se factura a su proveedor. Por defecto, las referencias solo asesoran una vez por turno del usuario.',
  moaAggregator: 'Agregador',
  moaAggregatorBilled: 'modelo que actúa · se factura por la ejecución',
  moaReferenceHint: 'asesora una vez por turno por defecto',
  tasks: esAuxTasks,
  subagents: {
    inherit: 'Heredar del agente principal',
    modelOnly: 'Modelo personalizado (credenciales del agente principal)',
    providerLabel: 'Proveedor de subagentes',
    modelLabel: 'Modelo de subagentes',
    direct: 'Endpoint directo',
    customProvider: 'Proveedor personalizado...',
    customProviderLabel: 'Proveedor personalizado de subagentes',
    parentModel: 'Usar el modelo del agente principal',
    parentModelWarning:
      'El proveedor elegido debe admitir el modelo del agente principal. Una ruta solo de proveedor puede fallar si el modelo no es compatible. Si ese proveedor define su propio modelo predeterminado, los subagentes usan ese.',
    directActive:
      'Hay un endpoint directo activo que tiene prioridad sobre el proveedor guardado. Su endpoint y credenciales se conservan mientras este modo esté seleccionado.',
    directClear:
      'Aplicar esta ruta borra el endpoint directo y su clave guardada. El resto de ajustes de solicitud se conserva.',
    catalogFailed: 'Catálogo de modelos no disponible. Introduce un proveedor y un modelo manualmente, o reintenta.',
    draftHint: 'Los cambios se quedan en local hasta que los apliques.',
    routing: esDelegationRouting
  }
}
