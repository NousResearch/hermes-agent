import type { DelegationRoutingTranslations } from './types_delegation_routing'

export const esDelegationRouting: DelegationRoutingTranslations = {
  title: 'Enrutamiento',
  settingsTitle: 'Valores predeterminados de subagentes',
  intro: 'Cómo obtiene su modelo cada subagente.',
  modes: {
    default: {
      label: 'Predeterminado',
      description: 'Todos los subagentes usan el modelo de subagente de abajo.'
    },
    model_map: {
      label: 'Rutas de modelo',
      description: 'Nombra algunos modelos. El agente elige uno para cada subagente.'
    },
    profile_routes: {
      label: 'Rutas de perfil',
      description: 'Los subagentes se ejecutan como uno de tus perfiles, con su propio modelo, habilidades y claves.'
    }
  },
  inUse: 'En uso',
  plugin: 'Plugin',
  pluginMissing: 'Rutas de modelo y Rutas de perfil provienen del plugin Advanced Delegation.',
  pluginDisabled: 'Advanced Delegation está instalado, pero desactivado para este perfil.',
  pluginNotLoaded:
    'Advanced Delegation está activado, pero Hermes aún no lo ha cargado. Reinicia Hermes para terminar.',
  installPlugin: 'Instalar plugin',
  installUnavailable:
    'Advanced Delegation aún no está en el catálogo de plugins. Instálalo desde Capacidades › Plugins.',
  enablePlugin: 'Activar',
  enableFailed: 'No se pudo activar Advanced Delegation',
  modeFailed: 'No se pudo cambiar el enrutamiento de subagentes',
  loadFailed: 'No se pudo cargar el enrutamiento de subagentes',
  blockedPill: 'No funciona',
  blockedDetail:
    'Este modo está guardado, así que Hermes rechaza cada subagente en lugar de usar Predeterminado sin avisar.',
  blockedMissing: 'Los subagentes no se iniciarán hasta que Advanced Delegation esté instalado.',
  blockedDisabled: 'Los subagentes no se iniciarán hasta que Advanced Delegation esté activado para este perfil.',
  blockedNotLoaded:
    'Los subagentes no se iniciarán hasta que Hermes cargue Advanced Delegation. Reinicia Hermes para terminar.',
  blockedUnavailable: label =>
    `Los subagentes no se iniciarán hasta que ${label} pueda funcionar aquí. Su plugin indica que no puede.`,
  blockedUnregistered: name =>
    `Los subagentes no se iniciarán hasta que se instale y active un plugin que proporcione «${name}».`,
  thirdPartyDescription: 'Proviene de un plugin. Sus ajustes están en ese plugin.',

  modelRoutesTitle: 'Rutas de modelo',
  modelRoutesEmpty: 'Aún no hay rutas. Hasta que añadas una, los subagentes usan el modelo del agente principal.',
  addRoute: 'Añadir ruta',
  editRoute: 'Editar',
  makeDefault: 'Hacer predeterminada',
  deleteRoute: 'Eliminar',
  defaultBadge: 'Predeterminada',
  routeActions: name => `Acciones para ${name}`,
  deleteConfirmTitle: name => `¿Eliminar la ruta “${name}”?`,
  deleteConfirmBody: 'Los subagentes ya no podrán elegirla.',
  deleteConfirmNextDefault: name => `Es la ruta predeterminada, así que “${name}” pasa a ser la predeterminada.`,
  deleteConfirmAction: 'Eliminar ruta',
  saveFailed: 'No se pudieron guardar las rutas',
  baseUrlRoute: url => `endpoint ${url}`,

  dialogNew: 'Nueva ruta de modelo',
  dialogEdit: 'Editar ruta de modelo',
  dialogDesc: 'El agente lee el nombre y la descripción para decidir qué ruta toma un subagente.',
  nameLabel: 'Nombre',
  namePlaceholder: 'p. ej. fast',
  nameHint: 'Corto, sin espacios. El agente elige las rutas por este nombre.',
  descriptionLabel: 'Cuándo usarla',
  descriptionPlaceholder: 'p. ej. Barato y rápido: búsquedas, resúmenes, ediciones sencillas.',
  descriptionHint: 'Una línea que el agente lee antes de elegir.',
  providerLabel: 'Proveedor',
  providerPlaceholder: 'Elige un proveedor',
  modelLabel: 'Modelo',
  effortLabel: 'Esfuerzo de razonamiento',
  effortInherit: 'Predeterminado del subagente',
  tierLabel: 'Velocidad',
  tierNormal: 'Normal',
  tierFast: 'Rápida',
  tierUltrafast: 'Ultrarrápida',
  tierHint: 'Solo en modelos que la ofrecen.',
  optional: 'opcional',
  errors: { required: 'Obligatorio', spaces: 'Sin espacios', duplicate: 'Otra ruta ya usa este nombre' },
  saveRoute: 'Guardar ruta',
  createRoute: 'Añadir ruta',

  profileRoutesTitle: 'Perfiles como los que pueden ejecutarse los subagentes',
  profileRoutesHint: 'Cada perfil marcado aporta su propio modelo, habilidades, personalidad y claves.',
  profileDescriptionPlaceholder: 'Cuándo usarlo (opcional)',
  profileDefault: 'Predeterminado',
  profileEmpty: 'Aún no hay otros perfiles. Crea uno desde la barra de perfiles.',
  profileMissing: name => `${name} (el perfil ya no existe)`,

  inactiveTitle: 'No se usa en este modo',
  inactiveModelMap: route =>
    `Las rutas de modelo eligen el modelo de cada subagente. Los subagentes que no eligen ninguna ruta usan “${route}”.`,
  inactiveProfiles: profile =>
    `Las rutas de perfil eligen el perfil de cada subagente. Los subagentes que no eligen ningún perfil se ejecutan como “${profile}”.`,
  inactiveModelMapNoDefault:
    'Las rutas de modelo eligen el modelo de cada subagente. Los subagentes que no eligen ruta usan el modelo del agente principal.',
  inactiveProfilesNoDefault:
    'Las rutas de perfil eligen el perfil de cada subagente. Los subagentes que no eligen perfil usan el modelo del agente principal.',
  inactivePlugin: label => `${label} decide el modelo de cada subagente, así que este ajuste no se aplica.`,
  useDefault: 'Cambiar a Predeterminado'
}
