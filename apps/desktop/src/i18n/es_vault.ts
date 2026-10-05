import type { Translations } from './types'

export const esVault: Translations['settings']['vault'] = {
  title: 'Contraseñas e inicios de sesión',
  blurb:
    'Di “inicia sesión en GitHub” y el agente lo hará por ti. La primera vez que encuentre una página de inicio de sesión, te pedirá los datos ahí mismo; después, simplemente funcionará. Las contraseñas se cifran en este equipo y se introducen directamente en la página: el modelo nunca las ve.',
  count: n => `${n} guardado${n === 1 ? '' : 's'}`,
  loadFailed: 'No se pudieron cargar los elementos guardados',
  empty: 'Todavía no hay nada guardado',
  emptyDesc:
    'No necesitas añadir nada aquí. Pide al agente que inicie sesión en un sitio y te pedirá los datos una vez, en ese momento. Usa Añadir si prefieres introducirlos de antemano.',
  add: 'Añadir',
  addTitle: 'Añadir un inicio de sesión, una tarjeta o una dirección',
  addDescription: 'Se guarda cifrado en este equipo. El agente nunca ve la contraseña.',
  added: 'Guardado.',
  adding: 'Guardando…',
  addConfirm: 'Guardar',
  kindField: 'Tipo',
  kinds: {
    login: 'Inicio de sesión',
    payment: 'Tarjeta de pago',
    address: 'Dirección'
  },
  labelField: 'Etiqueta',
  labelPlaceholder: 'p. ej., cuenta de trabajo de GitHub',
  labelRequired: 'La etiqueta es obligatoria.',
  originField: 'Origen del sitio',
  originPlaceholder: 'https://github.com',
  originPlaceholderCheckout: 'https://tienda.example.com',
  originInvalid: 'Introduce una URL válida, como https://example.com.',
  originAnySiteHint:
    'Déjalo en blanco para usar esta dirección en cualquier sitio; el agente te pide confirmar el sitio cada vez que la rellena.',
  anySite: 'Cualquier sitio',
  identifierTypeField: 'Tipo de identificador',
  identifierTypes: {
    email: 'Correo electrónico',
    phone: 'Teléfono',
    username: 'Nombre de usuario'
  },
  identifierField: 'Identificador',
  identifierShown: identifier => identifier,
  passwordField: 'Contraseña',
  loginFieldsRequired: 'El identificador y la contraseña son obligatorios.',
  cardNumberField: 'Número de tarjeta',
  cardNameField: 'Nombre en la tarjeta',
  expMonthField: 'Mes de venc.',
  expYearField: 'Año de venc.',
  cvcField: 'CVC',
  postalField: 'Código postal',
  addressLine1Field: 'Dirección, línea 1',
  addressLine2Field: 'Dirección, línea 2',
  cityField: 'Ciudad',
  stateField: 'Estado / región',
  countryField: 'País',
  optional: '(opcional)',
  createdOn: date => `Añadido el ${date}`,
  deleteAction: 'Quitar elemento guardado',
  otpField: 'Clave del autenticador',
  otpPlaceholder: 'Secreto Base32 o enlace otpauth://',
  otpHint:
    'La “clave de configuración” que muestra el sitio al activar la 2FA. Si la guardas, Hermes genera los códigos por sí mismo.',
  twoFactorBadge: '2FA automática',
  deleteTitle: '¿Eliminar este elemento?',
  deleteDescription: label => `Se quitará “${label}”. Esto no se puede deshacer.`,
  deleteConfirm: 'Eliminar',
  sources: {
    title: 'Gestores de contraseñas',
    blurb:
      'Los gestores de contraseñas instalados se detectan automáticamente. El agente te pide desbloquear uno la primera vez que necesita un inicio de sesión de él (una vez por sesión); solo se guarda en memoria un token de sesión, y el agente nunca ve tu contraseña maestra ni ningún inicio de sesión.',
    toggleFailed: 'No se pudo actualizar el gestor de contraseñas',
    notInstalled: name =>
      `No detectado. Instala la herramienta de línea de comandos de ${name} e inicia sesión en ella; Hermes la detectará automáticamente.`,
    disabledDesc: 'Detectado, pero desactivado para Hermes.',
    lockedDesc:
      'Detectado. El agente te pedirá desbloquearlo cuando necesite un inicio de sesión, o puedes desbloquearlo ahora.',
    unlockedDesc:
      'Desbloqueado para esta sesión. Se bloquea automáticamente tras 30 minutos de inactividad o al cerrar Hermes.',
    statusLocked: 'Bloqueado',
    statusNotDetected: 'No detectado',
    statusOff: 'Desactivado',
    statusUnlocked: 'Desbloqueado',
    unlock: 'Desbloquear',
    unlocking: 'Desbloqueando…',
    lock: 'Bloquear',
    unlocked: name => `${name} desbloqueado para esta sesión.`,
    unlockTitle: name => `Desbloquear ${name}`,
    unlockDescription:
      'Introduce tu contraseña maestra. Se entrega al gestor de contraseñas de este equipo y se descarta: nunca se guarda, se registra ni se muestra al agente.',
    masterPasswordPlaceholder: 'Contraseña maestra'
  }
}
