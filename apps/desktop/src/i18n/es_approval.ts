import type { ApprovalCopy } from './types_approval'

export const esApproval: Partial<ApprovalCopy> = {
  gatewayDisconnected:
    'Hermes está sin conexión ahora mismo. El comando sigue esperando tu respuesta (hasta que se agote el tiempo de aprobación). Reconéctate y vuelve a enviarla.',
  sendFailed: 'No se pudo enviar tu respuesta',
  reconnect: 'Reconectar',
  timedOutSystemLine:
    'Se agotó el tiempo de aprobación: el comando no se ejecutó. Pide a Hermes que lo intente de nuevo o sube el límite en Configuración → Seguridad → Tiempo de aprobación.',
  openSafetySettings: 'Abrir configuración de seguridad',
  run: 'Ejecutar',
  command: 'Comando',
  moreOptions: 'Más opciones de aprobación',
  allowSession: 'Permitir esta sesión',
  alwaysAllowMenu: 'Permitir siempre…',
  jumpToApproval: 'Aprobación necesaria',
  reject: 'Rechazar',
  alwaysTitle: '¿Permitir siempre este comando?',
  alwaysDescription: pattern =>
    `Esto añade el patrón “${pattern}” a tu allowlist permanente (~/.hermes/config.yaml). Hermes no volverá a preguntar por comandos como este, ni en esta sesión ni en futuras.`,
  alwaysAllow: 'Permitir siempre',
  deny: 'Denegar',
  allow: 'Permitir',
  allowHint: 'Hasta que se desactive el sandbox',
  denyReason: 'Motivo de la denegación',
  denyReasonPlaceholder: 'Dile a Hermes por qué (opcional)',
  access: {
    title: 'Acceso a carpeta',
    readOnly: 'Solo lectura',
    readWrite: 'Lectura y escritura',
    alwaysTitle: '¿Permitir siempre esta carpeta?',
    alwaysDescription:
      'La carpeta se añade al sandbox de este perfil. Hermes no volverá a preguntar por ella en ningún chat.'
  },
  network: {
    title: 'Acceso a sitio web',
    alwaysTitle: '¿Permitir siempre este sitio?',
    alwaysDescription:
      'El sitio se añade a los sitios permitidos de este perfil. Hermes no volverá a preguntar por él en ningún chat.'
  },
  schedule: {
    title: '¿Programar esta tarea?',
    allowSession: 'Permitir en este chat',
    alwaysAllowMenu: 'Permitir siempre tareas nuevas…',
    alwaysTitle: '¿Permitir siempre nuevas tareas programadas?',
    alwaysDescription:
      'Hermes creará tareas programadas sin preguntar antes. Puedes volver a activar la pregunta quitando cron_create de command_allowlist en config.yaml.',
    timedOutSystemLine:
      'La aprobación caducó: no se creó la tarea. Pide a Hermes que lo intente de nuevo o sube el límite en Ajustes → Seguridad → Tiempo de aprobación.'
  }
}
