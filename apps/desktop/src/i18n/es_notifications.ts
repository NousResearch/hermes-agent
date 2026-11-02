// Notification settings copy used when the desktop locale loads.
export const esNotifications = {
  title: 'Notificaciones',
  intro: 'Notificaciones del sistema operativo (no avisos dentro de la app). Por dispositivo.',
  enableAll: 'Activar notificaciones',
  enableAllDesc: 'Desactivado silencia todas las notificaciones siguientes.',
  focusedHint: 'Los avisos de finalización solo se activan cuando Hermes está en segundo plano.',
  kinds: {
    approval: {
      label: 'Se necesita aprobación',
      description: 'Un comando espera que lo apruebes o rechaces.'
    },
    input: {
      label: 'Se necesita información',
      description: 'Hermes hizo una pregunta o necesita una contraseña o un secreto.'
    },
    turnDone: {
      label: 'Respuesta lista',
      description: 'Terminó un turno mientras Hermes estaba en segundo plano.'
    },
    turnError: {
      label: 'El turno falló',
      description: 'Errores de turnos en segundo plano.'
    },
    backgroundDone: {
      label: 'Tarea en segundo plano finalizada',
      description: 'Se completó un comando de terminal en segundo plano.'
    },
    credits: {
      label: 'Avisos de crédito',
      description: 'El acceso a los créditos se pausa o se restablece.'
    },
    plugin: {
      label: 'Notificaciones de complementos',
      description: 'Un complemento de escritorio envió una notificación mientras Hermes estaba en segundo plano.'
    }
  },
  browserPermission: {
    title: 'Notificaciones del navegador',
    allow: 'Permitir',
    states: {
      default:
        'Muestra avisos de este navegador mientras la pestaña está en segundo plano. Hasta que lo permitas, solo parpadea el título de la pestaña.',
      granted:
        'Permitido. Los avisos aparecen mientras esta pestaña está en segundo plano; el título de la pestaña también parpadea.',
      denied: 'Bloqueado en la configuración del sitio de este navegador. El título de la pestaña sigue parpadeando.',
      insecure: 'Requiere HTTPS, como Tailscale serve. Con HTTP simple solo parpadea el título de la pestaña.',
      unsupported: 'Este navegador no puede mostrar notificaciones. El título de la pestaña sigue parpadeando.'
    }
  },
  test: 'Enviar notificación de prueba',
  testTitle: 'Hermes',
  testBody: 'Las notificaciones funcionan.',
  testSent:
    'Prueba enviada. Si no aparece nada, revisa los permisos de notificaciones del sistema operativo y el modo Concentración o No molestar.',
  testUnsupported: 'Este sistema no admite notificaciones nativas.',
  muteSoundsTitle: 'Silenciar sonidos de la interfaz',
  muteSoundsDesc:
    'Silencia los avisos automáticos de finalización, activación y reflexión, la respuesta háptica y los sonidos de las notificaciones del sistema. Las vistas previas de sonido y las respuestas habladas siguen reproduciéndose.',
  completionSoundTitle: 'Sonido de finalización',
  completionSoundDesc:
    'Reproduce un sonido cuando termina el turno de un agente. Elige un ajuste predefinido y pruébalo aquí.',
  completionSoundPreview: 'Vista previa'
}
