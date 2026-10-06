export const esSharedMetrics = {
  consentTitle: '¿Nos ayudas a mejorar Hermes?',
  consentBody:
    'Las métricas compartidas solo contienen contadores acotados. Nunca prompts, archivos, rutas ni textos de error. La recopilación es local. Enviarlas a Nous es una aceptación aparte.',
  whatIsCollected: 'Qué se recopila',
  collectedIntro: 'Solo contadores acotados:',
  collectedActivity:
    'Actividad, duración de sesiones, resultados y clases de error, incluido un motivo de una lista fija cuando una escritura en memoria o una compresión de contexto se rechaza, falla o se omite',
  collectedModels: 'Rutas de modelo y totales de tokens',
  collectedNames: 'Nombres de herramientas, comandos y elementos del catálogo integrados',
  collectedMilestones: 'Recuentos de configuración agrupados',
  collectedReliability:
    'Resultados y duración de actualizaciones, fallos, velocidad de inicio y de respuesta, estado de las plataformas de mensajería',
  collectedUsage:
    'Cómo se usa Hermes: precisión y eficiencia del agente (ediciones acertadas, bucles, recuperaciones, tokens y llamadas a herramientas por tarea, cortes de caché), tiempo activo por superficie y modo de Desktop, qué áreas, acciones y ajustes de la app se usan, se cierran enseguida o se desactivan, y resultados de la configuración de proveedores',
  collectedMachine:
    'Datos generales del equipo: rango de RAM, tipo de GPU, antigüedad y canal de la versión de Hermes, actualizaciones pendientes, si se usa un servidor de modelos local',
  installId:
    'Al enviar, cada paquete diario se sube al servicio de telemetría de Nous. Los paquetes llevan el ID de instalación de este perfil: un UUID aleatorio y estable sin información personal, que se restablece al borrar el directorio de métricas compartidas.',
  consentWindow:
    'Solo se envían los paquetes cuyo periodo de recopilación completo cae dentro de una ventana de consentimiento registrada; los datos de antes de aceptar, o de cualquier intervalo con el envío desactivado, se quedan en este equipo. Puedes volver a desactivar el envío cuando quieras.',
  readDocs: 'Leer todos los detalles',
  share: 'Recopilar y enviar a Nous',
  local: 'Recopilar solo en local',
  off: 'No, gracias',
  changeLater: 'Puedes cambiarlo cuando quieras en Ajustes → Seguridad.',
  saveFailed: 'No se pudo guardar tu elección',
  collectLabel: 'Recopilar estadísticas de uso',
  collectDesc: 'Contadores acotados guardados en este dispositivo. Nunca prompts, archivos, rutas ni textos de error.',
  sendLabel: 'Enviar estadísticas de uso a Nous',
  sendDesc:
    'Sube cada paquete diario al servicio de telemetría de Nous. Solo se envían datos de una ventana de consentimiento. Requiere la recopilación activada.',
  unavailable: 'Actualiza el backend de Hermes para cambiar este ajuste.',
  stripBody: 'Solo contadores acotados, nunca prompts ni archivos.',
  stripReaskBody:
    'Te lo preguntamos de nuevo: una versión anterior podía guardar «No, gracias» antes de que vieras esta pregunta.',
  stripChoices: { share: 'Enviar a Nous', local: 'Solo local', off: 'No, gracias' },
  stripDetails: 'Detalles'
}
