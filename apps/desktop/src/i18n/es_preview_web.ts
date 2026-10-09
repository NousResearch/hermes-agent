import type { TranslationOverride } from '@hermes/shared/i18n'

import type { Translations } from './types'

// The in-app browser pane's copy (`preview.web`), composed by es.ts.
export const esPreviewWeb: TranslationOverride<Translations['preview']['web']> = {
  embeddedPreviewHint:
    'Algunos sitios bloquean las vistas previas incrustadas. Abre la página original en una pestaña del navegador.',
  appFailedToBoot: 'La app de vista previa no arrancó',
  serverNotFound: 'Servidor no encontrado',
  remoteLoopback:
    'Esta dirección apunta al equipo que ejecuta tu agente, no a este. El panel del navegador carga las páginas localmente, así que un servidor de desarrollo remoto necesita un reenvío de puertos o un nombre de host accesible.',
  failedToLoad: 'No se pudo cargar la vista previa',
  tryAgain: 'Intentar de nuevo',
  restarting: 'Hermes se está reiniciando...',
  askRestart: 'Pedir a Hermes que reinicie el servidor',
  lookingRestart: taskId => `Hermes está buscando un servidor de vista previa para reiniciar (${taskId})`,
  restartingTitle: 'Reiniciando servidor de vista previa',
  restartingMessage: 'Hermes está trabajando en segundo plano. Mira la consola de vista previa para ver el progreso.',
  startRestartFailed: message => `No se pudo iniciar el reinicio del servidor: ${message}`,
  restartFailed: 'Falló el reinicio del servidor',
  hideConsole: 'Ocultar consola de vista previa',
  showConsole: 'Mostrar consola de vista previa',
  hideDevTools: 'Ocultar DevTools de vista previa',
  openDevTools: 'Abrir DevTools de vista previa',
  goBack: 'Atrás',
  goForward: 'Adelante',
  reload: 'Recargar página',
  address: 'Dirección',
  addressPlaceholder: 'Introduce una dirección',
  blankPageBody: 'Escribe una dirección arriba para navegar o pide a Hermes que abra una página.',
  finishedRestarting: message =>
    `Hermes terminó de reiniciar el servidor de vista previa${message ? `: ${message}` : ''}`,
  failedRestarting: message => `Falló el reinicio del servidor: ${message}`,
  unknownError: 'error desconocido',
  restartedTitle: 'Servidor de vista previa reiniciado',
  reloadingNow: 'Recargando la vista previa ahora.',
  restartFailedTitle: 'Falló el reinicio de la vista previa',
  restartFailedMessage: 'Hermes no pudo reiniciar el servidor.',
  stillWorking:
    'Hermes sigue trabajando, pero aún no llegó ningún resultado de reinicio. Puede que el comando del servidor siga en primer plano.',
  workspaceReloading: 'El espacio de trabajo cambió, recargando vista previa',
  fileChanged: url => `Archivo cambiado, recargando vista previa: ${url}`,
  filesChanged: (count, url) => `${count} cambios de archivo, recargando vista previa: ${url}`,
  watchFailed: message => `No se pudo vigilar el archivo de vista previa: ${message}`,
  moduleMimeDescription:
    'Los scripts de módulo se están sirviendo con el tipo MIME incorrecto. Normalmente significa que un servidor de archivos estáticos sirve una app Vite/React en lugar del dev server del proyecto.',
  loadFailedConsole: (code, message) => `Carga fallida${code ? ` (${code})` : ''}: ${message}`,
  unreachableDescription: 'No se pudo acceder a la página de vista previa.',
  openTarget: url => `Abrir ${url}`,
  fallbackTitle: 'Vista previa',
  annotate: 'Anotar',
  annotateOn: 'Dejar de anotar',
  annotateNeedPage: 'Primero abre una página en el navegador integrado.',
  annotateFailed: 'No se pudo iniciar el modo de anotación',
  commenting: 'Comentando',
  addComments: (count: number) => (count === 1 ? 'Añadir 1 comentario' : `Añadir ${count} comentarios`),
  commentPlaceholder: 'Añade un comentario...',
  commentTitle: (n: number) => `Comentario ${n}`,
  saveComment: 'Guardar',
  cancelComment: 'Cancelar comentario'
}
