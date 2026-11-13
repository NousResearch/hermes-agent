// Desktop-side /rollback /agents /config /history /save /export /import /timestamps lines
// (en_desktop_commands.ts is the source).
export const esDesktopCommands = {
  error: (message: string) => `error: ${message}`,
  rollbackUsage: 'uso: /rollback [list] | diff <n|hash> | <n|hash> [file] [--all]',
  checkpointsDisabled:
    'Los puntos de control no están activados para esta sesión (configura checkpoints.enabled en config.yaml).',
  checkpointsEmpty: 'Todavía no hay puntos de control para este directorio de trabajo.',
  checkpointsHeader: 'Puntos de control (los más recientes primero):',
  checkpointsHint:
    'Restaura con /rollback <n> [file] (añade --all para sobrescribir también tus propios cambios); previsualiza con /rollback diff <n>.',
  noDiff: 'No hay cambios desde ese punto de control.',
  rollbackFailed: (error: string) => `Falló la restauración: ${error}`,
  unknownError: 'error desconocido',
  restored: (hash: string) => (hash ? `Punto de control ${hash} restaurado.` : 'Punto de control restaurado.'),
  rewound: (count: string) => `Se retiraron ${count} mensaje(s) de esta conversación.`,
  reopenForRewind: 'Vuelve a abrir este chat para ver la conversación rebobinada.',
  agentsEmpty: 'No hay subagentes activos ni procesos en segundo plano en este chat.',
  subagentsHeader: (count: string) => `Subagentes (${count}):`,
  subagentFallback: 'subagente',
  statusRunning: 'en ejecución',
  toolCalls: (count: string) => `${count} llamadas a herramientas`,
  processesHeader: (count: string) => `Procesos en segundo plano (${count}):`,
  configHeader: 'Configuración (pantalla, aprobaciones, pegado y voz; no se muestra nada más):',
  configDefaults: '(valores predeterminados)',
  configFooter:
    'Cambia los ajustes en Ajustes, o edita config.yaml con `hermes config edit` en el host de la pasarela.',
  historyHeader: (count: string) => `Historial de la conversación (${count} mensajes):`,
  historyEmpty: 'Todavía no hay mensajes.',
  historyYou: 'Tú',
  historyHermes: 'Hermes',
  saveUsage: 'uso: /save [json|md|html] [filename] [redact]',
  saveLegacyRefused:
    'No guardado: esta pasarela no puede exportar una copia censurada, con nombre o que no sea JSON. Actualiza la pasarela o ejecuta /save sin opciones para escribir el JSON sin procesar en el host de la pasarela.',
  savedOnHost: (path: string) => `Guardado en el host de la pasarela: ${path}`,
  unknownPath: '(ruta desconocida)',
  saveUnsupported: 'error: esta versión de escritorio no puede guardar archivos',
  saveDialogTitle: 'Guardar conversación',
  savedTo: (path: string) => `Conversación guardada en ${path}`,
  saveCancelled: 'No guardado (cancelado).',
  exportRemote:
    '/export empaqueta un perfil desde el host de la pasarela, que no es este equipo. Ejecuta `hermes profile export` en ese host.',
  importRemote:
    '/import instala un perfil en el host de la pasarela, que no es este equipo. Ejecuta `hermes profile import <archive>` en ese host.',
  timestampsUsage: 'uso: /timestamps [on|off|status]',
  timestampsOn:
    'Marcas de tiempo: activadas (esta ventana; configura display.timestamps en config.yaml para conservarlo)',
  timestampsOff:
    'Marcas de tiempo: desactivadas (esta ventana; configura display.timestamps en config.yaml para conservarlo)'
}
