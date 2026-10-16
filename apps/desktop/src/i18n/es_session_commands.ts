// Typed /undo /retry /steer /queue lines (en_session_commands.ts is the source).
export const esSessionCommands = {
  undoUsage: 'uso: /undo [N]',
  undoNothing: 'nada que deshacer',
  undidOne: (count: string) => `se deshizo ${count} mensaje — el mensaje eliminado está en el compositor`,
  undidOther: (count: string) => `se deshicieron ${count} mensajes — el mensaje eliminado está en el compositor`,
  retryNothing: 'nada que reintentar',
  retryUnsupported: '/retry: no se pueden reintentar contenidos multimedia o partes desconocidas',
  steerUsage: 'uso: /steer <prompt>',
  steerQueued: (preview: string) => `indicación en cola — llega tras la próxima llamada a herramienta: "${preview}"`,
  steerNoActiveTurn: (preview: string) => `no hay turno activo — en cola para el siguiente: "${preview}"`,
  steerRejected: 'indicación rechazada — no hay turno activo, en cola para el siguiente turno',
  queued: (preview: string) => `en cola: "${preview}"`,
  queueFailed: 'No se pudo poner el mensaje en cola',
  countOne: (count: string) => `${count} mensaje en cola`,
  countOther: (count: string) => `${count} mensajes en cola`
}
