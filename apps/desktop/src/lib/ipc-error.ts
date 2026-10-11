// Electron wraps rejected IPC calls with the internal channel and error class.
const IPC_ERROR_PREFIX_RE = /^Error invoking remote method '[^']+':\s*(?:(?:[A-Za-z_$][\w$]*Error|Error):\s*)?/i

export function stripIpcErrorPrefix(message: string): string {
  const stripped = message.trim().replace(IPC_ERROR_PREFIX_RE, '').trim()

  return stripped || message
}
