// Electron wraps a rejected `ipcMain.handle` with the internal channel name
// and the main-process error class. Keep only the actionable message.
const IPC_ERROR_PREFIX_RE = /^Error invoking remote method '[^']+':\s*(?:(?:[A-Za-z_$][\w$]*Error|Error):\s*)?/

export function stripIpcErrorPrefix(message: string): string {
  const stripped = message.replace(IPC_ERROR_PREFIX_RE, '').trim()

  return stripped || message
}
