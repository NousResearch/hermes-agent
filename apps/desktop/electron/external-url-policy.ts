const SHELL_EXTERNAL_PROTOCOLS = new Set(['http:', 'https:', 'mailto:', 'obsidian:'])

/**
 * Protocols Electron may hand to the operating system after a user click.
 * Keep this list deliberately narrow: custom schemes can launch applications.
 */
export function isAllowedShellExternalProtocol(protocol: string): boolean {
  return SHELL_EXTERNAL_PROTOCOLS.has(protocol.toLowerCase())
}
