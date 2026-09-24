/** Whether the Desktop should expose managed local-model surfaces for a launch. */
export function localModelsEnabled(
  argv: readonly string[],
  platform: NodeJS.Platform = process.platform
): boolean {
  return argv.includes('--local') || platform === 'win32' || platform === 'darwin' || platform === 'linux'
}
