/** Enables Chromium's renderer accessibility tree for macOS assistive tools. */
export function enableMacRendererAccessibility(
  platform: NodeJS.Platform,
  app: Pick<Electron.App, 'setAccessibilitySupportEnabled'>
) {
  if (platform === 'darwin') {
    app.setAccessibilitySupportEnabled(true)
  }
}
