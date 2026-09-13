/**
 * Mark the CLI leg of the Electron POSIX updater so `hermes update` leaves the
 * running app bundle alone. Electron rebuilds and swaps that bundle after the
 * CLI update returns; a standalone CLI update remains responsible for syncing
 * an installed macOS app.
 */
function electronOwnedUpdateEnvironment(env = {}) {
  return { ...env, HERMES_DESKTOP_UPDATE_OWNER: 'electron' }
}

export { electronOwnedUpdateEnvironment }
