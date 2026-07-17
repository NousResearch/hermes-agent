// Classify WHY the Windows staged updater (HERMES_HOME/hermes-setup.exe) is
// missing when the user hits "Update", so the manual-fallback dialog doesn't
// tell installer users "You installed Hermes from the command line" (#66095).
//
// Two installer-deployed layouts can reach the manual fallback:
//   - The electron-builder NSIS/MSI desktop installer puts Hermes.exe OUTSIDE
//     the managed checkout (e.g. %LOCALAPPDATA%\Programs\Hermes) and its
//     first-launch bootstrap never stages hermes-setup.exe — that binary is
//     only staged by the Tauri Hermes-Setup installer.
//   - The Tauri Hermes-Setup installer normally stages hermes-setup.exe, but
//     the copy is best-effort (paths::copy_self_to_hermes_home); when it
//     failed, the only remaining breadcrumb is the log it writes to
//     HERMES_HOME/logs/bootstrap-installer.log.
// A genuine CLI install (`hermes desktop`) launches the packed exe from
// <checkout>/apps/desktop/release/*-unpacked/, i.e. INSIDE the update root.
//
// Only the Windows manual fallback consults this (POSIX takes the in-app
// update path), so path containment deliberately uses win32 semantics:
// case-insensitive, both separators. Pure so it unit-tests without Electron.

import path from 'node:path'

export type DesktopInstallKind = 'installer' | 'cli'

function isInsideWindowsPath(parent: string, child: string): boolean {
  const rel = path.win32.relative(parent.toLowerCase(), child.toLowerCase())

  return rel !== '' && !rel.startsWith('..') && !path.win32.isAbsolute(rel)
}

/**
 * Decide which manual-update message the Windows fallback should show.
 *
 * @param execPath        process.execPath of the running desktop app
 * @param updateRoot      resolveUpdateRoot() — the managed hermes-agent checkout
 * @param isPackaged      IS_PACKAGED; dev runs launch node_modules' electron.exe
 *                        (outside any checkout) and must not misclassify
 * @param hasInstallerLog whether HERMES_HOME/logs/bootstrap-installer.log exists
 *                        (only the Tauri Hermes-Setup installer writes it)
 */
export function classifyWindowsManualUpdate({
  execPath,
  updateRoot,
  isPackaged,
  hasInstallerLog
}: {
  execPath: string
  updateRoot: string
  isPackaged: boolean
  hasInstallerLog: boolean
}): DesktopInstallKind {
  if (hasInstallerLog) {
    return 'installer'
  }

  if (isPackaged && execPath && updateRoot && !isInsideWindowsPath(updateRoot, execPath)) {
    return 'installer'
  }

  return 'cli'
}
