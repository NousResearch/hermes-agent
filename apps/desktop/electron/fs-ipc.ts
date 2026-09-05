// IPC surface for local filesystem operations the renderer's project/file
// surfaces use: directory reads, reveal/open in the OS file manager, plugin
// roots + git installs, rename/write/trash. Extracted from main.ts; path
// hardening, HERMES_HOME resolution, and the git binary stay injected.
import fs from 'node:fs'
import path from 'node:path'

import { ipcMain, shell } from 'electron'

import { installDesktopPluginFromGit, probePluginRepo } from './desktop-plugin-install'
import { readDirForIpc } from './fs-read-dir'
import { gitRootForIpc } from './git-root'

export interface FsIpcDeps {
  hermesHome: string
  readActiveDesktopProfile: () => null | string
  expandUserPath: (value: string) => string
  resolveRequestedPathForIpc: (value: string, options: { purpose: string }) => string
  directoryExists: (value: string) => boolean
  resolveGitBinary: () => string
}

export function registerFsIpc({
  hermesHome,
  readActiveDesktopProfile,
  expandUserPath,
  resolveRequestedPathForIpc,
  directoryExists,
  resolveGitBinary
}: FsIpcDeps) {
  ipcMain.handle('hermes:fs:readDir', async (_event, dirPath) => readDirForIpc(dirPath))

  ipcMain.handle('hermes:fs:gitRoot', async (_event, startPath) => gitRootForIpc(startPath))

  // Reveal a path in the OS file manager (Finder / Explorer / Files).
  ipcMain.handle('hermes:fs:reveal', async (_event, targetPath) => {
    const target = String(targetPath || '').trim()

    if (!target) {
      return false
    }

    try {
      shell.showItemInFolder(target)

      return true
    } catch {
      return false
    }
  })

  // Open a DIRECTORY in the OS file manager, creating it first if needed. Unlike
  // `reveal` (which selects an existing item and silently no-ops on a missing
  // path — the "Open plugins folder" Windows bug), this is for the plugins door,
  // which often doesn't exist on first use. `shell.openPath` returns '' on
  // success or an error string; both mkdir + openPath failures are surfaced.
  ipcMain.handle('hermes:fs:openDir', async (_event, dirPath) => {
    const dir = String(dirPath || '').trim()

    if (!dir) {
      return { ok: false, error: 'no path' }
    }

    try {
      await fs.promises.mkdir(dir, { recursive: true })
      const error = await shell.openPath(path.normalize(dir))

      return error ? { ok: false, error } : { ok: true }
    } catch (error) {
      return { ok: false, error: error instanceof Error ? error.message : String(error) }
    }
  })

  // The LOCAL Desktop runtime-plugin root: `<HERMES_HOME>/desktop-plugins`,
  // resolved from the main-process HERMES_HOME (see resolveHermesHome) — NOT from
  // the connected backend. A remote backend reports its own `hermes_home` over
  // the gateway, which is a path on the REMOTE box; deriving the plugin dir from
  // it yields `undefined/desktop-plugins` (or a non-existent remote path) and the
  // on-disk plugin door silently breaks (#66899). Electron owns this resolution
  // so it stays valid in every connection mode. Created on demand, like openDir.
  async function localPluginsRoot(dirName: string): Promise<string> {
    // Profile-aware: a named Desktop profile gets its own plugin root under
    // profiles/<name>/, matching the profile-scoped hermes_home the backend
    // reported before this resolver existed. 'default'/unset pins the global root.
    const profile = readActiveDesktopProfile()
    const base = profile && profile !== 'default' ? path.join(hermesHome, 'profiles', profile) : hermesHome
    const dir = path.join(base, dirName)

    try {
      await fs.promises.mkdir(dir, { recursive: true })
    } catch {
      // Best-effort create; return the path regardless so the reveal action can
      // still surface a real openPath error and the scanner can retry later.
    }

    return dir
  }

  ipcMain.handle('hermes:fs:desktopPluginsRoot', async () => localPluginsRoot('desktop-plugins'))

  // The LOCAL logs root (`<HERMES_HOME>/logs`, profile-aware) — the error
  // card's "Open Logs" action reveals agent.log/gateway.log without the user
  // knowing where HERMES_HOME lives. Same Electron-local resolution as the
  // plugin roots: valid in every connection mode, created on demand.
  ipcMain.handle('hermes:fs:logsRoot', async () => localPluginsRoot('logs'))

  // The LOCAL agent-plugin root (`<HERMES_HOME>/plugins`), same Electron-local
  // resolution as above. This is the desktop half of a UNIFIED plugin package:
  // an agent plugin may ship `desktop/plugin.js` alongside its Python code (the
  // same shape as `dashboard/manifest.json`), and the renderer's disk door scans
  // this root for it — one installable folder serving both SDKs.
  ipcMain.handle('hermes:fs:agentPluginsRoot', async () => localPluginsRoot('plugins'))

  ipcMain.handle('hermes:plugin:probe', async (_event, payload) => {
    const identifier = String(payload?.identifier || payload?.repo || '').trim()

    if (!identifier) {
      return { ok: false, error: 'identifier is required', agent: false, desktop: false, warnings: [] }
    }

    return probePluginRepo(resolveGitBinary(), identifier)
  })

  ipcMain.handle('hermes:plugin:installDesktop', async (_event, payload) => {
    const identifier = String(payload?.identifier || payload?.repo || '').trim()

    if (!identifier) {
      return { ok: false, error: 'identifier is required' }
    }

    const desktopPluginsRoot = await localPluginsRoot('desktop-plugins')

    return installDesktopPluginFromGit(resolveGitBinary(), identifier, desktopPluginsRoot, Boolean(payload?.force))
  })

  // Rename a file/folder in place. The renderer passes the existing path + a new
  // base name; the destination is resolved in the SAME parent dir so a rename can
  // never move the item elsewhere or traverse out. Rejects on a name collision.
  ipcMain.handle('hermes:fs:rename', async (_event, targetPath, newName) => {
    const src = String(targetPath || '').trim()
    const name = String(newName || '').trim()

    if (!src || !name || name === '.' || name === '..' || name.includes('/') || name.includes('\\')) {
      throw new Error('Invalid rename')
    }

    const dst = path.join(path.dirname(src), name)

    if (dst === src) {
      return { path: dst }
    }

    if (fs.existsSync(dst)) {
      throw new Error(`"${name}" already exists`)
    }

    await fs.promises.rename(src, dst)

    return { path: dst }
  })

  // Write a small UTF-8 text file (e.g. a project's IDEA.md at creation). The path
  // is hardened (resolveRequestedPathForIpc) and the parent must already exist —
  // this never creates directory trees or escapes the allowed roots, and content
  // is size-capped so it can't be abused as a bulk-write primitive.
  ipcMain.handle('hermes:fs:writeText', async (_event, filePath, content) => {
    const raw = String(filePath || '').trim()

    if (!raw) {
      throw new Error('Invalid path')
    }

    const text = String(content ?? '')

    if (text.length > 1_000_000) {
      throw new Error('Content too large')
    }

    const resolved = resolveRequestedPathForIpc(expandUserPath(raw), { purpose: 'Write text file' })

    if (!directoryExists(path.dirname(resolved))) {
      throw new Error('Parent directory does not exist')
    }

    await fs.promises.writeFile(resolved, text, 'utf8')

    return { path: resolved }
  })

  // Move a file/folder to the OS trash (recoverable) — the VS Code "Delete"
  // default. `shell.trashItem` routes to Finder/Explorer/Files trash per platform.
  ipcMain.handle('hermes:fs:trash', async (_event, targetPath) => {
    const target = String(targetPath || '').trim()

    if (!target) {
      throw new Error('Invalid delete')
    }

    await shell.trashItem(target)

    return true
  })
}
