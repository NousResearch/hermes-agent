// pen.dev host discovery — the Electron-free half of the Pen canvas
// integration. Locates the user's installed pen.dev desktop app and exposes
// the paths hermes borrows from it at runtime:
//
//   - `out/editor/`   — the full canvas editor web bundle (served over the
//                       hermes-pen:// protocol into our <webview>)
//   - `@ha/*` modules — the host-side IPC/device libraries (required straight
//                       from the asar; they are plain CJS)
//   - `out/mcp-server-<platform>` — the MCP stdio binary the agent uses to
//                       drive a live canvas over the pencil socket
//   - `out/data/*.pen` — the document templates (blank canvas, design kits)
//
// Nothing is vendored: pen.dev updates itself, and hermes always hosts
// whatever version the user has installed. When Pen isn't installed the
// integration reports unavailable and every door stays closed.

import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

/** The socket app name hermes registers with. `pen interactive -a hermes` and
 *  `mcp-server --app hermes` both resolve `~/.pencil/socket/pencil-hermes.sock`
 *  from this name (see @ha/ipc getSocketPath). */
export const PEN_SOCKET_APP_NAME = 'hermes'

export interface PenInstallation {
  /** /Applications/Pen.app */
  appPath: string
  /** …/Contents/Resources/app.asar */
  asarPath: string
  /** …/Contents/Resources/app.asar.unpacked — folderPath for getMcpConfiguration */
  unpackedPath: string
  /** …/app.asar/out/editor — the editor web bundle root */
  editorRoot: string
  /** …/app.asar/out/data — .pen templates */
  templatesRoot: string
  /** …/app.asar.unpacked/out/mcp-server-<platform> */
  mcpServerPath: string
  /** Pen.app bundle version (best effort, '' when unreadable) */
  version: string
}

function mcpBinaryName(): string {
  const arch = process.arch === 'arm64' ? 'arm64' : 'x64'

  if (process.platform === 'win32') {
    return `mcp-server-windows-${arch}.exe`
  }

  if (process.platform === 'darwin') {
    return `mcp-server-darwin-${arch}`
  }

  return `mcp-server-linux-${arch}`
}

/** Candidate install locations, most specific first. */
function penAppCandidates(): string[] {
  if (process.platform === 'darwin') {
    return [
      path.join('/Applications', 'Pen.app'),
      path.join(os.homedir(), 'Applications', 'Pen.app')
    ]
  }

  // Linux/Windows installs land in per-user dirs; resources sit beside the
  // executable. Not wired yet — macOS is the only host we probe today.
  return []
}

function readBundleVersion(appPath: string): string {
  try {
    const plist = fs.readFileSync(path.join(appPath, 'Contents', 'Info.plist'), 'utf8')
    const match = plist.match(/<key>CFBundleShortVersionString<\/key>\s*<string>([^<]+)<\/string>/)

    return match?.[1] ?? ''
  } catch {
    return ''
  }
}

/** The editor bundle to serve, as an ordered ladder:
 *
 *    1. HERMES_PEN_EDITOR_ROOT        — explicit override, dev/testing
 *    2. ~/.hermes/pen/editor          — the BLESSED bundle: pen.dev hands
 *                                       partners the editor as a plain static
 *                                       dir (their pen-plugin zip); a symlink
 *                                       here decouples us from Pen.app's asar
 *                                       and from needing Pen.app at all for
 *                                       the canvas surface.
 *    3. Pen.app's asar (out/editor)   — the original recon path, still the
 *                                       floor so nothing regresses without
 *                                       the plugin.
 *
 *  A candidate is trusted only if its index.html actually exists — existence
 *  of the directory is not proof (dangling symlink, half-unzipped bundle). */
function resolveEditorRoot(asarEditorRoot: string): string {
  const candidates = [
    process.env.HERMES_PEN_EDITOR_ROOT || '',
    path.join(os.homedir(), '.hermes', 'pen', 'editor')
  ]

  for (const candidate of candidates) {
    if (candidate && fs.existsSync(path.join(candidate, 'index.html'))) {
      return candidate
    }
  }

  return asarEditorRoot
}

/** Locate the installed pen.dev desktop app, or null. Cheap enough to call on
 *  demand; existence is validated at every layer that relies on it. */
export function findPenInstallation(): PenInstallation | null {
  for (const appPath of penAppCandidates()) {
    const resources = path.join(appPath, 'Contents', 'Resources')
    const asarPath = path.join(resources, 'app.asar')
    const unpackedPath = path.join(resources, 'app.asar.unpacked')
    const mcpServerPath = path.join(unpackedPath, 'out', mcpBinaryName())

    // fs sees INTO the asar from Electron (asar support is patched into fs),
    // but the asar file itself is a real file for existsSync.
    if (!fs.existsSync(asarPath) || !fs.existsSync(mcpServerPath)) {
      continue
    }

    return {
      appPath,
      asarPath,
      unpackedPath,
      editorRoot: resolveEditorRoot(path.join(asarPath, 'out', 'editor')),
      templatesRoot: path.join(asarPath, 'out', 'data'),
      mcpServerPath,
      version: readBundleVersion(appPath)
    }
  }

  return null
}

/** Require one of Pen's host-side CJS modules straight out of its asar.
 *  Electron's patched `require`/`fs` read archive members transparently, so
 *  `@ha/ipc`, `@ha/shared`, and `@node-ipc/*` (all plain CJS dists) load as if
 *  they were on disk. The caller owns error handling — a Pen update could in
 *  principle move these, and the integration must degrade to "unavailable",
 *  never crash the app. */
export function requirePenModule(install: PenInstallation, modulePath: string): any {
   
  return require(path.join(install.asarPath, 'node_modules', modulePath))
}

/** Pen's session file — shared with Pen.app so one login covers both. */
export function penSessionFilePath(): string {
  return path.join(os.homedir(), '.pencil', 'session-desktop.json')
}

/** True when a pen.dev login token exists (Pen.app or `pen login`). */
export function penLoggedIn(): boolean {
  try {
    const session = JSON.parse(fs.readFileSync(penSessionFilePath(), 'utf8'))

    return Boolean(session?.email && session?.token)
  } catch {
    return false
  }
}

/** Where hermes keeps its temporary (unsaved) canvas documents. Mirrors Pen's
 *  own ~/.pencil/documents/<uuid>/ layout — same folder family, so pen.dev's
 *  recents/cleanup conventions treat them like any other temporary doc. */
export function penTemporaryDocumentsRoot(): string {
  return path.join(os.homedir(), '.pencil', 'documents')
}

/** The canvas LIBRARY — where hermes's own canvases live so they can be
 *  browsed, reopened, renamed, and deleted like any other artifact.
 *
 *  A temporary document (pen's ~/.pencil/documents/<uuid>/) is invisible and
 *  effectively disposable: nothing lists it and a restart strands it. Canvases
 *  created in hermes land here instead, one folder per canvas, so "my pens"
 *  is a real place on disk the user can also open in Pen.app or put in git. */
export function penLibraryRoot(): string {
  return path.join(os.homedir(), '.hermes', 'pens')
}

export interface PenLibraryEntry {
  /** Absolute path to the .pen file. */
  path: string
  /** Folder holding the .pen (deleting a canvas removes this). */
  folder: string
  name: string
  modifiedAt: number
  size: number
  /** Rendered canvas preview (preview.png beside the .pen, written from the
   *  editor's save-preview pushes), when one exists. */
  previewPath: null | string
}

/** Every canvas in the library, newest first. One shallow scan — the library
 *  is one folder per canvas, not an arbitrary tree. */
export function listPenLibrary(): PenLibraryEntry[] {
  const root = penLibraryRoot()
  const entries: PenLibraryEntry[] = []

  let folders: string[]

  try {
    folders = fs.readdirSync(root)
  } catch {
    return entries
  }

  for (const folder of folders) {
    const dir = path.join(root, folder)

    let files: string[]

    try {
      if (!fs.statSync(dir).isDirectory()) {
        continue
      }

      files = fs.readdirSync(dir)
    } catch {
      continue
    }

    for (const file of files) {
      if (!file.endsWith('.pen')) {
        continue
      }

      const full = path.join(dir, file)

      try {
        const stat = fs.statSync(full)
        const preview = path.join(dir, 'preview.png')

        entries.push({
          path: full,
          folder: dir,
          name: file.replace(/\.pen$/, ''),
          modifiedAt: stat.mtimeMs,
          size: stat.size,
          previewPath: fs.existsSync(preview) ? preview : null
        })
      } catch {
        // A canvas being written right now — skip it this pass.
      }
    }
  }

  return entries.sort((a, b) => b.modifiedAt - a.modifiedAt)
}

/** Reserve a path in the library for a new canvas. Never overwrites: a name
 *  collision gets a numeric suffix, the way a file manager would. */
export function penLibraryPathFor(name: string): string {
  const safe =
    name
      .replace(/\.pen$/i, '')
      .replace(/[/\\:*?"<>|]/g, '-')
      .trim() || 'Untitled'

  const root = penLibraryRoot()

  for (let n = 0; n < 1000; n += 1) {
    const candidate = n === 0 ? safe : `${safe} ${n + 1}`
    const folder = path.join(root, candidate)

    if (!fs.existsSync(folder)) {
      return path.join(folder, `${candidate}.pen`)
    }
  }

  return path.join(root, `${safe} ${Date.now()}`, `${safe}.pen`)
}

/** Delete a canvas — the whole folder, since that's what a canvas IS on disk.
 *  Refuses anything outside the library so a bad path can't take a user
 *  directory with it. */
export function deletePenFromLibrary(target: string): boolean {
  const root = penLibraryRoot()
  const folder = target.endsWith('.pen') ? path.dirname(target) : target
  const resolved = path.resolve(folder)

  if (!resolved.startsWith(path.resolve(root) + path.sep)) {
    return false
  }

  try {
    fs.rmSync(resolved, { recursive: true, force: true })

    return true
  } catch {
    return false
  }
}

/** Rename a canvas: the .pen and its folder move together, so the library
 *  stays one-folder-per-canvas. Returns the new path. */
export function renamePenInLibrary(target: string, nextName: string): null | string {
  const root = path.resolve(penLibraryRoot())
  const current = path.resolve(target)

  if (!current.startsWith(root + path.sep) || !fs.existsSync(current)) {
    return null
  }

  const destination = penLibraryPathFor(nextName)

  try {
    fs.mkdirSync(path.dirname(destination), { recursive: true })
    fs.renameSync(current, destination)

    const oldFolder = path.dirname(current)

    // Drop the now-empty folder the canvas came from.
    try {
      if (fs.readdirSync(oldFolder).length === 0) {
        fs.rmdirSync(oldFolder)
      }
    } catch {
      // Leftover siblings — leave the folder alone.
    }

    return destination
  } catch {
    return null
  }
}
