import { isDesktopFsRemoteMode, readDesktopDir } from '@/lib/desktop-fs'
import { localPreviewTarget } from '@/lib/local-preview'

/**
 * Resolve a transcript link to an existing directory on the Desktop machine.
 * Remote-backend paths deliberately stay out of this path: their filesystem is
 * not the Electron host filesystem and must keep using gateway-backed actions.
 */
export async function existingLocalDirectoryPath(rawTarget: string, cwd?: string | null): Promise<string | null> {
  if (isDesktopFsRemoteMode()) {
    return null
  }

  const target = localPreviewTarget(rawTarget, cwd)

  if (!target || target.kind !== 'file' || !target.path) {
    return null
  }

  // URL.pathname yields /C:/... for Windows file URLs. Electron filesystem
  // IPC expects the native drive path instead.
  const path = /^\/[a-z]:\//i.test(target.path) ? target.path.slice(1) : target.path

  try {
    await readDesktopDir(path)

    return path
  } catch {
    return null
  }
}
