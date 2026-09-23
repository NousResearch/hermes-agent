import fs from 'node:fs'

import { resolveDirectoryForIpc } from './hardening'
import type { WindowConnectionRoute } from './window-connection-route'

interface FolderLinkDeps {
  getRoute: (senderId: number) => WindowConnectionRoute | null
  ensureBackend: (senderId: number) => Promise<{ mode?: string } | null>
  openPath: (path: string) => Promise<string>
  fs?: typeof fs
}

// Reject, rather than expand/normalize, syntax with another OS or shell meaning.
export function assertFolderPath(
  value: unknown,
  platform: NodeJS.Platform = process.platform
): asserts value is string {
  if (
    typeof value !== 'string' ||
    !value ||
    value !== value.trim() ||
    // eslint-disable-next-line no-control-regex -- Native paths must not contain control characters.
    /[\x00-\x1f\x7f]/.test(value) ||
    /^[\\/]{2}/.test(value)
  ) {
    throw new Error('An absolute local folder path is required.')
  }

  const windows = platform === 'win32'

  if (windows ? !/^[a-z]:[\\/]/i.test(value) : !value.startsWith('/') || value.includes('\\')) {
    throw new Error('Folder path does not match this computer.')
  }

  if (
    windows &&
    (/[<>:"|?*]/.test(value.slice(2)) ||
      value
        .slice(3)
        .split(/[\\/]/)
        .some(
          segment => /[. ]$/.test(segment) || /^(?:con|prn|aux|nul|com[1-9¹²³]|lpt[1-9¹²³])(?:\.|$)/i.test(segment)
        ))
  ) {
    throw new Error('Unsupported Windows folder path.')
  }

  if (/\.app(?:[\\/]|$)/i.test(value)) {
    throw new Error('Application bundles cannot be opened as folders.')
  }
}

export async function openExistingDirectory(senderId: number, request: unknown, deps: FolderLinkDeps) {
  try {
    const input = request as { path?: unknown; owner?: { connectionId?: unknown; profile?: unknown } }
    const route = deps.getRoute(senderId)

    if (
      !route?.connectionId ||
      typeof input?.owner?.profile !== 'string' ||
      !input.owner.profile ||
      input.owner.connectionId !== route.connectionId ||
      input.owner.profile !== (route.profile ?? 'default')
    ) {
      throw new Error('Folder links require the current session’s exact local connection.')
    }

    const assertLocalRoute = async () => {
      if (deps.getRoute(senderId) !== route) {
        throw new Error('Connection changed. Try again.')
      }

      const backend = await deps.ensureBackend(senderId)

      if (deps.getRoute(senderId) !== route || backend?.mode !== 'local') {
        throw new Error('Folder links are only available on a confirmed local connection.')
      }
    }

    await assertLocalRoute()
    assertFolderPath(input?.path)
    const { realPath } = await resolveDirectoryForIpc(input.path, { fs: deps.fs ?? fs, purpose: 'Open folder' })
    assertFolderPath(realPath)

    if (!(await (deps.fs ?? fs).promises.stat(realPath)).isDirectory()) {
      throw new Error('Path is not a directory.')
    }

    await assertLocalRoute()
    const error = await deps.openPath(realPath)

    return error ? { ok: false, error } : { ok: true }
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : String(error) }
  }
}
