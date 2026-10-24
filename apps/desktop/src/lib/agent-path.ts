import { hermesApi } from '@/api/client'
import { desktopFsProfile, isDesktopFsRemoteMode } from '@/lib/desktop-fs'

/** Host path behind a docker agent's container path (`/workspace/x.md`), or null.
 *  Local mode reads through Electron, which can't see docker mounts; the local
 *  backend owns the `docker_volumes` mapping (#143098). Remote mode already
 *  reads through the backend, which translates on its own. */
export async function resolveAgentHostPath(path: string): Promise<null | string> {
  if (isDesktopFsRemoteMode() || !path.startsWith('/') || path.startsWith('//')) {
    return null
  }

  try {
    const result = await hermesApi<{ path?: string }>({
      path: `/api/fs/resolve-agent-path?path=${encodeURIComponent(path)}`,
      profile: desktopFsProfile()
    })

    return result?.path && result.path !== path ? result.path : null
  } catch {
    return null
  }
}
