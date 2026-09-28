/** When a composer attachment must cross as BYTES rather than ride its path.
 *
 * Shared by the submit-time upload pipeline (`uploadComposerAttachment`) and
 * the drop-time routing decision (`partitionDroppedFiles`) so the two never
 * drift apart. Pure: takes the path and backend facts as arguments.
 */

import { isWindowsAbsolutePath } from '@/lib/path-compare'

const POSIX_ABSOLUTE_PATH_RE = /^\/(?!\/)/

// Terminal backends whose execution environment has its own filesystem
// (docker/ssh/singularity/modal/...) cannot see the desktop's host paths —
// they must be crossed as bytes, like remote attachments. Mirrors the
// container_backend set in tools/terminal_tool.py::_get_env_config.
export const CONTAINER_TERMINAL_BACKENDS = new Set([
  'docker',
  'ssh',
  'singularity',
  'modal',
  'daytona',
  'vercel_sandbox',
  'apple_container'
])

// `mode: local` means the gateway was launched locally, not necessarily that
// Electron and the gateway share a filesystem. Windows Desktop can front a
// WSL/Docker backend whose cwd is POSIX, so a Windows host path must cross the
// boundary as bytes just like a remote attachment. Container terminal backends
// (docker, ssh, ...) always need bytes: the sandbox has its own filesystem and
// the host path would dangle inside it (#76577).
export function attachmentPathNeedsUpload(
  path: string,
  backendCwd?: null | string,
  terminalBackend?: null | string
): boolean {
  if (CONTAINER_TERMINAL_BACKENDS.has((terminalBackend || '').trim().toLowerCase())) {
    return true
  }

  return isWindowsAbsolutePath(path.trim()) && POSIX_ABSOLUTE_PATH_RE.test(backendCwd?.trim() || '')
}
