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
  'vercel_sandbox'
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

/** The slice of a composer attachment the upload-budget preflight looks at. */
export interface ByteUploadCandidate {
  kind: string
  label?: string
  path?: string
}

export interface AttachmentUploadLimits {
  maxFileBytes: number
  maxTotalBytes: number
}

export type AttachmentBudgetVerdict =
  | { ok: true; totalBytes: number }
  | { ok: false; reason: 'file'; label: string; sizeBytes: number; limitBytes: number }
  | { ok: false; reason: 'total'; totalBytes: number; limitBytes: number }

const mib = (bytes: number) => Math.max(1, Math.round(bytes / (1024 * 1024)))

/**
 * Pure size rule for one send's byte uploads. Every one of those files is
 * base64-loaded into the renderer, so an oversized file — or several files that
 * are each fine but together are not — can abort the renderer. Unsized entries
 * (a stat error) count as 0 here: the reader surfaces their real error later.
 */
export function checkAttachmentUploadBudget(
  files: ReadonlyArray<{ label: string; size?: number }>,
  limits: AttachmentUploadLimits
): AttachmentBudgetVerdict {
  let totalBytes = 0

  for (const file of files) {
    const size = Number.isFinite(file.size) && Number(file.size) > 0 ? Number(file.size) : 0

    if (limits.maxFileBytes > 0 && size > limits.maxFileBytes) {
      return { ok: false, reason: 'file', label: file.label, sizeBytes: size, limitBytes: limits.maxFileBytes }
    }

    totalBytes += size
  }

  if (limits.maxTotalBytes > 0 && totalBytes > limits.maxTotalBytes) {
    return { ok: false, reason: 'total', totalBytes, limitBytes: limits.maxTotalBytes }
  }

  return { ok: true, totalBytes }
}

export function attachmentBudgetErrorMessage(verdict: Exclude<AttachmentBudgetVerdict, { ok: true }>): string {
  return verdict.reason === 'file'
    ? `${verdict.label} is too large to upload to the remote gateway (${mib(verdict.sizeBytes)} MB; max ${mib(verdict.limitBytes)} MB).`
    : `These attachments total ${mib(verdict.totalBytes)} MB, over the ${mib(verdict.limitBytes)} MB upload limit for one message. Send them in separate messages.`
}

type StatFilesForAttach = (filePaths: string[]) => Promise<{
  limits: AttachmentUploadLimits
  files: Array<{ path: string; size?: number; error?: string }>
}>

/**
 * Size the given byte-upload candidates (stat only — no bytes are read) and
 * throw a user-facing Error when one file or the batch is over the limit.
 * Callers pass ONLY attachments that will actually cross as bytes. Older
 * shells without the stat bridge skip the batch check; the main-process
 * per-file cap on the attach reader still applies there.
 */
export async function assertAttachmentUploadBudget(
  candidates: ReadonlyArray<ByteUploadCandidate>,
  stat: StatFilesForAttach | undefined = typeof window === 'undefined'
    ? undefined
    : window.hermesDesktop?.statFilesForAttach
): Promise<void> {
  const sized = candidates.filter(
    candidate => (candidate.kind === 'file' || candidate.kind === 'image') && Boolean(candidate.path)
  )

  if (sized.length === 0 || !stat) {
    return
  }

  const report = await stat(sized.map(candidate => candidate.path as string))
  const sizeByPath = new Map(report.files.map(file => [file.path, file.size]))

  const verdict = checkAttachmentUploadBudget(
    sized.map(candidate => ({
      label: candidate.label || (candidate.path as string).split(/[\\/]/).pop() || 'attachment',
      size: sizeByPath.get(candidate.path as string)
    })),
    report.limits
  )

  if (!verdict.ok) {
    throw new Error(attachmentBudgetErrorMessage(verdict))
  }
}
