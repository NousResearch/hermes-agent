/**
 * Pure derivation of how the app names an update target: the label
 * (`v0.4.2`, `backend v0.4.2 (+12)`, `v0.4.2 · update`), its tooltip, and
 * whether an update is waiting.
 *
 * The statusbar and the command palette both name the same two targets, so the
 * wording lives here once — a palette row and its statusbar item can't drift
 * into describing the same install differently.
 */

import type { UpdateTarget } from '@/lib/update-copy'

export interface VersionStatusCopy {
  backendLabel: (version: string) => string
  backendVersion: (version: string) => string
  branch: (branch: string) => string
  clientLabel: (version: string) => string
  commit: (sha: string) => string
  commitsBehind: (count: number, branch: string) => string
  desktopVersion: (version: string) => string
  restart: string
  unknown: string
  update: string
  updateInProgress: string
}

export interface VersionStatusInput {
  /** True while an apply is in flight (including the restart hand-off). */
  applying: boolean
  /** Latest line from the apply stream — leads the tooltip while applying. */
  applyMessage?: string
  behind?: number
  branch?: string
  copy: VersionStatusCopy
  /** Remote mode: the client is one of two versions on screen, so it says so. */
  remote: boolean
  /** The apply reached the restart stage — labels `restart`, not `update`. */
  restarting: boolean
  /** Client only: short commit sha of the running build. */
  sha?: null | string
  target: UpdateTarget
  /** An update the commit count can't express (shallow clones, pip installs). */
  updateAvailable?: boolean
  version?: null | string
}

export interface VersionStatusResult {
  /** Secondary text beside the label — the commit sha, when it adds anything. */
  detail?: string
  /** An update is waiting: callers tint the row with it. */
  hasUpdate: boolean
  label: string
  tooltip?: string
  /** Nothing identifies this target yet — callers hide the row. */
  unknown: boolean
}

export function resolveVersionStatus({
  applyMessage,
  applying,
  behind = 0,
  branch,
  copy,
  remote,
  restarting,
  sha = null,
  target,
  updateAvailable,
  version = null
}: VersionStatusInput): VersionStatusResult {
  const client = target === 'client'
  const busy = applying || restarting
  // updateAvailable covers every "behind but uncountable" shape: shallow
  // installer clones (behind === null upstream, coalesced to 0 by callers),
  // SSH-official presence-only checks, and pip installs. It applies to BOTH
  // targets — the client statusbar item is how a shallow desktop install
  // learns it's stale at all.
  const available = behind > 0 || !!updateAvailable

  // A client with no version still identifies itself by sha; a backend can't.
  const named = version ?? (client ? sha : null) ?? copy.unknown

  const base = !client
    ? copy.backendLabel(named)
    : remote
      ? copy.clientLabel(named)
      : (version && `v${version}`) || named

  // Commits behind is the precise diff; `(update)` is the fallback for a
  // backend that knows it's stale but can't count (pip, non-git checkout).
  const hint = busy ? '' : behind > 0 ? ` (+${behind})` : available ? ` (${copy.update})` : ''

  const tooltip = [
    busy && (applyMessage || copy.updateInProgress),
    !busy && behind > 0 && copy.commitsBehind(behind, (client ? branch : 'main') || '...'),
    !busy && behind <= 0 && available && copy.update,
    version && (client ? copy.desktopVersion(version) : copy.backendVersion(version)),
    client && sha && copy.commit(sha),
    client && branch && copy.branch(branch)
  ]
    .filter(Boolean)
    .join(' · ')

  return {
    detail: client && version && sha && !busy && !remote ? sha : undefined,
    hasUpdate: !busy && available,
    label: busy ? `${base} · ${restarting ? copy.restart : copy.update}` : `${base}${hint}`,
    tooltip: tooltip || undefined,
    unknown: !version && !(client && sha)
  }
}
