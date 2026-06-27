// Pure formatting helpers for the install + boot-failure overlays. Kept out of
// the .tsx files so they can be unit-tested without a React/jsdom render
// (mirrors boot-failure-reauth.ts).

// Title-case a raw stage id as a last-resort fallback when no localized label
// exists. 'system-packages' -> 'System packages'; 'uv' stays 'uv'.
export function formatStageName(name: string): string {
  if (name.length <= 3) {
    return name
  }

  return name
    .split('-')
    .map((word, i) => (i === 0 ? word.charAt(0).toUpperCase() + word.slice(1) : word))
    .join(' ')
}

// Resolve a localized label for an installer stage id, falling back to
// formatStageName() for ids the catalog doesn't cover. The bootstrap protocol
// emits different stage ids per platform (install.ps1 splits prerequisites into
// uv/python/git/node/system-packages and uses dependencies/config-templates/
// configure/bootstrap-marker; install.sh uses prerequisites/python-deps/config/
// setup/complete), so the label catalog keys both naming schemes.
export function stageLabel(name: string, labels: Record<string, string>): string {
  return labels[name] ?? formatStageName(name)
}

// Map a raw bootstrap error string to a friendly, user-facing message. The raw
// transcript is preserved separately (the overlay keeps it behind the "show
// recent logs" expander); this only decides the primary line the user reads.
//
// Returns null when nothing matches so callers can decide whether to fall back
// to the generic copy or surface the raw string. Examples of raw inputs:
//   "Error invoking remote method 'hermes:connection': … Hermes bootstrap
//    failed at stage 'prerequisites': cancelled by user. Check …/desktop.log"
//   "Hermes install was cancelled."
//   "bootstrap cancelled by user"
export interface FriendlyBootErrorCopy {
  cancelled: string
  prerequisites: string
  network: string
  unknown: string
}

export function friendlyBootError(raw: string | null | undefined, copy: FriendlyBootErrorCopy): string | null {
  if (typeof raw !== 'string' || raw.trim() === '') {
    return null
  }

  const lower = raw.toLowerCase()

  // User aborted the install (clicked Cancel, or the elevation prompt was
  // dismissed). Surfaced as the friendliest case so a deliberate cancel never
  // looks like a crash.
  if (lower.includes('cancel')) {
    return copy.cancelled
  }

  // A network/download failure during the prerequisites or repository fetch is
  // by far the most common real failure on a fresh machine behind a flaky
  // connection — give an actionable hint instead of the raw stack.
  if (
    lower.includes('etimedout') ||
    lower.includes('econnrefused') ||
    lower.includes('enotfound') ||
    lower.includes('network') ||
    lower.includes('timed out') ||
    lower.includes('could not resolve') ||
    lower.includes('failed to download') ||
    lower.includes('connection reset')
  ) {
    return copy.network
  }

  // A prerequisites-stage failure that isn't a cancel/network case (missing
  // toolchain, permission, etc.).
  if (lower.includes("stage 'prerequisites'") || lower.includes('prerequisites')) {
    return copy.prerequisites
  }

  return copy.unknown
}
