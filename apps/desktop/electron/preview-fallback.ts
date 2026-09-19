import path from 'node:path'

// Resolve home-relative preview targets that the agent cwd cannot see.
//
// Attachment refs stored in chat history are frequently home-relative (e.g.
// "AppData/Local/hermes/attachments/foo.xlsx" on Windows or
// ".hermes/attachments/foo.xlsx" elsewhere): no leading slash, no `~`, no
// file: prefix. Resolving them against the agent working directory always
// ENOENTs even though the file is on disk, so the preview card goes dead.
//
// This module is dependency-free and intentionally does not touch the
// filesystem: it only proposes fallback candidates, the caller still resolves
// each one through the usual IPC path hardening.

// Whether a raw preview ref can be a home-relative attachment ref at all:
// absolute paths, file: URLs and ~/ refs already resolve through the primary
// path and must never enter the fallback.
export function isHomeRelativePreviewRef(raw) {
  const rawTarget = String(raw || '').trim()

  if (!rawTarget || /^file:/i.test(rawTarget) || rawTarget.startsWith('~')) {
    return false
  }

  // Anything carrying a URI scheme (https://, file:, mailto:) — including a
  // one-letter Windows drive prefix — is not a home-relative ref.
  if (/^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(rawTarget)) {
    return false
  }

  const normalized = rawTarget.replace(/\\/g, '/')

  return !(normalized.startsWith('/') || /^[a-zA-Z]:/.test(normalized))
}

// Fallback candidates for a home-relative ref: the ref joined under each base
// (user home first, then the attachments root), plus the basename alone under
// the attachments root for refs that lost their directory prefix entirely.
// `attachmentsRoot` must come from the app's hermes-home resolution, NOT a
// hardcoded segment: that already yields %LOCALAPPDATA%\hermes on Windows,
// ~/.hermes elsewhere, and honours HERMES_HOME overrides and legacy installs.
export function homeRelativePreviewFallbackCandidates(
  raw,
  { attachmentsRoot = '', homeDir = '' } = {}
) {
  if (!isHomeRelativePreviewRef(raw)) {
    return []
  }

  const normalized = String(raw || '').trim().replace(/\\/g, '/').replace(/\/+/g, '/')
  const bases = []

  if (homeDir) {
    bases.push(homeDir)
  }

  if (attachmentsRoot) {
    bases.push(attachmentsRoot)
  }

  const candidates = bases.map((base) => path.join(base, normalized))

  // The basename alone under the attachments root covers refs that lost
  // their directory prefix entirely (e.g. bare "foo.xlsx" from a trimmed
  // history ref). When the full ref already contains the attachments root,
  // the home-join above produced the same path — keep it deduped.
  const name = path.basename(normalized)

  if (attachmentsRoot && name) {
    candidates.push(path.join(attachmentsRoot, name))
  }

  return [...new Set(candidates)]
}
