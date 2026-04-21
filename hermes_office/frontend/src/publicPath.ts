/**
 * Gateway API server (user-facing) mounts the office SPA at ``/office/`` and
 * REST at ``/api/office/...``. Map ``/api/...`` → ``/api/office/...``.
 * Standalone ``hermes office`` serves at ``/`` — no prefix.
 */
export function officeUrl(path: string): string {
  const rel = path.startsWith("/") ? path : `/${path}`;
  if (typeof window === "undefined") return rel;
  const p = window.location.pathname;
  if (p.startsWith("/office")) {
    if (rel.startsWith("/api/")) {
      return "/api/office" + rel.slice(4);
    }
    return rel;
  }
  return rel;
}
