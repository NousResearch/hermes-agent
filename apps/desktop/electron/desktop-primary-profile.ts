/**
 * Resolve the Hermes profile the desktop launches its primary backend as.
 *
 * Precedence:
 *   1. `desktopStored` — the desktop's own per-machine preference file
 *      (`active-profile.json` in `app.getPath('userData')`). When set, this
 *      pins the backend to that profile via `hermes --profile <name>` and
 *      intentionally overrides any sticky CLI setting.
 *   2. `cliStickyFile` — the CLI's `~/.hermes/active_profile` file written
 *      by `hermes profile use <name>`. Mirrors what plain `hermes chat` on
 *      the same machine would resolve to, so the desktop and CLI stay in
 *      sync on first launch (before the user has set a desktop pref).
 *   3. `'default'` — the root profile (HERMES_HOME = ~/.hermes).
 *
 * The helper is pure: callers pass the pre-read values rather than us
 * touching the filesystem directly, so unit tests can drive every branch
 * without mocking fs / app paths.
 */
export function resolveDesktopPrimaryProfile(
  desktopStored: string | null,
  cliStickyFile: string | null
): string {
  if (desktopStored) {
    return desktopStored;
  }

  if (cliStickyFile) {
    const trimmed = cliStickyFile.trim();

    // Treat whitespace-only values as absent so a stray newline doesn't
    // break `--profile` resolution.
    if (trimmed) {
      return trimmed;
    }
  }

  return 'default';
}

// Mirror hermes_cli.profiles._PROFILE_ID_RE so we accept the same names the
// backend would. Kept loose (the desktop has its own copy in main.ts and the
// CLI has the canonical one); passing the same shape across the boundary is
// enough — the resolver here only checks it's a non-empty trimmed string.
export function isValidProfileName(name: string): boolean {
  if (!name) {
    return false;
  }

  const trimmed = name.trim();

  if (!trimmed) {
    return false;
  }

  return /^[a-z0-9][a-z0-9_-]{0,63}$/.test(trimmed) || trimmed === 'default';
}
