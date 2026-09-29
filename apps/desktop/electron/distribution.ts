/**
 * The ONE place the desktop names the repository this build is published from.
 * Agent Czesiek is a fork of Hermes; update checks and release lookups must
 * follow the repository the app ships from. Keep in sync with
 * `hermes_cli/distribution.py` (same slug, same override idea).
 */
export const DISTRIBUTION_REPO = 'aievolutionpl/AGENT_CZESIEK'

export const distributionHttpsUrl = (repo = DISTRIBUTION_REPO) => `https://github.com/${repo}.git`
export const distributionCanonical = (repo = DISTRIBUTION_REPO) => `github.com/${repo}`.toLowerCase()
export const distributionReleasesApi = (repo = DISTRIBUTION_REPO) =>
  `https://api.github.com/repos/${repo}/releases/latest`
export const distributionReleasesPage = (repo = DISTRIBUTION_REPO) => `https://github.com/${repo}/releases/latest`
