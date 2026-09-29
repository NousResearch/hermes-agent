/**
 * "Is there a newer Agent Czesiek?" for installed builds.
 *
 * An installed app ships its own sealed engine (see bundled-runtime.ts), so it
 * has no git checkout to `fetch` and never updates by itself: the way forward
 * is a newer installer from the distribution repository's latest GitHub
 * Release. This module is the pure half — parse the release JSON, compare
 * versions, pick the right installer for this OS/CPU — so it is testable
 * without Electron or a network.
 */

export interface ReleaseAssetInfo {
  name: string
  size?: number
  url: string
}

export interface ReleaseCheckResult {
  /** Direct download for this platform, when the release carries one. */
  asset: null | ReleaseAssetInfo
  currentVersion: string
  latestVersion: string
  /** First lines of the release notes, for a one-glance "what's new". */
  notes: string
  /** The release page — always present, the fallback when no asset matches. */
  pageUrl: string
  publishedAt: null | string
  updateAvailable: boolean
}

interface GithubAsset {
  browser_download_url?: unknown
  name?: unknown
  size?: unknown
}

interface GithubRelease {
  assets?: unknown
  body?: unknown
  draft?: unknown
  html_url?: unknown
  prerelease?: unknown
  published_at?: unknown
  tag_name?: unknown
}

const NUMERIC = /^v?(\d+)\.(\d+)(?:\.(\d+))?/i

/** Numeric x.y.z comparison; anything unparsable compares as "no newer". */
export function compareVersions(a: string, b: string): number {
  const pa = NUMERIC.exec(a.trim())
  const pb = NUMERIC.exec(b.trim())

  if (!pa || !pb) {
    return 0
  }

  for (let i = 1; i <= 3; i += 1) {
    const diff = Number(pa[i] ?? 0) - Number(pb[i] ?? 0)

    if (diff !== 0) {
      return diff > 0 ? 1 : -1
    }
  }

  return 0
}

const isHttps = (value: unknown): value is string => typeof value === 'string' && value.startsWith('https://')

function assetsOf(release: GithubRelease): ReleaseAssetInfo[] {
  if (!Array.isArray(release.assets)) {
    return []
  }

  return release.assets.flatMap((raw: GithubAsset) =>
    typeof raw?.name === 'string' && isHttps(raw.browser_download_url) && !/\.blockmap$/i.test(raw.name)
      ? [{ name: raw.name, size: typeof raw.size === 'number' ? raw.size : undefined, url: raw.browser_download_url }]
      : []
  )
}

/** The installer for this OS and CPU; null when the release has none (callers fall back to the page). */
export function pickInstallerAsset(
  assets: readonly ReleaseAssetInfo[],
  platform: string,
  arch: string
): null | ReleaseAssetInfo {
  const named = (re: RegExp) => assets.filter(asset => re.test(asset.name))

  if (platform === 'win32') {
    return named(/\.exe$/i)[0] ?? named(/\.msi$/i)[0] ?? null
  }

  if (platform === 'darwin') {
    const dmgs = named(/\.dmg$/i)
    const arm = /arm64|aarch64|apple/i
    const intel = /x64|x86_64|intel|amd64/i

    return dmgs.find(asset => (arch === 'arm64' ? arm : intel).test(asset.name)) ?? dmgs[0] ?? null
  }

  return named(/\.AppImage$/i)[0] ?? named(/\.deb$/i)[0] ?? named(/\.rpm$/i)[0] ?? null
}

function firstLines(body: unknown, max = 6): string {
  return typeof body === 'string'
    ? body
        .split('\n')
        .map(line => line.trim())
        .filter(Boolean)
        .slice(0, max)
        .join('\n')
    : ''
}

/** Turn one `releases/latest` payload into the answer the UI needs; throws on a payload that is not a usable release. */
export function evaluateRelease(
  payload: unknown,
  { arch, currentVersion, platform }: { arch: string; currentVersion: string; platform: string }
): ReleaseCheckResult {
  const release = (payload ?? {}) as GithubRelease
  const tag = typeof release.tag_name === 'string' ? release.tag_name : ''

  if (!tag || !NUMERIC.test(tag) || !isHttps(release.html_url) || release.draft === true || release.prerelease === true) {
    throw new Error('Latest release is missing, a draft, or has no version tag.')
  }

  return {
    asset: pickInstallerAsset(assetsOf(release), platform, arch),
    currentVersion,
    latestVersion: tag.replace(/^v/i, ''),
    notes: firstLines(release.body),
    pageUrl: release.html_url,
    publishedAt: typeof release.published_at === 'string' ? release.published_at : null,
    updateAvailable: compareVersions(tag, currentVersion) > 0
  }
}
