const OFFICIAL_RELEASES_API_URL = 'https://api.github.com/repos/NousResearch/hermes-agent/releases'
const OFFICIAL_RELEASE_TAG_RE = /^v(20\d{2})\.(\d{1,2})\.(\d{1,2})(?:\.(\d+))?$/

interface GithubReleaseRecord {
  body?: unknown
  draft?: unknown
  html_url?: unknown
  name?: unknown
  prerelease?: unknown
  published_at?: unknown
  tag_name?: unknown
}

interface GithubRelease {
  body: string
  name: string
  publishedAt: string
  tag: string
  url: string
}

export interface DesktopRelease {
  body: string
  name: string
  publishedAt: string
  sha: string
  tag: string
  url: string
}

export interface DesktopReleaseStatus {
  behind: number
  countKnown: boolean
  currentRelease: string | null
  releases: DesktopRelease[]
  targetRelease: string
  targetSha: string
}

type FetchLike = (input: string | URL | Request, init?: RequestInit) => Promise<Response>

function parseReleaseVersion(tag: string): number[] | null {
  const match = OFFICIAL_RELEASE_TAG_RE.exec(tag)

  return match ? match.slice(1).map(part => Number.parseInt(part || '0', 10)) : null
}

function compareReleaseTagsNewestFirst(left: string, right: string): number {
  const leftVersion = parseReleaseVersion(left)
  const rightVersion = parseReleaseVersion(right)

  if (!leftVersion || !rightVersion) {
    return left.localeCompare(right)
  }

  for (let index = 0; index < Math.max(leftVersion.length, rightVersion.length); index++) {
    const difference = (rightVersion[index] || 0) - (leftVersion[index] || 0)

    if (difference !== 0) {
      return difference
    }
  }

  return 0
}

function normaliseGithubRelease(record: GithubReleaseRecord): GithubRelease | null {
  const tag = typeof record.tag_name === 'string' ? record.tag_name.trim() : ''

  if (
    record.draft === true ||
    record.prerelease === true ||
    !OFFICIAL_RELEASE_TAG_RE.test(tag)
  ) {
    return null
  }

  return {
    body: typeof record.body === 'string' ? record.body : '',
    name: typeof record.name === 'string' && record.name.trim() ? record.name.trim() : tag,
    publishedAt: typeof record.published_at === 'string' ? record.published_at : '',
    tag,
    url: typeof record.html_url === 'string' ? record.html_url : ''
  }
}

async function fetchOfficialGithubReleases(fetchImpl: FetchLike = fetch): Promise<GithubRelease[]> {
  const records = new Map<string, GithubRelease>()

  for (let page = 1; page <= 20; page++) {
    const response = await fetchImpl(`${OFFICIAL_RELEASES_API_URL}?per_page=100&page=${page}`, {
      headers: {
        Accept: 'application/vnd.github+json',
        'User-Agent': 'Hermes-Desktop'
      }
    })

    if (!response.ok) {
      throw new Error(`GitHub Releases request failed (${response.status})`)
    }

    const payload = await response.json()

    if (!Array.isArray(payload)) {
      throw new Error('GitHub Releases returned an invalid response.')
    }

    for (const raw of payload) {
      const release = normaliseGithubRelease(raw || {})

      if (release) {
        records.set(release.tag, release)
      }
    }

    if (payload.length < 100) {
      break
    }
  }

  return [...records.values()].sort((left, right) => compareReleaseTagsNewestFirst(left.tag, right.tag))
}

function parseRemoteReleaseTags(output: string): Map<string, string> {
  const direct = new Map<string, string>()
  const peeled = new Map<string, string>()

  for (const line of String(output || '').split('\n')) {
    const [sha, ref] = line.trim().split(/\s+/, 2)
    const match = /^refs\/tags\/(v20\d{2}\.\d{1,2}\.\d{1,2}(?:\.\d+)?)(\^\{\})?$/.exec(ref || '')

    if (!match || !/^[0-9a-f]{40}$/i.test(sha || '')) {
      continue
    }

    const target = match[2] ? peeled : direct
    target.set(match[1], sha.toLowerCase())
  }

  return new Map([...direct, ...peeled])
}

function resolveDesktopReleaseStatus({
  currentSha,
  releaseRecords,
  remoteTags
}: {
  currentSha: string
  releaseRecords: GithubRelease[]
  remoteTags: Map<string, string>
}): DesktopReleaseStatus {
  const releases: DesktopRelease[] = releaseRecords
    .map(release => ({ ...release, sha: remoteTags.get(release.tag) || '' }))
    .filter(release => Boolean(release.sha))
    .sort((left, right) => compareReleaseTagsNewestFirst(left.tag, right.tag))

  if (releases.length === 0) {
    throw new Error('No supported official Hermes releases were found.')
  }

  const normalisedCurrentSha = String(currentSha || '').trim().toLowerCase()
  const currentIndex = releases.findIndex(release => release.sha === normalisedCurrentSha)
  const target = releases[0]

  return {
    behind: currentIndex >= 0 ? currentIndex : normalisedCurrentSha === target.sha ? 0 : 1,
    countKnown: currentIndex >= 0,
    currentRelease: currentIndex >= 0 ? releases[currentIndex].tag : null,
    releases,
    targetRelease: target.tag,
    targetSha: target.sha
  }
}

export {
  compareReleaseTagsNewestFirst,
  fetchOfficialGithubReleases,
  normaliseGithubRelease,
  OFFICIAL_RELEASE_TAG_RE,
  OFFICIAL_RELEASES_API_URL,
  parseReleaseVersion,
  parseRemoteReleaseTags,
  resolveDesktopReleaseStatus
}
