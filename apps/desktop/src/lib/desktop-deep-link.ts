export interface DesktopDeepLinkPayload {
  kind: string
  name: string
  params: Record<string, string>
}

const GITHUB_ISSUE_PATH = /^\/([A-Za-z0-9-]+)\/([A-Za-z0-9._-]+)\/issues\/([1-9]\d*)\/?$/

function githubIssuePrompt(value: string | undefined): string | null {
  if (!value || value.length > 2048) {
    return null
  }

  let url: URL

  try {
    url = new URL(value)
  } catch {
    return null
  }

  if (
    url.protocol !== 'https:' ||
    url.hostname.toLowerCase() !== 'github.com' ||
    url.port ||
    url.username ||
    url.password ||
    url.search ||
    url.hash
  ) {
    return null
  }

  const match = url.pathname.match(GITHUB_ISSUE_PATH)

  if (!match) {
    return null
  }

  const [, owner, repo, issue] = match
  const canonicalUrl = `https://github.com/${owner}/${repo}/issues/${issue}`

  return (
    'Investigate this GitHub issue in the current workspace. Read the issue and its comments, reproduce the ' +
    `problem, then implement and verify a focused fix:\n\n${canonicalUrl}`
  )
}

export function composerTextForDeepLink(payload: DesktopDeepLinkPayload | null | undefined): string | null {
  if (!payload) {
    return null
  }

  if (payload.kind === 'github-issue' && payload.name === 'open') {
    return githubIssuePrompt(payload.params?.url)
  }

  if (payload.kind !== 'blueprint' || !payload.name) {
    return null
  }

  const slots = Object.entries(payload.params || {})
    .map(([key, value]) => {
      const escaped = /\s/.test(value) ? `"${value.replace(/"/g, '\\"')}"` : value

      return `${key}=${escaped}`
    })
    .join(' ')

  return `/blueprint ${payload.name}${slots ? ' ' + slots : ''}`
}
