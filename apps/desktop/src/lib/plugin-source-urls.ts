const GITHUB_BROWSER_SEGMENTS = new Set(['tree', 'blob', 'commit'])

export interface PluginSourceLinks {
  gitUrl: string
  browseUrl: string | null
  subdir: string | null
}

function resolvePluginGitUrl(identifier: string): { gitUrl: string; subdir: string | null } {
  const trimmed = identifier.trim()

  if (!trimmed) {
    throw new Error('Plugin identifier is required.')
  }

  if (/^(https?:\/\/|git@|ssh:\/\/|file:\/\/)/.test(trimmed)) {
    if (trimmed.startsWith('https://github.com/')) {
      const rest = trimmed.slice('https://github.com/'.length).split(/[?#]/)[0].replace(/\/+$/, '')
      const parts = rest.split('/').filter(Boolean)

      if (parts.length >= 3 && parts[2] && GITHUB_BROWSER_SEGMENTS.has(parts[2])) {
        const repo = parts[1].replace(/\.git$/, '')
        let subdir: string | null = null

        if (parts[2] === 'tree' && parts.length >= 5) {
          subdir = parts.slice(4).join('/').replace(/\/+$/, '') || null
        }

        return { gitUrl: `https://github.com/${parts[0]}/${repo}.git`, subdir }
      }
    }

    if (trimmed.includes('#')) {
      const hashIdx = trimmed.indexOf('#')
      const gitUrl = trimmed.slice(0, hashIdx)
      const subdir = trimmed.slice(hashIdx + 1).replace(/^\/+|\/+$/g, '') || null

      return { gitUrl, subdir }
    }

    const marker = '.git/'

    if (trimmed.includes(marker)) {
      const idx = trimmed.indexOf(marker)
      const gitUrl = trimmed.slice(0, idx + marker.length - 1)
      const subdir = trimmed.slice(idx + marker.length).replace(/^\/+|\/+$/g, '') || null

      return { gitUrl, subdir }
    }

    return { gitUrl: trimmed, subdir: null }
  }

  const parts = trimmed.split('/').filter(Boolean)

  if (parts.length >= 2) {
    const [owner, repo, ...rest] = parts
    const gitUrl = `https://github.com/${owner}/${repo}.git`
    const subdir = rest.join('/').replace(/\/+$/, '') || null

    return { gitUrl, subdir }
  }

  throw new Error('Invalid plugin identifier.')
}

function githubBrowseBase(gitUrl: string): string | null {
  const sshMatch = gitUrl.match(/^git@github\.com:([^/]+)\/(.+?)(?:\.git)?$/i)

  if (sshMatch) {
    return `https://github.com/${sshMatch[1]}/${sshMatch[2].replace(/\.git$/, '')}`
  }

  try {
    const url = new URL(gitUrl)

    if (url.hostname.toLowerCase() === 'github.com') {
      const parts = url.pathname.replace(/\/+$/, '').split('/').filter(Boolean)

      if (parts.length >= 2) {
        return `https://github.com/${parts[0]}/${parts[1].replace(/\.git$/, '')}`
      }
    }
  } catch {
    return null
  }

  return null
}

function browseUrlFromGitUrl(gitUrl: string, subdir: string | null): string | null {
  const githubBase = githubBrowseBase(gitUrl)

  if (githubBase) {
    return subdir ? `${githubBase}/tree/HEAD/${subdir}` : githubBase
  }

  if (/^https?:\/\//.test(gitUrl)) {
    const base = gitUrl.replace(/\.git$/, '')

    return subdir ? `${base}/tree/HEAD/${subdir}` : base
  }

  return null
}

export function resolvePluginSourceLinks(identifier: string): PluginSourceLinks | null {
  try {
    const { gitUrl, subdir } = resolvePluginGitUrl(identifier)

    return {
      gitUrl,
      browseUrl: browseUrlFromGitUrl(gitUrl, subdir),
      subdir
    }
  } catch {
    return null
  }
}
