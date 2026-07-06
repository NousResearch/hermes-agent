'use strict'

const https = require('node:https')
const { execFile: defaultExecFile } = require('node:child_process')

const DEFAULT_TIMEOUT_MS = 5000
const RESPONSE_BYTE_LIMIT = 64 * 1024
const GITHUB_HOST_RE = /^(?:www\.)?github\.com$/i

function parseGithubIssueOrPullUrl(rawUrl) {
  let url
  try {
    url = new URL(String(rawUrl || '').trim())
  } catch {
    return null
  }

  if (!GITHUB_HOST_RE.test(url.hostname)) return null

  const parts = url.pathname.split('/').filter(Boolean).map(part => {
    try {
      return decodeURIComponent(part)
    } catch {
      return part
    }
  })

  if (parts.length < 4) return null

  const [owner, repo, type, numberRaw] = parts
  if (!owner || !repo || !numberRaw || !/^\d+$/.test(numberRaw)) return null
  if (type !== 'issues' && type !== 'pull') return null

  const number = Number(numberRaw)
  if (!Number.isSafeInteger(number) || number <= 0) return null

  return { kind: type === 'pull' ? 'pull' : 'issue', number, owner, repo }
}

function cleanGithubTitle(title) {
  return String(title || '').replace(/\s+/g, ' ').trim()
}

function formatGithubLinkTitle(parsed, title) {
  const clean = cleanGithubTitle(title)
  if (!parsed || !clean) return ''
  return `${parsed.owner}/${parsed.repo}#${parsed.number} — ${clean}`
}

function titleFromGhStdout(parsed, stdout) {
  try {
    const json = JSON.parse(String(stdout || ''))
    return formatGithubLinkTitle(parsed, json?.title)
  } catch {
    return ''
  }
}

function fetchGithubTitleWithGh(parsed, options = {}) {
  return new Promise(resolve => {
    const execFile = options.execFile || defaultExecFile
    const ghBinary = options.ghBinary || 'gh'
    const command = parsed.kind === 'pull' ? 'pr' : 'issue'
    const args = [command, 'view', String(parsed.number), '--repo', `${parsed.owner}/${parsed.repo}`, '--json', 'title']
    const env = {
      ...(options.env || process.env),
      GH_PROMPT_DISABLED: '1',
      GIT_TERMINAL_PROMPT: '0'
    }

    execFile(
      ghBinary,
      args,
      {
        windowsHide: true,
        ...(options.childOptions || {}),
        encoding: 'utf8',
        env,
        timeout: options.timeoutMs || DEFAULT_TIMEOUT_MS
      },
      (error, stdout) => {
        if (error) return resolve('')
        resolve(titleFromGhStdout(parsed, stdout))
      }
    )
  })
}

function githubApiUrl(parsed) {
  const endpoint = parsed.kind === 'pull' ? 'pulls' : 'issues'
  const owner = encodeURIComponent(parsed.owner)
  const repo = encodeURIComponent(parsed.repo)
  return `https://api.github.com/repos/${owner}/${repo}/${endpoint}/${parsed.number}`
}

function defaultRequestJson(url, options = {}) {
  return new Promise(resolve => {
    const request = https.request(
      url,
      {
        headers: options.headers || {},
        method: 'GET',
        timeout: options.timeoutMs || DEFAULT_TIMEOUT_MS
      },
      response => {
        const chunks = []
        let bytes = 0

        response.on('data', chunk => {
          if (bytes >= RESPONSE_BYTE_LIMIT) return
          const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk)
          const remaining = RESPONSE_BYTE_LIMIT - bytes
          const next = buffer.length > remaining ? buffer.subarray(0, remaining) : buffer
          chunks.push(next)
          bytes += next.length
        })

        response.on('end', () => {
          if (response.statusCode < 200 || response.statusCode >= 300 || !chunks.length) return resolve(null)
          try {
            resolve(JSON.parse(Buffer.concat(chunks).toString('utf8')))
          } catch {
            resolve(null)
          }
        })
      }
    )

    request.on('timeout', () => request.destroy())
    request.on('error', () => resolve(null))
    request.end()
  })
}

async function fetchGithubTitleWithToken(parsed, options = {}) {
  const env = options.env || process.env
  const token = env.GITHUB_TOKEN || env.GH_TOKEN
  if (!token) return ''

  const requestJson = options.requestJson || defaultRequestJson
  const json = await requestJson(githubApiUrl(parsed), {
    headers: {
      Accept: 'application/vnd.github+json',
      Authorization: `Bearer ${token}`,
      'User-Agent': 'Hermes-Desktop-LinkTitle',
      'X-GitHub-Api-Version': '2022-11-28'
    },
    timeoutMs: options.timeoutMs || DEFAULT_TIMEOUT_MS
  })

  return formatGithubLinkTitle(parsed, json?.title)
}

async function fetchGithubLinkTitle(rawUrl, options = {}) {
  const parsed = parseGithubIssueOrPullUrl(rawUrl)
  if (!parsed) return ''

  const ghTitle = await fetchGithubTitleWithGh(parsed, options).catch(() => '')
  if (ghTitle) return ghTitle

  return fetchGithubTitleWithToken(parsed, options).catch(() => '')
}

module.exports = {
  fetchGithubLinkTitle,
  fetchGithubTitleWithGh,
  fetchGithubTitleWithToken,
  formatGithubLinkTitle,
  githubApiUrl,
  parseGithubIssueOrPullUrl
}
