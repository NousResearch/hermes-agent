import { spawn } from 'node:child_process'
import fs from 'node:fs'
import https from 'node:https'
import path from 'node:path'

import {
  describeGitHubCredentialSource,
  forgetGhCliToken,
  githubApiHeaders,
  githubTokenRejected,
  resolveGitHubCredential
} from './github-api-auth'
import { describeGitSpawnFailure, GIT_UNUSABLE } from './select-runnable-binary'
import {
  branchTipApiUrl,
  cacheIsFresh,
  compareApiUrl,
  describeUpdateCheckFailure,
  githubRepoSlug,
  listLocalCommits,
  parseCompare,
  rateLimitFromHeaders,
  resolveBehindLocally
} from './update-api-check'
import { updateCheckAgent } from './update-api-proxy'
import { isOfficialSshRemote, OFFICIAL_REPO_HTTPS_URL } from './update-remote'
import { hiddenWindowsChildOptions } from './windows-child-options'

type DesktopUpdateCheckDependencies = {
  ACTIVE_HERMES_ROOT: string
  BrowserWindow: {
    getAllWindows: () => Array<{ webContents: { send: (channel: string, payload: unknown) => void } }>
  }
  DEFAULT_UPDATE_BRANCH: string
  DESKTOP_UPDATE_CHECK_CACHE_PATH: string
  DESKTOP_UPDATE_CONFIG_PATH: string
  IS_PACKAGED: boolean
  IS_WINDOWS: boolean
  SOURCE_REPO_ROOT: string
  directoryExists: (filePath: string) => boolean
  isHermesSourceRoot: (root: string) => boolean
  rememberLog: (line: string) => void
  resolveGitBinary: () => string
  writeFileAtomic: (targetPath: string, data: string, encoding?: BufferEncoding) => void
}

export function createDesktopUpdateCheckRuntime({
  ACTIVE_HERMES_ROOT,
  BrowserWindow,
  DEFAULT_UPDATE_BRANCH,
  DESKTOP_UPDATE_CHECK_CACHE_PATH,
  DESKTOP_UPDATE_CONFIG_PATH,
  IS_PACKAGED,
  IS_WINDOWS,
  SOURCE_REPO_ROOT,
  directoryExists,
  isHermesSourceRoot,
  rememberLog,
  resolveGitBinary,
  writeFileAtomic
}: DesktopUpdateCheckDependencies) {
  function readDesktopUpdateConfig() {
    try {
      const parsed = JSON.parse(fs.readFileSync(DESKTOP_UPDATE_CONFIG_PATH, 'utf8'))
      const branch = typeof parsed?.branch === 'string' ? parsed.branch.trim() : ''

      return { branch: branch || DEFAULT_UPDATE_BRANCH }
    } catch {
      return { branch: DEFAULT_UPDATE_BRANCH }
    }
  }

  function writeDesktopUpdateConfig(config) {
    fs.mkdirSync(path.dirname(DESKTOP_UPDATE_CONFIG_PATH), { recursive: true })
    writeFileAtomic(DESKTOP_UPDATE_CONFIG_PATH, JSON.stringify(config, null, 2))
  }

  // Match the backend's source resolution but bias toward a real git checkout.
  // Dev → SOURCE_REPO_ROOT. Packaged/CLI install → ACTIVE_HERMES_ROOT.
  // HERMES_DESKTOP_HERMES_ROOT always wins so devs can pin a worktree.
  function resolveUpdateRoot() {
    const candidates = [
      process.env.HERMES_DESKTOP_HERMES_ROOT && path.resolve(process.env.HERMES_DESKTOP_HERMES_ROOT),
      !IS_PACKAGED && isHermesSourceRoot(SOURCE_REPO_ROOT) ? SOURCE_REPO_ROOT : null,
      isHermesSourceRoot(ACTIVE_HERMES_ROOT) ? ACTIVE_HERMES_ROOT : null
    ].filter(Boolean)

    return candidates.find(c => directoryExists(path.join(c, '.git'))) || candidates[0] || ACTIVE_HERMES_ROOT
  }

  function runGit(args, options: any = {}): Promise<{ code: number; stdout: string; stderr: string }> {
    return new Promise((resolve, reject) => {
      const gitBinary = resolveGitBinary()

      const child = spawn(
        gitBinary,
        IS_WINDOWS ? ['-c', 'windows.appendAtomically=false', ...args] : args,
        hiddenWindowsChildOptions({
          cwd: options.cwd,
          env: { ...process.env, ...((options.env || {}) as any), GIT_TERMINAL_PROMPT: '0' },
          stdio: ['ignore', 'pipe', 'pipe']
        })
      )

      let stdout = ''
      let stderr = ''
      child.stdout.on('data', chunk => {
        const text = chunk.toString()
        stdout += text
        options.onLine?.('stdout', text)
      })
      child.stderr.on('data', chunk => {
        const text = chunk.toString()
        stderr += text
        options.onLine?.('stderr', text)
      })
      // A spawn-level failure means git itself never ran (missing, not
      // executable, wrong CPU architecture) — a local problem, not a network one.
      child.once('error', error => {
        const local = describeGitSpawnFailure(error, gitBinary)

        reject(local ? Object.assign(new Error(local), { kind: GIT_UNUSABLE, cause: error }) : error)
      })
      // 'close', not 'exit': exit can fire before the stdio pipes drain, and a
      // resolved-early `remote get-url` came back as "" often enough to route
      // passive checks down the wrong remote path.
      child.once('close', code => resolve({ code, stdout, stderr }))
    })
  }

  const firstLine = text => (text || '').split('\n').find(Boolean) || ''

  async function getOriginUrl(updateRoot) {
    const origin = await runGit(['remote', 'get-url', 'origin'], { cwd: updateRoot })

    return origin.code === 0 ? origin.stdout.trim() : ''
  }

  function emitUpdateProgress(payload) {
    const merged = { stage: 'idle', message: '', percent: null, error: null, ...payload, at: Date.now() }
    rememberLog(`[updates] ${merged.stage}: ${merged.message || merged.error || ''}`)

    for (const window of BrowserWindow.getAllWindows()) {
      window.webContents.send('hermes:updates:progress', merged)
    }
  }

  // Self-heal the tracked update branch: if origin no longer publishes it (e.g.
  // bb/gui was merged into main and deleted), fall back to main and persist so
  // every later check/apply follows main — no manual flip, even for already-
  // installed clients. Read-only ls-remote probe; only flips on a definitive
  // "ref absent" (exit 2), never on a transient network error, so a flaky
  // connection can't strand a user on the wrong branch.
  async function resolveHealedBranch(updateRoot, branch) {
    if (!branch || branch === 'main') {
      return branch || 'main'
    }

    const originUrl = await getOriginUrl(updateRoot)
    const remote = isOfficialSshRemote(originUrl) ? OFFICIAL_REPO_HTTPS_URL : 'origin'
    const probe = await runGit(['ls-remote', '--exit-code', '--heads', remote, branch], { cwd: updateRoot })

    if (probe.code !== 2) {
      return branch
    }

    rememberLog(`[updates] origin/${branch} is gone (merged?); falling back to main`)
    const config = readDesktopUpdateConfig()

    if (config.branch !== 'main') {
      writeDesktopUpdateConfig({ ...config, branch: 'main' })
    }

    return 'main'
  }

  // Passive checks never touch git's network side. Every client used to `git
  // fetch` twice per half hour; across the install base that was tens of
  // millions of pack negotiations a day against one repo (GitHub flagged it).
  // The REST API answers the same question in one 40-byte response, so the
  // check is API-first with a 24h on-disk cache keyed on local HEAD (applying an
  // update changes HEAD, which busts the cache immediately). `git fetch` runs only
  // inside applyUpdates. `force` (menu item, Settings "Check now") skips the
  // cache; the renderer's background poller never passes it.
  async function checkUpdates({ force = false }: { force?: boolean } = {}) {
    const updateRoot = resolveUpdateRoot()
    let { branch } = readDesktopUpdateConfig()
    const gitDir = path.join(updateRoot, '.git')

    if (!directoryExists(gitDir)) {
      return {
        supported: false,
        reason: 'not-a-git-checkout',
        message:
          "This copy of Hermes can't update itself from inside the app. Download the latest version from the Hermes website, " +
          `or reinstall Hermes to enable in-app updates. Details: ${updateRoot} has no version-control metadata.`,
        hermesRoot: updateRoot,
        branch
      }
    }

    const git = args => runGit(args, { cwd: updateRoot }).then(r => r.stdout.trim())

    const [currentSha, dirtyStr, currentBranch, originUrl] = await Promise.all([
      git(['rev-parse', 'HEAD']),
      git(['status', '--porcelain']),
      git(['rev-parse', '--abbrev-ref', 'HEAD']),
      getOriginUrl(updateRoot)
    ])

    const cached = readUpdateCheckCache()
    const now = Date.now()

    if (!force && cacheIsFresh(cached, { branch, currentSha, now })) {
      return { ...cached.status, dirty: dirtyStr.length > 0, currentBranch }
    }

    branch = await resolveHealedBranch(updateRoot, branch)
    const slug = githubRepoSlug(originUrl)

    const status = slug
      ? await checkUpdatesViaApi({ slug, branch, currentSha, updateRoot })
      : await checkUpdatesViaLsRemote({ updateRoot, branch, currentSha })

    const result = {
      supported: true,
      branch,
      currentBranch,
      currentSha,
      dirty: dirtyStr.length > 0,
      hermesRoot: updateRoot,
      fetchedAt: now,
      ...status
    }

    writeUpdateCheckCache({ fetchedAt: now, currentSha, branch, status: result })

    return result
  }

  function readUpdateCheckCache() {
    try {
      const parsed = JSON.parse(fs.readFileSync(DESKTOP_UPDATE_CHECK_CACHE_PATH, 'utf8'))

      return parsed && typeof parsed === 'object' && parsed.status ? parsed : null
    } catch {
      return null
    }
  }

  function writeUpdateCheckCache(entry) {
    try {
      fs.mkdirSync(path.dirname(DESKTOP_UPDATE_CHECK_CACHE_PATH), { recursive: true })
      writeFileAtomic(DESKTOP_UPDATE_CHECK_CACHE_PATH, JSON.stringify(entry))
    } catch (error) {
      rememberLog(`[updates] could not persist check cache: ${error?.message || error}`)
    }
  }

  // GitHub origins (official repo AND forks): tip SHA via the commits endpoint,
  // then the compare endpoint only when the tips differ — it yields the exact
  // behind count plus the commit list the overlay renders, replacing both
  // `rev-list --count` and `git log HEAD..origin/<branch>`.
  async function checkUpdatesViaApi({ slug, branch, currentSha, updateRoot }) {
    let targetSha

    try {
      targetSha = String(await fetchGitHubApi(branchTipApiUrl(slug, branch), 'application/vnd.github.sha')).trim()
    } catch (error) {
      return { error: 'fetch-failed', message: describeUpdateCheckFailure(error) }
    }

    if (!/^[0-9a-f]{40}$/i.test(targetSha)) {
      return { error: 'fetch-failed', message: 'GitHub API returned no tip SHA.' }
    }

    if (targetSha === currentSha) {
      return { behind: 0, updateAvailable: false, targetSha, commits: [] }
    }

    // Compare failure (rate-limited, local-only HEAD 404) keeps the honest
    // "update available, count unknown" — never a fabricated number.
    let compareError = null

    const compared = await fetchGitHubApi(compareApiUrl(slug, currentSha, targetSha))
      .then(parseCompare)
      .catch(error => {
        compareError = error

        return null
      })

    // ahead_by === 0 with differing tips: the remote tip is reachable from our
    // HEAD — a local commit sitting AHEAD, not behind. Flagging that as an update
    // nudges the user into wiping their work.
    if (compared?.behind === 0) {
      return { behind: 0, updateAvailable: false, targetSha, commits: [] }
    }

    // A local-only HEAD (a patched checkout's merge/rebase commits exist nowhere
    // upstream) makes the compare endpoint 404 FOREVER — the fallback below then
    // holds a permanent "update available" no update can ever clear, since every
    // update re-creates the local-only HEAD. When the already-fetched tip is in
    // the local object database, answer from the local graph instead — the same
    // ancestry guard the ls-remote path already applies and the backend's
    // banner._tips_behind uses. Only a tip we can't see at all stays "unknown".
    if (compared === null && compareError && updateRoot) {
      const local = await resolveBehindLocally(runGit, updateRoot, currentSha, targetSha)

      if (local !== null) {
        return {
          behind: local,
          updateAvailable: local > 0,
          targetSha,
          commits: local > 0 ? await listLocalCommits(runGit, updateRoot, currentSha, targetSha) : []
        }
      }
    }

    return {
      behind: compared ? compared.behind : null,
      updateAvailable: true,
      targetSha,
      commits: compared?.commits ?? []
    }
  }

  // Non-GitHub origins: one ls-remote for the tip SHA (still no pack transfer),
  // counting via the local graph only when the tip is already known locally.
  async function checkUpdatesViaLsRemote({ updateRoot, branch, currentSha }) {
    const target = await runGit(['ls-remote', 'origin', `refs/heads/${branch}`], { cwd: updateRoot })
    const targetSha = firstLine(target.stdout).split(/\s+/)[0] || ''

    if (target.code !== 0 || !targetSha) {
      return { error: 'fetch-failed', message: firstLine(target.stderr) || 'git ls-remote failed.' }
    }

    if (targetSha === currentSha) {
      return { behind: 0, updateAvailable: false, targetSha, commits: [] }
    }

    const known = (await runGit(['cat-file', '-e', `${targetSha}^{commit}`], { cwd: updateRoot })).code === 0

    const isAncestor =
      known && (await runGit(['merge-base', '--is-ancestor', targetSha, 'HEAD'], { cwd: updateRoot })).code === 0

    if (isAncestor) {
      return { behind: 0, updateAvailable: false, targetSha, commits: [] }
    }

    return { behind: null, updateAvailable: true, targetSha, commits: [] }
  }

  // GITHUB_TOKEN / GH_TOKEN from the environment, when present, moves the call
  // from the anonymous 60/hour-per-IP budget to the token's 5,000/hour one; the
  // header shape is otherwise unchanged. Read per request, never stored.
  //
  // Credential ladder (github-api-auth.ts): GITHUB_TOKEN / GH_TOKEN from the
  // launch env, then the gh CLI's login, then anonymous. A token GitHub rejects
  // (401: expired, revoked, malformed) must not turn a check that worked
  // anonymously into a hard failure, so the call is retried once without it; the
  // rejection is logged once per source per process, never with the token.
  const warnedRejectedGitHubTokenSources = new Set()

  async function fetchGitHubApi(url, accept = 'application/vnd.github+json') {
    const credential = await resolveGitHubCredential({ env: process.env })

    try {
      return await fetchGitHubApiOnce(url, accept, credential?.token ?? null)
    } catch (error) {
      if (!credential || !githubTokenRejected(error)) {
        throw error
      }

      if (credential.source === 'gh-cli') {
        // The user may re-login to gh; the next check asks it again.
        forgetGhCliToken()
      }

      if (!warnedRejectedGitHubTokenSources.has(credential.source)) {
        warnedRejectedGitHubTokenSources.add(credential.source)
        rememberLog(
          `[updates] api.github.com rejected ${describeGitHubCredentialSource(credential.source)} (HTTP 401); ` +
            'retrying the update check anonymously'
        )
      }

      return fetchGitHubApiOnce(url, accept, null)
    }
  }

  function fetchGitHubApiOnce(url, accept, token) {
    return new Promise((resolve, reject) => {
      const req = https.get(
        url,
        {
          agent: updateCheckAgent(url),
          headers: githubApiHeaders(
            {
              Accept: accept,
              // GitHub requires a UA on api.github.com; requests without one 403.
              'User-Agent': 'hermes-desktop-update-check'
            },
            token
          ),
          timeout: 10_000
        },
        res => {
          const chunks = []
          res.on('error', reject)
          res.on('data', chunk => chunks.push(chunk))
          res.on('end', () => {
            const body = Buffer.concat(chunks).toString('utf8')

            if ((res.statusCode || 500) >= 400) {
              reject(
                Object.assign(new Error(`HTTP ${res.statusCode}`), {
                  statusCode: res.statusCode,
                  ...rateLimitFromHeaders(res.headers),
                  authenticated: Boolean(token)
                })
              )

              return
            }

            if (accept === 'application/vnd.github.sha') {
              resolve(body)

              return
            }

            try {
              resolve(JSON.parse(body))
            } catch (error) {
              reject(error)
            }
          })
        }
      )

      req.on('timeout', () => req.destroy(new Error('timeout')))
      req.on('error', reject)
    })
  }

  return {
    checkUpdates,
    emitUpdateProgress,
    firstLine,
    readDesktopUpdateConfig,
    resolveHealedBranch,
    resolveUpdateRoot,
    runGit,
    writeDesktopUpdateConfig
  }
}
