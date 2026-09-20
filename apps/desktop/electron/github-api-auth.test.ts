import assert from 'node:assert/strict'

import { beforeEach, test } from 'vitest'

import {
  describeGitHubCredentialSource,
  findGhCli,
  forgetGhCliToken,
  GH_CLI_TIMEOUT_MS,
  githubApiHeaders,
  githubTokenFromEnv,
  githubTokenRejected,
  resolveGitHubCredential
} from './github-api-auth'

// The exact headers the anonymous update check sends today; the change must
// leave this shape byte-identical when no token is configured.
const UPDATE_CHECK_HEADERS = {
  Accept: 'application/vnd.github.sha',
  'User-Agent': 'hermes-desktop-update-check'
}

beforeEach(() => {
  forgetGhCliToken()
})

/** execFile stand-in that records every spawn and answers with a scripted result. */
function fakeExecFile(result: { error?: Error | null; stdout?: string }) {
  const calls: Array<{ file: string; args: string[]; options: Record<string, unknown> }> = []
  let stdinEnded = 0

  const execFileFn = ((file, args, options, callback) => {
    calls.push({ file, args, options })
    queueMicrotask(() => callback(result.error ?? null, result.stdout ?? '', ''))

    return {
      stdin: {
        end() {
          stdinEnded += 1
        }
      }
    }
  }) as any

  return { calls, execFileFn, stdinEnded: () => stdinEnded }
}

test('GITHUB_TOKEN wins over GH_TOKEN, blanks fall through, values are trimmed', () => {
  assert.equal(githubTokenFromEnv({ GITHUB_TOKEN: 'pat-a', GH_TOKEN: 'pat-b' }), 'pat-a')
  assert.equal(githubTokenFromEnv({ GITHUB_TOKEN: '   ', GH_TOKEN: '  pat-b  ' }), 'pat-b')
  assert.equal(githubTokenFromEnv({ GITHUB_TOKEN: '', GH_TOKEN: '' }), null)
  assert.equal(githubTokenFromEnv({}), null)
})

test('anonymous sends no Authorization key at all; a token adds the `token` scheme without touching the base', () => {
  const anonymous = githubApiHeaders(UPDATE_CHECK_HEADERS, null)

  assert.deepEqual(anonymous, UPDATE_CHECK_HEADERS)
  assert.equal('Authorization' in anonymous, false)

  const authed = githubApiHeaders(UPDATE_CHECK_HEADERS, 'pat-a')

  assert.equal(authed.Authorization, 'token pat-a')
  assert.equal(authed['User-Agent'], 'hermes-desktop-update-check')
  // The caller's base object is a constant; mutating it would leak the token
  // into every later request that builds from it.
  assert.equal('Authorization' in UPDATE_CHECK_HEADERS, false)
})

test('only an authenticated 401 counts as the credential being rejected', () => {
  // The anonymous-retry gate: a stale GITHUB_TOKEN or gh login must fall back
  // to the anonymous request that worked before credentials were wired in.
  assert.equal(githubTokenRejected({ statusCode: 401, authenticated: true }), true)
  assert.equal(githubTokenRejected({ statusCode: 401, authenticated: false }), false)
  assert.equal(githubTokenRejected({ statusCode: 403, authenticated: true }), false)
  assert.equal(githubTokenRejected({ statusCode: 429, authenticated: true }), false)
  assert.equal(githubTokenRejected(null), false)
})

// #112615: a Finder/launcher-started app inherits /usr/bin:/bin:/usr/sbin:/sbin,
// so `gh` from Homebrew (macOS) or the GitHub CLI installer (Windows) is
// invisible on PATH — the rung must look where those installers put it.
test('gh is found in the GUI-safe install locations a minimal launch PATH omits', () => {
  const brewGh = '/opt/homebrew/bin/gh'
  const macos = findGhCli({ PATH: '/usr/bin:/bin:/usr/sbin:/sbin' }, 'darwin', p => p === brewGh)

  assert.equal(macos, brewGh)

  const winGh = 'C:\\Program Files\\GitHub CLI\\gh.exe'
  const windows = findGhCli(
    { Path: 'C:\\Windows\\System32', ProgramFiles: 'C:\\Program Files' },
    'win32',
    p => p === winGh
  )

  assert.equal(windows, winGh)

  // PATH still wins over the fallbacks when both have one.
  const pathGh = '/usr/local/bin/gh'
  const both = findGhCli({ PATH: '/usr/local/bin' }, 'darwin', p => p === pathGh || p === brewGh)

  assert.equal(both, pathGh)
  assert.equal(
    findGhCli({ PATH: '/usr/bin' }, 'linux', () => false),
    null
  )
})

test('gh rung: argv-only spawn, stdin closed, bounded, cached for the process, and after the env rung', async () => {
  const exists = (p: string) => p === '/opt/homebrew/bin/gh'
  const env = { PATH: '/usr/bin:/bin' }
  const logged = fakeExecFile({ stdout: 'gho_from_gh\n' })

  const first = await resolveGitHubCredential({ env, platform: 'darwin', exists, execFileFn: logged.execFileFn })

  assert.deepEqual(first, { token: 'gho_from_gh', source: 'gh-cli' })
  assert.equal(logged.calls.length, 1)
  assert.equal(logged.calls[0].file, '/opt/homebrew/bin/gh')
  assert.deepEqual(logged.calls[0].args, ['auth', 'token'])
  assert.equal(logged.calls[0].options.shell, undefined)
  assert.equal(logged.calls[0].options.windowsHide, true)
  assert.equal(logged.calls[0].options.timeout, GH_CLI_TIMEOUT_MS)
  assert.equal(logged.stdinEnded(), 1)

  // Process-lifetime cache: the second check does not spawn gh again.
  await resolveGitHubCredential({ env, platform: 'darwin', exists, execFileFn: logged.execFileFn })
  assert.equal(logged.calls.length, 1)

  // Env token outranks gh and never spawns it.
  forgetGhCliToken()

  const fromEnv = await resolveGitHubCredential({
    env: { ...env, GH_TOKEN: 'pat-env' },
    platform: 'darwin',
    exists,
    execFileFn: logged.execFileFn
  })

  assert.deepEqual(fromEnv, { token: 'pat-env', source: 'env' })
  assert.equal(logged.calls.length, 1)

  // Logged-out gh (exit 1, empty stdout) and a missing gh both mean anonymous.
  forgetGhCliToken()
  const loggedOut = fakeExecFile({ error: Object.assign(new Error('exit 1'), { code: 1 }), stdout: '' })

  assert.equal(
    await resolveGitHubCredential({ env, platform: 'darwin', exists, execFileFn: loggedOut.execFileFn }),
    null
  )

  // "none" is not cached: a `gh auth login` after launch is picked up by the next check without a restart.
  const nowLogged = fakeExecFile({ stdout: 'gho_after_login\n' })

  assert.deepEqual(
    await resolveGitHubCredential({ env, platform: 'darwin', exists, execFileFn: nowLogged.execFileFn }),
    {
      token: 'gho_after_login',
      source: 'gh-cli'
    }
  )

  forgetGhCliToken()
  assert.equal(
    await resolveGitHubCredential({ env, platform: 'darwin', exists: () => false, execFileFn: logged.execFileFn }),
    null
  )
  assert.equal(logged.calls.length, 1)
})

test('the rejected-credential log line names the source, never the token', () => {
  assert.match(describeGitHubCredentialSource('env'), /GITHUB_TOKEN \/ GH_TOKEN/)
  assert.match(describeGitHubCredentialSource('gh-cli'), /gh auth token/)
})
