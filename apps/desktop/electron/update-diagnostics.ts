/**
 * Pure diagnostics for a failed desktop update check.
 *
 * `checkUpdates()` in main.ts shells out to git; every nonzero exit used to come
 * back to the UI as a bare `error: 'fetch-failed'` plus one raw stderr line.
 * Both renderer surfaces then discarded even that line and worded the failure
 * as "We couldn't reach the update server." That is true for only a subset of
 * the failures: a misconfigured `origin`, a stale `.git/*.lock`, a disabled
 * credential prompt or a TLS-intercepting proxy all arrive with exit code 128
 * and look identical to an outage in the UI. On Windows those non-network
 * failures are the common ones, so the bare wording sent users chasing the
 * wrong problem.
 *
 * Extracted from main.ts so the classification is unit-testable without booting
 * Electron (main.ts requires('electron') at load).
 */

type Env = Record<string, string | undefined>

/** Proxy variables git honours. Their values routinely embed credentials
 *  (`http://user:pass@host`), so only the NAMES are ever surfaced. */
const PROXY_ENV_KEYS = ['HTTPS_PROXY', 'https_proxy', 'HTTP_PROXY', 'http_proxy', 'ALL_PROXY', 'all_proxy'] as const
const NO_PROXY_ENV_KEYS = ['NO_PROXY', 'no_proxy'] as const

export interface UpdateCheckFailure {
  /** Remote the check contacted: the origin URL, or the official HTTPS URL
   *  substituted for passive SSH-official checks. */
  remote?: string
  branch?: string
  /** git's exit code, or null when the process never started (spawn ENOENT). */
  code?: number | null
  /** Raw git stderr. */
  stderr?: string
  /** Injectable for tests; defaults to process.env. */
  env?: Env
}

// Ordered; first match wins. Specific causes must precede the generic
// "unable to access" rule, which git/curl also use for 429s and TLS failures.
const FAILURE_RULES: readonly (readonly [RegExp, string])[] = [
  [/HTTP 429|returned error: 429|rate limit/i, 'GitHub is rate limiting or having an outage (HTTP 429)'],
  [/HTTP (?:500|502|503|504)|returned error: (?:500|502|503|504)/i, 'GitHub appears to be having an outage (HTTP 5xx)'],
  [
    /SSL certificate problem|certificate verify failed|unable to get local issuer certificate|schannel: next InitializeSecurityContext failed/i,
    'TLS certificate not trusted (an intercepting proxy or antivirus root CA is the usual cause)'
  ],
  [
    /Unable to create .*\.lock|\.lock'?: File exists|index\.lock/i,
    'git metadata is locked by another git process (stale .git/*.lock)'
  ],
  [
    /does not appear to be a git repository/i,
    'the remote is not a git repository (misconfigured remote URL, not a network failure)'
  ],
  [
    /Repository not found|repository .* not found|ERROR: Repository not found/i,
    'the remote repository was not found (renamed, private, or wrong URL)'
  ],
  [
    /could not read Username|terminal prompts disabled|Authentication failed|Permission denied \(publickey\)/i,
    'the remote asked for credentials, and background checks never prompt'
  ],
  [
    /Could not resolve host|Temporary failure in name resolution|getaddrinfo|Name or service not known/i,
    'DNS lookup failed'
  ],
  [
    /unable to access|Failed to connect|Connection (?:refused|timed out|reset)|network is unreachable|Could not read from remote repository|early EOF|proxy/i,
    'network/connection failure'
  ]
]

/** Map git stderr to a one-line cause. Never claims the network when stderr says otherwise. */
export function classifyUpdateCheckFailure(stderr?: string): string {
  const text = stderr || ''

  return FAILURE_RULES.find(([pattern]) => pattern.test(text))?.[1] ?? 'unclassified git failure'
}

/** Which proxy variables the spawning process actually had. git reads these, so
 *  "none set" in a GUI process whose shell exports them is itself the finding. */
export function proxyHint(env: Env = process.env): string {
  const keys = [...PROXY_ENV_KEYS, ...NO_PROXY_ENV_KEYS].filter(key => (env[key] || '').trim())

  return keys.length ? `proxy: ${keys.join(', ')} set` : 'proxy: none set'
}

/** `user:pass@` in a URL is never worth emitting into a UI or a log. */
function redactCredentials(text: string): string {
  return text.replace(/(\/\/)[^/@\s]*@/g, '$1***@')
}

/**
 * Build the message the update surfaces show verbatim: the remote actually
 * contacted, git's exit code, the classified cause, git's own first stderr line
 * and whether any proxy variable was set.
 */
export function describeUpdateCheckFailure(failure: UpdateCheckFailure): string {
  const remote = redactCredentials(failure.remote || 'origin')
  const where = failure.branch ? `${remote} (${failure.branch})` : remote
  const code = failure.code === null || failure.code === undefined ? 'unknown' : failure.code
  const rawLine = (failure.stderr || '').split('\n').find(line => line.trim())
  const line = redactCredentials(rawLine?.trim() || '(git exited with no stderr)')

  return `git failed (exit ${code}) contacting ${where} — ${classifyUpdateCheckFailure(failure.stderr)}. ${line} [${proxyHint(failure.env)}]`
}
