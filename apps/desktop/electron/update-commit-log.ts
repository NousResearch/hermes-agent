// Commit-log acquisition + parsing for the desktop update overlay's changelog.
// Extracted from checkUpdates() in main.ts so the official-SSH path is unit
// testable (checkUpdates itself is I/O-bound and untested). The parser is pure;
// the two readers take an injected `run` so tests can assert the exact git
// invocations without spawning git — mirroring the pure-helper pattern in
// update-count.ts.

type GitResult = { code: number; stdout: string; stderr: string }
type GitRun = (args: string[]) => Promise<GitResult>

type CommitEntry = { sha: string; summary: string; author: string; at: number }

// Unit/record separators keep `%s` (commit summary) safe from embedded newlines
// and spaces when splitting the `git log` output back into fields.
const COMMIT_LOG_SEP = '\x1f'
const COMMIT_LOG_REC = '\x1e'
const COMMIT_LOG_FORMAT = `%H${COMMIT_LOG_SEP}%s${COMMIT_LOG_SEP}%an${COMMIT_LOG_SEP}%at${COMMIT_LOG_REC}`
const COMMIT_LOG_LIMIT = 40

function parseCommitLog(stdout): CommitEntry[] {
  return stdout
    .split(COMMIT_LOG_REC)
    .map(line => line.trim())
    .filter(Boolean)
    .map(line => {
      const [sha, summary, author, at] = line.split(COMMIT_LOG_SEP)

      return { sha, summary, author, at: Number.parseInt(at, 10) * 1000 }
    })
}

// Read the commit changelog for an arbitrary range (e.g. `HEAD..origin/main` on
// the fetched HTTPS path, or `HEAD..<sha>` on the official-SSH path).
async function readCommitLogRange(run: GitRun, range): Promise<CommitEntry[]> {
  const { stdout } = await run([
    'log',
    range,
    `--pretty=format:${COMMIT_LOG_FORMAT}`,
    '-n',
    String(COMMIT_LOG_LIMIT)
  ])

  return parseCommitLog(stdout)
}

// Obtain the changelog for the official-SSH remote WITHOUT `git fetch origin`.
// Fetching `origin` would hit the SSH remote and can trigger a passkey/FIDO2
// hardware-touch prompt (the whole reason checkUpdates() takes a separate branch
// for official-SSH installs). The remote tip was already discovered via anonymous
// `git ls-remote` against the HTTPS URL, so fetch the same objects over anonymous
// HTTPS into FETCH_HEAD and read the log up to `targetSha`.
//
// Best-effort: any failure yields [] so the overlay falls back to its generic
// "Improvements and fixes" summary rather than erroring the whole update check.
// FAIL-BEFORE: this path did not exist — the SSH branch hardcoded `commits: []`,
// so official-SSH installs NEVER saw a real changelog (#69081).
async function readOfficialSshCommitLog(
  run: GitRun,
  httpsUrl,
  branch,
  targetSha
): Promise<CommitEntry[]> {
  const fetched = await run(['fetch', '--quiet', httpsUrl, branch])

  if (fetched.code !== 0) {
    return []
  }

  return readCommitLogRange(run, `HEAD..${targetSha}`)
}

export {
  COMMIT_LOG_LIMIT,
  parseCommitLog,
  readCommitLogRange,
  readOfficialSshCommitLog
}
export type { CommitEntry, GitRun }
