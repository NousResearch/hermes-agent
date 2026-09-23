/**
 * Deterministic Windows Task Scheduler spec for the unattended self-update —
 * pure, dependency-free core (no `node:`, no Electron, no fs).
 *
 * ROLE: the in-app timer (electron/unattended-update.ts) covers the "app is
 * OPEN at the scheduled minute" half of the unattended runway; this module
 * covers the "app is CLOSED" half. The Desktop registers ONE per-user, least-
 * privilege scheduled task that fires at the same local wall-clock time and
 * invokes ONLY the repo-owned updater hand-off (scripts/desktop-update/
 * windows.ps1, or the staged hermes-setup.exe fallback). No arbitrary user
 * command, no branch chosen at run time, no backend, no remote control, no
 * elevation, no SYSTEM account.
 *
 * SAFETY CONTRACT (why the action can never be an injection):
 *
 *   * FIXED SHAPE — the action is built from a constant token sequence plus
 *     exactly four validated values (repo script path, install root, relaunch
 *     exe, branch). The tokens are the same ones the in-app hand-off already
 *     uses (see wrapHandoffForDetachedConsole + applyUpdates in main.ts), so
 *     "invokes the existing updater hand-off" is literally true.
 *   * PATH ALLOWLIST — `isTaskPathSafe` REJECTS every character the command
 *     line could treat specially (`" & | < > ^ % ! ` ` backtick`, control
 *     chars), so a path can never smuggle a second command out of its quotes.
 *     Trailing separators are rejected too — a path ending in `\` would
 *     escape the closing quote under cmd/CreateProcess parsing.
 *   * BRANCH ALLOWLIST — refs are restricted to `[A-Za-z0-9._/-]` (no leading
 *     dash, so a ref can never be mistaken for a switch).
 *   * DETERMINISTIC — the spec is a pure function of its inputs; the same
 *     inputs always yield the same action, /tr value and schtasks argv. The
 *     exact named task is what gets deleted on disable.
 *   * VERIFIED READ-BACK — the registered task is re-read through
 *     `schtasks /query /xml` and must either match our action exactly
 *     (`taskActionEquals`) or at least our fixed structural shape
 *     (`taskActionMatchesOurShape`); a task under our name that matches
 *     NEITHER is treated as foreign and NEVER overwritten or deleted.
 *   * LEAST PRIVILEGE — schtasks is invoked with NO `/ru`/`/rp`: the task
 *     belongs to the current user, runs only while that user is logged on,
 *     with an interactive (non-elevated) token, and `/rl LIMITED` forbids
 *     running with the highest available privileges. No stored credentials.
 *
 * The task itself is a one-shot nightly trigger whose real work happens in
 * windows.ps1 `-Unattended` mode: when the Desktop is already running the
 * script exits 0 without touching anything (the open app's own timer owns the
 * slot), and at the end it relaunches the Desktop so the update is seamless.
 */

/** The exact, immutable name of the scheduled task. Uninstall deletes THIS. */
export const UNATTENDED_TASK_NAME = 'HermesDesktopUnattendedUpdate'

/**
 * Stamp file (in HERMES_HOME) written by windows.ps1 `-Unattended` when a
 * scheduled run actually executes. The booted Desktop reads it as a
 * "recently ran" cooldown signal, so the relaunched app cannot double-run
 * the same nightly slot. Filename must match scripts/desktop-update/windows.ps1.
 */
export const UNATTENDED_TASK_STAMP_FILENAME = '.hermes-unattended-last-run'

export type UnattendedTaskStateKind =
  /** The exact named task exists and its action matches our spec. */
  | 'installed'
  /** The task exists and is OUR shape, but its paths/config differ (e.g. the
   *  checkout moved). Safe to overwrite via `/create /f`. */
  | 'stale'
  /** No task under our name exists. */
  | 'absent'
  /** A task under our name exists whose action is not ours — never touched. */
  | 'foreign'
  /** Not Windows, or this install has no usable updater/paths. */
  | 'unsupported'
  /** The schtasks invocation itself failed. */
  | 'error'

export interface UnattendedTaskState {
  kind: UnattendedTaskStateKind
  /** Human-readable detail for logs / the Settings row (never secrets). */
  message?: string
}

export type UnattendedTaskHandoff = 'script' | 'binary'

export interface ScheduledTaskSpecInput {
  /** Absolute path to the repo-owned windows.ps1 (handoff 'script'). */
  scriptPath?: string
  /** Absolute path to the staged hermes-setup.exe (handoff 'binary'). */
  updaterBinaryPath?: string
  /** Absolute repo checkout root (HERMES_HOME\hermes-agent). */
  installRoot: string
  /** Absolute path of the executable to relaunch after the update
   *  (Electron process.execPath — Hermes.exe on a packaged install). */
  relaunchExe: string
  /** Branch to update against (pinned from the app's own config at
   *  registration; never a run-time user input). */
  branch: string
  /** Local wall-clock hour, 0-23. */
  hour: number
  /** Local wall-clock minute, 0-59. */
  minute: number
}

export interface ScheduledTaskSpec {
  taskName: string
  handoff: UnattendedTaskHandoff
  /** Absolute path of whatever the task will execute. */
  updaterPath: string
  installRoot: string
  relaunchExe: string
  branch: string
  hour: number
  minute: number
  /** Task Scheduler start time, zero-padded HH:MM. */
  startClock: string
  /** The complete /TR action command line. */
  action: string
  /** argv for `schtasks.exe /create` (idempotent: `/f`). */
  schtasksArgs: string[]
  /** argv for `schtasks.exe /delete`. */
  schtasksDeleteArgs: string[]
  /** argv for `schtasks.exe /query /xml` (read-back verification). */
  schtasksQueryArgs: string[]
}

/** Characters that must NEVER appear inside a path embedded in a task action.
 *  `%`/`!` expand even inside double quotes in cmd; `&|<>^"` are operators or
 *  quoting; backtick is PowerShell escape; control chars are whitespace/
 *  newline tricks. Everything else (including non-ASCII letters) is fine.
 *  Done as a code-point scan (no control chars inside a regex literal). */
const UNSAFE_ASCII_PATH_CHARS = new Set(['"', '&', '|', '<', '>', '^', '%', '!', '`'])

function containsUnsafePathChar(p: string): boolean {
  for (let i = 0; i < p.length; i++) {
    const code = p.charCodeAt(i)

    if (code < 0x20 || code === 0x7f) {
      return true
    }

    if (code <= 0x7f && UNSAFE_ASCII_PATH_CHARS.has(p[i])) {
      return true
    }
  }

  return false
}

const MAX_TASK_PATH_LENGTH = 300

/** Branch refs allowed in a task action: alnum start, then `[A-Za-z0-9._/-]`.
 *  No leading dash (can't be parsed as a switch), no space, no shell chars. */
const SAFE_BRANCH_RE = /^[A-Za-z0-9][A-Za-z0-9._/-]*$/
const MAX_BRANCH_LENGTH = 200

/** True when `p` may be embedded in the task action command line. */
export function isTaskPathSafe(p: string): boolean {
  if (typeof p !== 'string' || p.length === 0 || p.length > MAX_TASK_PATH_LENGTH) {
    return false
  }

  if (containsUnsafePathChar(p)) {
    return false
  }

  // "C:\foo\" before a closing quote would escape it under cmd/CreateProcess
  // parsing — the classic trailing-backslash quote break. Reject outright.
  if (p.endsWith('\\') || p.endsWith('/')) {
    return false
  }

  // No leading/trailing whitespace: a value that pads its way into another
  // token is a foot-gun even when every other check passes.
  if (p.trim() !== p) {
    return false
  }

  return true
}

/** True when `branch` may be passed to the updater in a task action. */
export function isTaskBranchSafe(branch: string): boolean {
  return (
    typeof branch === 'string' && branch.length > 0 && branch.length <= MAX_BRANCH_LENGTH && SAFE_BRANCH_RE.test(branch)
  )
}

function assertSafeTime(hour: number, minute: number): { hour: number; minute: number } {
  const h = Math.floor(Number(hour))
  const m = Math.floor(Number(minute))

  if (!Number.isFinite(h) || !Number.isFinite(m) || h < 0 || h > 23 || m < 0 || m > 59) {
    throw new Error(`invalid unattended task time ${hour}:${minute}`)
  }

  return { hour: h, minute: m }
}

function zeroPad(n: number): string {
  return String(n).padStart(2, '0')
}

/**
 * Build the complete, deterministic task spec — or THROW (fail closed).
 *
 * Exactly one of scriptPath / updaterBinaryPath must be given; script is the
 * preferred hand-off (repo-refreshed, per the frozen-binary policy), binary
 * is the fallback for checkouts that predate windows.ps1 — mirroring the
 * in-app applyUpdates choice. The action embeds only validated values.
 */
export function buildScheduledTaskSpec(input: ScheduledTaskSpecInput): ScheduledTaskSpec {
  const { installRoot, relaunchExe, branch } = input
  const hasScript = typeof input.scriptPath === 'string' && input.scriptPath.length > 0
  const hasBinary = typeof input.updaterBinaryPath === 'string' && input.updaterBinaryPath.length > 0

  if (hasScript === hasBinary) {
    throw new Error('exactly one of scriptPath / updaterBinaryPath must be provided')
  }

  if (!isTaskPathSafe(installRoot)) {
    throw new Error('unsafe install root for scheduled task')
  }

  if (!isTaskPathSafe(relaunchExe)) {
    throw new Error('unsafe relaunch executable for scheduled task')
  }

  if (!isTaskBranchSafe(branch)) {
    throw new Error('unsafe branch ref for scheduled task')
  }

  const time = assertSafeTime(input.hour, input.minute)
  const updaterPath = hasScript ? (input.scriptPath as string) : (input.updaterBinaryPath as string)

  if (!isTaskPathSafe(updaterPath)) {
    throw new Error('unsafe updater path for scheduled task')
  }

  const startClock = `${zeroPad(time.hour)}:${zeroPad(time.minute)}`
  const quoted = (p: string) => `"${p}"`

  // The canned script hand-off — same tokens as the in-app path, minus the
  // `cmd start` wrapper (Task Scheduler is already the detacher) and plus the
  // -Unattended -NoUi switches: guarded relaunch, silent at 2am. DesktopPid 0
  // means "closed — nothing to wait out" (windows.ps1 skips the wait).
  const action = hasScript
    ? [
        'powershell.exe',
        '-NoProfile',
        '-ExecutionPolicy',
        'Bypass',
        '-WindowStyle',
        'Hidden',
        '-File',
        quoted(updaterPath),
        '-InstallRoot',
        quoted(installRoot),
        '-Branch',
        branch,
        '-DesktopPid',
        '0',
        '-RelaunchExe',
        quoted(relaunchExe),
        '-Unattended',
        '-NoUi'
      ].join(' ')
    : `${quoted(updaterPath)} --update --branch ${branch}`

  return {
    taskName: UNATTENDED_TASK_NAME,
    handoff: hasScript ? 'script' : 'binary',
    updaterPath,
    installRoot,
    relaunchExe,
    branch,
    hour: time.hour,
    minute: time.minute,
    startClock,
    action,
    // NO /ru, NO /rp: per-user task, runs only while THIS user is logged on
    // with an interactive (non-elevated) token. /rl LIMITED = never run with
    // the highest available privileges. /f = idempotent force-recreate.
    schtasksArgs: [
      '/create',
      '/tn',
      UNATTENDED_TASK_NAME,
      '/tr',
      `"${action}"`,
      '/sc',
      'daily',
      '/st',
      startClock,
      '/rl',
      'LIMITED',
      '/f'
    ],
    schtasksDeleteArgs: ['/delete', '/tn', UNATTENDED_TASK_NAME, '/f'],
    schtasksQueryArgs: ['/query', '/tn', UNATTENDED_TASK_NAME, '/xml']
  }
}

// ── Read-back verification (`schtasks /query /xml`) ─────────────────────────

export interface ParsedTaskAction {
  command: string
  arguments: string
}

function decodeXmlBuffer(buffer: string | Buffer): string {
  const raw = typeof buffer === 'string' ? Buffer.from(buffer, 'latin1') : buffer

  if (raw.length >= 2 && raw[0] === 0xff && raw[1] === 0xfe) {
    return raw.toString('utf16le')
  }

  if (raw.length >= 2 && raw[0] === 0xfe && raw[1] === 0xff) {
    return raw.swap16().toString('utf16le')
  }

  if (raw.length >= 3 && raw[0] === 0xef && raw[1] === 0xbb && raw[2] === 0xbf) {
    return raw.toString('utf8')
  }

  return raw.toString('utf8')
}

function unescapeXmlEntities(s: string): string {
  return s
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'")
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&amp;/g, '&')
}

/**
 * Extract <Command> and <Arguments> from a `schtasks /query /xml` document.
 * schtasks can emit UTF-16 (BOM'd) regardless of the console codepage, so the
 * buffer is decoded by BOM before parsing. Only the leaf elements are matched
 * (a generic tag regex would let the <Task> root swallow the whole document).
 * Returns null when the document has no Exec action (or is garbage) — the
 * caller then cannot verify ownership and must not touch the task.
 */
const COMMAND_RE = /<Command>([\s\S]*?)<\/Command>/g
const ARGUMENTS_RE = /<Arguments>([\s\S]*?)<\/Arguments>/g

export function parseTaskActionFromXml(xml: string | Buffer): ParsedTaskAction | null {
  const text = decodeXmlBuffer(xml)
  // `g`-flagged regexes keep lastIndex — reset before each exec so repeated
  // calls on different documents never resume in the middle of a match.
  COMMAND_RE.lastIndex = 0
  const command = COMMAND_RE.exec(text)?.[1]
  ARGUMENTS_RE.lastIndex = 0
  const argumentsText = ARGUMENTS_RE.exec(text)?.[1]

  if (typeof command !== 'string' || typeof argumentsText !== 'string') {
    return null
  }

  const cleaned = (s: string) => unescapeXmlEntities(s.trim())

  return { command: cleaned(command), arguments: cleaned(argumentsText) }
}

/** Collapse runs of whitespace so read-back vs. spec comparison is immune to
 *  incidental spacing differences in the stored action. */
function normalizeForCompare(s: string): string {
  return s.split(/\s+/).join(' ').trim()
}

/** Task Scheduler splits the stored action at the first token; Command may
 *  come back with or without a surrounding quote pair. Strip one pair when
 *  present so both sides compare equal. */
function stripSurroundingQuotes(s: string): string {
  return s.length >= 2 && s.startsWith('"') && s.endsWith('"') ? s.slice(1, -1) : s
}

/**
 * Strip a single quote pair around the FIRST token only. Task Scheduler may
 * echo the stored <Command> with or without quotes; a quoted executable path
 * is the same command. Quotes INSIDE later tokens are significant and kept.
 */
function stripFirstTokenQuotes(s: string): string {
  const tokens = s.split(/\s+/)

  if (tokens.length > 1 && tokens[0].startsWith('"') && tokens[0].endsWith('"')) {
    tokens[0] = tokens[0].slice(1, -1)
  }

  return tokens.join(' ').trim()
}

/**
 * Exact ownership check: does the read-back action equal OUR spec's action?
 * A task passes only when every token, path, branch and flag matches what we
 * would register today — a diverging task is not ours.
 */
export function taskActionEquals(parsed: ParsedTaskAction, spec: ScheduledTaskSpec): boolean {
  const command = normalizeForCompare(parsed.command)
  const args = normalizeForCompare(parsed.arguments)

  if (!command || !args) {
    return false
  }

  const parsedAction = stripFirstTokenQuotes(`${command} ${args}`)
  const expectedAction = stripFirstTokenQuotes(normalizeForCompare(spec.action))

  return parsedAction === expectedAction
}

/** Shape check: action s (single line) structurally looks like OURS — fixed
 *  token sequence, any safe paths/branch — regardless of exact values. This
 *  distinguishes "our task with moved paths" (safe to overwrite) from a
 *  genuinely foreign task (never touched). */
export function isOurActionShape(action: string): boolean {
  if (typeof action !== 'string') {
    return false
  }

  const scriptShape =
    /^powershell\.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File ".+" -InstallRoot ".+" -Branch [A-Za-z0-9][A-Za-z0-9._/-]* -DesktopPid 0 -RelaunchExe ".+" -Unattended -NoUi$/

  const binaryShape = /^"[^"]+\\hermes-setup\.exe" --update --branch [A-Za-z0-9][A-Za-z0-9._/-]*$/

  return scriptShape.test(action) || binaryShape.test(action)
}

/** Classify a read-back task against our current spec. */
export function classifyTaskState(
  parsed: ParsedTaskAction | null,
  spec: ScheduledTaskSpec | null
): UnattendedTaskState {
  if (!parsed) {
    return { kind: 'foreign', message: 'existing task action could not be read' }
  }

  if (!spec) {
    // No spec to compare against — if the shape is ours we still own it,
    // otherwise it is foreign.
    if (isOurActionShape(normalizeTaskAction(parsed))) {
      return { kind: 'installed', message: 'existing task matches the Hermes updater hand-off shape' }
    }

    return { kind: 'foreign', message: 'a task with this name is not ours' }
  }

  if (taskActionEquals(parsed, spec)) {
    return { kind: 'installed' }
  }

  if (isOurActionShape(normalizeTaskAction(parsed))) {
    return { kind: 'stale', message: 'existing task is ours but out of date (paths or schedule changed)' }
  }

  return { kind: 'foreign', message: 'a task with this name is not ours' }
}

function normalizeTaskAction(parsed: ParsedTaskAction): string {
  return normalizeForCompare(`${stripSurroundingQuotes(parsed.command)} ${normalizeForCompare(parsed.arguments)}`)
}
