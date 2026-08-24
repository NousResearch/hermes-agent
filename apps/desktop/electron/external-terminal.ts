// Launching the Hermes TUI in the user's OWN terminal emulator.
//
// This is deliberately NOT the in-app terminal pane: the point of the verb is
// to hand a session to the terminal the user already lives in, running
// `hermes --tui --resume <id>` there. Two problems have to be solved for that
// to work anywhere:
//
//  1. WHAT to run. The desktop's Hermes runtime is often a venv Python invoked
//     as `python -m hermes_cli.main`, not a `hermes` on PATH — so the command
//     and its PYTHONPATH have to be carried over verbatim. We write them into a
//     small launcher script instead of trying to quote a nested command through
//     a terminal emulator's `-e` argument, which every emulator parses
//     differently.
//  2. WHERE to run it. There is no portable "default terminal" API, so each
//     platform gets its own resolution:
//       - macOS: `open` the `.command` script with NO `-a`, letting
//         LaunchServices route it to whichever app the user has bound to shell
//         scripts (Terminal.app by default, iTerm2/Ghostty/WezTerm when they've
//         claimed it). That is the closest thing macOS has to "their terminal".
//       - Linux: an ordered ladder of emulators, led by Debian's
//         `x-terminal-emulator` alternative — which IS the user's configured
//         choice — before falling back to the common concrete emulators.
//       - Windows: Windows Terminal when installed, else a `cmd.exe` console.
//
// Everything here is pure so it can be unit-tested without Electron; the side
// effects (writing the script, spawning) live in main.ts.

/** Argv for resuming a session in the TUI, profile-pinned when we know it. */
export function tuiResumeArgs(sessionId: string, profile?: string): string[] {
  const head = profile ? ['--profile', profile] : []

  return [...head, '--tui', '--resume', sessionId]
}

/** Single-quote a value for /bin/sh (the POSIX launcher script). */
export function posixQuote(value: string): string {
  return `'${String(value ?? '').replaceAll("'", `'\\''`)}'`
}

/** Quote a value for a cmd.exe script line. */
export function windowsQuote(value: string): string {
  return `"${String(value ?? '').replaceAll('"', '""')}"`
}

/**
 * The environment the launcher script exports.
 *
 * PATH is deliberately dropped: the script runs inside a login shell that
 * already has the user's own PATH, and the desktop's PATH (assembled for a
 * headless child) is the wrong answer for an interactive terminal. The Hermes
 * command is invoked by absolute path, so nothing here depends on PATH.
 */
export function terminalScriptEnv(
  backendEnv: Record<string, string | undefined> = {},
  hermesHome?: string
): Record<string, string> {
  const out: Record<string, string> = {}

  for (const [key, value] of Object.entries(backendEnv)) {
    if (key.toUpperCase() === 'PATH' || value === undefined || value === '') {
      continue
    }

    out[key] = value
  }

  if (hermesHome) {
    out.HERMES_HOME = hermesHome
  }

  return out
}

export interface TerminalScriptSpec {
  command: string
  args: string[]
  cwd: string
  env?: Record<string, string>
  platform?: NodeJS.Platform
}

/**
 * The launcher script contents. `exec` on POSIX so the terminal window belongs
 * to the TUI itself rather than an idle shell wrapping it.
 */
export function buildTerminalScript({ command, args, cwd, env = {}, platform = process.platform }: TerminalScriptSpec) {
  const entries = Object.entries(env)

  if (platform === 'win32') {
    return [
      '@echo off',
      `cd /d ${windowsQuote(cwd)}`,
      ...entries.map(([key, value]) => `set ${windowsQuote(`${key}=${value}`)}`),
      [command, ...args].map(windowsQuote).join(' '),
      ''
    ].join('\r\n')
  }

  return [
    '#!/bin/sh',
    `cd ${posixQuote(cwd)} || exit 1`,
    ...entries.map(([key, value]) => `export ${key}=${posixQuote(value)}`),
    `exec ${[command, ...args].map(posixQuote).join(' ')}`,
    ''
  ].join('\n')
}

export function terminalScriptExtension(platform: NodeJS.Platform = process.platform): string {
  if (platform === 'win32') {
    return '.cmd'
  }

  // `.command` is the UTI macOS binds to a terminal app; on Linux the
  // extension is cosmetic (we always name the interpreter explicitly).
  return platform === 'darwin' ? '.command' : '.sh'
}

// Linux emulators in resolution order, with the flag that precedes a program
// to run. `x-terminal-emulator` is Debian/Ubuntu's alternatives symlink to the
// user's chosen terminal, so it leads; the rest are the common concretes.
const LINUX_TERMINALS: Array<{ command: string; flag: string }> = [
  { command: 'x-terminal-emulator', flag: '-e' },
  { command: 'gnome-terminal', flag: '--' },
  { command: 'konsole', flag: '-e' },
  { command: 'xfce4-terminal', flag: '-x' },
  { command: 'tilix', flag: '-e' },
  { command: 'kitty', flag: '' },
  { command: 'alacritty', flag: '-e' },
  { command: 'wezterm', flag: '-e' },
  { command: 'foot', flag: '' },
  { command: 'xterm', flag: '-e' }
]

export interface TerminalLaunchOptions {
  scriptPath: string
  findOnPath: (command: string) => null | string
  platform?: NodeJS.Platform
}

/**
 * Resolve the argv that opens `scriptPath` in a terminal window, or null when
 * no terminal emulator could be found (Linux boxes with none installed).
 */
export function resolveTerminalLaunch({
  scriptPath,
  findOnPath,
  platform = process.platform
}: TerminalLaunchOptions): { command: string; args: string[] } | null {
  if (platform === 'darwin') {
    // No `-a`: LaunchServices picks the user's handler for shell scripts.
    return { command: 'open', args: [scriptPath] }
  }

  if (platform === 'win32') {
    const windowsTerminal = findOnPath('wt.exe')

    if (windowsTerminal) {
      return { command: windowsTerminal, args: ['cmd.exe', '/k', scriptPath] }
    }

    return { command: 'cmd.exe', args: ['/c', 'start', '', 'cmd.exe', '/k', scriptPath] }
  }

  for (const { command, flag } of LINUX_TERMINALS) {
    const resolved = findOnPath(command)

    if (resolved) {
      return { command: resolved, args: [...(flag ? [flag] : []), '/bin/sh', scriptPath] }
    }
  }

  return null
}
