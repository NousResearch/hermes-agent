type TerminalShellEnvOptions = {
  appVersion: string
  currentEnv?: NodeJS.ProcessEnv
}

function buildTerminalShellEnv({ appVersion, currentEnv = process.env }: TerminalShellEnvOptions) {
  const env = { ...currentEnv }

  // Electron is commonly launched through `npm run dev`; do not leak npm's
  // managed prefix into a user's interactive shell (nvm/proto warn loudly).
  for (const key of Object.keys(env)) {
    if (key === 'npm_config_prefix' || key.startsWith('npm_config_') || key.startsWith('npm_package_')) {
      delete env[key]
    }
  }

  // Strip color/theme-detection vars that ride along when Electron is launched
  // from a non-tty agent shell (Cursor's runner sets NO_COLOR/FORCE_COLOR=0
  // /TERM=dumb; some terminals set COLORFGBG which would flip Hermes' TUI into
  // light-mode). Our PTY is a real xterm-compat terminal — force truecolor.
  delete env.NO_COLOR
  delete env.FORCE_COLOR
  delete env.COLORFGBG

  env.COLORTERM = 'truecolor'
  env.TERM = 'xterm-256color'
  env.TERM_PROGRAM = 'Hermes'
  env.TERM_PROGRAM_VERSION = appVersion

  // Let a hermes/--tui launched in this pane know it's embedded in the desktop
  // GUI (build_environment_hints surfaces this). Distinct from HERMES_DESKTOP,
  // which marks the agent *backend* and gates cron/gateway behavior.
  env.HERMES_DESKTOP_TERMINAL = '1'

  return env
}

export { buildTerminalShellEnv }
