interface AppEnvSandbox {
  hermesHome: string
  userDataDir: string
}

const INHERITED_DESKTOP_OVERRIDE_NAMES = [
  'HERMES_DESKTOP_DEV_SERVER',
  'HERMES_DESKTOP_REMOTE_URL',
  'HERMES_DESKTOP_REMOTE_TOKEN',
  'HERMES_DESKTOP_BOOT_FAKE',
  'HERMES_DESKTOP_BOOT_FAKE_ERROR',
  'HERMES_DESKTOP_IS_PACKAGED',
  'HERMES_DESKTOP_FORCE_DEV',
] as const

const CREDENTIAL_SUFFIXES: string[] = [
  '_API_KEY',
  '_TOKEN',
  '_SECRET',
  '_PASSWORD',
  '_CREDENTIALS',
  '_ACCESS_KEY',
  '_PRIVATE_KEY',
  '_OAUTH_TOKEN',
]

const CREDENTIAL_NAMES = new Set([
  'ANTHROPIC_BASE_URL',
  'ANTHROPIC_TOKEN',
  'AWS_ACCESS_KEY_ID',
  'AWS_SECRET_ACCESS_KEY',
  'AWS_SESSION_TOKEN',
  'CUSTOM_API_KEY',
  'GEMINI_BASE_URL',
  'OPENAI_BASE_URL',
  'OPENROUTER_BASE_URL',
  'OLLAMA_BASE_URL',
  'GROQ_BASE_URL',
  'XAI_BASE_URL',
])

function isCredentialEnvVar(name: string): boolean {
  if (CREDENTIAL_NAMES.has(name)) {
    return true
  }

  return CREDENTIAL_SUFFIXES.some((suffix) => name.endsWith(suffix))
}

function stripCredentials(env: Record<string, string | undefined>): Record<string, string> {
  const clean: Record<string, string> = {}

  for (const [key, value] of Object.entries(env)) {
    if (!value) {
      continue
    }

    if (isCredentialEnvVar(key)) {
      continue
    }

    clean[key] = value
  }

  return clean
}

export function buildAppEnvFromParent(
  parentEnv: Record<string, string | undefined>,
  sandbox: AppEnvSandbox,
  repoRoot: string,
  extra: Record<string, string> = {},
): Record<string, string> {
  const clean = stripCredentials(parentEnv)

  for (const name of INHERITED_DESKTOP_OVERRIDE_NAMES) {
    delete clean[name]
  }

  // XDG_RUNTIME_DIR is needed for Electron on Linux when running in a
  // headless/CI context — without it the zygote may fail to initialize.
  if (!clean.XDG_RUNTIME_DIR && parentEnv.XDG_RUNTIME_DIR) {
    clean.XDG_RUNTIME_DIR = parentEnv.XDG_RUNTIME_DIR
  }

  // DISPLAY — needed for Electron to open a window.
  if (!clean.DISPLAY && parentEnv.DISPLAY) {
    clean.DISPLAY = parentEnv.DISPLAY
  }

  return {
    ...clean,
    HERMES_HOME: sandbox.hermesHome,
    HERMES_DESKTOP_USER_DATA_DIR: sandbox.userDataDir,
    HERMES_DESKTOP_IGNORE_EXISTING: '1',
    // Electron 41 can classify `electron <desktop-dir>` as packaged on
    // Windows. E2E development fixtures need an explicit mode; the packaged
    // fixture removes this before it launches the real bundle.
    HERMES_DESKTOP_FORCE_DEV: '1',
    HERMES_DESKTOP_HERMES_ROOT: repoRoot,
    HERMES_DESKTOP_APP_NAME: `HermesE2E-${Date.now()}`,
    // `app.close()` in teardown must exit even when a spec leaves a turn
    // mid-flight — otherwise the quit confirmation waits on a click that no
    // one is there to make, and the worker dies on a teardown timeout.
    HERMES_DESKTOP_SKIP_QUIT_CONFIRM: '1',
    ...extra,
  }
}
