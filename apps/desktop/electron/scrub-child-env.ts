/** Environment maps accepted by Node child-process APIs. */
export type EnvMap = Record<string, string | undefined>

// Mirrors the credential entries owned by Hermes' provider/tool/messaging
// registries. Keep unrelated operator credentials (for example NPM_TOKEN),
// the general AWS chain, and CLAUDE_CODE_OAUTH_TOKEN available to the user's
// shell; Python applies the same ownership boundary in environments/local.py.
const HERMES_CREDENTIAL_NAMES = new Set([
  'A2A_BEARER_TOKEN',
  'A2A_PEER_TOKENS',
  'ACTUAL_API_KEY',
  'AIRTABLE_API_KEY',
  'AI_GATEWAY_API_KEY',
  'ALIBABA_CODING_PLAN_API_KEY',
  'ANTHROPIC_API_KEY',
  'ANTHROPIC_TOKEN',
  'API_SERVER_KEY',
  'ARCEEAI_API_KEY',
  'AZURE_FOUNDRY_API_KEY',
  'BLUEBUBBLES_PASSWORD',
  'BRAVE_SEARCH_API_KEY',
  'BROWSERBASE_API_KEY',
  'BROWSER_USE_API_KEY',
  'BRV_API_KEY',
  'BUZZ_PRIVATE_KEY',
  'CAMOFOX_API_KEY',
  'CLOAKBROWSER_PROXY',
  'COMMANDCODE_API_KEY',
  'COPILOT_GITHUB_TOKEN',
  'CUSTOM_API_KEY',
  'DASHSCOPE_API_KEY',
  'DAYTONA_API_KEY',
  'DEEPINFRA_API_KEY',
  'DEEPSEEK_API_KEY',
  'DINGTALK_CLIENT_SECRET',
  'DISCORD_BOT_TOKEN',
  'ELEVENLABS_API_KEY',
  'EMAIL_PASSWORD',
  'EXA_API_KEY',
  'FAL_KEY',
  'FEISHU_APP_SECRET',
  'FIRECRAWL_API_KEY',
  'FIREWORKS_API_KEY',
  'FREEBUFF_PROXY_API_KEY',
  'FREEBUFF_TOKEN',
  'FREELLMAPI_API_KEY',
  'GATEWAY_PROXY_KEY',
  'GATEWAY_RELAY_DELIVERY_KEY',
  'GATEWAY_RELAY_ID',
  'GATEWAY_RELAY_SECRET',
  'GEMINI_API_KEY',
  'GH_TOKEN',
  'GITHUB_APP_ID',
  'GITHUB_APP_INSTALLATION_ID',
  'GITHUB_APP_PRIVATE_KEY_PATH',
  'GITHUB_TOKEN',
  'GLM_API_KEY',
  'GMI_API_KEY',
  'GOOGLE_API_KEY',
  'GOOGLE_APPLICATION_CREDENTIALS',
  'GOOGLE_CHAT_SERVICE_ACCOUNT_JSON',
  'HASS_TOKEN',
  'HERMES_LANGFUSE_SECRET_KEY',
  'HF_TOKEN',
  'HINDSIGHT_API_KEY',
  'HONCHO_API_KEY',
  'IRC_NICKSERV_PASSWORD',
  'IRC_SERVER_PASSWORD',
  'KILOCODE_API_KEY',
  'KIMI_API_KEY',
  'KIMI_CN_API_KEY',
  'KIMI_CODING_API_KEY',
  'KREA_API_KEY',
  'LANGFUSE_SECRET_KEY',
  'LINEAR_API_KEY',
  'LINE_CHANNEL_ACCESS_TOKEN',
  'LINE_CHANNEL_SECRET',
  'LM_API_KEY',
  'MATTERMOST_TOKEN',
  'MATRIX_ACCESS_TOKEN',
  'MATRIX_PASSWORD',
  'MATRIX_RECOVERY_KEY',
  'MEM0_API_KEY',
  'META_API_KEY',
  'META_MODEL_API_KEY',
  'MINIMAX_API_KEY',
  'MINIMAX_CN_API_KEY',
  'MISTRAL_API_KEY',
  'MODAL_TOKEN_ID',
  'MODAL_TOKEN_SECRET',
  'MODEL_API_KEY',
  'NOTION_API_KEY',
  'NOVITA_API_KEY',
  'NTFY_TOKEN',
  'NVIDIA_API_KEY',
  'OLLAMA_API_KEY',
  'OPENCODE_API_KEY',
  'OPENCODE_GO_API_KEY',
  'OPENCODE_ZEN_API_KEY',
  'OPENAI_API_KEY',
  'OPENROUTER_API_KEY',
  'OPENVIKING_API_KEY',
  'PARALLEL_API_KEY',
  'PHOTON_PROJECT_SECRET',
  'PORCUPINE_ACCESS_KEY',
  'QQ_CLIENT_SECRET',
  'RETAINDB_API_KEY',
  'SITDECK_PASSWORD',
  'SLACK_APP_TOKEN',
  'SLACK_BOT_TOKEN',
  'STEPFUN_API_KEY',
  'SUDO_PASSWORD',
  'SUPERMEMORY_API_KEY',
  'TAVILY_API_KEY',
  'TEAMS_CLIENT_SECRET',
  'TELEGRAM_BOT_TOKEN',
  'TENOR_API_KEY',
  'TERMINAL_SSH_KEY',
  'TOGETHER_API_KEY',
  'TOOL_GATEWAY_USER_TOKEN',
  'TWILIO_AUTH_TOKEN',
  'UPSTAGE_API_KEY',
  'VERCEL_OIDC_TOKEN',
  'VERCEL_TOKEN',
  'VERTEX_CREDENTIALS_PATH',
  'VOICE_TOOLS_OPENAI_KEY',
  'WEBHOOK_SECRET',
  'WECOM_CALLBACK_CORP_SECRET',
  'WECOM_CALLBACK_ENCODING_AES_KEY',
  'WECOM_CALLBACK_TOKEN',
  'WECOM_SECRET',
  'WORLDMONITOR_API_KEY',
  'XAI_API_KEY',
  'XIAOMI_API_KEY',
  'ZAI_API_KEY',
  'Z_AI_API_KEY'
])

const FORWARDED_ENV_PREFIXES = ['APPTAINERENV_', 'SINGULARITYENV_']

function effectiveEnvName(name: string): string {
  let normalized = name.trim().toUpperCase()
  let changed = true

  while (changed) {
    changed = false

    for (const prefix of FORWARDED_ENV_PREFIXES) {
      if (normalized.startsWith(prefix)) {
        normalized = normalized.slice(prefix.length)
        changed = true
      }
    }
  }

  return normalized
}

export function isHermesCredentialEnvVar(name: string): boolean {
  const normalized = effectiveEnvName(name)

  if (!normalized) {
    return false
  }

  if (normalized.startsWith('_HERMES_FORCE_')) {
    return true
  }

  if (HERMES_CREDENTIAL_NAMES.has(normalized)) {
    return true
  }

  if (/^HERMES_[A-Z0-9_]+_(?:API_KEY|TOKEN|SECRET|PASSWORD|PRIVATE_KEY)$/.test(normalized)) {
    return true
  }

  if (/^AUXILIARY_[A-Z0-9_]+_(?:API_KEY|BASE_URL)$/.test(normalized)) {
    return true
  }

  return /^GATEWAY_RELAY_[A-Z0-9_]+_(?:SECRET|KEY|TOKEN)$/.test(normalized)
}

/** Merge sources while removing only credentials owned by Hermes. */
export function scrubDesktopChildEnv(...sources: Array<EnvMap | null | undefined>): Record<string, string> {
  const out: Record<string, string> = {}

  for (const source of sources) {
    for (const [key, value] of Object.entries(source || {})) {
      if (value == null || isHermesCredentialEnvVar(key)) {
        continue
      }

      out[key] = String(value)
    }
  }

  return out
}

export interface DesktopServeChildEnvOptions {
  source?: EnvMap
  backendEnv?: EnvMap
  hermesHome: string
  terminalCwd: string
  dashboardSessionToken: string
  parentIdentityEnv?: EnvMap
  webDist: string
  readyFile?: string | null
}

/** Build the local Desktop backend env and re-add only its freshly minted token. */
export function buildDesktopServeChildEnv({
  source,
  backendEnv,
  hermesHome,
  terminalCwd,
  dashboardSessionToken,
  parentIdentityEnv,
  webDist,
  readyFile
}: DesktopServeChildEnvOptions): Record<string, string> {
  const env = scrubDesktopChildEnv(source, backendEnv, parentIdentityEnv)

  env.HERMES_HOME = hermesHome
  env.TERMINAL_CWD = terminalCwd
  env.HERMES_DASHBOARD_SESSION_TOKEN = dashboardSessionToken
  env.HERMES_DESKTOP = '1'
  env.HERMES_WEB_DIST = webDist

  if (readyFile) {
    env.HERMES_DESKTOP_READY_FILE = readyFile
  }

  return env
}
