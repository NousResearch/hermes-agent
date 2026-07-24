// The backend localizes auth.no_provider_configured per display.language
// (locales/*.yaml), so the CJK openings of that catalog entry are recognized
// alongside the English shapes and the stable error code.
const PROVIDER_SETUP_ERROR_RE =
  /No (?:inference|Hermes) provider(?: is)? configured|no_provider_configured|set an API key|未配置推理提供方|未設定推理提供方|推論プロバイダーが設定されていません/i

const SESSION_INFO_CREDENTIAL_WARNING_RE = /^No API key configured for provider '[^']*'\. First message will fail\.$/

export function isProviderSetupErrorMessage(message: null | string | undefined): boolean {
  const text = message?.trim()

  if (!text) {
    return false
  }

  return PROVIDER_SETUP_ERROR_RE.test(text) || SESSION_INFO_CREDENTIAL_WARNING_RE.test(text)
}
