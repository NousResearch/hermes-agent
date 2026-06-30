const PROVIDER_SETUP_ERROR_RE =
  /No (?:inference|Hermes) provider(?: is)? configured|No usable credentials found for \S+|no_provider_configured|OPENROUTER_API_KEY|OPENAI_API_KEY|ANTHROPIC_API_KEY|set an API key/i

export function isProviderSetupErrorMessage(message: null | string | undefined): boolean {
  const text = message?.trim()

  if (!text) {
    return false
  }

  return PROVIDER_SETUP_ERROR_RE.test(text)
}
