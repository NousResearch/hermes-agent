const PROVIDER_SETUP_ERROR_RE =
  /No (?:inference|Hermes) provider(?: is)? configured|no_provider_configured|No Codex credentials stored|Codex auth (?:state is missing tokens|is missing (?:access_token|refresh_token))|OPENROUTER_API_KEY|OPENAI_API_KEY|ANTHROPIC_API_KEY|set an API key/i

export function isProviderSetupErrorMessage(message: null | string | undefined): boolean {
  const text = message?.trim()

  if (!text) {
    return false
  }

  return PROVIDER_SETUP_ERROR_RE.test(text)
}
