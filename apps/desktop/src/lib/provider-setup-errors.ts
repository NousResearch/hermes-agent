// "Hermes" stays in the alternation: the Python gateway (tui_gateway/server.py)
// still emits "No Hermes provider is configured." on the wire — that's backend
// copy, not desktop branding. "IX Agency" covers the rebranded desktop copy.
const PROVIDER_SETUP_ERROR_RE =
  /No (?:inference|Hermes|IX Agency) provider(?: is)? configured|no_provider_configured|OPENROUTER_API_KEY|OPENAI_API_KEY|ANTHROPIC_API_KEY|set an API key/i

export function isProviderSetupErrorMessage(message: null | string | undefined): boolean {
  const text = message?.trim()

  if (!text) {
    return false
  }

  return PROVIDER_SETUP_ERROR_RE.test(text)
}
