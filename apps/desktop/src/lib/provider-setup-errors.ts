// Fallback for gateways that send no `code` on the error event. Every alternative here is a
// sentence the backend actually produces today: `agent_init` raises "No LLM provider configured"
// on a blank install and `missing_provider_credentials_message()` the "is set in config.yaml
// but …" form, and both arrive wrapped in `agent_init_failed_message()`. Matching is by the
// noun phrase, never the surrounding wording, and deliberately NOT by "provider configured"
// alone: the auxiliary-model warning says "No auxiliary LLM provider configured" and is not a
// provider-setup failure.
const PROVIDER_SETUP_ERROR_RE =
  /No (?:inference|Hermes|LLM) provider(?: is)? configured|no_provider_configured|set an API key|is set in config\.yaml but no (?:API key|credentials)/i

const SESSION_INFO_CREDENTIAL_WARNING_RE = /^No API key configured for provider '[^']*'\. First message will fail\.$/

/** True when the gateway named this failure itself: preferred over reading the sentence. */
export function isProviderSetupErrorCode(code: null | string | undefined): boolean {
  return code === 'provider_not_configured'
}

export function isProviderSetupErrorMessage(message: null | string | undefined): boolean {
  const text = message?.trim()

  if (!text) {
    return false
  }

  return PROVIDER_SETUP_ERROR_RE.test(text) || SESSION_INFO_CREDENTIAL_WARNING_RE.test(text)
}
