import type { Translations } from './types'

type ThreadCopy = Translations['assistant']['thread']

// The error card's per-code title and body (lib/error-surface-copy.ts), keyed by error_surface.code.
export const enErrorCodes: ThreadCopy['errorCodes'] = {
  auth: {
    title: provider => `${provider} rejected your sign-in`,
    body: provider =>
      `The credentials saved for ${provider} were not accepted. Fix them in Settings or switch provider, then send your message again.`
  },
  auth_permanent: {
    title: provider => `${provider} rejected your sign-in`,
    body: provider =>
      `The credentials saved for ${provider} are invalid or were revoked. Update them or switch provider, then send your message again.`
  },
  billing: {
    title: 'Out of credits',
    body: provider => `Your ${provider} account has no credits left. Top up or switch provider, then send again.`
  },
  rate_limit: {
    title: 'The AI service is busy',
    body: provider => `${provider} is limiting requests right now. Wait a minute, then retry.`
  },
  upstream_rate_limit: {
    title: 'The AI service is busy',
    body: provider => `${provider} is limiting requests right now. Wait a minute, then retry.`
  },
  overloaded: {
    title: 'The AI service is overloaded',
    body: provider => `${provider} is having problems right now. Retry in a moment or switch provider.`
  },
  server_error: {
    title: 'The AI service had a problem',
    body: provider => `${provider} returned a server error. Retry in a moment or switch provider.`
  },
  timeout: {
    title: 'Could not reach the AI service',
    body: provider =>
      `${provider} could not be reached or did not answer in time. Check your internet connection, then retry.`
  },
  stream_drop: {
    title: 'The reply was cut off',
    body: 'The connection dropped before the reply finished. Retry to send it again.'
  },
  no_reply: {
    title: "The reply didn't finish",
    body: 'Hermes ended this turn without a reply. Retry to send it again.'
  },
  upstream_blocked: {
    title: 'A firewall blocked the request',
    body: provider =>
      `A firewall or CDN in front of ${provider} blocked the request before it reached the model — your key is probably fine. Set a User-Agent header via the provider's extra_headers in Settings, or switch provider, then send your message again.`
  },
  ssl_cert_verification: {
    title: 'Secure connection failed',
    body: provider =>
      `Hermes could not verify the secure connection to ${provider}. Check your network or proxy settings, or switch provider, then send your message again.`
  },
  context_overflow: {
    title: 'This conversation is too long',
    body: 'The conversation no longer fits the model. Compress it or start a new chat, then send again.'
  },
  payload_too_large: {
    title: 'This message is too large',
    body: 'The request was too big for the model. Compress the conversation or start a new chat, then send again.'
  },
  model_not_found: {
    title: 'This model is not available',
    body: provider =>
      `${provider} does not offer this model on your account. Choose another model, then send your message again.`
  },
  provider_policy_blocked: {
    title: 'This model is blocked by your account settings',
    body: provider =>
      `${provider} would not route this request under your account's data or privacy settings. Choose another model or switch provider.`
  },
  content_policy_blocked: {
    title: 'The AI service declined this request',
    body: provider => `${provider} would not answer this message. Edit it and send again.`
  },
  format_error: {
    title: 'The AI service rejected the request',
    body: provider =>
      `${provider} did not accept how this request was built. Switch provider or send diagnostics so we can look into it.`
  },
  truncated: {
    title: 'The reply was cut short',
    body: 'The model stopped before finishing. Retry to get a complete reply.'
  },
  invalid_response: {
    title: 'The AI service sent an unreadable reply',
    body: provider => `${provider} returned something Hermes could not read. Retry in a moment.`
  },
  empty_response: {
    title: 'The AI service sent an empty reply',
    body: provider => `${provider} returned nothing for this message. Retry in a moment.`
  },
  loop_error: {
    title: 'Hermes got stuck in a loop',
    body: 'The reply kept repeating the same steps, so Hermes stopped it. Retry, or start a new chat if it happens again.'
  },
  SESSION_OWNER_BUSY: {
    title: 'Hermes is busy in this chat',
    body: 'Another turn is running here and your message could not join its queue. Nothing was sent. Retry once that turn ends.'
  },
  cron_session_read_only: {
    title: 'This chat belongs to a scheduled job',
    body: 'It only runs that job, so your message was not run. Start a new chat to send it.'
  },
  session_read_only: {
    title: 'This chat is read-only',
    body: 'It is stored history Hermes can show but not continue, so your message was not run. Start a new chat to send it.'
  },
  workspace_unavailable: {
    title: "This chat's folder is gone",
    body: 'The folder this chat works in was moved or deleted, so your message was not run. Recreate the folder, or use Move to project to give the chat another one, then retry.'
  },
  launch_credentials_unavailable: {
    title: "This chat's API key changed",
    body: 'This chat started with an API key that has since changed or been removed, so your message was not sent. An open chat never switches to a different key on its own. Start a new session to use your current key.'
  },
  SESSION_OWNER_UNCONFIRMED: {
    title: 'Your message may already be queued',
    body: 'The gateway did not confirm your message in time. If it was queued, it runs once the earlier turn finishes or is resolved. Check the chat before sending it again.'
  },
  SESSION_NOT_OWNED: {
    title: 'This chat is open somewhere else',
    body: 'This chat is currently open in another Hermes window or terminal. Close it there and send your message again, or start a new chat here.'
  },
  disk_full: {
    title: 'Disk full',
    body: 'Your disk is full, so Hermes could not save this conversation. Free some space, then retry.'
  },
  // Nous free tier. The body is normally the backend's own sentence (it names the wait
  // and the way forward); these bodies stand in for an older backend that sent none.
  free_tier_disabled: {
    title: 'Using Hermes without signing in is switched off right now',
    body: "Sign in with a Nous account to keep chatting, it's free."
  },
  free_tier_rate_limited: {
    title: "You've used up the allowance for chatting without signing in",
    body: "It refreshes shortly. Sign in with a Nous account for a bigger allowance, it's free."
  },
  free_tier_at_capacity: {
    title: 'Chatting without signing in is really busy right now',
    body: "Sign in to skip the queue, it's free, or try again in a little while."
  },
  free_tier_model_not_free: {
    title: "That model isn't available without signing in",
    body: "Hermes uses the free model for now. Sign in with a Nous account for more models, it's free."
  },
  free_tier_route: {
    title: "Hermes couldn't reach the free model on this route",
    body: "Sign in with a Nous account, it's free, or check the NOUS_INFERENCE_BASE_URL setting."
  },
  free_tier_outage: {
    title: 'The free model is having trouble responding right now',
    body: 'Try sending your message again in a minute.'
  },
  free_tier_refused: {
    title: "Hermes couldn't send that without signing in",
    body: 'Signing in with a Nous account is free.'
  },
  usage_window: {
    title: 'Included usage used up',
    body: "Your plan's included usage is used up until it resets. Add credits in /topup to keep going."
  }
}
