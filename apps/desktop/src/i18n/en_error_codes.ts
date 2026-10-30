import type { Translations } from './types'

type ThreadCopy = Translations['assistant']['thread']

// The error card's per-code title and body (lib/error-surface-copy.ts), keyed by error_surface.code.
export const enErrorCodes: ThreadCopy['errorCodes'] = {
  auth: {
    title: provider => `${provider} rejected your API key`,
    body: () => 'The saved key is invalid or was revoked. Update it, then send your message again.'
  },
  auth_permanent: {
    title: provider => `${provider} rejected your API key`,
    body: () => 'The saved key is invalid or was revoked. Update it, then send your message again.'
  },
  billing: {
    title: provider => `${provider} is out of credits`,
    body: provider => `Add credits to keep using paid models on ${provider}.`
  },
  rate_limit: {
    title: provider => `${provider} is busy right now`,
    body: provider => `You've hit ${provider}'s rate limit. Wait a moment, then retry.`
  },
  upstream_rate_limit: {
    title: (_provider, model) => `${model || 'This model'} is busy right now`,
    body: (provider, model) =>
      `${model || 'This model'} is rate-limited upstream of ${provider}, not on your account. Retry in a moment.`
  },
  overloaded: {
    title: provider => `${provider} is busy right now`,
    body: provider => `${provider} is overloaded and couldn't take this request. Retry in a moment.`
  },
  server_error: {
    title: provider => `${provider} is having problems`,
    body: provider => `${provider} returned a server error. Retry in a moment.`
  },
  timeout: {
    title: provider => `Can't reach ${provider}`,
    body: provider => `Hermes couldn't reach ${provider}. Check your connection, then retry.`
  },
  stream_drop: {
    title: 'The reply was cut off',
    body: 'The connection dropped before the reply finished. Retry to send it again.'
  },
  no_reply: {
    title: "The reply didn't arrive",
    body: 'Hermes finished this turn, but no reply reached this window. Retry to send it again.'
  },
  upstream_blocked: {
    title: provider => `${provider} blocked this request`,
    body: provider =>
      `A firewall in front of ${provider} refused it before it reached the model. Switch provider, or set a custom User-Agent for it.`
  },
  ssl_cert_verification: {
    title: provider => `Can't verify ${provider}'s connection`,
    body: provider =>
      `A proxy or an outdated certificate store blocked the secure connection to ${provider}. The setup guide shows how to fix it.`
  },
  context_overflow: {
    title: 'This chat is too long',
    body: (_provider, model) =>
      `The conversation no longer fits ${model ? `${model}'s` : "the model's"} context window. Compress it to keep going.`
  },
  payload_too_large: {
    title: 'This message is too large',
    body: provider =>
      `The request, including attachments, is over ${provider}'s size limit. Compress the conversation or remove attachments.`
  },
  model_not_found: {
    title: (_provider, model) => `${model || 'This model'} isn't available`,
    body: (provider, model) =>
      `${provider} doesn't offer ${model || 'this model'} on your account. Choose another model.`
  },
  provider_policy_blocked: {
    title: (provider, model) => `${provider} settings block ${model || 'this model'}`,
    body: provider =>
      `Your ${provider} privacy settings exclude every endpoint for this model. Change them on ${provider}, or choose another model.`
  },
  content_policy_blocked: {
    title: provider => `${provider} declined this message`,
    body: provider => `${provider}'s safety filter refused it. Edit the message and send it again.`
  },
  format_error: {
    title: provider => `${provider} rejected this request`,
    body: provider =>
      `${provider} couldn't accept how this conversation is structured. Switch provider, or start a new chat if it keeps happening.`
  },
  truncated: {
    title: 'The reply was cut short',
    body: (_provider, model) =>
      `${model || 'The model'} hit its output limit before finishing, so nothing was run. Continue to pick up where it stopped.`
  },
  invalid_response: {
    title: "Couldn't read the reply",
    body: provider => `${provider} sent back an empty or broken reply. Retry in a moment.`
  },
  empty_response: {
    title: "Couldn't get a reply",
    body: provider => `${provider} returned nothing for this message. Retry in a moment.`
  },
  loop_error: {
    title: "Hermes couldn't finish this step",
    body: 'Something went wrong handling the reply. Retry; if it happens again, send diagnostics from the details.'
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
  agent_init_failed: {
    title: "Couldn't start this chat",
    body: provider =>
      `Hermes couldn't connect to ${provider} for this chat. Retry; if it keeps failing, check the provider in Settings.`
  },
  SESSION_NOT_OWNED: {
    title: 'Chat open somewhere else',
    body: 'This chat is open in another Hermes window or terminal. Start a new chat here, or close it there first.'
  },
  disk_full: {
    title: 'Your disk is full',
    body: "Hermes couldn't save this conversation. Free up space, then retry."
  },
  // Nous free tier. The body is normally the backend's own sentence (it names the wait
  // and the way forward); these bodies stand in for an older backend that sent none.
  free_tier_disabled: {
    title: 'Sign in to keep chatting',
    body: "Sign in with a Nous account to keep chatting, it's free."
  },
  free_tier_rate_limited: {
    title: 'Free allowance used up',
    body: "It refreshes shortly. Sign in with a Nous account for a bigger allowance, it's free."
  },
  free_tier_at_capacity: {
    title: 'Free tier is busy',
    body: "Sign in to skip the queue, it's free, or try again in a little while."
  },
  free_tier_model_not_free: {
    title: 'This model needs a sign-in',
    body: "Hermes uses the free model for now. Sign in with a Nous account for more models, it's free."
  },
  free_tier_route: {
    title: "Can't reach the free model",
    body: "Sign in with a Nous account, it's free, or check the NOUS_INFERENCE_BASE_URL setting."
  },
  free_tier_outage: {
    title: 'Free model is having trouble',
    body: 'Try sending your message again in a minute.'
  },
  free_tier_refused: {
    title: 'Sign in to keep chatting',
    body: 'Signing in with a Nous account is free.'
  },
  usage_window: {
    title: 'Included usage used up',
    body: "Your plan's included usage is used up until it resets. Add credits in /topup to keep going."
  }
}
