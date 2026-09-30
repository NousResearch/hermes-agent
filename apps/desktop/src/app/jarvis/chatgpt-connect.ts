/**
 * Sign in with a ChatGPT subscription: the engine's `openai-codex` device-code login, end to end.
 *
 * Start a login session, hand the user a code and a page to enter it on, poll until OpenAI answers, then
 * read the served catalog and pick the best GPT model it lists. All I/O is injected so the flow is
 * testable without a network, like the OpenRouter quick connect next to it.
 */

import type { ProfileScope } from '@/api/client'
import type { ModelOptionProvider, ModelOptionsResponse, OAuthPollResponse, OAuthStartResponse } from '@/types/hermes'

export const CHATGPT_PROVIDER_SLUG = 'openai-codex'

/** Best first. The first prefix the subscription actually lists wins, so a new generation is picked up without a release. */
const MODEL_PREFERENCE = ['gpt-6', 'gpt-5.6', 'gpt-5.5', 'gpt-5.4', 'gpt-5.3', 'gpt-5.2', 'gpt-5.1', 'gpt-5'] as const

/** Small, routing and dated variants are not "the ChatGPT model". */
const LESSER = /(mini|nano|lite|preview|\d{4}-\d{2}-\d{2})/i

export function chatGptProvider(providers: readonly ModelOptionProvider[] | undefined): ModelOptionProvider | undefined {
  return providers?.find(provider => provider.slug === CHATGPT_PROVIDER_SLUG)
}

export function chatGptConnected(providers: readonly ModelOptionProvider[] | undefined): boolean {
  const provider = chatGptProvider(providers)

  return Boolean(provider && provider.authenticated !== false && (provider.models?.length ?? 0) > 0)
}

/** The model a fresh ChatGPT sign-in should start on, or undefined when the subscription lists none. */
export function chatGptWorkModel(providers: readonly ModelOptionProvider[] | undefined): string | undefined {
  const provider = chatGptProvider(providers)
  const models = provider?.models ?? []

  if (!provider || provider.authenticated === false || models.length === 0) {
    return undefined
  }

  for (const prefix of MODEL_PREFERENCE) {
    const hit = models.find(model => model.toLowerCase().startsWith(prefix) && !LESSER.test(model))

    if (hit) {
      return hit
    }
  }

  return models.find(model => !LESSER.test(model)) ?? models[0]
}

export interface ChatGptConnectDeps {
  cancel: (sessionId: string, scope?: ProfileScope) => Promise<unknown>
  loadOptions: (scope?: ProfileScope) => Promise<ModelOptionsResponse>
  openUrl: (url: string) => Promise<unknown> | void
  poll: (sessionId: string, scope?: ProfileScope) => Promise<OAuthPollResponse>
  /** Make the chosen model the default; omitted when the caller commits it later (onboarding). */
  setDefaultModel?: (provider: string, model: string, scope?: ProfileScope) => Promise<unknown>
  sleep: (ms: number) => Promise<void>
  start: (scope?: ProfileScope) => Promise<OAuthStartResponse>
}

export type ChatGptConnectResult =
  | { model?: string; ok: true; options: ModelOptionsResponse }
  | { message?: string; ok: false; reason: 'cancelled' | 'denied' | 'error' | 'expired' }

export interface ChatGptConnectOptions {
  /** Called once the code is known: show it, the browser is already being opened. */
  onCode: (code: { url: string; userCode: string }) => void
  pollIntervalMs?: number
  signal?: AbortSignal
}

export async function connectChatGpt(
  deps: ChatGptConnectDeps,
  scope: ProfileScope | undefined,
  { onCode, pollIntervalMs = 3000, signal }: ChatGptConnectOptions
): Promise<ChatGptConnectResult> {
  const args = scope === undefined ? [] : [scope]
  let start: OAuthStartResponse

  try {
    start = await deps.start(...args)
  } catch (error) {
    return { message: error instanceof Error ? error.message : undefined, ok: false, reason: 'error' }
  }

  if (start.flow !== 'device_code') {
    await deps.cancel(start.session_id, ...args).catch(() => undefined)

    return { ok: false, reason: 'error' }
  }

  onCode({ url: start.verification_url, userCode: start.user_code })
  void Promise.resolve(deps.openUrl(start.verification_url)).catch(() => undefined)

  const interval = Math.max(1000, (start.poll_interval || 0) * 1000, pollIntervalMs)
  const deadline = Date.now() + Math.max(1, start.expires_in) * 1000

  while (Date.now() < deadline) {
    if (signal?.aborted) {
      await deps.cancel(start.session_id, ...args).catch(() => undefined)

      return { ok: false, reason: 'cancelled' }
    }

    await deps.sleep(interval)

    if (signal?.aborted) {
      continue
    }

    let status: OAuthPollResponse

    try {
      status = await deps.poll(start.session_id, ...args)
    } catch (error) {
      return { message: error instanceof Error ? error.message : undefined, ok: false, reason: 'error' }
    }

    if (status.status === 'approved') {
      const options = await deps.loadOptions(...args)
      const model = chatGptWorkModel(options.providers)

      if (model && deps.setDefaultModel) {
        await deps.setDefaultModel(CHATGPT_PROVIDER_SLUG, model, ...args)
      }

      return { model, ok: true, options }
    }

    if (status.status === 'denied' || status.status === 'expired') {
      return { message: status.error_message ?? undefined, ok: false, reason: status.status }
    }

    if (status.status === 'error') {
      return { message: status.error_message ?? undefined, ok: false, reason: 'error' }
    }
  }

  await deps.cancel(start.session_id, ...args).catch(() => undefined)

  return { ok: false, reason: 'expired' }
}
