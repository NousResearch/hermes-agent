import { atom } from 'nanostores'

import {
  cancelOAuthSession,
  getGlobalModelOptions,
  getRecommendedDefaultModel,
  listOAuthProviders,
  pollOAuthSession,
  setEnvVar,
  setModelAssignment,
  startOAuthLogin,
  submitOAuthCode,
  validateProviderCredential
} from '@/hermes'
import { evaluateRuntimeReadiness, type RuntimeReadinessResult } from '@/lib/runtime-readiness'
import { notify, notifyError } from '@/store/notifications'
import type { ModelOptionProvider, OAuthProvider, OAuthStartResponse } from '@/types/hermes'

type PkceStart = Extract<OAuthStartResponse, { flow: 'pkce' }>
type DeviceStart = Extract<OAuthStartResponse, { flow: 'device_code' }>

export type OnboardingMode = 'apikey' | 'oauth'

export type OnboardingFlow =
  | { status: 'idle' }
  | { provider: OAuthProvider; status: 'starting' }
  | { code: string; provider: OAuthProvider; start: PkceStart; status: 'awaiting_user' }
  | { copied: boolean; provider: OAuthProvider; start: DeviceStart; status: 'polling' }
  | { provider: OAuthProvider; start: OAuthStartResponse; status: 'submitting' }
  | { copied: boolean; provider: OAuthProvider; status: 'external_pending' }
  | { provider: OAuthProvider; status: 'success' }
  | {
      // After successful credential acquisition, before completing
      // onboarding: show the user which model they're getting and let
      // them change it. providerSlug is the model.options slug for the
      // just-authenticated provider (used to persist the chosen model
      // via /api/model/set). The change-model UI uses the existing
      // ModelPickerDialog, which fetches its own model list from
      // /api/model/options — no need to cache the list here.
      currentModel: string
      label: string
      providerSlug: string
      saving: boolean
      status: 'confirming_model'
    }
  | { message: string; provider?: OAuthProvider; start?: OAuthStartResponse; status: 'error' }

export interface DesktopOnboardingState {
  /** null until the first runtime check resolves. Seeded from localStorage so
   *  returning users skip the boot overlay entirely instead of flashing it
   *  every reload. */
  configured: boolean | null
  flow: OnboardingFlow
  mode: OnboardingMode
  providers: null | OAuthProvider[]
  reason: null | string
  requested: boolean
  /** True when the user explicitly chose "I'll choose a provider later" on the
   *  first-run picker. Persisted to localStorage so the blocking overlay never
   *  re-nags on subsequent launches — the user can connect a provider any time
   *  from Settings → Providers (or the model picker's "Add provider"). Distinct
   *  from `configured`: the app still has no usable provider, so chat won't work
   *  until one is connected; we just stop forcing the choice up front. */
  firstRunSkipped: boolean
  /** True when the user explicitly opened the provider selector to add /
   *  switch providers from an already-configured app (e.g. via the model
   *  picker's "Add provider" button). Forces the overlay to show the picker
   *  even when configured === true, and adds a close affordance. */
  manual: boolean
  /** True when the overlay was opened specifically to configure a local /
   *  custom OpenAI-compatible endpoint (e.g. from Settings → Model's "Set up
   *  custom endpoint"). Forces the API-key form with the local option
   *  preselected instead of the OAuth picker. */
  localEndpoint: boolean
}

export interface OnboardingContext {
  onCompleted?: () => void
  requestGateway: <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>
}

const CONFIGURED_CACHE_KEY = 'hermes-desktop-onboarded-v1'
const SKIP_CACHE_KEY = 'hermes-onboarding-skipped-v1'
const POLL_MS = 2000
const COPY_FLASH_MS = 1500
export const DEFAULT_ONBOARDING_REASON = 'No inference provider is configured.'
export const DEFAULT_MANUAL_ONBOARDING_REASON = 'Add or switch inference provider.'

function readCachedConfigured(): boolean | null {
  if (typeof window === 'undefined') {
    return null
  }

  try {
    return window.localStorage.getItem(CONFIGURED_CACHE_KEY) === '1' ? true : null
  } catch {
    return null
  }
}

function writeCachedConfigured(value: boolean) {
  if (typeof window === 'undefined') {
    return
  }

  try {
    if (value) {
      window.localStorage.setItem(CONFIGURED_CACHE_KEY, '1')
    } else {
      window.localStorage.removeItem(CONFIGURED_CACHE_KEY)
    }
  } catch {
    // localStorage unavailable — degrade silently.
  }
}

function readCachedSkipped(): boolean {
  if (typeof window === 'undefined') {
    return false
  }

  try {
    return window.localStorage.getItem(SKIP_CACHE_KEY) === '1'
  } catch {
    return false
  }
}

function writeCachedSkipped(value: boolean) {
  if (typeof window === 'undefined') {
    return
  }

  try {
    if (value) {
      window.localStorage.setItem(SKIP_CACHE_KEY, '1')
    } else {
      window.localStorage.removeItem(SKIP_CACHE_KEY)
    }
  } catch {
    // localStorage unavailable — degrade silently.
  }
}

const INITIAL: DesktopOnboardingState = {
  configured: readCachedConfigured(),
  flow: { status: 'idle' },
  mode: 'oauth',
  providers: null,
  reason: null,
  requested: false,
  firstRunSkipped: readCachedSkipped(),
  manual: false,
  localEndpoint: false
}

export const $desktopOnboarding = atom<DesktopOnboardingState>(INITIAL)

let pollTimer: number | null = null
let providersRefreshPromise: null | Promise<void> = null

const errMessage = (e: unknown) => (e instanceof Error ? e.message : String(e))

const patch = (update: Partial<DesktopOnboardingState>) =>
  $desktopOnboarding.set({ ...$desktopOnboarding.get(), ...update })

const setFlow = (flow: OnboardingFlow) => patch(flow.status === 'idle' ? { flow } : { flow, reason: null })

const sessionIdFor = (flow: OnboardingFlow) => ('start' in flow && flow.start ? flow.start.session_id : undefined)

function clearPoll() {
  if (pollTimer !== null) {
    window.clearInterval(pollTimer)
    pollTimer = null
  }
}

async function checkRuntime(ctx: OnboardingContext, requestedProvider?: string): Promise<RuntimeReadinessResult> {
  return evaluateRuntimeReadiness(ctx.requestGateway, {
    defaultReason: DEFAULT_ONBOARDING_REASON,
    requestedProvider,
    unknownReady: false
  })
}

function shouldPreserveConfiguredOnFallback(runtime: RuntimeReadinessResult, state: DesktopOnboardingState): boolean {
  // A fallback result means both runtime probes were non-authoritative
  // (transport timeout/disconnect). Keep a previously verified configured
  // state instead of forcing the blocking onboarding overlay.
  return runtime.source === 'fallback' && state.configured === true && !state.requested
}

function notifyReady(provider: string) {
  notify({ kind: 'success', title: 'Hermes is ready', message: `${provider} connected.` })
}

// Human-friendly labels for tools auto-routed through the Nous Tool Gateway,
// mirroring hermes_cli/nous_subscription._GATEWAY_TOOL_LABELS so the GUI and
// CLI describe the same thing.
const GATEWAY_TOOL_LABELS: Record<string, string> = {
  browser: 'browser automation',
  image_gen: 'image generation',
  tts: 'text-to-speech',
  video_gen: 'video generation',
  web: 'web search & extract'
}

// When switching to Nous auto-routes unconfigured tools through the Tool
// Gateway, tell the user which ones — same information the CLI prints. Silent
// when nothing changed (subscriber already configured, has own keys, etc.).
function notifyGatewayTools(tools: string[] | undefined) {
  if (!tools || tools.length === 0) {
    return
  }

  const labels = tools.map(t => GATEWAY_TOOL_LABELS[t] ?? t)
  const list = labels.length === 1 ? labels[0] : `${labels.slice(0, -1).join(', ')} and ${labels[labels.length - 1]}`

  notify({
    durationMs: 8000,
    kind: 'info',
    message: `${list} now run through your Nous subscription — no separate API keys needed.`,
    title: 'Tool Gateway enabled'
  })
}

// After credentials are persisted, ask the backend which provider+models
// are now authenticated. Pick the first curated model for the matching
// provider as a sensible default, persist it via /api/model/set, and
// transition to the model-confirmation step. If anything goes wrong
// fetching options (no providers returned, network error), the caller
// falls through to completing onboarding without showing the confirm
// card — the user gets the undefined-model auto-selection behaviour
// we had before, which works but is surprising. The confirm step is
// opportunistic polish, not a hard requirement for onboarding.
async function fetchProviderDefaultModel(
  preferredSlugs: string[]
): Promise<null | { providerSlug: string; defaultModel: string }> {
  let options

  try {
    options = await getGlobalModelOptions({ includeUnconfigured: true, explicitOnly: false })
  } catch {
    return null
  }

  const providers = options?.providers ?? []

  if (providers.length === 0) {
    return null
  }

  // Try each preferred slug (lowercased), fall back to the first provider
  // returned (model.options orders by recency / authenticated state, so
  // the just-authenticated provider is usually first anyway).
  const lower = preferredSlugs.map(s => s.toLowerCase())

  const matched =
    providers.find((p: ModelOptionProvider) => lower.includes(String(p.slug).toLowerCase())) ?? providers[0]

  const models = matched.models ?? []

  if (models.length === 0) {
    return null
  }

  // Prefer the backend's recommended default — it mirrors the curation
  // `hermes model` does (for Nous it honors the user's free/paid tier, so a
  // free user gets a free model rather than a paid default like opus). Fall
  // back to the first curated model if the endpoint can't resolve one.
  let defaultModel = String(models[0])

  try {
    const recommended = await getRecommendedDefaultModel(String(matched.slug))

    if (recommended.model && models.map(String).includes(recommended.model)) {
      defaultModel = recommended.model
    } else if (recommended.model) {
      // Recommended model isn't in the curated options list (e.g. a Portal
      // free-recommendation the picker list didn't include); trust it anyway.
      defaultModel = recommended.model
    }
  } catch {
    // Endpoint unavailable — keep models[0]. Non-fatal: the confirm card still
    // shows and the user can change it.
  }

  return {
    providerSlug: String(matched.slug),
    defaultModel
  }
}

// After OAuth/API-key success: reload the backend env, verify runtime,
// then either show the model-confirm step or fall straight through to
// completion if we can't determine a default.
//
// onFail receives the runtime-readiness `reason` from checkRuntime so
// the caller can fold it into a user-facing error — same contract as
// reloadAndConnect used to have (which this replaces).
async function completeWithModelConfirm(
  ctx: OnboardingContext,
  providerLabel: string,
  preferredSlugs: string[],
  onFail: (reason: null | string) => void,
  // When true, a failing runtime check no longer blocks progression — the
  // user is allowed through onboarding regardless. Used by the API-key path,
  // where we intentionally don't validate the key (it blocked too many users).
  ignoreRuntimeGate = false
) {
  await ctx.requestGateway('reload.env').catch(() => undefined)

  const defaults = await fetchProviderDefaultModel(preferredSlugs)

  if (defaults) {
    // Persist the chosen provider/model before the runtime gate so a stale
    // config provider (e.g. anthropic from a prior failed setup) cannot make
    // setup.runtime_check validate the wrong backend after a fresh OAuth login.
    try {
      const res = await setModelAssignment({
        scope: 'main',
        provider: defaults.providerSlug,
        model: defaults.defaultModel
      })

      notifyGatewayTools(res.gateway_tools)
    } catch {
      // Persistence failed — still run the scoped runtime check below and
      // show the confirm card so the user can pick something explicitly.
    }
  }

  const runtime = await checkRuntime(ctx, preferredSlugs[0])

  if (!runtime.ready && !ignoreRuntimeGate) {
    onFail(runtime.reason)

    return
  }

  if (!defaults) {
    // Couldn't get a sensible default — proceed without confirm step.
    notifyReady(providerLabel)
    completeDesktopOnboarding()
    ctx.onCompleted?.()

    return
  }

  setFlow({
    status: 'confirming_model',
    providerSlug: defaults.providerSlug,
    currentModel: defaults.defaultModel,
    label: providerLabel,
    saving: false
  })
}

function providerResolutionFailure(reason: null | string) {
  const detail = reason?.trim()

  return detail
    ? `Connected, but Hermes still cannot resolve a usable provider. ${detail}`
    : 'Connected, but Hermes still cannot resolve a usable provider.'
}

async function refreshProviders() {
  if (providersRefreshPromise) {
    await providersRefreshPromise

    return
  }

  providersRefreshPromise = (async () => {
    try {
      const { providers } = await listOAuthProviders()
      patch({ mode: providers.length > 0 ? 'oauth' : 'apikey', providers })
    } catch {
      patch({ mode: 'apikey', providers: [] })
    } finally {
      providersRefreshPromise = null
    }
  })()

  await providersRefreshPromise
}

export function requestDesktopOnboarding(reason = DEFAULT_ONBOARDING_REASON) {
  patch({ reason: reason.trim() || DEFAULT_ONBOARDING_REASON, requested: true })
}

// Open the onboarding provider selector on demand from an already-configured
// app — e.g. the model picker's "Add provider" button. Reuses the entire
// onboarding flow (OAuth rows, API-key form, model-confirm) instead of
// duplicating provider UI. Sets manual=true so the overlay shows the picker
// even though configured===true, and refreshes the provider list.
export function startManualOnboarding(reason: null | string = DEFAULT_MANUAL_ONBOARDING_REASON) {
  patch({
    manual: true,
    requested: true,
    localEndpoint: false,
    // `null` opts out of the prompt banner entirely (e.g. when the user already
    // picked a specific provider and we auto-start its sign-in).
    reason: reason ? reason.trim() || DEFAULT_ONBOARDING_REASON : null,
    flow: { status: 'idle' }
  })
  void refreshProviders()
}

// Open the onboarding overlay directly on the local / custom endpoint form
// (URL + optional API key), bypassing the OAuth picker. Used by Settings →
// Model's "Set up custom endpoint" so it lands on a form that can actually
// configure the endpoint instead of dead-ending on the OAuth provider list
// (`custom` is not an OAuth provider, so the generic manual flow would just
// re-show the picker — the original "booted back to the first screen" loop).
export function startManualLocalEndpoint(reason: null | string = null) {
  pendingProviderOAuthId = null
  patch({
    manual: true,
    requested: true,
    localEndpoint: true,
    mode: 'apikey',
    reason: reason ? reason.trim() || DEFAULT_ONBOARDING_REASON : null,
    flow: { status: 'idle' }
  })
}

// One-shot hand-off used when the dedicated Providers settings page launches a
// specific provider's sign-in: we open the manual onboarding overlay AND
// remember which provider to start, so the overlay drives that exact OAuth
// flow instead of re-showing the picker the user just clicked through.
// Module-level (not store state) because it's consumed immediately on the next
// overlay render and never needs to persist or re-render anything itself.
let pendingProviderOAuthId: null | string = null

export function startManualProviderOAuth(providerId: string, reason: null | string = null) {
  pendingProviderOAuthId = providerId
  startManualOnboarding(reason)
}

// Read the pending provider id without clearing it. The overlay only clears it
// (via clearPendingProviderOAuth) once it has actually launched that provider,
// so a transient empty/failed provider fetch doesn't drop the hand-off and the
// deep-link can still auto-start after the list loads.
export function peekPendingProviderOAuth(): null | string {
  return pendingProviderOAuthId
}

export function clearPendingProviderOAuth() {
  pendingProviderOAuthId = null
}

// Dismiss a manually-opened provider selector without touching the existing
// (working) configuration. Only valid in the manual path — the unconfigured
// first-run flow has no close affordance because the app can't run yet.
export function closeManualOnboarding() {
  pendingProviderOAuthId = null

  patch({ manual: false, requested: false, localEndpoint: false, flow: { status: 'idle' } })
}

export function completeDesktopOnboarding() {
  clearPoll()
  writeCachedConfigured(true)
  // A real provider is now connected, so any earlier "choose later" skip is
  // moot — clear it so the flag never lingers in a configured install.
  writeCachedSkipped(false)
  $desktopOnboarding.set({
    configured: true,
    flow: { status: 'idle' },
    mode: 'oauth',
    providers: null,
    reason: null,
    requested: false,
    firstRunSkipped: false,
    manual: false,
    localEndpoint: false
  })
}

// "I'll choose a provider later" on the first-run picker. Persists the skip so
// the blocking overlay never re-nags on future launches, and dismisses it now
// so the user lands in the app. Chat won't work until a provider is connected
// (from Settings → Providers or the model picker's "Add provider") — this only
// stops forcing the choice up front. Distinct from completeDesktopOnboarding,
// which marks the app actually configured.
export function dismissFirstRunOnboarding() {
  clearPoll()
  writeCachedSkipped(true)
  patch({ firstRunSkipped: true, requested: false, manual: false, localEndpoint: false, flow: { status: 'idle' } })
}

export function setOnboardingMode(mode: OnboardingMode) {
  patch({ mode })
}

export async function refreshOnboarding(ctx: OnboardingContext) {
  // Manual mode (user opened the selector from a working app): never
  // auto-dismiss on runtime-ready — the whole point is to let them add /
  // switch a provider while already configured. Just ensure the provider
  // list is loaded and show the picker.
  if ($desktopOnboarding.get().manual) {
    await refreshProviders()

    return false
  }

  const runtime = await checkRuntime(ctx)

  if (runtime.ready) {
    completeDesktopOnboarding()
    ctx.onCompleted?.()

    return true
  }

  const state = $desktopOnboarding.get()

  if (shouldPreserveConfiguredOnFallback(runtime, state)) {
    // Gateway probes timed out but the user was already configured — don't
    // downgrade to the blocking onboarding overlay. Surface a non-blocking
    // notification with a stable id so repeated calls during an outage dedup
    // instead of stacking toasts.
    notify({
      id: 'runtime-not-ready',
      kind: 'error',
      title: 'Runtime not ready',
      message:
        'Hermes Desktop could not verify the running backend on startup. Some features may be unavailable until the gateway is reachable.'
    })

    return false
  }

  const reason = runtime.reason || state.reason || DEFAULT_ONBOARDING_REASON

  writeCachedConfigured(false)
  patch({ configured: false, reason })

  if (state.providers !== null && !state.requested) {
    return false
  }

  await refreshProviders()

  return false
}

// Open a sign-in URL via the desktop bridge, falling back to window.open
// when the bridge isn't present (e.g. the web dashboard / dev preview) so
// the flow never silently stalls in a waiting state. Mirrors the pattern in
// apps/desktop/src/app/artifacts/index.tsx.
async function openSignInUrl(url: string) {
  if (window.hermesDesktop?.openExternal) {
    try {
      await window.hermesDesktop.openExternal(url)

      return
    } catch {
      // Bridge present but failed (no OS handler, user denied, etc.). Fall
      // through to window.open so the sign-in URL still opens and the flow
      // doesn't strand a pending OAuth session in a waiting state.
    }
  }

  window.open(url, '_blank', 'noopener,noreferrer')
}

export async function startProviderOAuth(provider: OAuthProvider, ctx: OnboardingContext) {
  clearPoll()

  if (provider.flow === 'external') {
    setFlow({ status: 'external_pending', provider, copied: false })

    return
  }

  setFlow({ status: 'starting', provider })

  try {
    const start = await startOAuthLogin(provider.id)
    const browserUrl = start.flow === 'device_code' ? start.verification_url : start.auth_url
    await openSignInUrl(browserUrl)

    if (start.flow === 'pkce') {
      setFlow({ status: 'awaiting_user', provider, start, code: '' })

      return
    }

    setFlow({ status: 'polling', provider, start, copied: false })
    pollTimer = window.setInterval(() => void pollSession(provider, start, ctx), POLL_MS)
  } catch (error) {
    setFlow({ status: 'error', provider, message: `Could not start sign-in: ${errMessage(error)}` })
  }
}

// Poll a session-backed device-code flow until it resolves.
async function pollSession(provider: OAuthProvider, start: DeviceStart, ctx: OnboardingContext) {
  try {
    const { error_message, status } = await pollOAuthSession(provider.id, start.session_id)

    if (status === 'approved') {
      clearPoll()
      setFlow({ status: 'success', provider })
      await completeWithModelConfirm(ctx, provider.name, [provider.id], reason =>
        setFlow({
          status: 'error',
          provider,
          message: providerResolutionFailure(reason)
        })
      )
    } else if (status !== 'pending') {
      clearPoll()
      setFlow({ status: 'error', provider, start, message: error_message || `Sign-in ${status}.` })
    }
  } catch (error) {
    clearPoll()
    setFlow({ status: 'error', provider, start, message: `Polling failed: ${errMessage(error)}` })
  }
}

export function setOnboardingCode(code: string) {
  const { flow } = $desktopOnboarding.get()

  if (flow.status === 'awaiting_user') {
    setFlow({ ...flow, code })
  }
}

export async function submitOnboardingCode(ctx: OnboardingContext) {
  const { flow } = $desktopOnboarding.get()

  if (flow.status !== 'awaiting_user' || !flow.code.trim()) {
    return
  }

  const { provider, start, code } = flow
  setFlow({ status: 'submitting', provider, start })

  try {
    const resp = await submitOAuthCode(provider.id, start.session_id, code.trim())

    if (resp.ok && resp.status === 'approved') {
      setFlow({ status: 'success', provider })
      await completeWithModelConfirm(ctx, provider.name, [provider.id], reason =>
        setFlow({
          status: 'error',
          provider,
          message: providerResolutionFailure(reason)
        })
      )
    } else {
      setFlow({ status: 'error', provider, start, message: resp.message || 'Token exchange failed.' })
    }
  } catch (error) {
    setFlow({ status: 'error', provider, start, message: errMessage(error) })
  }
}

export function cancelOnboardingFlow() {
  clearPoll()
  const sessionId = sessionIdFor($desktopOnboarding.get().flow)

  if (sessionId) {
    cancelOAuthSession(sessionId).catch(() => undefined)
  }

  setFlow({ status: 'idle' })
}

async function copyAndFlash(text: string, predicate: (flow: OnboardingFlow) => boolean) {
  try {
    await navigator.clipboard.writeText(text)
  } catch {
    return
  }

  const { flow } = $desktopOnboarding.get()

  if (!predicate(flow) || !('copied' in flow)) {
    return
  }

  setFlow({ ...flow, copied: true })
  window.setTimeout(() => {
    const current = $desktopOnboarding.get().flow

    if (predicate(current) && 'copied' in current) {
      setFlow({ ...current, copied: false })
    }
  }, COPY_FLASH_MS)
}

export async function copyDeviceCode() {
  const { flow } = $desktopOnboarding.get()

  if (flow.status !== 'polling') {
    return
  }

  const sid = flow.start.session_id
  await copyAndFlash(flow.start.user_code, f => f.status === 'polling' && f.start.session_id === sid)
}

export async function copyExternalCommand() {
  const { flow } = $desktopOnboarding.get()

  if (flow.status !== 'external_pending') {
    return
  }

  const id = flow.provider.id
  await copyAndFlash(flow.provider.cli_command, f => f.status === 'external_pending' && f.provider.id === id)
}

export async function recheckExternalSignin(ctx: OnboardingContext) {
  const { flow } = $desktopOnboarding.get()

  if (flow.status !== 'external_pending') {
    return
  }

  const { provider } = flow
  await completeWithModelConfirm(ctx, provider.name, [provider.id], reason =>
    setFlow({
      status: 'error',
      provider,
      message:
        reason?.trim() ||
        `Hermes still cannot reach ${provider.name}. Run \`${provider.cli_command}\` in a terminal first.`
    })
  )
}

export async function saveOnboardingApiKey(
  envKey: string,
  value: string,
  label: string,
  ctx: OnboardingContext,
  // Optional endpoint key — only meaningful for the "Local / custom endpoint"
  // option, whose primary `value` is the base URL. Ignored for plain API-key
  // providers (their key IS `value`).
  endpointApiKey?: string
) {
  const trimmed = value.trim()

  if (!trimmed) {
    return { ok: false, message: 'Enter a value first.' }
  }

  // The "Local / custom endpoint" option carries a base URL (in `value`) plus
  // an optional API key. It must be wired into config (provider=custom +
  // base_url + model + api_key), not dropped into .env — runtime resolution
  // ignores OPENAI_BASE_URL.
  if (envKey === 'OPENAI_BASE_URL') {
    return saveOnboardingLocalEndpoint(trimmed, endpointApiKey?.trim() ?? '', ctx)
  }

  // No key validation here on purpose: we previously live-probed the key and
  // hard-blocked on a runtime check after saving, which rejected too many
  // legitimate users (corporate proxies, regional blocks, flaky/rate-limited
  // provider probes, self-hosted endpoints). We now save the value as-is and
  // let the user proceed; an actually-bad key surfaces later at chat time.
  try {
    await setEnvVar(envKey, trimmed)
    // For API-key flows we don't have a definitive provider id (the
    // user picked which API key they're entering, but the corresponding
    // backend slug — e.g. OPENROUTER_API_KEY → "openrouter" — is the
    // env-key prefix stripped). Pass a couple of likely candidates;
    // fetchProviderDefaultModel falls back to the first authenticated
    // provider returned by /api/model/options if none match.
    const slugCandidates = [envKey.replace(/_API_KEY$/, '').toLowerCase(), label.toLowerCase()]
    // ignoreRuntimeGate=true: never block onboarding on the runtime check.
    await completeWithModelConfirm(ctx, label, slugCandidates, () => undefined, true)

    return { ok: true }
  } catch (error) {
    notifyError(error, `Could not save ${label}`)

    return { ok: false, message: errMessage(error) }
  }
}

// Configure a local / self-hosted OpenAI-compatible endpoint (vLLM, llama.cpp,
// Ollama, …). Unlike API-key providers, a local endpoint is defined by its URL
// and usually needs NO key. The runtime resolver reads model.base_url from
// config (it ignores the OPENAI_BASE_URL env var), so we persist
// provider=custom + base_url + model via /api/model/set rather than dropping an
// env var that resolution never consults.
//
// The model is auto-discovered from the endpoint's /v1/models (surfaced by the
// validate probe). The optional API key is forwarded to the probe (so hosted
// endpoints that gate /v1/models behind auth still enumerate models) and
// persisted to model.api_key so the runtime can authenticate.
//
// We deliberately don't route through completeWithModelConfirm: that path
// re-assigns the model from /api/model/options WITHOUT a base_url, which would
// wipe the base_url we just wrote. We have a concrete model already, so we
// verify the runtime directly and finish.
export async function saveOnboardingLocalEndpoint(baseUrl: string, apiKey: string, ctx: OnboardingContext) {
  const url = baseUrl.trim()
  const key = apiKey.trim()

  if (!url) {
    return { ok: false, message: 'Enter the endpoint URL first.' }
  }

  // Probe connectivity + discover the served models. Any HTTP response proves
  // the endpoint is up; an unreachable probe hard-blocks because we can't
  // resolve a model to route to.
  let model = ''
  let probeMessage = ''

  try {
    const probe = await validateProviderCredential('OPENAI_BASE_URL', url, key)

    if (!probe.ok && probe.reachable) {
      return { ok: false, message: probe.message || 'Could not reach that endpoint.' }
    }

    if (!probe.reachable) {
      return { ok: false, message: probe.message || `Could not reach ${url}.` }
    }

    model = (probe.models?.[0] ?? '').trim()
    probeMessage = (probe.message ?? '').trim()
  } catch {
    return { ok: false, message: `Could not reach ${url}.` }
  }

  if (!model) {
    // The probe distinguishes "endpoint answered but errored" (message set,
    // e.g. a proxy or gateway answered HTTP 502 at /v1/models) from a healthy
    // endpoint that genuinely serves no models.
    return {
      ok: false,
      message: probeMessage
        ? `Connected to ${url}, but ${probeMessage}`
        : `Connected to ${url}, but it advertised no models at /v1/models. Start a model on that endpoint and try again.`
    }
  }

  try {
    await setModelAssignment({ scope: 'main', provider: 'custom', model, base_url: url, api_key: key })
    await ctx.requestGateway('reload.env').catch(() => undefined)

    const runtime = await checkRuntime(ctx)

    if (!runtime.ready) {
      const detail = (runtime.reason ?? '').trim()

      return { ok: false, message: detail || `Saved, but Hermes still cannot reach ${url}.` }
    }

    notifyReady('Local / custom endpoint')
    completeDesktopOnboarding()
    ctx.onCompleted?.()

    return { ok: true }
  } catch (error) {
    notifyError(error, 'Could not save local endpoint')

    return { ok: false, message: errMessage(error) }
  }
}

// User picked a different model from the dropdown on the confirm card.
// Persists immediately so the displayed value is always what's on disk.
export async function setOnboardingModel(model: string) {
  const { flow } = $desktopOnboarding.get()

  if (flow.status !== 'confirming_model') {
    return
  }

  // Optimistic update so the dropdown feels instant; revert on failure.
  const previous = flow.currentModel
  setFlow({ ...flow, currentModel: model, saving: true })

  try {
    await setModelAssignment({
      scope: 'main',
      provider: flow.providerSlug,
      model
    })
    const current = $desktopOnboarding.get().flow

    if (current.status === 'confirming_model') {
      setFlow({ ...current, currentModel: model, saving: false })
    }
  } catch (error) {
    notifyError(error, 'Could not change model')
    const current = $desktopOnboarding.get().flow

    if (current.status === 'confirming_model') {
      setFlow({ ...current, currentModel: previous, saving: false })
    }
  }
}

// User clicked "Start chatting" on the confirm card. Finalizes onboarding
// — the model was already persisted by completeWithModelConfirm (or by
// setOnboardingModel if they changed it), so all that's left is to mark
// onboarding done and unblock the rest of the app.
export function confirmOnboardingModel(ctx: OnboardingContext) {
  const { flow } = $desktopOnboarding.get()

  if (flow.status !== 'confirming_model') {
    return
  }

  // No success toast here: the confirm-model screen already showed "<provider>
  // connected." notifyReady is reserved for completion paths that SKIP this
  // screen (no-default fallthrough, local endpoint) so feedback isn't lost.
  completeDesktopOnboarding()
  ctx.onCompleted?.()
}
