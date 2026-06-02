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
type LoopbackStart = Extract<OAuthStartResponse, { flow: 'loopback' }>

export type OnboardingMode = 'apikey' | 'oauth'

export type OnboardingFlow =
  | { status: 'idle' }
  | { provider: OAuthProvider; status: 'starting' }
  | { code: string; provider: OAuthProvider; start: PkceStart; status: 'awaiting_user' }
  | { copied: boolean; provider: OAuthProvider; start: DeviceStart; status: 'polling' }
  // Loopback PKCE (xAI Grok): browser opens, the local backend's 127.0.0.1
  // listener catches the redirect, and we poll until the worker finishes.
  // No code to paste and no user_code to show — just a waiting state.
  | { provider: OAuthProvider; start: LoopbackStart; status: 'awaiting_browser' }
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
  /** True when the user explicitly opened the provider selector to add /
   *  switch providers from an already-configured app (e.g. via the model
   *  picker's "Add provider" button). Forces the overlay to show the picker
   *  even when configured === true, and adds a close affordance. */
  manual: boolean
}

export interface OnboardingContext {
  onCompleted?: () => void
  requestGateway: <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>
}

const CONFIGURED_CACHE_KEY = 'hermes-desktop-onboarded-v1'
const POLL_MS = 2000
const COPY_FLASH_MS = 1500
const DEFAULT_ONBOARDING_REASON = 'No inference provider is configured.'

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

const INITIAL: DesktopOnboardingState = {
  configured: readCachedConfigured(),
  flow: { status: 'idle' },
  mode: 'oauth',
  providers: null,
  reason: null,
  requested: false,
  manual: false
}

export const $desktopOnboarding = atom<DesktopOnboardingState>(INITIAL)

let pollTimer: number | null = null
let providersRefreshPromise: null | Promise<void> = null

const errMessage = (e: unknown) => (e instanceof Error ? e.message : String(e))

const patch = (update: Partial<DesktopOnboardingState>) =>
  $desktopOnboarding.set({ ...$desktopOnboarding.get(), ...update })

const setFlow = (flow: OnboardingFlow) => patch({ flow })

const sessionIdFor = (flow: OnboardingFlow) => ('start' in flow && flow.start ? flow.start.session_id : undefined)

function clearPoll() {
  if (pollTimer !== null) {
    window.clearInterval(pollTimer)
    pollTimer = null
  }
}

async function checkRuntime(ctx: OnboardingContext): Promise<RuntimeReadinessResult> {
  return evaluateRuntimeReadiness(ctx.requestGateway, {
    defaultReason: DEFAULT_ONBOARDING_REASON,
    unknownReady: false
  })
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
    options = await getGlobalModelOptions()
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
  onFail: (reason: null | string) => void
) {
  await ctx.requestGateway('reload.env').catch(() => undefined)
  const runtime = await checkRuntime(ctx)

  if (!runtime.ready) {
    onFail(runtime.reason)

    return
  }

  const defaults = await fetchProviderDefaultModel(preferredSlugs)

  if (!defaults) {
    // Couldn't get a sensible default — proceed without confirm step.
    notifyReady(providerLabel)
    completeDesktopOnboarding()
    ctx.onCompleted?.()

    return
  }

  // Persist the default model BEFORE showing the confirm card so that:
  // (1) "current default: X" shown in the UI is what's actually written
  //     to config — no lying.
  // (2) If the user clicks "Start chatting" without changing anything,
  //     no extra write is needed.
  // (3) If they bail out (e.g., refresh the page), they still end up
  //     with a working config, not an empty-model fallback.
  try {
    const res = await setModelAssignment({
      scope: 'main',
      provider: defaults.providerSlug,
      model: defaults.defaultModel
    })
    notifyGatewayTools(res.gateway_tools)
  } catch {
    // Persistence failed — still show the confirm card so the user can
    // pick something explicitly. The backend will pick its own default
    // at chat time if we end up never persisting.
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
export function startManualOnboarding(reason = 'Add or switch inference provider.') {
  patch({
    manual: true,
    requested: true,
    reason: reason.trim() || DEFAULT_ONBOARDING_REASON,
    flow: { status: 'idle' }
  })
  void refreshProviders()
}

// Dismiss a manually-opened provider selector without touching the existing
// (working) configuration. Only valid in the manual path — the unconfigured
// first-run flow has no close affordance because the app can't run yet.
export function closeManualOnboarding() {
  patch({ manual: false, requested: false, flow: { status: 'idle' } })
}

export function completeDesktopOnboarding() {
  clearPoll()
  writeCachedConfigured(true)
  $desktopOnboarding.set({
    configured: true,
    flow: { status: 'idle' },
    mode: 'oauth',
    providers: null,
    reason: null,
    requested: false,
    manual: false
  })
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

    if (start.flow === 'loopback') {
      // No code to paste: the redirect lands on the backend's loopback
      // listener. Just wait and poll the session until the worker finishes.
      setFlow({ status: 'awaiting_browser', provider, start })
      pollTimer = window.setInterval(() => void pollSession(provider, start, ctx), POLL_MS)

      return
    }

    setFlow({ status: 'polling', provider, start, copied: false })
    pollTimer = window.setInterval(() => void pollSession(provider, start, ctx), POLL_MS)
  } catch (error) {
    setFlow({ status: 'error', provider, message: `Could not start sign-in: ${errMessage(error)}` })
  }
}

// Poll a session-backed flow (device_code or loopback) until it resolves.
// Both shapes only need the session_id to poll; the start is threaded
// through to the error flow so the user can retry from the same context.
async function pollSession(provider: OAuthProvider, start: DeviceStart | LoopbackStart, ctx: OnboardingContext) {
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

export async function saveOnboardingApiKey(envKey: string, value: string, label: string, ctx: OnboardingContext) {
  const trimmed = value.trim()

  if (!trimmed) {
    return { ok: false, message: 'Enter a value first.' }
  }

  // Live-probe the credential BEFORE persisting so a mistyped key never lands
  // in .env. A rejected key (reachable && !ok) hard-blocks; an unreachable
  // probe (offline / provider down) falls through and saves with the usual
  // runtime check, so we don't strand offline users.
  try {
    const probe = await validateProviderCredential(envKey, trimmed)
    if (!probe.ok && probe.reachable) {
      return { ok: false, message: probe.message || `That ${label} key was rejected.` }
    }
  } catch {
    // Validation endpoint unavailable — don't block; fall through to save.
  }

  try {
    await setEnvVar(envKey, trimmed)
    let stillFailing = false
    let runtimeFailure: null | string = null
    // For API-key flows we don't have a definitive provider id (the
    // user picked which API key they're entering, but the corresponding
    // backend slug — e.g. OPENROUTER_API_KEY → "openrouter" — is the
    // env-key prefix stripped). Pass a couple of likely candidates;
    // fetchProviderDefaultModel falls back to the first authenticated
    // provider returned by /api/model/options if none match.
    const slugCandidates = [envKey.replace(/_API_KEY$/, '').toLowerCase(), label.toLowerCase()]
    await completeWithModelConfirm(ctx, label, slugCandidates, reason => {
      stillFailing = true
      runtimeFailure = reason
    })

    if (stillFailing) {
      const failureDetail = (runtimeFailure ?? '').trim()

      return {
        ok: false,
        message: failureDetail || `Saved, but Hermes still cannot reach ${label}. Double-check the value.`
      }
    }

    return { ok: true }
  } catch (error) {
    notifyError(error, `Could not save ${label}`)

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

  notifyReady(flow.label)
  completeDesktopOnboarding()
  ctx.onCompleted?.()
}
