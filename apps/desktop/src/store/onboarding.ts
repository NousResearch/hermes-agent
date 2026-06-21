import { atom } from 'nanostores'

import {
  cancelOAuthSession,
  listOAuthProviders,
  pollOAuthSession,
  setEnvVar,
  startOAuthLogin,
  submitOAuthCode
} from '@/hermes'
import { notify, notifyError } from '@/store/notifications'
import type { OAuthProvider, OAuthStartResponse } from '@/types/hermes'

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
  | { message: string; provider?: OAuthProvider; start?: OAuthStartResponse; status: 'error' }

export interface DesktopOnboardingState {
  /** null until the first runtime check resolves; lets the overlay render
   *  during boot without flickering for already-configured users. */
  configured: boolean | null
  flow: OnboardingFlow
  mode: OnboardingMode
  providers: null | OAuthProvider[]
  reason: null | string
  requested: boolean
}

export interface OnboardingContext {
  onCompleted?: () => void
  requestGateway: <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>
}

const INITIAL: DesktopOnboardingState = {
  configured: null,
  flow: { status: 'idle' },
  mode: 'oauth',
  providers: null,
  reason: null,
  requested: false
}

const POLL_MS = 2000
const COPY_FLASH_MS = 1500

export const $desktopOnboarding = atom<DesktopOnboardingState>(INITIAL)

let pollTimer: number | null = null

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

async function safeReq<T>(ctx: OnboardingContext, method: string, fallback: T): Promise<T> {
  try {
    return await ctx.requestGateway<T>(method)
  } catch {
    return fallback
  }
}

async function checkRuntime(ctx: OnboardingContext) {
  const [status, runtime] = await Promise.all([
    safeReq<{ provider_configured?: boolean }>(ctx, 'setup.status', {}),
    safeReq<{ error?: string; ok?: boolean }>(ctx, 'setup.runtime_check', { ok: false })
  ])

  return runtime.ok !== undefined ? Boolean(runtime.ok) : Boolean(status.provider_configured)
}

function notifyReady(provider: string) {
  notify({ kind: 'success', title: '元话 Agent is ready', message: `${provider} connected.` })
}

async function reloadAndConnect(ctx: OnboardingContext, providerName: string, onFail: () => void) {
  await ctx.requestGateway('reload.env').catch(() => undefined)
  const ok = await checkRuntime(ctx)

  if (ok) {
    notifyReady(providerName)
    completeDesktopOnboarding()
    ctx.onCompleted?.()
  } else {
    onFail()
  }
}

export function requestDesktopOnboarding(reason = 'No inference provider is configured.') {
  patch({ reason, requested: true })
}

export function completeDesktopOnboarding() {
  clearPoll()
  $desktopOnboarding.set({ ...INITIAL, configured: true })
}

export function setOnboardingMode(mode: OnboardingMode) {
  patch({ mode })
}

export async function refreshOnboarding(ctx: OnboardingContext) {
  if (await checkRuntime(ctx)) {
    completeDesktopOnboarding()
    ctx.onCompleted?.()

    return true
  }

  patch({ configured: false })

  if ($desktopOnboarding.get().providers !== null) {
    return false
  }

  try {
    const { providers } = await listOAuthProviders()
    patch({ providers, mode: providers.length > 0 ? 'oauth' : 'apikey' })
  } catch {
    patch({ providers: [], mode: 'apikey' })
  }

  return false
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
    await window.hermesDesktop?.openExternal(start.flow === 'pkce' ? start.auth_url : start.verification_url)

    if (start.flow === 'pkce') {
      setFlow({ status: 'awaiting_user', provider, start, code: '' })

      return
    }

    setFlow({ status: 'polling', provider, start, copied: false })
    pollTimer = window.setInterval(() => void pollDevice(provider, start, ctx), POLL_MS)
  } catch (error) {
    setFlow({ status: 'error', provider, message: `Could not start sign-in: ${errMessage(error)}` })
  }
}

async function pollDevice(provider: OAuthProvider, start: DeviceStart, ctx: OnboardingContext) {
  try {
    const { error_message, status } = await pollOAuthSession(provider.id, start.session_id)

    if (status === 'approved') {
      clearPoll()
      setFlow({ status: 'success', provider })
      await reloadAndConnect(ctx, provider.name, () =>
        setFlow({
          status: 'error',
          provider,
          message: 'Connected, but Hermes still cannot resolve a usable provider.'
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
      await reloadAndConnect(ctx, provider.name, () =>
        setFlow({
          status: 'error',
          provider,
          message: 'Connected, but Hermes still cannot resolve a usable provider.'
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
  await reloadAndConnect(ctx, provider.name, () =>
    setFlow({
      status: 'error',
      provider,
      message: `Hermes still cannot reach ${provider.name}. Run \`${provider.cli_command}\` in a terminal first.`
    })
  )
}

export async function saveOnboardingApiKey(envKey: string, value: string, label: string, ctx: OnboardingContext) {
  const trimmed = value.trim()

  if (!trimmed) {
    return { ok: false, message: 'Enter a value first.' }
  }

  try {
    await setEnvVar(envKey, trimmed)
    let stillFailing = false
    await reloadAndConnect(ctx, label, () => {
      stillFailing = true
    })

    if (stillFailing) {
      return { ok: false, message: `Saved, but Hermes still cannot reach ${label}. Double-check the value.` }
    }

    return { ok: true }
  } catch (error) {
    notifyError(error, `Could not save ${label}`)

    return { ok: false, message: errMessage(error) }
  }
}
