import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { useEffect, useMemo, useRef, useState } from 'react'

import { ModelPickerDialog } from '@/components/model-picker'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { ErrorIcon } from '@/components/ui/error-state'
import { Input } from '@/components/ui/input'
import { Loader } from '@/components/ui/loader'
import { getGlobalModelOptions } from '@/hermes'
import { useI18n } from '@/i18n'
import { Check, ChevronDown, ChevronLeft, ChevronRight, ExternalLink, KeyRound, Loader2, Terminal } from '@/lib/icons'
import { isProviderSetupErrorMessage } from '@/lib/provider-setup-errors'
import { cn } from '@/lib/utils'
import { $desktopBoot, type DesktopBootState } from '@/store/boot'
import {
  $desktopOnboarding,
  cancelOnboardingFlow,
  clearPendingProviderOAuth,
  closeManualOnboarding,
  confirmOnboardingModel,
  copyDeviceCode,
  copyExternalCommand,
  DEFAULT_MANUAL_ONBOARDING_REASON,
  DEFAULT_ONBOARDING_REASON,
  dismissFirstRunOnboarding,
  type OnboardingContext,
  type OnboardingFlow,
  peekPendingProviderOAuth,
  recheckExternalSignin,
  refreshOnboarding,
  saveOnboardingApiKey,
  setOnboardingCode,
  setOnboardingMode,
  setOnboardingModel,
  startProviderOAuth,
  submitOnboardingCode
} from '@/store/onboarding'
import type { ModelOptionProvider, OAuthProvider } from '@/types/hermes'

interface DesktopOnboardingOverlayProps {
  enabled: boolean
  onCompleted?: () => void
  requestGateway: OnboardingContext['requestGateway']
}

export interface ApiKeyOption {
  description?: string
  docsUrl: string
  envKey: string
  id: string
  name: string
  placeholder?: string
  short?: string
}

const API_KEY_OPTIONS: ApiKeyOption[] = [
  {
    id: 'openrouter',
    name: 'OpenRouter',
    envKey: 'OPENROUTER_API_KEY',
    docsUrl: 'https://openrouter.ai/keys'
  },
  {
    id: 'openai',
    name: 'OpenAI',
    envKey: 'OPENAI_API_KEY',
    docsUrl: 'https://platform.openai.com/api-keys'
  },
  {
    id: 'gemini',
    name: 'Google Gemini',
    envKey: 'GEMINI_API_KEY',
    docsUrl: 'https://aistudio.google.com/app/apikey'
  },
  {
    id: 'xai',
    name: 'xAI Grok',
    envKey: 'XAI_API_KEY',
    docsUrl: 'https://console.x.ai/'
  },
  {
    id: 'local',
    name: 'Local / custom endpoint',
    envKey: 'OPENAI_BASE_URL',
    docsUrl: 'https://github.com/NousResearch/hermes-agent#bring-your-own-endpoint',
    placeholder: 'http://127.0.0.1:8000/v1'
  }
]

// Build the FULL API-key provider catalog from the backend model options so the
// onboarding / Providers key form lists every `api_key` provider `hermes model`
// knows about — not just the hand-curated five. Curated entries keep their
// richer copy + placeholders and float to the top (recommended defaults); every
// other api_key provider is appended with a generic "paste {KEY}" affordance.
// OAuth / external providers are intentionally excluded here — they go through
// the OAuth picker / sign-in flow, not a pasted key.
function useApiKeyCatalog(): ApiKeyOption[] {
  const [rows, setRows] = useState<ModelOptionProvider[]>([])

  useEffect(() => {
    let cancelled = false

    // Best-effort — on failure the curated defaults still render. Wrapped in
    // Promise.resolve().then so a synchronous throw (e.g. no desktop bridge in
    // tests) is funneled into the same .catch instead of escaping.
    void Promise.resolve()
      .then(() => getGlobalModelOptions())
      .then(res => {
        if (!cancelled) {
          setRows(res.providers ?? [])
        }
      })
      .catch(() => {
        // Ignore — fall back to the curated API_KEY_OPTIONS only.
      })

    return () => {
      cancelled = true
    }
  }, [])

  return useMemo(() => {
    const curatedByEnv = new Map(API_KEY_OPTIONS.map(o => [o.envKey, o]))
    const derived: ApiKeyOption[] = []
    const seenEnv = new Set<string>(API_KEY_OPTIONS.map(o => o.envKey))

    for (const row of rows) {
      // Only api_key providers can be activated with a pasted key. Skip OAuth /
      // external / managed flows and anything missing an env var to write to.
      if (row.auth_type && row.auth_type !== 'api_key') {
        continue
      }

      const envKey = row.key_env

      if (!envKey || seenEnv.has(envKey)) {
        continue
      }

      seenEnv.add(envKey)
      derived.push({
        id: row.slug,
        name: row.name,
        envKey,
        description: `Direct API access to ${row.name}.`,
        docsUrl: ''
      })
    }

    // Curated first (recommended order), then the rest alphabetically so the
    // long tail is scannable.
    derived.sort((a, b) => a.name.localeCompare(b.name))

    return [...API_KEY_OPTIONS.filter(o => curatedByEnv.has(o.envKey)), ...derived]
  }, [rows])
}

const PROVIDER_DISPLAY: Record<string, { order: number; title: string }> = {
  nous: { order: 0, title: 'Nous Portal' },
  'openai-codex': { order: 1, title: 'OpenAI OAuth (ChatGPT)' },
  'minimax-oauth': { order: 2, title: 'MiniMax' },
  'qwen-oauth': { order: 3, title: 'Qwen Code' },
  'xai-oauth': { order: 4, title: 'xAI Grok' },
  // Both Anthropic entries sit at the bottom: the API-key path first, then
  // the subscription OAuth path (only works with extra usage credits).
  anthropic: { order: 5, title: 'Anthropic API Key' },
  'claude-code': { order: 6, title: 'Anthropic OAuth: Required Extra Usage Credits to Use Subscription' }
}

const assetPath = (path: string) => `${import.meta.env.BASE_URL}${path.replace(/^\/+/, '')}`

export const providerTitle = (p: OAuthProvider) => PROVIDER_DISPLAY[p.id]?.title ?? p.name
const orderOf = (p: OAuthProvider) => PROVIDER_DISPLAY[p.id]?.order ?? 99

export const sortProviders = (providers: OAuthProvider[]) =>
  [...providers].sort((a, b) => orderOf(a) - orderOf(b) || a.name.localeCompare(b.name))

// Exit choreography, mirroring the gateway "connecting" overlay's timing:
// text-out (360ms: CONNECTED fades down, rest scrambles+fades) → hold (300ms)
// → surface-out (520ms, held back by [transition-delay:660ms]). Finalize after.
const ONBOARDING_EXIT_MS = 1180

export function DesktopOnboardingOverlay({ enabled, onCompleted, requestGateway }: DesktopOnboardingOverlayProps) {
  const { t } = useI18n()
  const onboarding = useStore($desktopOnboarding)
  const boot = useStore($desktopBoot)
  const ctxRef = useRef<OnboardingContext>({ requestGateway, onCompleted })
  ctxRef.current = { requestGateway, onCompleted }

  const ctx = useMemo<OnboardingContext>(
    () => ({
      requestGateway: (...args) => ctxRef.current.requestGateway(...args),
      onCompleted: () => ctxRef.current.onCompleted?.()
    }),
    []
  )

  // Cinematic exit on "Begin": dissolve the panel + overlay (revealing the chat
  // behind), THEN finalize so the unmount lands after the fade — mirrors the
  // connecting overlay's exit choreography instead of cutting instantly.
  const [leaving, setLeaving] = useState(false)

  const finalizeOnboarding = () => {
    if (leaving) {
      return
    }

    const reduce = typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches

    if (reduce) {
      confirmOnboardingModel(ctx)

      return
    }

    setLeaving(true)
    window.setTimeout(() => confirmOnboardingModel(ctx), ONBOARDING_EXIT_MS)
  }

  useEffect(() => {
    if (enabled || onboarding.requested) {
      void refreshOnboarding(ctx)
    }
  }, [ctx, enabled, onboarding.requested])

  // When the Providers settings page asked to connect a specific provider, the
  // store stashed its id. Once the provider list has loaded and we're back at
  // an idle picker, launch that exact OAuth flow so the user lands directly in
  // sign-in instead of the picker they just came from.
  useEffect(() => {
    if (!onboarding.manual || onboarding.providers === null || onboarding.flow.status !== 'idle') {
      return
    }

    const pendingId = peekPendingProviderOAuth()

    if (!pendingId) {
      return
    }

    const provider = onboarding.providers.find(p => p.id === pendingId)

    if (provider) {
      // Only clear once we've committed to launching it, so a failed/empty
      // provider fetch doesn't silently drop the hand-off.
      clearPendingProviderOAuth()
      void startProviderOAuth(provider, ctx)
    } else if (onboarding.providers.length > 0) {
      // The list loaded but the id isn't a real provider — drop the stale
      // hand-off. An empty list means the fetch isn't ready yet, so keep it
      // and let a later refresh retry.
      clearPendingProviderOAuth()
    }
  }, [ctx, onboarding.flow.status, onboarding.manual, onboarding.providers])

  // Mount from frame 1 so we replace the boot overlay seamlessly. The
  // configured field stays null until the runtime check resolves; only then
  // do we know whether to dismiss (true) or surface the picker (false).
  // EXCEPTION: manual mode (user opened the selector from a working app to
  // add/switch a provider) shows the overlay regardless of configured state.
  if (onboarding.configured === true && !onboarding.manual) {
    return null
  }

  // The user chose "I'll choose a provider later" on first run. Stay out of the
  // way on every subsequent launch — they re-enter via Settings → Providers
  // (manual mode), which sets manual=true and bypasses this gate.
  if (onboarding.firstRunSkipped && !onboarding.manual) {
    return null
  }

  const { flow } = onboarding
  // Show the launch reason only when it's a meaningful, caller-supplied prompt —
  // suppress the generic defaults (useless noise) and provider-setup errors
  // (those are surfaced by FlowPanel, not as a banner).
  const rawReason = onboarding.reason?.trim() || null

  const reason =
    rawReason &&
    !isProviderSetupErrorMessage(rawReason) &&
    rawReason !== DEFAULT_ONBOARDING_REASON &&
    rawReason !== DEFAULT_MANUAL_ONBOARDING_REASON
      ? rawReason
      : null

  // In manual mode the app is already configured, so the flow is "ready"
  // immediately — no runtime gate needed. Otherwise wait for the readiness
  // check (configured === false) before showing the picker.
  const ready = onboarding.manual || (enabled && onboarding.configured === false)
  const showPicker = flow.status === 'idle' || flow.status === 'success'
  // The final "you're in" screen drops the card chrome and floats centered on
  // the surface — same bare, cinematic treatment as the connecting overlay.
  const bare = ready && !showPicker && flow.status === 'confirming_model'

  return (
    <div
      className={cn(
        'fixed inset-0 z-1300 flex items-center justify-center bg-(--ui-chat-surface-background) p-6 transition-opacity duration-[520ms] ease-out',
        // On the bare confirm screen, hold the surface (text-out + hold) so the
        // per-element exit plays before it dissolves.
        bare && leaving ? '[transition-delay:660ms]' : '',
        leaving ? 'pointer-events-none opacity-0' : 'opacity-100'
      )}
    >
      <div
        className={cn(
          'relative w-full max-w-[45rem] transition-all duration-500 ease-out',
          bare
            ? ''
            : 'overflow-hidden rounded-xl border border-(--stroke-nous) bg-(--ui-chat-bubble-background) shadow-nous',
          // Bare confirm screen orchestrates its own per-element exit; the
          // carded states use the simple lift/blur dissolve.
          leaving && !bare
            ? '-translate-y-1 scale-[0.985] opacity-0 blur-[2px]'
            : 'translate-y-0 scale-100 opacity-100 blur-0'
        )}
      >
        {showPicker || !ready ? <Header /> : null}
        {onboarding.manual ? (
          <Button
            aria-label={t.common.close}
            className="absolute right-3 top-3 z-10 text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground"
            onClick={() => closeManualOnboarding()}
            size="icon-sm"
            variant="ghost"
          >
            <Codicon name="close" size="1rem" />
          </Button>
        ) : null}
        <div className="grid gap-3 p-5">
          {reason ? <ReasonNotice reason={reason} /> : null}
          {ready ? (
            showPicker ? (
              <Picker ctx={ctx} />
            ) : (
              <FlowPanel ctx={ctx} flow={flow} leaving={leaving} onBegin={finalizeOnboarding} />
            )
          ) : (
            <Preparing boot={boot} />
          )}
        </div>
      </div>
    </div>
  )
}

// The launch reason is a prompt ("why am I seeing this"), not an error. Only
// rendered for meaningful caller-supplied reasons (defaults are filtered out
// upstream), so it never shows the generic "no provider configured" noise.
function ReasonNotice({ reason }: { reason: string }) {
  return (
    <div className="rounded-2xl border border-(--ui-stroke-tertiary) bg-(--ui-bg-tertiary)/40 px-4 py-3 text-sm text-muted-foreground">
      {reason}
    </div>
  )
}

function Preparing({ boot }: { boot: DesktopBootState }) {
  const { t } = useI18n()
  const progress = Math.max(2, Math.min(100, Math.round(boot.progress)))
  const hasError = Boolean(boot.error)
  const installing = boot.phase.startsWith('runtime.')

  return (
    <div className="grid gap-3" role="status">
      <p className="text-sm text-muted-foreground">
        {installing ? t.onboarding.preparingInstall : t.onboarding.starting}
      </p>
      <div className="h-2 overflow-hidden rounded-full bg-muted">
        <div
          className={cn(
            'h-full rounded-full bg-primary transition-[width] duration-300 ease-out',
            hasError && 'bg-destructive'
          )}
          style={{ width: `${progress}%` }}
        />
      </div>
      <div className="flex items-center justify-between gap-3 text-xs text-muted-foreground">
        <span className="truncate">{boot.message}</span>
        <span>{progress}%</span>
      </div>
      {hasError ? <p className="text-xs text-destructive">{boot.error}</p> : null}
    </div>
  )
}

function Header() {
  const { t } = useI18n()

  return (
    <div className="bg-(--ui-chat-bubble-background) px-5 pt-5 pb-1">
      <h2 className="text-[0.9375rem] font-semibold tracking-tight">{t.onboarding.headerTitle}</h2>
      <p className="mt-1 max-w-xl text-[0.8125rem] leading-5 text-(--ui-text-tertiary)">{t.onboarding.headerDesc}</p>
    </div>
  )
}

export const FEATURED_ID = 'nous'
const SHOW_ALL_KEY = 'hermes-onboarding-show-all-v1'

const readShowAll = () => {
  try {
    return window.localStorage.getItem(SHOW_ALL_KEY) === '1'
  } catch {
    return false
  }
}

const persistShowAll = (value: boolean) => {
  try {
    window.localStorage.setItem(SHOW_ALL_KEY, value ? '1' : '0')
  } catch {
    // localStorage unavailable — degrade silently.
  }

  return value
}

export function Picker({ ctx }: { ctx: OnboardingContext }) {
  const { t } = useI18n()
  const { localEndpoint, manual, mode, providers } = useStore($desktopOnboarding)
  const [showAll, setShowAll] = useState(readShowAll)
  const ordered = useMemo(() => (providers ? sortProviders(providers) : []), [providers])
  const hasOauth = ordered.length > 0
  const apiKeyOptions = useApiKeyCatalog()

  // localEndpoint forces the key form regardless of `mode` (which a manual
  // provider refresh may flip back to 'oauth'); it preselects the local option
  // and hides the "back to sign in" link since the user came specifically to
  // configure a custom endpoint.
  if (localEndpoint || mode === 'apikey' || !hasOauth) {
    return (
      <div className="grid gap-3">
        <ApiKeyForm
          canGoBack={hasOauth && !localEndpoint}
          initialEnvKey={localEndpoint ? 'OPENAI_BASE_URL' : undefined}
          onBack={() => setOnboardingMode('oauth')}
          onSave={(envKey, value, name, apiKey) => saveOnboardingApiKey(envKey, value, name, ctx, apiKey)}
          options={apiKeyOptions}
        />
        {manual ? null : (
          <div className="flex justify-center border-t border-(--ui-stroke-tertiary) pt-3">
            <ChooseLaterLink />
          </div>
        )}
      </div>
    )
  }

  if (providers === null) {
    return <Status>{t.onboarding.lookingUpProviders}</Status>
  }

  const select = (p: OAuthProvider) => void startProviderOAuth(p, ctx)
  const featured = ordered.find(p => p.id === FEATURED_ID) ?? null
  const rest = featured ? ordered.filter(p => p.id !== FEATURED_ID) : ordered
  // Collapse the secondary providers behind a disclosure only when Nous
  // Portal is present to anchor the choice — otherwise show the full list.
  const collapsible = Boolean(featured) && rest.length > 0
  const showRest = !collapsible || showAll

  return (
    <div className="grid gap-2">
      <div className="grid max-h-[60dvh] gap-2 overflow-y-auto p-1">
        {featured ? <FeaturedProviderRow onSelect={select} provider={featured} /> : null}
        {showRest ? (
          <>
            {rest.map(p => (
              <ProviderRow key={p.id} onSelect={select} provider={p} />
            ))}
            <KeyProviderRow onClick={() => setOnboardingMode('apikey')} />
          </>
        ) : null}
      </div>
      {collapsible ? (
        <Button
          className="mt-1 self-center font-medium"
          onClick={() => setShowAll(persistShowAll(!showAll))}
          size="xs"
          type="button"
          variant="text"
        >
          {showAll ? t.onboarding.collapse : t.onboarding.otherProviders}
          <ChevronDown className={cn('size-3.5 transition', showAll && 'rotate-180')} />
        </Button>
      ) : null}
      <div className="flex items-center justify-between gap-3 pt-1">
        {/* First run only: let the user defer the choice and land in the app.
            In manual mode the overlay already has a close affordance, so the
            "choose later" escape would be redundant — hide it. */}
        {manual ? <span /> : <ChooseLaterLink />}
        <Button
          className="-mr-2 font-medium"
          onClick={() => setOnboardingMode('apikey')}
          size="xs"
          type="button"
          variant="text"
        >
          {t.onboarding.haveApiKey}
        </Button>
      </div>
    </div>
  )
}

// "I'll choose a provider later" — dismisses the first-run picker and persists
// the skip so it never re-nags. The user connects a provider any time from
// Settings → Providers. Rendered only on the unconfigured first-run flow.
function ChooseLaterLink() {
  const { t } = useI18n()

  return (
    <Button className="font-medium" onClick={() => dismissFirstRunOnboarding()} size="xs" type="button" variant="text">
      {t.onboarding.chooseLater}
    </Button>
  )
}

export function FeaturedProviderRow({
  onSelect,
  provider
}: {
  onSelect: (provider: OAuthProvider) => void
  provider: OAuthProvider
}) {
  const { t } = useI18n()
  const loggedIn = provider.status?.logged_in

  return (
    <button
      className="group relative flex w-full items-center justify-between gap-4 rounded-[8px] bg-primary/[0.06] px-3 py-2.5 text-left transition-colors hover:bg-primary/10"
      onClick={() => onSelect(provider)}
      type="button"
    >
      <span aria-hidden className="arc-border arc-reverse arc-nous" />
      <div className="min-w-0">
        <div className="flex items-center gap-2">
          <img alt="" className="size-5 shrink-0 rounded" src={assetPath('apple-touch-icon.png')} />
          <span className="text-[length:var(--conversation-text-font-size)] font-semibold">
            {providerTitle(provider)}
          </span>
          {loggedIn ? (
            <ConnectedTag />
          ) : (
            <span className="inline-flex items-center gap-1.5 bg-primary px-2 py-0.5 text-[0.64rem] font-semibold uppercase tracking-[0.16em] text-primary-foreground">
              <span aria-hidden="true" className="dither inline-block size-2 shrink-0" />
              {t.onboarding.recommended}
            </span>
          )}
        </div>
        <p className="mt-1 text-xs leading-5 text-muted-foreground">{t.onboarding.featuredPitch}</p>
      </div>
      <ChevronRight className="size-4 shrink-0 text-primary transition group-hover:translate-x-0.5" />
    </button>
  )
}

function ConnectedTag() {
  const { t } = useI18n()

  return (
    <span className="inline-flex items-center gap-1 bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary">
      <Check className="size-3" />
      {t.onboarding.connected}
    </span>
  )
}

const PROVIDER_ROW_CLASS =
  'group flex w-full items-center justify-between gap-3 rounded-[6px] px-3 py-2.5 text-left transition-colors hover:bg-(--ui-control-hover-background)'

export function KeyProviderRow({ onClick }: { onClick: () => void }) {
  const { t } = useI18n()

  return (
    <button className={PROVIDER_ROW_CLASS} onClick={onClick} type="button">
      <div className="min-w-0">
        <span className="text-[length:var(--conversation-text-font-size)] font-semibold">OpenRouter</span>
        <p className="mt-1 text-xs leading-5 text-muted-foreground">{t.onboarding.openRouterPitch}</p>
      </div>
      <ChevronRight className="size-4 text-muted-foreground transition group-hover:text-foreground" />
    </button>
  )
}

export function ProviderRow({
  onSelect,
  provider
}: {
  onSelect: (provider: OAuthProvider) => void
  provider: OAuthProvider
}) {
  const { t } = useI18n()
  const loggedIn = provider.status?.logged_in
  const Trail = provider.flow === 'external' ? Terminal : ChevronRight

  return (
    <button className={PROVIDER_ROW_CLASS} onClick={() => onSelect(provider)} type="button">
      <div className="min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-[length:var(--conversation-text-font-size)] font-semibold">
            {providerTitle(provider)}
          </span>
          {loggedIn ? <ConnectedTag /> : null}
        </div>
        <p className="mt-1 text-xs leading-5 text-muted-foreground">{t.onboarding.flowSubtitles[provider.flow]}</p>
      </div>
      <Trail className="size-4 text-muted-foreground transition group-hover:text-foreground" />
    </button>
  )
}

// Presentational two-column key picker. Onboarding feeds it its curated
// options + a ctx-bound save; the Providers settings page feeds it the full
// provider catalog + a setEnvVar-backed save (plus `isSet`/`onClear` so it can
// double as a manage surface). Keep it free of store/ctx coupling so both
// surfaces render the identical form.
export function ApiKeyForm({
  canGoBack,
  initialEnvKey,
  isSet,
  onBack,
  onClear,
  onSave,
  options = API_KEY_OPTIONS,
  redactedValue
}: {
  canGoBack: boolean
  /** Preselect a specific option by env key (e.g. 'OPENAI_BASE_URL' to land on
   *  the local / custom endpoint form). Falls back to the first option. */
  initialEnvKey?: string
  isSet?: (envKey: string) => boolean
  onBack: () => void
  onClear?: (envKey: string) => void
  onSave: (envKey: string, value: string, name: string, apiKey?: string) => Promise<{ message?: string; ok: boolean }>
  options?: ApiKeyOption[]
  redactedValue?: (envKey: string) => null | string | undefined
}) {
  const { t } = useI18n()

  const [option, setOption] = useState<ApiKeyOption>(() => options.find(o => o.envKey === initialEnvKey) ?? options[0])

  const [value, setValue] = useState('')
  // Optional endpoint API key, only used by the local / custom endpoint option
  // (whose `value` is the base URL). Cleared whenever the option changes.
  const [localKey, setLocalKey] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<null | string>(null)
  // `options` can change at runtime when callers filter the catalog (e.g. the
  // Providers page wiring its search into this grid). Keep the selection valid
  // by snapping back to the first remaining option when the current one drops.
  useEffect(() => {
    if (options.length > 0 && !options.some(o => o.envKey === option.envKey)) {
      setOption(options[0])
      setValue('')
      setLocalKey('')
      setError(null)
    }
  }, [option.envKey, options])
  // The catalog grid can be tall, leaving the entry field far below the fold.
  // On selection we scroll the field into view and focus it so it's always
  // obvious where to paste next.
  const entryRef = useRef<HTMLDivElement>(null)

  const pick = (o: ApiKeyOption) => {
    setOption(o)
    setValue('')
    setLocalKey('')
    setError(null)
    requestAnimationFrame(() => {
      entryRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' })
      entryRef.current?.querySelector('input')?.focus()
    })
  }

  const isLocal = option.envKey === 'OPENAI_BASE_URL'
  const alreadySet = isSet?.(option.envKey) ?? false
  // When set, surface the backend's redacted value (e.g. "sk-12…wxyz") as the
  // placeholder so users can eyeball that the right key is in place.
  const currentRedacted = alreadySet ? (redactedValue?.(option.envKey) ?? null) : null
  // Only require a non-empty value — no length/format validation, so a short
  // or unusual key can't block the user from continuing.
  const canSave = value.trim().length >= 1
  const optionCopy = t.onboarding.apiKeyOptions[option.id]
  const optionDescription = optionCopy?.description ?? option.description

  const submit = async () => {
    if (!canSave || saving) {
      return
    }

    setSaving(true)
    setError(null)
    const result = await onSave(option.envKey, value, option.name, isLocal ? localKey : undefined)

    if (result.ok) {
      setValue('')
      setLocalKey('')
    } else {
      setError(result.message ?? t.onboarding.couldNotSave)
    }

    setSaving(false)
  }

  return (
    <div className="grid gap-4">
      {canGoBack ? (
        <Button className="-mt-1 self-start font-medium" onClick={onBack} size="xs" type="button" variant="text">
          <ChevronLeft className="size-3" />
          {t.onboarding.backToSignIn}
        </Button>
      ) : null}

      <div className="grid max-h-[42dvh] gap-2 overflow-y-auto p-1 sm:grid-cols-2">
        {options.map(o => (
          <button
            className={cn(
              'rounded-2xl border bg-background/60 p-3 text-left transition hover:bg-accent/50',
              option.envKey === o.envKey ? 'border-primary ring-2 ring-primary/20' : 'border-transparent'
            )}
            key={o.envKey}
            onClick={() => pick(o)}
            type="button"
          >
            <div className="flex items-center justify-between gap-2">
              <span className="text-sm font-medium">{o.name}</span>
              {isSet?.(o.envKey) ? <Check className="size-3.5 text-muted-foreground" /> : null}
            </div>
            {(t.onboarding.apiKeyOptions[o.id]?.short ?? o.short) ? (
              <p className="mt-1 text-xs text-muted-foreground">{t.onboarding.apiKeyOptions[o.id]?.short ?? o.short}</p>
            ) : null}
          </button>
        ))}
      </div>

      <div className="grid scroll-mt-4 gap-2" ref={entryRef}>
        <div className="flex items-center justify-between gap-3">
          <p className="text-sm leading-6 text-muted-foreground">{optionDescription}</p>
          {option.docsUrl ? <DocsLink href={option.docsUrl}>{t.onboarding.getKey}</DocsLink> : null}
        </div>
        <Input
          autoComplete="off"
          autoFocus
          className="font-mono"
          onChange={e => setValue(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && void submit()}
          placeholder={
            currentRedacted ??
            (alreadySet ? t.onboarding.replaceCurrent : option.placeholder || t.onboarding.pasteApiKey)
          }
          type={isLocal ? 'text' : 'password'}
          value={value}
        />
        {isLocal ? (
          <Input
            autoComplete="off"
            className="font-mono"
            onChange={e => setLocalKey(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && void submit()}
            placeholder={t.onboarding.localApiKeyPlaceholder}
            type="password"
            value={localKey}
          />
        ) : null}
        {error ? <p className="text-xs text-destructive">{error}</p> : null}
      </div>

      <div className="flex items-center justify-between gap-3">
        <div>
          {alreadySet && onClear ? (
            <Button onClick={() => onClear(option.envKey)} size="sm" variant="ghost">
              {t.common.remove}
            </Button>
          ) : null}
        </div>
        <Button disabled={!canSave || saving} onClick={() => void submit()}>
          {saving ? <Loader2 className="animate-spin" /> : <KeyRound />}
          {saving ? t.onboarding.connecting : alreadySet ? t.onboarding.update : t.common.connect}
        </Button>
      </div>
    </div>
  )
}

function FlowPanel({
  ctx,
  flow,
  leaving,
  onBegin
}: {
  ctx: OnboardingContext
  flow: OnboardingFlow
  leaving: boolean
  onBegin: () => void
}) {
  const { t } = useI18n()
  const title = 'provider' in flow && flow.provider ? providerTitle(flow.provider) : ''

  if (flow.status === 'starting') {
    return <Status>{t.onboarding.startingSignIn(title)}</Status>
  }

  if (flow.status === 'submitting') {
    return <Status>{t.onboarding.verifyingCode(title)}</Status>
  }

  if (flow.status === 'success') {
    return <DecodedLabel text={t.onboarding.connectedPicking(title)} />
  }

  if (flow.status === 'confirming_model') {
    return <ConfirmingModelPanel flow={flow} leaving={leaving} onBegin={onBegin} />
  }

  if (flow.status === 'error') {
    return (
      <div className="grid gap-3">
        <div className="flex items-center gap-1.5 text-sm text-destructive">
          <ErrorIcon className="shrink-0" size="0.875rem" />
          <span>{flow.message || t.onboarding.signInFailed}</span>
        </div>
        <div className="flex justify-end">
          <Button onClick={cancelOnboardingFlow} variant="outline">
            {t.onboarding.pickDifferentProvider}
          </Button>
        </div>
      </div>
    )
  }

  if (flow.status === 'awaiting_user') {
    return (
      <Step title={t.onboarding.signInWith(title)}>
        <ol className="list-decimal space-y-1 pl-5 text-sm text-muted-foreground">
          <li>{t.onboarding.openedBrowser(title)}</li>
          <li>{t.onboarding.authorizeThere}</li>
          <li>{t.onboarding.copyAuthCode}</li>
        </ol>
        <Input
          autoFocus
          onChange={e => setOnboardingCode(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && void submitOnboardingCode(ctx)}
          placeholder={t.onboarding.pasteAuthCode}
          value={flow.code}
        />
        <FlowFooter left={<DocsLink href={flow.start.auth_url}>{t.onboarding.reopenAuthPage}</DocsLink>}>
          <CancelBtn />
          <Button disabled={!flow.code.trim()} onClick={() => void submitOnboardingCode(ctx)}>
            {t.common.continue}
          </Button>
        </FlowFooter>
      </Step>
    )
  }

  if (flow.status === 'awaiting_browser') {
    return (
      <Step title={t.onboarding.signInWith(title)}>
        <p className="text-sm text-muted-foreground">{t.onboarding.autoBrowser(title)}</p>
        <FlowFooter left={<DocsLink href={flow.start.auth_url}>{t.onboarding.reopenSignInPage}</DocsLink>}>
          <span className="flex items-center gap-2 text-xs text-muted-foreground">
            <Loader2 className="size-3 animate-spin" />
            {t.onboarding.waitingAuthorize}
          </span>
          <CancelBtn size="sm" />
        </FlowFooter>
      </Step>
    )
  }

  if (flow.status === 'external_pending') {
    return (
      <Step title={t.onboarding.signInWith(title)}>
        <p className="text-sm text-muted-foreground">{t.onboarding.externalPending(title)}</p>
        <CodeBlock copied={flow.copied} onCopy={() => void copyExternalCommand()} text={flow.provider.cli_command} />
        <FlowFooter
          left={
            flow.provider.docs_url ? (
              <DocsLink href={flow.provider.docs_url}>{t.onboarding.docs(title)}</DocsLink>
            ) : null
          }
        >
          <CancelBtn />
          <Button onClick={() => void recheckExternalSignin(ctx)}>{t.onboarding.signedIn}</Button>
        </FlowFooter>
      </Step>
    )
  }

  if (flow.status !== 'polling') {
    return null
  }

  return (
    <Step title={t.onboarding.signInWith(title)}>
      <p className="text-sm text-muted-foreground">{t.onboarding.deviceCodeOpened(title)}</p>
      <DeviceCode code={flow.start.user_code} copied={flow.copied} onCopy={() => void copyDeviceCode()} />
      <FlowFooter left={<DocsLink href={flow.start.verification_url}>{t.onboarding.reopenVerification}</DocsLink>}>
        <span className="flex items-center gap-2 text-xs text-muted-foreground">
          <Loader2 className="size-3 animate-spin" />
          {t.onboarding.waitingAuthorize}
        </span>
        <CancelBtn size="sm" />
      </FlowFooter>
    </Step>
  )
}

function Step({ children, title }: { children: React.ReactNode; title: string }) {
  return (
    <div className="grid gap-4">
      <h3 className="text-sm font-semibold">{title}</h3>
      {children}
    </div>
  )
}

// Device-code display: OTP-style — each character in its own readonly cell.
// The whole row is the copy button (no side button, no checkmark); on copy the
// cells flash emerald for feedback. Dashes render as quiet separators.
function DeviceCode({ code, copied, onCopy }: { code: string; copied: boolean; onCopy: () => void }) {
  const { t } = useI18n()

  return (
    <button
      aria-label={t.onboarding.copy}
      className="group flex w-full items-center justify-center gap-1.5"
      onClick={onCopy}
      type="button"
    >
      {[...code].map((ch, i) =>
        ch === '-' || ch === ' ' ? (
          <span className="w-1.5 text-center text-lg text-muted-foreground" key={i}>
            –
          </span>
        ) : (
          <span
            className={cn(
              'flex size-10 items-center justify-center rounded-md border font-mono text-xl font-semibold uppercase transition-colors',
              copied
                ? 'border-primary/50 text-primary'
                : 'border-(--stroke-nous) text-foreground group-hover:border-(--ui-stroke-secondary)'
            )}
            key={i}
          >
            {ch}
          </span>
        )
      )}
    </button>
  )
}

function CodeBlock({ copied, onCopy, text }: { copied: boolean; onCopy: () => void; text: string }) {
  const { t } = useI18n()

  return (
    <div className="flex items-center justify-between gap-3 rounded-md border border-(--stroke-nous) px-3 py-2">
      <code className="min-w-0 flex-1 truncate font-mono text-sm">
        <span className="mr-2 select-none text-muted-foreground">$</span>
        {text}
      </code>
      <Button onClick={onCopy} size="sm" variant="outline">
        {copied ? t.common.copied : t.onboarding.copy}
      </Button>
    </div>
  )
}

function FlowFooter({ children, left }: { children: React.ReactNode; left?: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <div className="min-w-0">{left}</div>
      <div className="flex items-center gap-3">{children}</div>
    </div>
  )
}

function CancelBtn({ size = 'default' }: { size?: 'default' | 'sm' }) {
  const { t } = useI18n()

  return (
    <Button onClick={cancelOnboardingFlow} size={size} variant="ghost">
      {t.common.cancel}
    </Button>
  )
}

// Borrowed from the gateway "connecting" overlay: a mono, letter-spaced label
// that decodes left-to-right from scrambled glyphs into the real text, with a
// blinking block cursor. Ties onboarding's success moment to that same motif.
// Cuneiform glyphs (array, since each is a surrogate pair) for the scramble.
// Hero "X CONNECTED" decode uses the SAME ascii map as the connecting overlay.
const ASCII_GLYPHS = [...'/\\|-_=+<>~:*']
const pickAscii = () => ASCII_GLYPHS[(Math.random() * ASCII_GLYPHS.length) | 0]
// Cuneiform is reserved for the subtle "other text" (model name + BEGIN) easter egg.
const SCRAMBLE_GLYPHS = [...'𒀀𒀁𒀂𒀅𒀊𒀖𒀜𒀭𒀲𒀸𒁀𒁉𒁒𒁕𒁹𒂊𒃻𒄆𒄴𒅀𒆍𒇽𒈨𒉡']
const GLYPH_SET = new Set(SCRAMBLE_GLYPHS)
const pickGlyph = () => SCRAMBLE_GLYPHS[(Math.random() * SCRAMBLE_GLYPHS.length) | 0]
// How many trailing characters of each word scramble during decode-in.
const DECODE_TAIL = 4

// Renders text where cuneiform scramble-glyphs are dropped to a smaller em-size
// (resolved Latin chars stay full size) — keeps the easter-egg glyphs subtle.
function GlyphText({ text }: { text: string }) {
  return (
    <>
      {Array.from(text, (ch, i) =>
        GLYPH_SET.has(ch) ? (
          <span className="text-[0.62em]" key={i}>
            {ch}
          </span>
        ) : (
          ch
        )
      )}
    </>
  )
}

function useDecoded(text: string): string {
  const [out, setOut] = useState(text)

  useEffect(() => {
    if (typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) {
      setOut(text)

      return
    }

    // Each WORD keeps its head static and only churns its tail (last few chars),
    // resolving left-to-right across all tails — same anchor-the-prefix trick the
    // connecting overlay uses ("CONN" static, "ECTING" churns), applied per word
    // so both the provider and "CONNECTED" decode and time stays constant.
    const chars = [...text]
    const scrambleable = chars.map(() => false)

    for (let i = 0; i < chars.length; ) {
      if (!/[a-z0-9]/i.test(chars[i])) {
        i += 1

        continue
      }

      let j = i

      while (j < chars.length && /[a-z0-9]/i.test(chars[j])) {
        j += 1
      }

      for (let k = Math.max(i, j - DECODE_TAIL); k < j; k += 1) {
        scrambleable[k] = true
      }

      i = j
    }

    const tailIndices = chars.map((_, idx) => idx).filter(idx => scrambleable[idx])
    let resolved = 0

    const id = window.setInterval(() => {
      resolved += 0.5
      const settled = new Set(tailIndices.slice(0, Math.floor(resolved)))

      setOut(chars.map((ch, idx) => (scrambleable[idx] && !settled.has(idx) ? pickAscii() : ch)).join(''))

      if (Math.floor(resolved) >= tailIndices.length) {
        window.clearInterval(id)
      }
    }, 45)

    return () => window.clearInterval(id)
  }, [text])

  return out
}

// Continuously scrambles alphanumeric chars while `active` (used on exit so the
// model name / button decay into ascii noise as they fade).
function useScramble(text: string, active: boolean): string {
  const [out, setOut] = useState(text)

  useEffect(() => {
    if (!active) {
      setOut(text)

      return
    }

    const id = window.setInterval(() => {
      setOut(Array.from(text, ch => (/[a-z0-9]/i.test(ch) ? pickGlyph() : ch)).join(''))
    }, 45)

    return () => window.clearInterval(id)
  }, [text, active])

  return out
}

function DecodedLabel({ leaving, text }: { leaving?: boolean; text: string }) {
  const decoded = useDecoded(text.toUpperCase())

  return (
    <span
      className={cn(
        'inline-flex items-center font-mono text-xs font-semibold uppercase tracking-[0.28em] tabular-nums text-primary transition duration-[360ms] ease-out',
        leaving ? 'translate-y-2 opacity-0 saturate-0' : 'translate-y-0 opacity-100 saturate-100'
      )}
    >
      <GlyphText text={decoded} />
      <span
        aria-hidden="true"
        className="dither ml-1.5 -mr-[0.875rem] inline-block size-2 shrink-0 -translate-y-px rounded-[1px] text-primary"
        style={{ animation: 'ob-decode-cursor 1s step-end infinite' }}
      />
      <style>{'@keyframes ob-decode-cursor { 0%, 49% { opacity: 1 } 50%, 100% { opacity: 0 } }'}</style>
    </span>
  )
}

// Terminal-flavored CTA to match the connecting overlay's hacker aesthetic:
// mono, uppercase, letter-spaced, wrapped in primary brackets that light up on
// hover. The whole onboarding "you're in" moment leans into this motif.
function HackeryButton({
  disabled,
  label,
  loading,
  onClick
}: {
  disabled?: boolean
  label: React.ReactNode
  loading?: boolean
  onClick: () => void
}) {
  return (
    <button
      className={cn(
        'group inline-flex items-center gap-2 rounded-md border border-(--stroke-nous) px-6 py-2.5',
        'font-mono text-xs font-semibold uppercase text-primary',
        'transition-all duration-150 hover:border-primary/60 hover:bg-primary/[0.06]',
        'disabled:pointer-events-none disabled:opacity-50'
      )}
      disabled={disabled}
      onClick={onClick}
      type="button"
    >
      <span className="text-primary/40 transition-colors group-hover:text-primary">[</span>
      {loading ? <Loader2 className="size-3 animate-spin" /> : null}
      <span className="-mr-[0.25em] pl-[0.25em] tracking-[0.25em]">{label}</span>
      <span className="text-primary/40 transition-colors group-hover:text-primary">]</span>
    </button>
  )
}

function ConfirmingModelPanel({
  flow,
  leaving,
  onBegin
}: {
  flow: Extract<OnboardingFlow, { status: 'confirming_model' }>
  leaving: boolean
  onBegin: () => void
}) {
  const { t } = useI18n()
  const scrambledModel = useScramble(flow.currentModel, leaving)
  const scrambledBegin = useScramble(t.onboarding.startChatting, leaving)
  // Local state controls whether the model picker dialog is open.
  // We reuse the existing ModelPickerDialog component (the same picker
  // available from the chat shell) rather than building an inline
  // dropdown — gives us search, multi-provider listing if relevant, and
  // a familiar UI for users who'll see this picker again later.
  const [pickerOpen, setPickerOpen] = useState(false)

  // Pull pricing + tier for the just-picked default so the confirm card
  // shows the same $/Mtok + Free/Pro info the picker and CLI do.
  const options = useQuery({
    queryKey: ['onboarding-model-options', flow.providerSlug],
    queryFn: () => getGlobalModelOptions()
  })

  const providerRow = options.data?.providers?.find(
    p => String(p.slug).toLowerCase() === flow.providerSlug.toLowerCase()
  )

  const price = providerRow?.pricing?.[flow.currentModel]
  const freeTier = providerRow?.free_tier

  return (
    <div className="grid place-items-center gap-7 py-6 text-center">
      <DecodedLabel leaving={leaving} text={t.onboarding.connectedProvider(flow.label)} />

      <div
        className={cn(
          'grid justify-items-center gap-1.5 transition duration-[360ms] ease-out',
          leaving ? 'opacity-0 saturate-0' : 'opacity-100 saturate-100'
        )}
      >
        <div className="flex items-center gap-2">
          <span className="font-mono text-[0.625rem] uppercase tracking-[0.2em] text-muted-foreground">
            {t.onboarding.defaultModel}
          </span>
          {freeTier === true && (
            <span className="rounded-sm bg-emerald-500/15 px-1 py-0.5 text-[0.6rem] font-semibold uppercase tracking-wide text-emerald-600 dark:text-emerald-400">
              {t.onboarding.freeTier}
            </span>
          )}
          {freeTier === false && (
            <span className="rounded-sm bg-primary/15 px-1 py-0.5 text-[0.6rem] font-semibold uppercase tracking-wide text-primary">
              {t.onboarding.pro}
            </span>
          )}
        </div>
        <p className="font-mono text-base">
          <GlyphText text={scrambledModel} />
        </p>
        {price && (price.input || price.output) && (
          <p className="font-mono text-xs text-muted-foreground">
            {price.free ? t.onboarding.free : t.onboarding.price(price.input || '?', price.output || '?')}
          </p>
        )}
        <Button
          className="mt-0.5 text-xs"
          disabled={flow.saving}
          onClick={() => setPickerOpen(true)}
          size="inline"
          variant="text"
        >
          {t.onboarding.change}
        </Button>
      </div>

      <div
        className={cn(
          'transition duration-[360ms] ease-out',
          leaving ? 'opacity-0 saturate-0' : 'opacity-100 saturate-100'
        )}
      >
        <HackeryButton
          disabled={flow.saving}
          label={<GlyphText text={scrambledBegin} />}
          loading={flow.saving}
          onClick={onBegin}
        />
      </div>

      {/*
        ModelPickerDialog defaults to z-130 on its content, which renders
        UNDER the onboarding overlay (z-1300) and breaks pointer events.
        Bump it above with z-[1310] so the picker sits on top of the
        onboarding panel. The dialog's own dim-backdrop layer stays at
        its default z-120 — the onboarding overlay is already dimming
        the rest of the screen, so we don't want a second backdrop.
      */}
      <ModelPickerDialog
        contentClassName="z-[1310]"
        currentModel={flow.currentModel}
        currentProvider={flow.providerSlug}
        onOpenChange={setPickerOpen}
        onSelect={({ model }) => {
          void setOnboardingModel(model)
          setPickerOpen(false)
        }}
        open={pickerOpen}
      />
    </div>
  )
}

function DocsLink({ children, href }: { children: React.ReactNode; href: string }) {
  return (
    <Button asChild size="xs" variant="text">
      <a href={href} rel="noreferrer" target="_blank">
        <ExternalLink className="size-3" />
        {children}
      </a>
    </Button>
  )
}

function Status({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex items-center gap-2.5 py-1 text-sm text-muted-foreground" role="status">
      <Loader className="size-7" type="lemniscate-bloom" />
      {children}
    </div>
  )
}
