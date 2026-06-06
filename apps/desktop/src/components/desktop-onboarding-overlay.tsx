import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { useEffect, useMemo, useRef, useState } from 'react'

import { ModelPickerDialog } from '@/components/model-picker'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Input } from '@/components/ui/input'
import { getGlobalModelOptions } from '@/hermes'
import { useI18n } from '@/i18n'
import {
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ExternalLink,
  KeyRound,
  Loader2,
  Sparkles,
  Terminal
} from '@/lib/icons'
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
import type { OAuthProvider } from '@/types/hermes'

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

const providerTitle = (p: OAuthProvider) => PROVIDER_DISPLAY[p.id]?.title ?? p.name
const orderOf = (p: OAuthProvider) => PROVIDER_DISPLAY[p.id]?.order ?? 99

export const sortProviders = (providers: OAuthProvider[]) =>
  [...providers].sort((a, b) => orderOf(a) - orderOf(b) || a.name.localeCompare(b.name))

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
  const rawReason = onboarding.reason?.trim() || null
  const reason = rawReason && !isProviderSetupErrorMessage(rawReason) ? rawReason : null
  // In manual mode the app is already configured, so the flow is "ready"
  // immediately — no runtime gate needed. Otherwise wait for the readiness
  // check (configured === false) before showing the picker.
  const ready = onboarding.manual || (enabled && onboarding.configured === false)
  const showPicker = flow.status === 'idle' || flow.status === 'success'

  return (
    <div className="fixed inset-0 z-1300 flex items-center justify-center bg-(--ui-chat-surface-background) p-6">
      <div className="relative w-full max-w-[45rem] overflow-hidden rounded-xl border border-(--ui-stroke-secondary) bg-(--ui-chat-bubble-background) shadow-sm">
        <Header />
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
          {ready ? showPicker ? <Picker ctx={ctx} /> : <FlowPanel ctx={ctx} flow={flow} /> : <Preparing boot={boot} />}
        </div>
      </div>
    </div>
  )
}

// The launch reason is a prompt ("why am I seeing this"), not an error — real
// provider-setup failures are filtered out upstream and surfaced by FlowPanel.
// Keep it neutral so it never reads as a failure.
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
        {installing
          ? t.onboarding.preparingInstall
          : t.onboarding.starting}
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
    <div className="border-b border-(--ui-stroke-tertiary) bg-(--ui-chat-bubble-background) px-5 py-4">
      <div className="flex items-start gap-3">
        <div className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-(--ui-bg-tertiary) text-(--ui-text-tertiary)">
          <Sparkles className="size-5" />
        </div>
        <div>
          <h2 className="text-[0.9375rem] font-semibold tracking-tight">{t.onboarding.headerTitle}</h2>
          <p className="mt-1 max-w-xl text-[0.8125rem] leading-5 text-(--ui-text-tertiary)">
            {t.onboarding.headerDesc}
          </p>
        </div>
      </div>
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
  const { manual, mode, providers } = useStore($desktopOnboarding)
  const [showAll, setShowAll] = useState(readShowAll)
  const ordered = useMemo(() => (providers ? sortProviders(providers) : []), [providers])
  const hasOauth = ordered.length > 0

  if (mode === 'apikey' || !hasOauth) {
    return (
      <div className="grid gap-3">
        <ApiKeyForm
          canGoBack={hasOauth}
          onBack={() => setOnboardingMode('oauth')}
          onSave={(envKey, value, name) => saveOnboardingApiKey(envKey, value, name, ctx)}
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
      {featured ? <FeaturedProviderRow onSelect={select} provider={featured} /> : null}
      {showRest ? (
        <>
          {rest.map(p => (
            <ProviderRow key={p.id} onSelect={select} provider={p} />
          ))}
          <KeyProviderRow onClick={() => setOnboardingMode('apikey')} />
        </>
      ) : null}
      {collapsible ? (
        <button
          className="flex items-center justify-center gap-1.5 pt-1 text-xs font-medium text-muted-foreground transition hover:text-foreground"
          onClick={() => setShowAll(persistShowAll(!showAll))}
          type="button"
        >
          {showAll ? t.onboarding.collapse : t.onboarding.otherProviders}
          <ChevronDown className={cn('size-3.5 transition', showAll && 'rotate-180')} />
        </button>
      ) : null}
      <div className="flex items-center justify-between gap-3 pt-1">
        {/* First run only: let the user defer the choice and land in the app.
            In manual mode the overlay already has a close affordance, so the
            "choose later" escape would be redundant — hide it. */}
        {manual ? <span /> : <ChooseLaterLink />}
        <button
          className="text-xs font-medium text-muted-foreground hover:text-foreground"
          onClick={() => setOnboardingMode('apikey')}
          type="button"
        >
          {t.onboarding.haveApiKey}
        </button>
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
    <button
      className="text-xs font-medium text-muted-foreground hover:text-foreground"
      onClick={() => dismissFirstRunOnboarding()}
      type="button"
    >
      {t.onboarding.chooseLater}
    </button>
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

export function KeyProviderRow({ onClick }: { onClick: () => void }) {
  const { t } = useI18n()

  return (
    <button
      className="group flex w-full items-center justify-between gap-3 rounded-[6px] px-3 py-2.5 text-left transition-colors hover:bg-(--ui-control-hover-background)"
      onClick={onClick}
      type="button"
    >
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
    <button
      className="group flex w-full items-center justify-between gap-3 rounded-[6px] px-3 py-2.5 text-left transition-colors hover:bg-(--ui-control-hover-background)"
      onClick={() => onSelect(provider)}
      type="button"
    >
      <div className="min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-[length:var(--conversation-text-font-size)] font-semibold">
            {providerTitle(provider)}
          </span>
          {loggedIn ? <ConnectedTag /> : null}
        </div>
        <p className="mt-1 text-xs leading-5 text-muted-foreground">
          {t.onboarding.flowSubtitles[provider.flow]}
        </p>
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
  isSet,
  onBack,
  onClear,
  onSave,
  options = API_KEY_OPTIONS,
  redactedValue
}: {
  canGoBack: boolean
  isSet?: (envKey: string) => boolean
  onBack: () => void
  onClear?: (envKey: string) => void
  onSave: (envKey: string, value: string, name: string) => Promise<{ message?: string; ok: boolean }>
  options?: ApiKeyOption[]
  redactedValue?: (envKey: string) => null | string | undefined
}) {
  const { t } = useI18n()
  const [option, setOption] = useState<ApiKeyOption>(options[0])
  const [value, setValue] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<null | string>(null)
  // `options` can change at runtime when callers filter the catalog (e.g. the
  // Providers page wiring its search into this grid). Keep the selection valid
  // by snapping back to the first remaining option when the current one drops.
  useEffect(() => {
    if (options.length > 0 && !options.some(o => o.id === option.id)) {
      setOption(options[0])
      setValue('')
      setError(null)
    }
  }, [option.id, options])
  // The catalog grid can be tall, leaving the entry field far below the fold.
  // On selection we scroll the field into view and focus it so it's always
  // obvious where to paste next.
  const entryRef = useRef<HTMLDivElement>(null)

  const pick = (o: ApiKeyOption) => {
    setOption(o)
    setValue('')
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
    const result = await onSave(option.envKey, value, option.name)

    if (result.ok) {
      setValue('')
    } else {
      setError(result.message ?? t.onboarding.couldNotSave)
    }

    setSaving(false)
  }

  return (
    <div className="grid gap-4">
      {canGoBack ? (
        <button
          className="-mt-1 flex items-center gap-1 self-start text-xs font-medium text-muted-foreground hover:text-foreground"
          onClick={onBack}
          type="button"
        >
          <ChevronLeft className="size-3" />
          {t.onboarding.backToSignIn}
        </button>
      ) : null}

      <div className="grid gap-2 sm:grid-cols-2">
        {options.map(o => (
          <button
            className={cn(
              'rounded-2xl border bg-background/60 p-3 text-left transition hover:bg-accent/50',
              option.id === o.id ? 'border-primary ring-2 ring-primary/20' : 'border-border'
            )}
            key={o.id}
            onClick={() => pick(o)}
            type="button"
          >
            <div className="flex items-center justify-between gap-2">
              <span className="text-sm font-medium">{o.name}</span>
              {option.id === o.id ? (
                <Check className="size-4 text-primary" />
              ) : isSet?.(o.envKey) ? (
                <Check className="size-3.5 text-muted-foreground" />
              ) : null}
            </div>
            {(t.onboarding.apiKeyOptions[o.id]?.short ?? o.short) ? (
              <p className="mt-1 text-xs text-muted-foreground">
                {t.onboarding.apiKeyOptions[o.id]?.short ?? o.short}
              </p>
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
            currentRedacted ?? (alreadySet ? t.onboarding.replaceCurrent : option.placeholder || t.onboarding.pasteApiKey)
          }
          type={isLocal ? 'text' : 'password'}
          value={value}
        />
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
          {saving ? <Loader2 className="size-4 animate-spin" /> : <KeyRound className="size-4" />}
          {saving ? t.onboarding.connecting : alreadySet ? t.onboarding.update : t.common.connect}
        </Button>
      </div>
    </div>
  )
}

function FlowPanel({ ctx, flow }: { ctx: OnboardingContext; flow: OnboardingFlow }) {
  const { t } = useI18n()
  const title = 'provider' in flow && flow.provider ? providerTitle(flow.provider) : ''

  if (flow.status === 'starting') {
    return <Status>{t.onboarding.startingSignIn(title)}</Status>
  }

  if (flow.status === 'submitting') {
    return <Status>{t.onboarding.verifyingCode(title)}</Status>
  }

  if (flow.status === 'success') {
    return (
      <div className="flex items-center gap-2 rounded-2xl border border-primary/30 bg-primary/10 px-4 py-3 text-sm text-primary">
        <Check className="size-4" />
        {t.onboarding.connectedPicking(title)}
      </div>
    )
  }

  if (flow.status === 'confirming_model') {
    return <ConfirmingModelPanel ctx={ctx} flow={flow} />
  }

  if (flow.status === 'error') {
    return (
      <div className="grid gap-3">
        <div className="rounded-2xl border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
          {flow.message || t.onboarding.signInFailed}
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
        <p className="text-sm text-muted-foreground">
          {t.onboarding.autoBrowser(title)}
        </p>
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
        <p className="text-sm text-muted-foreground">
          {t.onboarding.externalPending(title)}
        </p>
        <CodeBlock copied={flow.copied} onCopy={() => void copyExternalCommand()} text={flow.provider.cli_command} />
        <FlowFooter
          left={flow.provider.docs_url ? <DocsLink href={flow.provider.docs_url}>{t.onboarding.docs(title)}</DocsLink> : null}
        >
          <CancelBtn />
          <Button onClick={() => void recheckExternalSignin(ctx)}>
            <Check className="size-4" />
            {t.onboarding.signedIn}
          </Button>
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
      <CodeBlock copied={flow.copied} large onCopy={() => void copyDeviceCode()} text={flow.start.user_code} />
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

function CodeBlock({
  copied,
  large,
  onCopy,
  text
}: {
  copied: boolean
  large?: boolean
  onCopy: () => void
  text: string
}) {
  const { t } = useI18n()

  return (
    <div className="flex items-center justify-between gap-3 rounded-2xl border border-border bg-secondary/30 px-4 py-3">
      <code className={cn('font-mono', large ? 'text-2xl tracking-[0.4em]' : 'text-sm')}>{text}</code>
      <Button onClick={onCopy} size="sm" variant="outline">
        {copied ? <Check className="size-4" /> : t.onboarding.copy}
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

function ConfirmingModelPanel({
  ctx,
  flow
}: {
  ctx: OnboardingContext
  flow: Extract<OnboardingFlow, { status: 'confirming_model' }>
}) {
  const { t } = useI18n()
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
    <div className="grid gap-4">
      <div className="flex items-center gap-2 rounded-2xl border border-primary/30 bg-primary/10 px-4 py-3 text-sm text-primary">
        <Check className="size-4 shrink-0" />
        <span>{t.onboarding.connectedProvider(flow.label)}</span>
      </div>

      <div className="grid gap-3 rounded-2xl border border-border bg-background/60 p-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <p className="text-xs uppercase tracking-wide text-muted-foreground">{t.onboarding.defaultModel}</p>
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
            <p className="mt-1 truncate font-mono text-sm">{flow.currentModel}</p>
            {price && (price.input || price.output) && (
              <p className="mt-1 font-mono text-xs text-muted-foreground">
                {price.free ? t.onboarding.free : t.onboarding.price(price.input || '?', price.output || '?')}
              </p>
            )}
          </div>
          <Button disabled={flow.saving} onClick={() => setPickerOpen(true)} size="sm" variant="outline">
            {t.onboarding.change}
          </Button>
        </div>
      </div>

      <div className="flex justify-end">
        <Button disabled={flow.saving} onClick={() => confirmOnboardingModel(ctx)}>
          {flow.saving ? <Loader2 className="size-4 animate-spin" /> : <Sparkles className="size-4" />}
          {t.onboarding.startChatting}
        </Button>
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
    <Button asChild size="xs" variant="ghost">
      <a href={href} rel="noreferrer" target="_blank">
        <ExternalLink className="size-3" />
        {children}
      </a>
    </Button>
  )
}

function Status({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex items-center gap-3 rounded-2xl bg-muted/30 px-4 py-6 text-sm text-muted-foreground">
      <Loader2 className="size-4 animate-spin" />
      {children}
    </div>
  )
}
