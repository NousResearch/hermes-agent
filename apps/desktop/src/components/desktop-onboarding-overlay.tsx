import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { useEffect, useMemo, useRef, useState } from 'react'

import { ModelPickerDialog } from '@/components/model-picker'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { getGlobalModelOptions } from '@/hermes'
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
  closeManualOnboarding,
  confirmOnboardingModel,
  copyDeviceCode,
  copyExternalCommand,
  type OnboardingContext,
  type OnboardingFlow,
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

interface ApiKeyOption {
  description: string
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
    short: 'one key, many models',
    envKey: 'OPENROUTER_API_KEY',
    description: 'Hosts hundreds of models behind a single key. Good default for new installs.',
    docsUrl: 'https://openrouter.ai/keys'
  },
  {
    id: 'openai',
    name: 'OpenAI',
    short: 'GPT-class models',
    envKey: 'OPENAI_API_KEY',
    description: 'Direct access to OpenAI models.',
    docsUrl: 'https://platform.openai.com/api-keys'
  },
  {
    id: 'gemini',
    name: 'Google Gemini',
    short: 'Gemini models',
    envKey: 'GEMINI_API_KEY',
    description: 'Direct access to Google Gemini models.',
    docsUrl: 'https://aistudio.google.com/app/apikey'
  },
  {
    id: 'xai',
    name: 'xAI Grok',
    short: 'Grok models',
    envKey: 'XAI_API_KEY',
    description: 'Direct access to xAI Grok models.',
    docsUrl: 'https://console.x.ai/'
  },
  {
    id: 'local',
    name: 'Local / custom endpoint',
    short: 'self-hosted',
    envKey: 'OPENAI_BASE_URL',
    description: 'Point Hermes at a local or self-hosted OpenAI-compatible endpoint (vLLM, llama.cpp, Ollama, etc).',
    docsUrl: 'https://github.com/NousResearch/hermes-agent#bring-your-own-endpoint',
    placeholder: 'http://127.0.0.1:8000/v1'
  }
]

const PROVIDER_DISPLAY: Record<string, { order: number; title: string }> = {
  nous: { order: 0, title: 'Nous Portal' },
  anthropic: { order: 1, title: 'Anthropic Claude' },
  'openai-codex': { order: 2, title: 'OpenAI Codex / ChatGPT' },
  'minimax-oauth': { order: 3, title: 'MiniMax' },
  'xai-oauth': { order: 4, title: 'xAI Grok' },
  'claude-code': { order: 5, title: 'Claude Code' },
  'qwen-oauth': { order: 6, title: 'Qwen Code' }
}

const assetPath = (path: string) => `${import.meta.env.BASE_URL}${path.replace(/^\/+/, '')}`

const FLOW_SUBTITLES: Record<OAuthProvider['flow'], string> = {
  pkce: 'Opens your browser to sign in, then continues here',
  device_code: 'Opens a verification page in your browser — Hermes connects automatically',
  loopback: 'Opens your browser to sign in — Hermes connects automatically',
  external: 'Sign in once in your terminal, then come back to chat'
}

const providerTitle = (p: OAuthProvider) => PROVIDER_DISPLAY[p.id]?.title ?? p.name
const orderOf = (p: OAuthProvider) => PROVIDER_DISPLAY[p.id]?.order ?? 99

const sortProviders = (providers: OAuthProvider[]) =>
  [...providers].sort((a, b) => orderOf(a) - orderOf(b) || a.name.localeCompare(b.name))

export function DesktopOnboardingOverlay({ enabled, onCompleted, requestGateway }: DesktopOnboardingOverlayProps) {
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

  // Mount from frame 1 so we replace the boot overlay seamlessly. The
  // configured field stays null until the runtime check resolves; only then
  // do we know whether to dismiss (true) or surface the picker (false).
  // EXCEPTION: manual mode (user opened the selector from a working app to
  // add/switch a provider) shows the overlay regardless of configured state.
  if (onboarding.configured === true && !onboarding.manual) {
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
      <div className="w-full max-w-[45rem] overflow-hidden rounded-xl border border-(--ui-stroke-secondary) bg-(--ui-chat-bubble-background) shadow-sm">
        <Header />
        <div className="grid gap-3 p-5">
          {onboarding.manual ? (
            <div className="flex justify-end">
              <button
                className="text-xs font-medium text-muted-foreground transition hover:text-foreground"
                onClick={() => closeManualOnboarding()}
                type="button"
              >
                Close
              </button>
            </div>
          ) : null}
          {reason ? <ReasonNotice reason={reason} /> : null}
          {ready ? showPicker ? <Picker ctx={ctx} /> : <FlowPanel ctx={ctx} flow={flow} /> : <Preparing boot={boot} />}
        </div>
      </div>
    </div>
  )
}

function ReasonNotice({ reason }: { reason: string }) {
  return (
    <div className="rounded-2xl border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
      {reason}
    </div>
  )
}

function Preparing({ boot }: { boot: DesktopBootState }) {
  const progress = Math.max(2, Math.min(100, Math.round(boot.progress)))
  const hasError = Boolean(boot.error)
  const installing = boot.phase.startsWith('runtime.')

  return (
    <div className="grid gap-3" role="status">
      <p className="text-sm text-muted-foreground">
        {installing
          ? 'Hermes is finishing install. This usually takes under a minute on first run.'
          : 'Starting Hermes…'}
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
  return (
    <div className="border-b border-(--ui-stroke-tertiary) bg-(--ui-chat-bubble-background) px-5 py-4">
      <div className="flex items-start gap-3">
        <div className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-(--ui-bg-tertiary) text-(--ui-text-tertiary)">
          <Sparkles className="size-5" />
        </div>
        <div>
          <h2 className="text-[0.9375rem] font-semibold tracking-tight">Let's get you setup with Hermes Agent</h2>
          <p className="mt-1 max-w-xl text-[0.8125rem] leading-5 text-(--ui-text-tertiary)">
            Connect a model provider to start chatting. Most options take one click.
          </p>
        </div>
      </div>
    </div>
  )
}

const FEATURED_ID = 'nous'
const FEATURED_PITCH = 'One subscription, 300+ frontier models — the recommended way to run Hermes'
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
  const { mode, providers } = useStore($desktopOnboarding)
  const [showAll, setShowAll] = useState(readShowAll)
  const ordered = useMemo(() => (providers ? sortProviders(providers) : []), [providers])
  const hasOauth = ordered.length > 0

  if (mode === 'apikey' || !hasOauth) {
    return <ApiKeyForm canGoBack={hasOauth} ctx={ctx} />
  }

  if (providers === null) {
    return <Status>Looking up providers...</Status>
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
          {showAll ? 'Collapse' : 'Other providers'}
          <ChevronDown className={cn('size-3.5 transition', showAll && 'rotate-180')} />
        </button>
      ) : null}
      <div className="flex justify-end pt-1">
        <button
          className="text-xs font-medium text-muted-foreground hover:text-foreground"
          onClick={() => setOnboardingMode('apikey')}
          type="button"
        >
          I have an API key
        </button>
      </div>
    </div>
  )
}

function FeaturedProviderRow({
  onSelect,
  provider
}: {
  onSelect: (provider: OAuthProvider) => void
  provider: OAuthProvider
}) {
  const loggedIn = provider.status?.logged_in

  return (
    <button
      className={cn(
        'group flex w-full items-center justify-between gap-4 rounded-2xl border-2 border-primary/50 bg-primary/5 p-4 text-left transition hover:border-primary hover:bg-primary/10',
        loggedIn && 'border-primary'
      )}
      onClick={() => onSelect(provider)}
      type="button"
    >
      <div className="min-w-0">
        <div className="flex items-center gap-2">
          <img alt="" className="size-5 shrink-0 rounded" src={assetPath('apple-touch-icon.png')} />
          <span className="text-base font-semibold">{providerTitle(provider)}</span>
          {loggedIn ? (
            <ConnectedTag />
          ) : (
            <span className="inline-flex items-center gap-1.5 bg-primary px-2 py-0.5 text-[0.64rem] font-semibold uppercase tracking-[0.16em] text-primary-foreground">
              <span aria-hidden="true" className="dither inline-block size-2 shrink-0" />
              Recommended
            </span>
          )}
        </div>
        <p className="mt-1 text-xs leading-5 text-muted-foreground">{FEATURED_PITCH}</p>
      </div>
      <ChevronRight className="size-5 shrink-0 text-primary transition group-hover:translate-x-0.5" />
    </button>
  )
}

function ConnectedTag() {
  return (
    <span className="inline-flex items-center gap-1 bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary">
      <Check className="size-3" />
      Connected
    </span>
  )
}

function KeyProviderRow({ onClick }: { onClick: () => void }) {
  return (
    <button
      className="group flex w-full items-center justify-between gap-3 rounded-2xl border border-border bg-background/60 p-3 text-left transition hover:border-primary/40 hover:bg-accent/40"
      onClick={onClick}
      type="button"
    >
      <div className="min-w-0">
        <span className="text-sm font-semibold">OpenRouter</span>
        <p className="mt-1 text-xs leading-5 text-muted-foreground">One key, hundreds of models — a solid default</p>
      </div>
      <ChevronRight className="size-4 text-muted-foreground transition group-hover:text-foreground" />
    </button>
  )
}

function ProviderRow({ onSelect, provider }: { onSelect: (provider: OAuthProvider) => void; provider: OAuthProvider }) {
  const loggedIn = provider.status?.logged_in
  const Trail = provider.flow === 'external' ? Terminal : ChevronRight

  return (
    <button
      className={cn(
        'group flex w-full items-center justify-between gap-3 rounded-2xl border border-border bg-background/60 p-3 text-left transition hover:border-primary/40 hover:bg-accent/40',
        loggedIn && 'border-primary/30'
      )}
      onClick={() => onSelect(provider)}
      type="button"
    >
      <div className="min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold">{providerTitle(provider)}</span>
          {loggedIn ? <ConnectedTag /> : null}
        </div>
        <p className="mt-1 text-xs leading-5 text-muted-foreground">{FLOW_SUBTITLES[provider.flow]}</p>
      </div>
      <Trail className="size-4 text-muted-foreground transition group-hover:text-foreground" />
    </button>
  )
}

function ApiKeyForm({ canGoBack, ctx }: { canGoBack: boolean; ctx: OnboardingContext }) {
  const [option, setOption] = useState<ApiKeyOption>(API_KEY_OPTIONS[0])
  const [value, setValue] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<null | string>(null)

  const isLocal = option.envKey === 'OPENAI_BASE_URL'
  // Only require a non-empty value — no length/format validation, so a short
  // or unusual key can't block the user from continuing.
  const canSave = value.trim().length >= 1

  const submit = async () => {
    if (!canSave || saving) {
      return
    }

    setSaving(true)
    setError(null)
    const result = await saveOnboardingApiKey(option.envKey, value, option.name, ctx)

    if (result.ok) {
      setValue('')
    } else {
      setError(result.message ?? 'Could not save credential.')
    }

    setSaving(false)
  }

  return (
    <div className="grid gap-4">
      {canGoBack ? (
        <button
          className="-mt-1 flex items-center gap-1 self-start text-xs font-medium text-muted-foreground hover:text-foreground"
          onClick={() => setOnboardingMode('oauth')}
          type="button"
        >
          <ChevronLeft className="size-3" />
          Back to sign in
        </button>
      ) : null}

      <div className="grid gap-2 sm:grid-cols-2">
        {API_KEY_OPTIONS.map(o => (
          <button
            className={cn(
              'rounded-2xl border bg-background/60 p-3 text-left transition hover:bg-accent/50',
              option.id === o.id ? 'border-primary ring-2 ring-primary/20' : 'border-border'
            )}
            key={o.id}
            onClick={() => {
              setOption(o)
              setValue('')
              setError(null)
            }}
            type="button"
          >
            <div className="flex items-center justify-between gap-2">
              <span className="text-sm font-medium">{o.name}</span>
              {option.id === o.id ? <Check className="size-4 text-primary" /> : null}
            </div>
            {o.short ? <p className="mt-1 text-xs text-muted-foreground">{o.short}</p> : null}
          </button>
        ))}
      </div>

      <div className="grid gap-2">
        <div className="flex items-center justify-between gap-3">
          <p className="text-sm leading-6 text-muted-foreground">{option.description}</p>
          {option.docsUrl ? <DocsLink href={option.docsUrl}>Get a key</DocsLink> : null}
        </div>
        <Input
          autoComplete="off"
          autoFocus
          className="font-mono"
          onChange={e => setValue(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && void submit()}
          placeholder={option.placeholder || 'Paste API key'}
          type={isLocal ? 'text' : 'password'}
          value={value}
        />
        {error ? <p className="text-xs text-destructive">{error}</p> : null}
      </div>

      <div className="flex justify-end">
        <Button disabled={!canSave || saving} onClick={() => void submit()}>
          {saving ? <Loader2 className="size-4 animate-spin" /> : <KeyRound className="size-4" />}
          {saving ? 'Connecting' : 'Connect'}
        </Button>
      </div>
    </div>
  )
}

function FlowPanel({ ctx, flow }: { ctx: OnboardingContext; flow: OnboardingFlow }) {
  const title = 'provider' in flow && flow.provider ? providerTitle(flow.provider) : ''

  if (flow.status === 'starting') {
    return <Status>Starting sign-in for {title}...</Status>
  }

  if (flow.status === 'submitting') {
    return <Status>Verifying your code with {title}...</Status>
  }

  if (flow.status === 'success') {
    return (
      <div className="flex items-center gap-2 rounded-2xl border border-primary/30 bg-primary/10 px-4 py-3 text-sm text-primary">
        <Check className="size-4" />
        {title} connected. Picking a default model...
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
          {flow.message || 'Sign-in failed. Try again.'}
        </div>
        <div className="flex justify-end">
          <Button onClick={cancelOnboardingFlow} variant="outline">
            Pick a different provider
          </Button>
        </div>
      </div>
    )
  }

  if (flow.status === 'awaiting_user') {
    return (
      <Step title={`Sign in with ${title}`}>
        <ol className="list-decimal space-y-1 pl-5 text-sm text-muted-foreground">
          <li>We opened {title} in your browser.</li>
          <li>Authorize Hermes there.</li>
          <li>Copy the authorization code and paste it below.</li>
        </ol>
        <Input
          autoFocus
          onChange={e => setOnboardingCode(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && void submitOnboardingCode(ctx)}
          placeholder="Paste authorization code"
          value={flow.code}
        />
        <FlowFooter left={<DocsLink href={flow.start.auth_url}>Re-open authorization page</DocsLink>}>
          <CancelBtn />
          <Button disabled={!flow.code.trim()} onClick={() => void submitOnboardingCode(ctx)}>
            Continue
          </Button>
        </FlowFooter>
      </Step>
    )
  }

  if (flow.status === 'awaiting_browser') {
    return (
      <Step title={`Sign in with ${title}`}>
        <p className="text-sm text-muted-foreground">
          We opened {title} in your browser. Authorize Hermes there and you'll be connected
          automatically — nothing to copy or paste.
        </p>
        <FlowFooter left={<DocsLink href={flow.start.auth_url}>Re-open sign-in page</DocsLink>}>
          <span className="flex items-center gap-2 text-xs text-muted-foreground">
            <Loader2 className="size-3 animate-spin" />
            Waiting for you to authorize...
          </span>
          <CancelBtn size="sm" />
        </FlowFooter>
      </Step>
    )
  }

  if (flow.status === 'external_pending') {
    return (
      <Step title={`Sign in with ${title}`}>
        <p className="text-sm text-muted-foreground">
          {title} signs in through its own CLI. Run this command in a terminal, then come back and pick "I've signed
          in":
        </p>
        <CodeBlock copied={flow.copied} onCopy={() => void copyExternalCommand()} text={flow.provider.cli_command} />
        <FlowFooter
          left={flow.provider.docs_url ? <DocsLink href={flow.provider.docs_url}>{title} docs</DocsLink> : null}
        >
          <CancelBtn />
          <Button onClick={() => void recheckExternalSignin(ctx)}>
            <Check className="size-4" />
            I've signed in
          </Button>
        </FlowFooter>
      </Step>
    )
  }

  if (flow.status !== 'polling') {
    return null
  }

  return (
    <Step title={`Sign in with ${title}`}>
      <p className="text-sm text-muted-foreground">We opened {title} in your browser. Enter this code there:</p>
      <CodeBlock copied={flow.copied} large onCopy={() => void copyDeviceCode()} text={flow.start.user_code} />
      <FlowFooter left={<DocsLink href={flow.start.verification_url}>Re-open verification page</DocsLink>}>
        <span className="flex items-center gap-2 text-xs text-muted-foreground">
          <Loader2 className="size-3 animate-spin" />
          Waiting for you to authorize...
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
  return (
    <div className="flex items-center justify-between gap-3 rounded-2xl border border-border bg-secondary/30 px-4 py-3">
      <code className={cn('font-mono', large ? 'text-2xl tracking-[0.4em]' : 'text-sm')}>{text}</code>
      <Button onClick={onCopy} size="sm" variant="outline">
        {copied ? <Check className="size-4" /> : 'Copy'}
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
  return (
    <Button onClick={cancelOnboardingFlow} size={size} variant="ghost">
      Cancel
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
        <span>{flow.label} connected.</span>
      </div>

      <div className="grid gap-3 rounded-2xl border border-border bg-background/60 p-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <p className="text-xs uppercase tracking-wide text-muted-foreground">Default model</p>
              {freeTier === true && (
                <span className="rounded-sm bg-emerald-500/15 px-1 py-0.5 text-[0.6rem] font-semibold uppercase tracking-wide text-emerald-600 dark:text-emerald-400">
                  Free tier
                </span>
              )}
              {freeTier === false && (
                <span className="rounded-sm bg-primary/15 px-1 py-0.5 text-[0.6rem] font-semibold uppercase tracking-wide text-primary">
                  Pro
                </span>
              )}
            </div>
            <p className="mt-1 truncate font-mono text-sm">{flow.currentModel}</p>
            {price && (price.input || price.output) && (
              <p className="mt-1 font-mono text-xs text-muted-foreground">
                {price.free ? 'Free' : `${price.input || '?'} in / ${price.output || '?'} out per Mtok`}
              </p>
            )}
          </div>
          <Button disabled={flow.saving} onClick={() => setPickerOpen(true)} size="sm" variant="outline">
            Change
          </Button>
        </div>
      </div>

      <div className="flex justify-end">
        <Button disabled={flow.saving} onClick={() => confirmOnboardingModel(ctx)}>
          {flow.saving ? <Loader2 className="size-4 animate-spin" /> : <Sparkles className="size-4" />}
          Start chatting
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
