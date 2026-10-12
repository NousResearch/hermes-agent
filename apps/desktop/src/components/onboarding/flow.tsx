import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { useCallback, useEffect, useRef, useState } from 'react'

import { ModelPickerDialog } from '@/components/model-picker'
import { Button } from '@/components/ui/button'
import { ErrorIcon } from '@/components/ui/error-state'
import { Input } from '@/components/ui/input'
import { Loader } from '@/components/ui/loader'
import { getGlobalModelOptions, listOAuthProviders, profileScopeKey } from '@/hermes'
import { type Translations, useI18n } from '@/i18n'
import { ExternalLink, Loader2 } from '@/lib/icons'
import { isMacPlatform, isWindowsPlatform } from '@/lib/platform'
import { isLocalScope } from '@/lib/simple-agent-defaults'
import { cn } from '@/lib/utils'
import { $showsAdvancedChrome } from '@/store/interface-mode'
import {
  cancelOnboardingFlow,
  copyDeviceCode,
  copyExternalCommand,
  type OnboardingContext,
  type OnboardingFlow,
  recheckExternalSignin,
  setOnboardingCode,
  setOnboardingModel,
  startManualOnboarding,
  startProviderOAuth,
  submitOnboardingCode
} from '@/store/onboarding'
import type { OAuthProvider } from '@/types/hermes'

import { DecodedLabel, GlyphText, HackeryButton, useScramble } from './glyph'
import { providerTitle, SIMPLE_SIGN_IN_TILES } from './providers'

export function FlowPanel({
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
  const advanced = useStore($showsAdvancedChrome)
  const provider = 'provider' in flow ? flow.provider : null
  const simpleTitle = simpleTileTitle(provider, advanced)
  const title = simpleTitle ?? (provider ? providerTitle(provider) : '')

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
    return <ConfirmingModelPanel ctx={ctx} flow={flow} leaving={leaving} onBegin={onBegin} />
  }

  if (flow.status === 'error') {
    // Recovery in the order a stuck user needs it: retry the same provider
    // (when we know which one failed), fall back to a pasted API key, or go
    // back to the provider list. Raw error text stays behind Details.
    const failedProvider = flow.provider
    const { backLabel, detail, headline } = flowErrorCopy(flow, simpleTitle, t)

    return (
      <div className="grid gap-3">
        <div className="flex items-center gap-1.5 text-sm text-destructive">
          <ErrorIcon className="shrink-0" size="0.875rem" />
          <span>{headline}</span>
        </div>
        {detail ? (
          <details className="text-xs text-muted-foreground">
            <summary className="cursor-pointer select-none">{t.onboarding.errorDetails}</summary>
            <pre className="mt-1 whitespace-pre-wrap wrap-break-word font-mono text-[0.6875rem]">{detail}</pre>
          </details>
        ) : null}
        <div className="flex flex-wrap justify-end gap-2">
          <Button onClick={cancelOnboardingFlow} variant="text">
            {backLabel}
          </Button>
          {/* API keys are an Advanced door. */}
          {advanced ? (
            <Button onClick={() => startManualOnboarding(null, ctx.scope ?? ctx.profile)} variant="outline">
              {t.onboarding.useApiKeyInstead}
            </Button>
          ) : null}
          {failedProvider ? (
            <Button onClick={() => void startProviderOAuth(failedProvider, ctx)}>{t.onboarding.tryAgain}</Button>
          ) : null}
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
          onKeyDown={e => e.key === 'Enter' && !e.nativeEvent.isComposing && void submitOnboardingCode(ctx)}
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

  if (flow.status === 'external_pending') {
    const commandStep = (
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

    // The terminal signs in THIS machine's Hermes; a remote backend keeps the command card.
    return simpleTitle && isLocalScope(ctx.scope ?? {}) ? (
      <TerminalSignInStep ctx={ctx} fallback={commandStep} flow={flow} title={title} />
    ) : (
      commandStep
    )
  }

  if (flow.status !== 'polling') {
    return null
  }

  return <DeviceCodeStep flow={flow} title={title} />
}

// Simple names a sign-in by its tile ("ChatGPT"), never a catalog label like
// "ChatGPT or Codex Subscription"; null in Advanced or for a provider with no tile.
function simpleTileTitle(provider: null | OAuthProvider | undefined, advanced: boolean): null | string {
  return advanced ? null : (SIMPLE_SIGN_IN_TILES.find(tile => tile.providerId === provider?.id)?.title ?? null)
}

// Simple names the tile in plain words; the backend's wording (CLI commands,
// env vars, catalog labels, "use an API key") stays behind Details.
function flowErrorCopy(
  flow: Extract<OnboardingFlow, { status: 'error' }>,
  simpleTitle: null | string,
  t: Translations
): { backLabel: string; detail?: string; headline: string } {
  if (!simpleTitle) {
    return {
      backLabel: t.onboarding.pickDifferentProvider,
      detail: flow.detail,
      headline: flow.message || t.onboarding.signInFailed
    }
  }

  return {
    backLabel: t.onboarding.pickDifferentAccount,
    detail: [flow.message, flow.detail].filter(Boolean).join('\n\n') || undefined,
    headline: t.onboarding.simpleSignInNotFinished(simpleTitle)
  }
}

// One terminal per sign-in attempt (a flow object), even when StrictMode
// replays mount effects in dev.
const terminalLaunches = new WeakSet<OnboardingFlow>()

// The keys that paste the copied code into the terminal and submit it, per OS.
const PASTE_KEY = isMacPlatform() ? '⌘V' : isWindowsPlatform() ? 'Ctrl+V' : 'Ctrl+Shift+V'
const ENTER_KEY = isMacPlatform() ? 'Return' : 'Enter'

// Simple's Claude sign-in: Electron main opens Hermes' own CLI login in the
// user's terminal from an allowlisted argv (the backend's `cli_command` stays
// display-only). No terminal could be opened → the copy-the-command step.
function TerminalSignInStep({
  ctx,
  fallback,
  flow,
  title
}: {
  ctx: OnboardingContext
  fallback: React.ReactNode
  flow: Extract<OnboardingFlow, { status: 'external_pending' }>
  title: string
}) {
  const { t } = useI18n()
  const [failed, setFailed] = useState(false)
  const profile = ctx.scope?.profile ?? ctx.profile
  const providerId = flow.provider.id

  const open = useCallback(() => {
    void Promise.resolve()
      .then(() => window.hermesDesktop?.openAuthInTerminal?.(providerId, { profile: profile ?? undefined }))
      .then(
        result => setFailed(!result?.ok),
        () => setFailed(true)
      )
  }, [profile, providerId])

  useEffect(() => {
    if (!failed && !terminalLaunches.has(flow)) {
      terminalLaunches.add(flow)
      open()
    }
  }, [failed, flow, open])

  useNoticeTerminalSignIn(ctx, providerId)

  if (failed) {
    return fallback
  }

  return (
    <Step title={t.onboarding.signInWith(title)}>
      <p className="text-sm text-muted-foreground">{t.onboarding.terminalSignInOpened(PASTE_KEY, ENTER_KEY)}</p>
      <FlowFooter
        left={
          <Button onClick={open} size="sm" variant="text">
            {t.onboarding.openTerminalAgain}
          </Button>
        }
      >
        <CancelBtn />
        <Button onClick={() => void recheckExternalSignin(ctx)}>{t.onboarding.signedIn}</Button>
      </FlowFooter>
    </Step>
  )
}

const TERMINAL_SIGN_IN_POLL_MS = 3_000

// The user finishes in a terminal we cannot watch, so ask the backend quietly
// (every few seconds and on focus) and move on by ourselves once a NEW
// credential appears: one that already existed (an env key, a stale token from a
// failed attempt) must not cut the sign-in short. A failed check is just a missed
// tick; "I've signed in" still works.
function useNoticeTerminalSignIn(ctx: OnboardingContext, providerId: string) {
  // A re-render hands in a new ctx; that must not restart the poll.
  const ctxRef = useRef(ctx)
  ctxRef.current = ctx

  useEffect(() => {
    let done = false
    let sawSignedOut = false

    const check = () => {
      const current = ctxRef.current

      void Promise.resolve()
        .then(() => listOAuthProviders(current.scope ?? current.profile))
        .then(
          ({ providers }) => {
            const signedIn = Boolean(providers.find(p => p.id === providerId)?.status.logged_in)

            if (!signedIn) {
              sawSignedOut = true
            } else if (sawSignedOut && !done) {
              done = true
              void recheckExternalSignin(ctxRef.current)
            }
          },
          () => undefined
        )
    }

    const timer = window.setInterval(check, TERMINAL_SIGN_IN_POLL_MS)
    window.addEventListener('focus', check)

    return () => {
      done = true
      window.clearInterval(timer)
      window.removeEventListener('focus', check)
    }
  }, [providerId])
}

// Sign-ins whose code Simple already copied, so a remount never writes the clipboard again.
const copiedSignIns = new Set<string>()

function DeviceCodeStep({ flow, title }: { flow: Extract<OnboardingFlow, { status: 'polling' }>; title: string }) {
  const { t } = useI18n()
  const advanced = useStore($showsAdvancedChrome)
  const sessionId = flow.start.session_id

  // Simple copies the code the moment the browser opens, once per sign-in, so
  // pasting is the only step left. Mode is read at start on purpose: a toggle
  // mid-sign-in must not trigger a second clipboard write.
  useEffect(() => {
    if (!$showsAdvancedChrome.get() && !copiedSignIns.has(sessionId)) {
      copiedSignIns.add(sessionId)
      void copyDeviceCode()
    }
  }, [sessionId])

  return (
    <Step title={t.onboarding.signInWith(title)}>
      <p className="text-sm text-muted-foreground">
        {advanced ? t.onboarding.deviceCodeOpened(title) : t.onboarding.simpleDeviceCodeOpened}
      </p>
      {advanced ? null : <p className="text-sm text-muted-foreground">{t.onboarding.simpleDeviceCodeCopied}</p>}
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
export function DeviceCode({ code, copied, onCopy }: { code: string; copied: boolean; onCopy: () => void }) {
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

function ConfirmingModelPanel({
  flow,
  leaving,
  onBegin,
  ctx
}: {
  flow: Extract<OnboardingFlow, { status: 'confirming_model' }>
  leaving: boolean
  onBegin: () => void
  ctx: OnboardingContext
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
    queryKey: ['onboarding-model-options', profileScopeKey(ctx.scope ?? ctx.profile), flow.providerSlug],
    queryFn: () => getGlobalModelOptions({ includeUnconfigured: true, explicitOnly: false }, ctx.scope ?? ctx.profile)
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
        ModelPickerDialog's content sits on the modal rung, which is below the
        onboarding overlay — it would render underneath and swallow pointer
        events. Lift it to the rung above onboarding. Its own dim-backdrop
        layer stays on the modal-backdrop rung: onboarding already dims the
        rest of the screen, so a second backdrop would double up.
      */}
      <ModelPickerDialog
        contentClassName="z-(--z-onboarding-popover)"
        currentModel={flow.currentModel}
        currentProvider={flow.providerSlug}
        onOpenChange={setPickerOpen}
        onSelect={({ model, provider }) => {
          // The picker lists models from every configured provider, not just
          // the one the user just signed in with. Persist the assignment
          // against the provider that actually serves the picked model (and
          // sync the card label to it) — otherwise a foreign model gets
          // paired with the sign-in provider and chat errors out.
          const picked = options.data?.providers?.find(
            p => String(p.slug).toLowerCase() === String(provider).toLowerCase()
          )

          void setOnboardingModel(model, provider, picked?.name)
          setPickerOpen(false)
        }}
        open={pickerOpen}
        ownerConnectionId={ctx.scope ? (ctx.scope.connectionId ?? null) : undefined}
        profile={ctx.profile}
        request={ctx.requestGateway}
      />
    </div>
  )
}

export function DocsLink({ children, href }: { children: React.ReactNode; href: string }) {
  return (
    <Button asChild size="xs" variant="text">
      <a href={href} rel="noreferrer" target="_blank">
        <ExternalLink className="size-3" />
        {children}
      </a>
    </Button>
  )
}

export function Status({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex items-center gap-2.5 py-1 text-sm text-muted-foreground" role="status">
      <Loader className="size-7" type="lemniscate-bloom" />
      {children}
    </div>
  )
}
