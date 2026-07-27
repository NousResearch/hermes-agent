import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'

import { ModelPickerDialog } from '@/components/model-picker'
import { Button } from '@/components/ui/button'
import { ErrorIcon } from '@/components/ui/error-state'
import { Input } from '@/components/ui/input'
import { Loader } from '@/components/ui/loader'
import { getGlobalModelOptions } from '@/hermes'
import { useI18n } from '@/i18n'
import { ExternalLink, Loader2 } from '@/lib/icons'
import { cn } from '@/lib/utils'
import {
  cancelOnboardingFlow,
  copyDeviceCode,
  copyExternalCommand,
  type OnboardingContext,
  type OnboardingFlow,
  recheckExternalSignin,
  setOnboardingCode,
  setOnboardingModel,
  submitOnboardingCode
} from '@/store/onboarding'

import { DecodedLabel, GlyphText, HackeryButton, useScramble } from './glyph'
import { providerTitle } from './providers'

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
    return <ConfirmingModelPanel flow={flow} leaving={leaving} onBegin={onBegin} profile={ctx.profile} />
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

function ConfirmingModelPanel({
  flow,
  leaving,
  onBegin,
  profile
}: {
  flow: Extract<OnboardingFlow, { status: 'confirming_model' }>
  leaving: boolean
  onBegin: () => void
  profile?: string
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
    queryFn: () => getGlobalModelOptions({ includeUnconfigured: true, explicitOnly: false })
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
        profile={profile}
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
