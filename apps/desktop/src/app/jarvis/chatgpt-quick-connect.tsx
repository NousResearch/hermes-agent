import { useEffect, useRef, useState } from 'react'

import type { ProfileScope } from '@/api/client'
import { ModelBrandIcon } from '@/components/model-brand-icon'
import { Button } from '@/components/ui/button'
import { cancelOAuthSession, pollOAuthSession, setGlobalModel, startOAuthLogin } from '@/hermes'
import { useI18n } from '@/i18n'
import { openExternalLink } from '@/lib/external-link'
import { Loader2 } from '@/lib/icons'
import { cn } from '@/lib/utils'
import type { ModelOptionsResponse } from '@/types/hermes'

import { CHATGPT_PROVIDER_SLUG, type ChatGptConnectDeps, type ChatGptConnectResult, connectChatGpt } from './chatgpt-connect'

const COPY = {
  en: {
    cancel: 'Cancel',
    code: 'Your code',
    denied: 'Sign-in was declined. You can try again.',
    expired: 'The code expired. Start again.',
    failed: 'Could not sign in.',
    hint: 'Use the GPT you already pay for — no API key, no extra bill.',
    open: 'Open the sign-in page',
    start: 'Sign in with ChatGPT',
    title: 'ChatGPT subscription',
    waiting: 'Waiting for you to confirm in the browser…'
  },
  pl: {
    cancel: 'Anuluj',
    code: 'Twój kod',
    denied: 'Logowanie odrzucone. Możesz spróbować jeszcze raz.',
    expired: 'Kod wygasł. Zacznij od nowa.',
    failed: 'Nie udało się zalogować.',
    hint: 'Użyj GPT, za który już płacisz — bez klucza API i bez dodatkowego rachunku.',
    open: 'Otwórz stronę logowania',
    start: 'Zaloguj się kontem ChatGPT',
    title: 'Subskrypcja ChatGPT',
    waiting: 'Czekam, aż potwierdzisz w przeglądarce…'
  }
} as const

export interface ChatGptQuickConnectProps {
  className?: string
  /** Reads the served catalog after login. */
  loadOptions: (scope?: ProfileScope) => Promise<ModelOptionsResponse>
  onConnected: (result: Extract<ChatGptConnectResult, { ok: true }>) => void
  scope?: ProfileScope
  /** Make the chosen model the default straight away (the rail); onboarding commits it on the last step. */
  setDefault?: boolean
  tone?: 'app' | 'dark'
}

/** "Sign in with ChatGPT": the subscription as Czesiek's brain, through the engine's device-code login. */
export function ChatGptQuickConnect({ className, loadOptions, onConnected, scope, setDefault = false, tone = 'app' }: ChatGptQuickConnectProps) {
  const { locale } = useI18n()
  const copy = locale === 'pl' ? COPY.pl : COPY.en
  const [phase, setPhase] = useState<'idle' | 'starting' | 'waiting'>('idle')
  const [code, setCode] = useState<null | { url: string; userCode: string }>(null)
  const [error, setError] = useState('')
  const controller = useRef<AbortController | null>(null)
  const dark = tone === 'dark'

   
  useEffect(() => () => controller.current?.abort(), [])

  const begin = async () => {
    controller.current?.abort()
    const abort = new AbortController()

    controller.current = abort
    setPhase('starting')
    setError('')
    setCode(null)

    const deps: ChatGptConnectDeps = {
      cancel: (sessionId, s) => cancelOAuthSession(sessionId, s),
      loadOptions,
      openUrl: url => openExternalLink(url),
      poll: (sessionId, s) => pollOAuthSession(CHATGPT_PROVIDER_SLUG, sessionId, s),
      setDefaultModel: setDefault ? (provider, model, s) => setGlobalModel(provider, model, s) : undefined,
      sleep: ms => new Promise(resolve => window.setTimeout(resolve, ms)),
      start: s => startOAuthLogin(CHATGPT_PROVIDER_SLUG, s)
    }

    const result = await connectChatGpt(deps, scope, {
      onCode: next => {
        setCode(next)
        setPhase('waiting')
      },
      signal: abort.signal
    })

    if (abort.signal.aborted && !result.ok) {
      return
    }

    setPhase('idle')
    setCode(null)

    if (result.ok) {
      onConnected(result)
    } else {
      setError(result.reason === 'denied' ? copy.denied : result.reason === 'expired' ? copy.expired : result.message || copy.failed)
    }
  }

  const stop = () => {
    controller.current?.abort()
    setPhase('idle')
    setCode(null)
  }

  return (
    <div className={cn('grid gap-2', className)} data-testid="chatgpt-quick-connect">
      <div className="flex items-center gap-3">
        <ModelBrandIcon hints={['openai']} model="openai/gpt" />
        <div className="min-w-0">
          <p className="text-sm font-semibold">{copy.title}</p>
          <p className={cn('text-xs leading-5', dark ? 'text-[#C7CBD1]' : 'text-(--ui-text-secondary)')}>{copy.hint}</p>
        </div>
      </div>
      {phase === 'idle' ? (
        <Button className="min-h-11" onClick={() => void begin()} type="button">
          {copy.start}
        </Button>
      ) : (
        <div className="grid gap-2">
          {code ? (
            <div className={cn('jarvis-well grid gap-1 px-3 py-2', dark && 'bg-black/30')}>
              <span className="text-xs opacity-70">{copy.code}</span>
              <span className="select-all font-mono text-xl font-semibold tracking-[0.2em]" data-testid="chatgpt-code">{code.userCode}</span>
              <button className="w-fit text-xs text-(--ui-accent) underline-offset-4 hover:underline" onClick={() => openExternalLink(code.url)} type="button">
                {copy.open}
              </button>
            </div>
          ) : null}
          <div className="flex items-center gap-2 text-xs opacity-80">
            <Loader2 className="size-4 animate-spin" />
            {copy.waiting}
            <Button className="ml-auto" onClick={stop} size="sm" type="button" variant="ghost">
              {copy.cancel}
            </Button>
          </div>
        </div>
      )}
      {error ? (
        <p className="text-xs text-red-400" role="alert">
          {error}
        </p>
      ) : null}
    </div>
  )
}
