import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { useNavigate } from 'react-router'

import { getBrowserStatus } from '@/api/browser'
import { CONNECTION_STATUS_KEY, getConnectionStatus } from '@/api/connections'
import { getVaultGraph, VAULT_RAIL_KEY } from '@/api/vault'
import { useI18n } from '@/i18n'
import { Check, ChevronRight, Sparkles, X } from '@/lib/icons'
import { persistString, storedString } from '@/lib/storage'
import { cn } from '@/lib/utils'
import { $activeGatewayProfile } from '@/store/profile'

import { requestComposerInsert } from '../chat/composer/focus'
import { GoogleConnectDialog } from '../connections/google-connect-dialog'
import { SETTINGS_ROUTE, SKILLS_ROUTE, STARMAP_ROUTE } from '../routes'

import { RailCard } from './rail-cards'
import { buildSetupChecklist, type SetupStepId } from './setup-progress'

const DISMISSED_KEY = 'czesiek:start-card-dismissed:v1'
const BRIEFING_KEY = 'czesiek:start-briefing-tried:v1'

const COPY = {
  en: {
    dismiss: 'Hide this list',
    done: 'Done',
    progress: (done: number, total: number) => `${done} of ${total} ready`,
    steps: {
      briefing: { hint: 'Mail, calendar, tasks and news in one go.', prompt: 'Give me my daily report: mail, calendar, open tasks and the main news.', title: 'Try the daily report' },
      browser: { hint: 'Sign in to Gmail once in its own window.', title: 'Let Czesiek sign in to your browser' },
      firstTask: { hint: 'Start small: it learns how you work.', prompt: 'What can you do for me here? Suggest three things I can ask right now.', title: 'Give it a first task' },
      google: { hint: 'Mail, calendar and Drive, in a few clicks.', title: 'Connect Google' },
      memory: { hint: 'Your notes, linked — Czesiek remembers them.', title: 'Open your memory map' },
      model: { hint: 'Pick the AI model Czesiek thinks with.', title: 'Connect an AI model' }
    },
    title: 'Get started'
  },
  pl: {
    dismiss: 'Ukryj tę listę',
    done: 'Gotowe',
    progress: (done: number, total: number) => `Gotowe ${done} z ${total}`,
    steps: {
      briefing: { hint: 'Mail, kalendarz, zadania i newsy naraz.', prompt: 'Daj mi raport dnia: maile, kalendarz, otwarte zadania i najważniejsze newsy.', title: 'Wypróbuj raport dnia' },
      browser: { hint: 'Zaloguj się raz do Gmaila w jego oknie.', title: 'Pozwól Czeskowi logować się w przeglądarce' },
      firstTask: { hint: 'Zacznij od małego: uczy się, jak pracujesz.', prompt: 'Co możesz dla mnie zrobić? Zaproponuj trzy rzeczy, które mogę Ci zlecić od razu.', title: 'Zlec pierwsze zadanie' },
      google: { hint: 'Mail, kalendarz i Dysk w kilka kliknięć.', title: 'Połącz Google' },
      memory: { hint: 'Twoje notatki, połączone — Czesiek je pamięta.', title: 'Otwórz mapę pamięci' },
      model: { hint: 'Wybierz model AI, którym Czesiek myśli.', title: 'Połącz model AI' }
    },
    title: 'Zacznij tutaj'
  }
} as const

function dismissedNow(): boolean {
  return storedString(DISMISSED_KEY) === '1'
}

/**
 * The first screen after setup: what makes Czesiek useful, how far along the user is, and one click
 * for each next step. Progress comes from real state, never from a flag the user ticks; it hides
 * itself when everything is done, or when the user says so.
 */
export function JarvisStartCard({
  connected,
  hasHistory,
  modelReady
}: {
  connected: boolean
  hasHistory: boolean
  modelReady: boolean
}) {
  const { locale } = useI18n()
  const copy = locale === 'pl' ? COPY.pl : COPY.en
  const navigate = useNavigate()
  const profile = useStore($activeGatewayProfile)
  const [dismissed, setDismissed] = useState(dismissedNow)
  const [briefingTried, setBriefingTried] = useState(() => storedString(BRIEFING_KEY) === '1')
  const [googleOpen, setGoogleOpen] = useState(false)

  const status = useQuery({ enabled: connected && !dismissed, queryFn: () => getConnectionStatus(), queryKey: CONNECTION_STATUS_KEY, staleTime: 30_000 })
  const vault = useQuery({ enabled: connected && !dismissed, queryFn: () => getVaultGraph(), queryKey: [VAULT_RAIL_KEY, profile], staleTime: 30_000 })
  const browser = useQuery({ enabled: connected && !dismissed, queryFn: () => getBrowserStatus(), queryKey: ['browser-modes', null], staleTime: 30_000 })

  if (dismissed || !status.data) {
    return null
  }

  const checklist = buildSetupChecklist({
    briefingTried,
    browserSignedIn: browser.data ? Boolean(browser.data.own.google_signed_in || browser.data.copy.google_signed_in) : undefined,
    googleConnected: status.data.google === 'connected',
    hasHistory,
    modelReady,
    vaultExists: vault.data?.vault.exists
  })

  if (checklist.complete) {
    return null
  }

  const act = (id: SetupStepId) => {
    if (id === 'model') {
      navigate(`${SETTINGS_ROUTE}?tab=providers`)
    } else if (id === 'google') {
      setGoogleOpen(true)
    } else if (id === 'memory') {
      navigate(`${STARMAP_ROUTE}?view=vault`)
    } else if (id === 'browser') {
      navigate(SKILLS_ROUTE)
    } else {
      if (id === 'briefing') {
        persistString(BRIEFING_KEY, '1')
        setBriefingTried(true)
      }

      requestComposerInsert(copy.steps[id].prompt, { mode: 'block', target: 'main' })
    }
  }

  return (
    <RailCard
      action={
        <button
          aria-label={copy.dismiss}
          className="grid size-8 place-items-center rounded-lg text-(--ui-text-tertiary) outline-none hover:text-(--ui-text-primary) focus-visible:outline focus-visible:outline-solid focus-visible:outline-2 focus-visible:outline-(--ui-accent)"
          onClick={() => {
            persistString(DISMISSED_KEY, '1')
            setDismissed(true)
          }}
          title={copy.dismiss}
          type="button"
        >
          <X className="size-4" />
        </button>
      }
      icon={Sparkles}
      testId="start"
      title={copy.title}
    >
      <div className="mb-3">
        <div className="mb-1 flex items-center justify-between text-xs text-(--ui-text-secondary)">
          <span>{copy.progress(checklist.doneCount, checklist.total)}</span>
        </div>
        <div
          aria-valuemax={checklist.total}
          aria-valuemin={0}
          aria-valuenow={checklist.doneCount}
          className="jarvis-progress h-1.5 overflow-hidden rounded-full"
          role="progressbar"
        >
          <span className="jarvis-progress-fill block h-full rounded-full" style={{ width: `${(checklist.doneCount / checklist.total) * 100}%` }} />
        </div>
      </div>
      <ul className="grid grid-cols-1 gap-1.5">
        {checklist.steps.map(step => {
          const text = copy.steps[step.id]
          const next = checklist.next === step.id

          return (
            <li key={step.id}>
              <button
                className={cn('jarvis-choice flex min-h-12 w-full items-center gap-3 rounded-xl px-2.5 py-2 text-left outline-none focus-visible:outline focus-visible:outline-solid focus-visible:outline-2 focus-visible:outline-(--ui-accent)', next ? 'jarvis-choice-on' : 'jarvis-well')}
                data-done={step.done}
                data-step={step.id}
                disabled={step.done}
                onClick={() => act(step.id)}
                type="button"
              >
                <span className={cn('grid size-6 shrink-0 place-items-center rounded-full text-xs', step.done ? 'bg-emerald-500/20 text-emerald-500' : 'bg-(--ui-bg-quaternary) text-(--ui-text-tertiary)')}>
                  {step.done ? <Check className="size-3.5" /> : checklist.steps.indexOf(step) + 1}
                </span>
                <span className="min-w-0 flex-1">
                  <span className={cn('block text-sm font-medium leading-snug', step.done ? 'text-(--ui-text-tertiary) line-through' : 'text-(--ui-text-primary)')}>{text.title}</span>
                  {step.done ? null : <span className="block text-xs leading-snug text-(--ui-text-secondary)">{text.hint}</span>}
                </span>
                {step.done ? null : <ChevronRight className="size-4 shrink-0 text-(--ui-text-tertiary)" />}
              </button>
            </li>
          )
        })}
      </ul>
      <GoogleConnectDialog onChanged={() => void status.refetch()} onClose={() => setGoogleOpen(false)} open={googleOpen} />
    </RailCard>
  )
}
