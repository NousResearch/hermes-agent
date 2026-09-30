import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'

import { getTrust, setTrustLevel, TRUST_LEVELS, type TrustLevel, type TrustLogEntry } from '@/api/trust'
import { useI18n } from '@/i18n'
import { ShieldLock } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { notifyError } from '@/store/notifications'

const COPY = {
  en: {
    decisions: { approved: 'You approved', auto: 'Done on its own', blocked: 'Blocked', denied: 'You declined' },
    empty: 'Nothing yet. Every send, delete or change Czesiek makes in Google will be listed here.',
    google: 'Google (mail, calendar, Drive, Docs, Sheets)',
    levels: {
      ask: { hint: 'It shows what it wants to do and waits for your yes. Recommended.', name: 'Act after I agree' },
      auto: { hint: 'It sends, creates and deletes without asking. Only for things you fully trust.', name: 'Act on its own' },
      propose: { hint: 'It reads and tells you exactly what it would do, but cannot do it.', name: 'Read and propose' },
      read: { hint: 'It only reads. Any send, delete or change is blocked.', name: 'Read only' }
    },
    recent: 'What Czesiek did lately',
    scope: 'Applies to writes through Google. Other services are read-only for now.',
    title: 'How far Czesiek may go',
    when: (ts: number) => new Date(ts * 1000).toLocaleString('en', { day: 'numeric', hour: '2-digit', minute: '2-digit', month: 'short' })
  },
  pl: {
    decisions: { approved: 'Zatwierdzone przez Ciebie', auto: 'Zrobione samodzielnie', blocked: 'Zablokowane', denied: 'Odrzucone przez Ciebie' },
    empty: 'Jeszcze nic. Każde wysłanie, usunięcie lub zmianę w Google, którą zrobi Czesiek, zobaczysz tutaj.',
    google: 'Google (mail, kalendarz, Dysk, Dokumenty, Arkusze)',
    levels: {
      ask: { hint: 'Pokazuje, co chce zrobić, i czeka na Twoje „tak”. Zalecane.', name: 'Działa po zgodzie' },
      auto: { hint: 'Wysyła, tworzy i usuwa bez pytania. Tylko dla rzeczy, którym w pełni ufasz.', name: 'Działa sam' },
      propose: { hint: 'Czyta i mówi dokładnie, co by zrobił, ale nie może tego wykonać.', name: 'Czyta i proponuje' },
      read: { hint: 'Tylko czyta. Każde wysłanie, usunięcie i zmiana jest blokowane.', name: 'Tylko czyta' }
    },
    recent: 'Co Czesiek zrobił ostatnio',
    scope: 'Dotyczy zapisów przez Google. Pozostałe usługi są na razie tylko do odczytu.',
    title: 'Jak daleko może pójść Czesiek',
    when: (ts: number) => new Date(ts * 1000).toLocaleString('pl', { day: 'numeric', hour: '2-digit', minute: '2-digit', month: 'short' })
  }
} as const

const TONE: Record<TrustLogEntry['decision'], string> = {
  approved: 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-400',
  auto: 'bg-sky-500/15 text-sky-600 dark:text-sky-400',
  blocked: 'bg-amber-500/15 text-amber-600 dark:text-amber-400',
  denied: 'bg-rose-500/15 text-rose-600 dark:text-rose-400'
}

/** Per-integration trust level (enforced by the approval gate, not advisory) and the log of what happened. */
export function TrustCard() {
  const { locale } = useI18n()
  const copy = locale === 'pl' ? COPY.pl : COPY.en
  const client = useQueryClient()
  const { data } = useQuery({ queryFn: () => getTrust(), queryKey: ['trust'], refetchInterval: 15_000 })
  const [busy, setBusy] = useState(false)

  if (!data) {
    return null
  }

  const level: TrustLevel = data.levels.google ?? 'ask'

  const choose = async (next: TrustLevel) => {
    if (next === level || busy) {
      return
    }

    setBusy(true)

    try {
      client.setQueryData(['trust'], await setTrustLevel('google', next))
    } catch (error) {
      notifyError(error, copy.title)
    } finally {
      setBusy(false)
    }
  }

  return (
    <section className="jarvis-panel jarvis-rise grid gap-3 rounded-3xl p-4" data-testid="trust-card">
      <header className="flex items-start gap-3">
        <span className="jarvis-icon-chip grid size-9 shrink-0 place-items-center rounded-xl">
          <ShieldLock className="size-5" />
        </span>
        <div className="min-w-0">
          <h3 className="text-base font-semibold text-(--ui-text-primary)">{copy.title}</h3>
          <p className="text-sm text-(--ui-text-secondary)">{copy.google}</p>
        </div>
      </header>
      <div aria-label={copy.title} className="jarvis-well grid grid-cols-2 gap-1 p-1 sm:grid-cols-4" role="radiogroup">
        {TRUST_LEVELS.map(item => (
          <button
            aria-checked={level === item}
            className={cn(
              'jarvis-segment min-h-11 rounded-xl px-2 text-xs font-medium leading-tight outline-none focus-visible:outline focus-visible:outline-solid focus-visible:outline-2 focus-visible:outline-(--ui-accent)',
              level === item ? 'jarvis-segment-on' : 'text-(--ui-text-secondary) hover:text-(--ui-text-primary)'
            )}
            disabled={busy}
            key={item}
            onClick={() => void choose(item)}
            role="radio"
            type="button"
          >
            {copy.levels[item].name}
          </button>
        ))}
      </div>
      <p className="text-xs text-(--ui-text-secondary)">{copy.levels[level].hint}</p>
      <p className="text-xs text-(--ui-text-tertiary)">{copy.scope}</p>
      <div>
        <h4 className="mb-1.5 text-xs font-semibold uppercase tracking-[0.12em] text-(--ui-text-tertiary)">{copy.recent}</h4>
        {data.log.length === 0 ? (
          <p className="text-xs text-(--ui-text-tertiary)">{copy.empty}</p>
        ) : (
          <ul className="grid gap-1.5">
            {data.log.map((entry, index) => (
              <li className="jarvis-well flex items-center gap-2 px-3 py-2" key={`${entry.ts}-${index}`}>
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-sm text-(--ui-text-primary)">{locale === 'pl' ? entry.label_pl : entry.label}</span>
                  <span className="block truncate text-xs text-(--ui-text-tertiary)">
                    {entry.preview ? `${entry.preview} · ` : ''}
                    {copy.when(entry.ts)}
                  </span>
                </span>
                <span className={cn('shrink-0 rounded-full px-2 py-0.5 text-xs font-medium', TONE[entry.decision])}>{copy.decisions[entry.decision]}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </section>
  )
}
