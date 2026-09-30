/**
 * Setup step 1: what Jarvis is before anything is configured.
 *
 * The rest of the wizard asks for decisions (engine, voice, access); this step
 * asks for none. It gives the words the later steps rely on — a brain that
 * plans, hands that act, memory, voice, and approvals that keep a risky step
 * behind a "yes" — and shows what the person can do once setup is done.
 */

import logoUrl from '@/assets/czesiek-logo.png'
import type { Translations } from '@/i18n'
import { Bookmark, Brain, Check, ChevronRight, Lock, Mic, ShieldLock, Wrench } from '@/lib/icons'

type WelcomeCopy = Translations['jarvisOnboarding']['welcome']
type Pillar = keyof WelcomeCopy['pillars']

const PILLARS: readonly { icon: React.ComponentType<{ className?: string }>; id: Exclude<Pillar, 'approvals'> }[] = [
  { icon: Brain, id: 'brain' },
  { icon: Wrench, id: 'hands' },
  { icon: Bookmark, id: 'memory' },
  { icon: Mic, id: 'voice' }
]

export function WelcomeStep({ copy }: { copy: WelcomeCopy }) {
  return (
    <div className="grid gap-5" data-testid="jarvis-onboarding-welcome">
      <div className="flex items-center gap-4">
        <img
          alt=""
          className="jarvis-float size-16 shrink-0 object-contain drop-shadow-[0_0_14px_rgb(124_92_255/0.5)]"
          src={logoUrl}
        />
        <div className="min-w-0">
          <p className="text-lg font-semibold">{copy.title}</p>
          <p className="mt-1 max-w-2xl text-sm leading-6 text-(--ui-text-secondary)">{copy.body}</p>
        </div>
      </div>

      <section aria-label={copy.pillarsLabel} className="grid gap-2 sm:grid-cols-2">
        {PILLARS.map(({ icon: Icon, id }) => (
          <div className="jarvis-well jarvis-rise flex gap-3 p-3" key={id}>
            <span className="jarvis-icon-chip grid size-9 shrink-0 place-items-center rounded-xl">
              <Icon className="size-4" />
            </span>
            <span className="min-w-0">
              <span className="block text-sm font-semibold">{copy.pillars[id].title}</span>
              <span className="block text-sm leading-5 text-(--ui-text-secondary)">{copy.pillars[id].body}</span>
            </span>
          </div>
        ))}
        <div className="jarvis-rise flex gap-3 rounded-xl border border-emerald-400/30 bg-emerald-400/8 p-3 sm:col-span-2">
          <span className="grid size-9 shrink-0 place-items-center rounded-xl bg-emerald-400/15 text-emerald-400">
            <ShieldLock className="size-4" />
          </span>
          <span className="min-w-0">
            <span className="block text-sm font-semibold">{copy.pillars.approvals.title}</span>
            <span className="block text-sm leading-5 text-(--ui-text-secondary)">{copy.pillars.approvals.body}</span>
          </span>
        </div>
      </section>

      <section aria-label={copy.flowLabel}>
        <p className="mb-2 text-xs font-semibold uppercase tracking-[0.12em] text-(--ui-text-tertiary)">{copy.flowLabel}</p>
        <ol className="flex flex-wrap items-center gap-2">
          {copy.flow.map((stage, index) => (
            <li className="flex items-center gap-2" key={stage}>
              <span className="jarvis-well flex items-center gap-2 rounded-full px-3 py-1.5 text-sm">
                <span className="jarvis-segment-on grid size-5 place-items-center rounded-full text-xs">
                  {index + 1}
                </span>
                {stage}
              </span>
              {index < copy.flow.length - 1 ? (
                <ChevronRight aria-hidden="true" className="size-4 text-(--ui-text-tertiary)" />
              ) : null}
            </li>
          ))}
        </ol>
      </section>

      <section aria-label={copy.examplesLabel}>
        <p className="mb-2 text-xs font-semibold uppercase tracking-[0.12em] text-(--ui-text-tertiary)">{copy.examplesLabel}</p>
        <ul className="grid gap-2 sm:grid-cols-2">
          {copy.examples.map(example => (
            <li className="flex items-start gap-2 text-sm leading-5 text-(--ui-text-primary)" key={example}>
              <Check className="mt-0.5 size-4 shrink-0 text-emerald-300" />
              {example}
            </li>
          ))}
        </ul>
      </section>

      <p className="flex items-start gap-2 rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-bg-tertiary) p-3 text-sm leading-5 text-(--ui-text-tertiary)">
        <Lock className="mt-0.5 size-4 shrink-0" />
        {copy.privacy}
      </p>
    </div>
  )
}
