import type { ReactNode } from 'react'

import { Brain, Check, Clock, Lock, NotebookTabs, Sparkles } from '@/lib/icons'

import { Pill } from './primitives'

const PLANNED_MODES = [
  {
    title: 'Ask first',
    description: 'Hermes asks before saving memories, drafting skills, or proposing automations.'
  },
  {
    title: 'Auto-draft',
    description: 'Hermes may prepare draft memories, skills, and job suggestions for review.'
  },
  {
    title: 'Learn mode',
    description: 'Opt-in metadata analysis for recurring workflow opportunities. Nothing becomes active automatically.'
  },
  {
    title: 'Teach me this workflow',
    description: 'Explicit bounded observation of one workflow to help draft a reusable skill.'
  }
]

const GUARDRAILS = [
  'No background sampler, storage, or automation is included in this first step.',
  'Learned workflows stay inactive until the user explicitly approves them.',
  'Default collection is intended to be metadata-only in a later implementation.',
  'Computer Use-style observation belongs only in explicit Teach Mode sessions.'
]

function InfoCard({
  icon: Icon,
  title,
  children
}: {
  icon: typeof Sparkles
  title: string
  children: ReactNode
}) {
  return (
    <div className="rounded-lg bg-background/55 p-3">
      <div className="mb-1.5 flex items-center gap-2 text-sm font-medium">
        <Icon className="size-4 text-muted-foreground" />
        {title}
      </div>
      <div className="text-[0.72rem] leading-relaxed text-muted-foreground">{children}</div>
    </div>
  )
}

export function LearnPanel() {
  return (
    <div className="mt-3 grid gap-3 rounded-xl bg-background/60 p-3">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <Brain className="size-4 text-muted-foreground" />
            <span className="text-sm font-medium">Learn</span>
            <Pill>Panel skeleton</Pill>
            <Pill>Not running</Pill>
          </div>
          <p className="mt-1 max-w-2xl text-[0.72rem] leading-relaxed text-muted-foreground">
            A proposed Desktop surface for identifying recurring workflows and drafting skills or automations for
            explicit review. This panel is informational only while the product shape is reviewed in #51451.
          </p>
        </div>
      </div>

      <div className="grid gap-2 md:grid-cols-2">
        <InfoCard icon={Sparkles} title="Purpose">
          Wrap existing Hermes primitives — memory, skills, session search, and cron suggestions — behind one review
          surface for high-value, low-risk workflow opportunities.
        </InfoCard>
        <InfoCard icon={Lock} title="Safety invariant">
          Learn does not automate from observation. Learn converts approved signals into reviewable proposals.
        </InfoCard>
      </div>

      <div className="grid gap-2 rounded-lg bg-background/55 p-3">
        <div className="flex items-center gap-2 text-sm font-medium">
          <NotebookTabs className="size-4 text-muted-foreground" />
          Planned modes
        </div>
        <div className="grid gap-2 md:grid-cols-2">
          {PLANNED_MODES.map(mode => (
            <div className="rounded-md bg-muted/20 p-2" key={mode.title}>
              <div className="text-[0.78rem] font-medium">{mode.title}</div>
              <p className="mt-0.5 text-[0.7rem] leading-relaxed text-muted-foreground">{mode.description}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="grid gap-2 rounded-lg bg-background/55 p-3">
        <div className="flex items-center gap-2 text-sm font-medium">
          <Clock className="size-4 text-muted-foreground" />
          Initial guardrails
        </div>
        <div className="grid gap-1.5">
          {GUARDRAILS.map(item => (
            <div className="flex gap-2 text-[0.72rem] leading-relaxed text-muted-foreground" key={item}>
              <Check className="mt-0.5 size-3.5 shrink-0" />
              <span>{item}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
