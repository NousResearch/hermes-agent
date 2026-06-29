import { useStore } from '@nanostores/react'
import { useMemo } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { $cronJobs } from '@/store/cron'
import { $profiles } from '@/store/profile'
import {
  $currentCwd,
  $currentModel,
  $currentProvider,
  $gatewayState,
  $messagingSessions,
  $sessions,
  $workingSessionIds
} from '@/store/session'

import { PAGE_INSET_X } from '../layout-constants'
import { AGENTS_ROUTE, CRON_ROUTE } from '../routes'

interface MissionMetric {
  label: string
  value: string
  detail: string
}

interface MissionLink {
  href: string
  icon: string
  label: string
  meta: string
}

function gatewayLabel(state: string): string {
  if (!state || state === 'idle') {
    return 'idle'
  }

  return state.replace(/_/g, ' ')
}

function modelLabel(provider: string, model: string): string {
  if (!provider && !model) {
    return 'No model selected'
  }

  if (!provider) {
    return model
  }

  if (!model) {
    return provider
  }

  return `${provider} · ${model}`
}

export function MissionControlView() {
  const sessions = useStore($sessions)
  const messagingSessions = useStore($messagingSessions)
  const workingSessionIds = useStore($workingSessionIds)
  const cronJobs = useStore($cronJobs)
  const profiles = useStore($profiles)
  const gatewayState = useStore($gatewayState)
  const currentProvider = useStore($currentProvider)
  const currentModel = useStore($currentModel)
  const currentCwd = useStore($currentCwd)

  const metrics = useMemo<MissionMetric[]>(
    () => [
      {
        detail: 'Loaded conversations available in the current Desktop session list.',
        label: 'Sessions',
        value: String(sessions.length)
      },
      {
        detail: 'Sessions currently reporting active work in the Desktop runtime.',
        label: 'Active runs',
        value: String(workingSessionIds.length)
      },
      {
        detail: 'Scheduled jobs reported by the Hermes backend scheduler.',
        label: 'Automations',
        value: String(cronJobs.length)
      },
      {
        detail: 'Platform conversations available from connected messaging adapters.',
        label: 'Messaging',
        value: String(messagingSessions.length)
      },
      {
        detail: 'Profiles available to this Desktop session.',
        label: 'Profiles',
        value: String(profiles.length)
      }
    ],
    [cronJobs.length, messagingSessions.length, profiles.length, sessions.length, workingSessionIds.length]
  )

  const links: MissionLink[] = [
    {
      href: AGENTS_ROUTE,
      icon: 'server-process',
      label: 'Spawn tree',
      meta: 'Inspect subagents attached to the active chat.'
    },
    {
      href: CRON_ROUTE,
      icon: 'clock',
      label: 'Automations',
      meta: 'Manage scheduled jobs and recent cron runs.'
    }
  ]

  return (
    <section className="flex h-full min-h-0 flex-col overflow-y-auto bg-background pt-(--titlebar-height)">
      <div className={`${PAGE_INSET_X} mx-auto flex w-full max-w-6xl flex-1 flex-col gap-6 pb-10 pt-8`}>
        <header className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
          <div className="space-y-2">
            <p className="text-xs font-medium uppercase tracking-[0.22em] text-muted-foreground/70">Mission Control</p>
            <h1 className="text-2xl font-semibold tracking-tight text-foreground">Hermes activity at a glance</h1>
            <p className="max-w-2xl text-sm leading-relaxed text-muted-foreground">
              A read-only overview of the live Desktop runtime: conversations, active work, scheduled jobs, messaging,
              profiles, and the current gateway context.
            </p>
          </div>
          <div className="rounded-xl border border-border/70 bg-card/70 px-4 py-3 text-sm shadow-sm">
            <div className="flex items-center gap-2 text-foreground">
              <span className="size-2 rounded-full bg-primary" />
              <span className="font-medium capitalize">{gatewayLabel(gatewayState)}</span>
            </div>
            <p className="mt-1 max-w-72 truncate text-xs text-muted-foreground" title={modelLabel(currentProvider, currentModel)}>
              {modelLabel(currentProvider, currentModel)}
            </p>
          </div>
        </header>

        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
          {metrics.map(metric => (
            <article className="rounded-xl border border-border/70 bg-card/70 p-4 shadow-sm" key={metric.label}>
              <p className="text-xs font-medium uppercase tracking-[0.14em] text-muted-foreground/70">{metric.label}</p>
              <p className="mt-3 text-3xl font-semibold tracking-tight text-foreground">{metric.value}</p>
              <p className="mt-2 text-xs leading-relaxed text-muted-foreground/80">{metric.detail}</p>
            </article>
          ))}
        </div>

        <div className="grid gap-4 lg:grid-cols-[1.2fr_0.8fr]">
          <section className="rounded-2xl border border-border/70 bg-card/70 p-5 shadow-sm">
            <div className="flex items-start justify-between gap-4">
              <div>
                <h2 className="text-sm font-semibold text-foreground">Operational surfaces</h2>
                <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
                  Mission Control starts as an observability layer. Future iterations can add safe actions for task boards,
                  worker status, run history, and profile-scoped operations without duplicating existing Hermes views.
                </p>
              </div>
              <Codicon className="mt-0.5 text-muted-foreground" name="type-hierarchy-sub" size="1.1rem" />
            </div>

            <div className="mt-5 grid gap-3 md:grid-cols-3">
              {['Observe runtime state', 'Inspect worker activity', 'Coordinate safe actions'].map((item, index) => (
                <div className="rounded-xl border border-border/60 bg-background/50 p-3" key={item}>
                  <p className="text-[0.7rem] font-medium uppercase tracking-[0.16em] text-muted-foreground/70">
                    Step {index + 1}
                  </p>
                  <p className="mt-2 text-sm font-medium text-foreground">{item}</p>
                </div>
              ))}
            </div>
          </section>

          <aside className="rounded-2xl border border-border/70 bg-card/70 p-5 shadow-sm">
            <h2 className="text-sm font-semibold text-foreground">Related control surfaces</h2>
            <div className="mt-4 grid gap-2">
              {links.map(link => (
                <Button asChild className="h-auto justify-start gap-3 px-3 py-3 text-left" key={link.href} variant="outline">
                  <a href={link.href}>
                    <Codicon className="shrink-0 text-muted-foreground" name={link.icon} size="1rem" />
                    <span className="min-w-0">
                      <span className="block text-sm font-medium text-foreground">{link.label}</span>
                      <span className="block truncate text-xs text-muted-foreground">{link.meta}</span>
                    </span>
                  </a>
                </Button>
              ))}
            </div>
            <div className="mt-4 rounded-xl border border-dashed border-border/80 bg-background/40 p-3">
              <p className="text-xs font-medium text-muted-foreground">Workspace</p>
              <p className="mt-1 truncate text-sm text-foreground" title={currentCwd || 'No workspace selected'}>
                {currentCwd || 'No workspace selected'}
              </p>
            </div>
          </aside>
        </div>
      </div>
    </section>
  )
}
