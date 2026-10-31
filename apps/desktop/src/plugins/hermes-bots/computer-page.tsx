import { Button, Codicon } from '@hermes/plugin-sdk'

import { useRoster } from './data'
import { displayName } from './labels'
import { openBotScreen } from './screen-open'
import type { RosterRow } from './types'

function botIsReachable(bot: RosterRow): boolean {
  return !bot.ghost && bot.sourceReachable !== false && !bot.sourceMissing
}

/** First-class Computer page that reuses the existing Bot Desktop screen implementation. */
export function ComputerPage() {
  const { data: snapshot, isLoading, error } = useRoster()
  const roster = Array.isArray(snapshot?.profiles) ? snapshot.profiles : []
  const bots = [...roster].sort((a, b) => displayName(a, null).localeCompare(displayName(b, null)))

  return (
    <main className="flex h-full min-h-0 flex-col overflow-auto bg-background text-foreground">
      <header className="border-b border-border px-6 py-5">
        <div className="flex items-center gap-3">
          <div className="flex size-11 items-center justify-center rounded-xl border border-border bg-muted">
            <Codicon name="device-desktop" size="1.35rem" />
          </div>
          <div className="min-w-0">
            <h1 className="text-xl font-semibold">Computer</h1>
            <p className="mt-1 text-sm text-muted-foreground">
              Open an agent's visual desktop. It uses the existing gateway profile and computer-use backend.
            </p>
          </div>
        </div>
      </header>

      <section className="mx-auto flex w-full max-w-4xl flex-col gap-3 px-6 py-5">
        <div className="rounded-xl border border-border bg-card p-4">
          <div className="flex items-start gap-3">
            <Codicon name="info" size="1rem" />
            <div className="text-sm text-muted-foreground">
              The screen is provided by the configured Hermes gateway. Terminal commands and browser actions
              still run through that profile; this page does not create a second task runtime or an isolated
              filesystem. On unsupported hosts, the screen controls report that limitation.
            </div>
          </div>
        </div>

        {isLoading && (
          <div className="rounded-xl border border-border p-5 text-sm text-muted-foreground">
            Loading available profiles…
          </div>
        )}
        {error && (
          <div role="alert" className="rounded-xl border border-destructive/40 p-4 text-sm">
            Could not load profiles: {error instanceof Error ? error.message : String(error)}
          </div>
        )}
        {!isLoading && bots.length === 0 && (
          <div className="rounded-xl border border-border p-6 text-center">
            <Codicon name="device-desktop" size="1.5rem" />
            <h2 className="mt-3 font-medium">No agent profiles found</h2>
            <p className="mt-1 text-sm text-muted-foreground">
              Connect Hermes to a gateway or create an agent profile, then return here.
            </p>
          </div>
        )}

        {bots.map(bot => {
          const reachable = botIsReachable(bot)
          const label = displayName(bot, null)

          return (
            <article key={`${bot.connectionId ?? connectionId ?? 'local'}:${bot.name}`} className="flex flex-col gap-3 rounded-xl border border-border bg-card p-4 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex min-w-0 items-center gap-3">
                <div className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-muted">
                  <Codicon name="device-desktop" size="1.1rem" />
                </div>
                <div className="min-w-0">
                  <h2 className="truncate font-medium">{label}</h2>
                  <p className="mt-1 text-xs text-muted-foreground">
                    {bot.connectionLabel || bot.connectionId || 'Gateway profile'}
                    {bot.ghost || bot.sourceMissing || bot.sourceReachable === false ? ' · Offline' : ' · Profile desktop'}
                  </p>
                  {bot.description && <p className="mt-1 line-clamp-2 text-sm text-muted-foreground">{bot.description}</p>}
                </div>
              </div>
              <Button
                disabled={!reachable}
                onClick={() => openBotScreen(bot)}
                className="shrink-0"
              >
                <Codicon name="device-desktop" size="0.95rem" />
                Open computer
              </Button>
            </article>
          )
        })}
      </section>
    </main>
  )
}
