import { type ReactElement, useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import type { DesktopUninstallMode, DesktopUninstallResult, DesktopUninstallSummary } from '@/global'
import { AlertTriangle, Loader2, Trash2 } from '@/lib/icons'
import { cn } from '@/lib/utils'

import { SectionHeading } from './primitives'

interface ModeOption {
  mode: DesktopUninstallMode
  title: string
  description: string
  /** Shown in the confirm step so people know exactly what disappears. */
  consequence: string
  /** True when the option removes the Python agent (hidden if no agent). */
  needsAgent: boolean
}

const OPTIONS: ModeOption[] = [
  {
    mode: 'gui',
    title: 'Uninstall Chat GUI only',
    description: 'Remove this desktop app. The Hermes agent, your config, and chats all stay.',
    consequence: 'the desktop Chat GUI (this app and its data)',
    needsAgent: false
  },
  {
    mode: 'lite',
    title: 'Uninstall GUI + agent, keep my data',
    description: 'Remove the app and the Hermes agent, but keep config, chats, and secrets for a future reinstall.',
    consequence: 'the Chat GUI and the Hermes agent (config, chats, and secrets are kept)',
    needsAgent: true
  },
  {
    mode: 'full',
    title: 'Uninstall everything',
    description: 'Remove the app, the agent, and all user data — config, chats, scheduled jobs, secrets, logs.',
    consequence: 'EVERYTHING — the Chat GUI, the Hermes agent, and all of your config, chats, secrets, and logs',
    // full removes the agent (and user data), so it's an agent-removing option:
    // hide it on a lite client with no local agent, same as lite. A lite client
    // connecting to a remote backend has no local agent OR local user data the
    // GUI installed, so gui-only is the correct (and only) option there.
    needsAgent: true
  }
]

export function UninstallSection(): ReactElement | null {
  const [summary, setSummary] = useState<DesktopUninstallSummary | null>(null)
  const [pending, setPending] = useState<DesktopUninstallMode | null>(null)
  const [running, setRunning] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)

  useEffect((): (() => void) | undefined => {
    let alive: boolean = true
    const bridge: Window['hermesDesktop']['uninstall'] | undefined = window.hermesDesktop?.uninstall

    if (!bridge) {
      return
    }

    void bridge
      .summary()
      .then((result: DesktopUninstallSummary): void => {
        if (alive) {
          setSummary(result)
        }
      })
      .catch((): void => {
        // A failed ownership probe must not offer a destructive fallback.
        if (alive) {
          setSummary(null)
        }
      })

    return (): void => {
      alive = false
    }
  }, [])

  const bridge: Window['hermesDesktop']['uninstall'] | undefined = window.hermesDesktop?.uninstall

  if (!bridge || summary?.code_removal_allowed !== true) {
    return null
  }

  // An owned GUI can remain after its local agent has been removed.
  const agentInstalled: boolean = summary.agent_installed
  const visibleOptions: ModeOption[] = OPTIONS.filter((opt: ModeOption): boolean => agentInstalled || !opt.needsAgent)

  const handleConfirm = async (): Promise<void> => {
    if (!pending) {
      return
    }

    setRunning(true)
    setError(null)

    try {
      const result: DesktopUninstallResult = await bridge.run(pending)

      if (!result.ok) {
        setError(result.message || result.error || 'Uninstall could not start.')
        setRunning(false)
        setPending(null)
      }
      // On success the app quits shortly; keep the spinner up until it does.
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      setRunning(false)
      setPending(null)
    }
  }

  const pendingOption: ModeOption | null = OPTIONS.find((opt: ModeOption): boolean => opt.mode === pending) ?? null

  return (
    <div className="mx-auto mt-8 w-full max-w-2xl">
      <SectionHeading icon={AlertTriangle} title="Danger zone" />

      <div className="rounded-xl border border-destructive/30 bg-destructive/5 px-4 py-3">
        {pendingOption ? (
          <div>
            <p className="text-sm font-medium text-destructive">Confirm uninstall</p>
            <p className="mt-1 text-xs text-muted-foreground">
              This removes {pendingOption.consequence}. This can&apos;t be undone.
            </p>
            {summary?.running_app_path && (
              <p className="mt-1 font-mono text-[0.68rem] text-muted-foreground/60">App: {summary.running_app_path}</p>
            )}
            {error && <p className="mt-2 text-xs text-destructive">{error}</p>}
            <div className="mt-3 flex flex-wrap items-center gap-3">
              <Button disabled={running} onClick={(): void => void handleConfirm()} size="sm" variant="destructive">
                {running && <Loader2 className="size-3 animate-spin" />}
                {running ? 'Uninstalling…' : 'Yes, uninstall'}
              </Button>
              <Button disabled={running} onClick={(): void => setPending(null)} size="sm" variant="text">
                Cancel
              </Button>
            </div>
          </div>
        ) : (
          <div className="flex flex-col gap-2">
            <p className="text-sm font-medium">Uninstall Hermes</p>
            <p className="text-xs text-muted-foreground">
              Choose how much to remove. The app closes to finish the job; reopen the installer any time to come back.
            </p>
            <div className="mt-1 flex flex-col gap-2">
              {visibleOptions.map((opt: ModeOption): ReactElement => (
                <button
                  className={cn(
                    'flex items-start gap-3 rounded-lg border border-border/60 bg-background/40 px-3 py-2.5 text-left transition',
                    'hover:border-destructive/40 hover:bg-destructive/5'
                  )}
                  key={opt.mode}
                  onClick={(): void => {
                    setError(null)
                    setPending(opt.mode)
                  }}
                  type="button"
                >
                  <Trash2 className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
                  <span className="min-w-0">
                    <span className="block text-sm font-medium text-foreground">{opt.title}</span>
                    <span className="mt-0.5 block text-xs text-muted-foreground">{opt.description}</span>
                  </span>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
