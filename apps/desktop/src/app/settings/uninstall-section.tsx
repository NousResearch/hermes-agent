import { type ReactElement, useContext, useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import type { DesktopUninstallMode, DesktopUninstallResult, DesktopUninstallSummary } from '@/global'
import { useI18n } from '@/i18n'
import { AlertTriangle, Loader2, Trash2 } from '@/lib/icons'
import { cn } from '@/lib/utils'

import { SectionHeading, SettingsBreadcrumbContext } from './primitives'

interface ModeOption {
  mode: DesktopUninstallMode
  title: string
  description: string
  /** Shown in the confirm step so people know exactly what disappears. */
  consequence: string
  /** True when the option removes the Python agent (hidden if no agent). */
  needsAgent: boolean
}

export function UninstallSection(): ReactElement | null {
  const hasBreadcrumb = useContext(SettingsBreadcrumbContext)
  const { t } = useI18n()
  const u = t.settings.uninstallSection

  const options: ModeOption[] = [
    {
      mode: 'gui',
      title: u.options.gui.title,
      description: u.options.gui.description,
      consequence: u.options.gui.consequence,
      needsAgent: false
    },
    {
      mode: 'lite',
      title: u.options.lite.title,
      description: u.options.lite.description,
      consequence: u.options.lite.consequence,
      needsAgent: true
    },
    {
      mode: 'full',
      title: u.options.full.title,
      description: u.options.full.description,
      consequence: u.options.full.consequence,
      // full removes the agent (and user data), so it's an agent-removing option:
      // hide it on a lite client with no local agent, same as lite. A lite client
      // connecting to a remote backend has no local agent OR local user data the
      // GUI installed, so gui-only is the correct (and only) option there.
      needsAgent: true
    }
  ]

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
  const visibleOptions: ModeOption[] = options.filter((opt: ModeOption): boolean => agentInstalled || !opt.needsAgent)

  const handleConfirm: () => Promise<void> = async (): Promise<void> => {
    if (!pending) {
      return
    }

    setRunning(true)
    setError(null)

    try {
      const result: DesktopUninstallResult = await bridge.run(pending)

      if (!result.ok) {
        setError(result.message || result.error || u.couldNotStart)
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

  const pendingOption: ModeOption | null = options.find((opt: ModeOption): boolean => opt.mode === pending) ?? null

  return (
    <div className={cn('mx-auto w-full max-w-2xl', !hasBreadcrumb && 'mt-8')}>
      <SectionHeading icon={AlertTriangle} page title={t.settings.uninstallSection.dangerZone} />

      <div className="rounded-xl border border-destructive/30 bg-destructive/5 px-4 py-3">
        {pendingOption ? (
          <div>
            <p className="text-sm font-medium text-destructive">{t.settings.uninstallSection.confirmUninstall}</p>
            <p className="mt-1 text-xs text-muted-foreground">{u.confirmBody(pendingOption.consequence)}</p>
            {summary?.running_app_path && (
              <p className="mt-1 font-mono text-[0.68rem] text-muted-foreground/60">
                {u.appLabel} {summary.running_app_path}
              </p>
            )}
            {error && <p className="mt-2 text-xs text-destructive">{error}</p>}
            <div className="mt-3 flex flex-wrap items-center gap-3">
              <Button disabled={running} onClick={(): void => void handleConfirm()} size="sm" variant="destructive">
                {running && <Loader2 className="size-3 animate-spin" />}
                {running ? u.uninstalling : u.yesUninstall}
              </Button>
              <Button disabled={running} onClick={(): void => setPending(null)} size="sm" variant="text">
                {t.common.cancel}
              </Button>
            </div>
          </div>
        ) : (
          <div className="flex flex-col gap-2">
            <p className="text-sm font-medium">{t.settings.uninstallSection.uninstallHermes}</p>
            <p className="text-xs text-muted-foreground">{u.chooseHowMuch}</p>
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
