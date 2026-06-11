import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { PageLoader } from '@/components/page-loader'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import {
  deleteEnvVar,
  getActionStatus,
  getToolsetConfig,
  revealEnvVar,
  runToolsetPostSetup,
  selectToolsetProvider,
  setEnvVar
} from '@/hermes'
import { useI18n } from '@/i18n'
import { Check, Loader2, Save, Terminal } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { upsertDesktopActionTask } from '@/store/activity'
import { notify, notifyError } from '@/store/notifications'
import type { ActionStatusResponse, ToolEnvVar, ToolProvider, ToolsetConfig } from '@/types/hermes'

import { EnvVarActionsMenu, EnvVarActionsTrigger } from './env-var-actions-menu'
import { Pill } from './primitives'

interface ToolsetConfigPanelProps {
  toolset: string
  /** Called after a key is saved/cleared or a provider chosen, so the parent
   *  can refresh the "Configured / Needs keys" pill. */
  onConfiguredChange?: () => void
}

function providerConfigured(provider: ToolProvider, envState: Record<string, boolean>): boolean {
  if (provider.env_vars.length === 0) {
    return true
  }

  return provider.env_vars.every(ev => envState[ev.key])
}

interface EnvVarFieldProps {
  envVar: ToolEnvVar
  isSet: boolean
  onSaved: (key: string) => void
  onCleared: (key: string) => void
}

function EnvVarField({ envVar, isSet, onSaved, onCleared }: EnvVarFieldProps) {
  const { t } = useI18n()
  const copy = t.settings.toolsets
  const [editing, setEditing] = useState(false)
  const [value, setValue] = useState('')
  const [revealed, setRevealed] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  async function handleSave() {
    if (!value) {
      return
    }

    setBusy(true)

    try {
      await setEnvVar(envVar.key, value)
      setEditing(false)
      setValue('')
      onSaved(envVar.key)
      notify({ kind: 'success', title: copy.savedTitle, message: copy.savedMessage(envVar.key) })
    } catch (err) {
      notifyError(err, copy.failedSave(envVar.key))
    } finally {
      setBusy(false)
    }
  }

  async function handleClear() {
    if (!window.confirm(copy.removeConfirm(envVar.key))) {
      return
    }

    setBusy(true)

    try {
      await deleteEnvVar(envVar.key)
      setRevealed(null)
      onCleared(envVar.key)
      notify({ kind: 'success', title: copy.removedTitle, message: copy.removedMessage(envVar.key) })
    } catch (err) {
      notifyError(err, copy.failedRemove(envVar.key))
    } finally {
      setBusy(false)
    }
  }

  async function handleReveal() {
    if (revealed !== null) {
      setRevealed(null)

      return
    }

    try {
      const result = await revealEnvVar(envVar.key)
      setRevealed(result.value)
    } catch (err) {
      notifyError(err, copy.failedReveal(envVar.key))
    }
  }

  return (
    <div className="grid gap-2 rounded-lg bg-background/55 p-2.5">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-mono text-xs font-medium">{envVar.key}</span>
            <Pill tone={isSet ? 'primary' : 'muted'}>
              {isSet && <Check className="size-3" />}
              {isSet ? copy.set : copy.notSet}
            </Pill>
          </div>
          {envVar.prompt && envVar.prompt !== envVar.key && (
            <p className="mt-0.5 text-[0.7rem] text-muted-foreground">{envVar.prompt}</p>
          )}
        </div>
        {!editing && (
          <EnvVarActionsMenu
            clearDisabled={busy}
            docsUrl={envVar.url}
            isRevealed={revealed !== null}
            isSet={isSet}
            label={envVar.key}
            onClear={() => void handleClear()}
            onEdit={() => setEditing(true)}
            onReveal={() => void handleReveal()}
          >
            <EnvVarActionsTrigger label={envVar.key} onClick={event => event.stopPropagation()} />
          </EnvVarActionsMenu>
        )}
      </div>

      {isSet && revealed !== null && (
        <div className="rounded-md bg-background px-2.5 py-1.5 font-mono text-xs text-foreground">
          {revealed || '---'}
        </div>
      )}

      {editing && (
        <div className="flex flex-wrap items-center gap-2">
          <Input
            autoFocus
            className="min-w-52 flex-1 font-mono"
            onChange={e => setValue(e.target.value)}
            placeholder={envVar.prompt || envVar.key}
            type={envVar.default ? 'text' : 'password'}
            value={value}
          />
          <Button disabled={busy || !value} onClick={() => void handleSave()} size="sm">
            {busy ? <Loader2 className="size-3.5 animate-spin" /> : <Save />}
            {t.common.save}
          </Button>
          <Button onClick={() => setEditing(false)} size="sm" variant="text">
            {t.common.cancel}
          </Button>
        </div>
      )}
    </div>
  )
}

interface PostSetupRunnerProps {
  toolset: string
  /** The provider's post_setup hook key (e.g. "camofox", "ddgs"). */
  postSetupKey: string
  /** Refresh the parent config after the install finishes (a backend may now
   *  report itself configured). */
  onComplete?: () => void
}

/**
 * Runs a provider's post-setup install hook (npm / pip / binary) via the
 * `/api/tools/toolsets/{name}/post-setup` spawn-action and tails the resulting
 * log inline — the GUI equivalent of the install step `hermes tools` runs
 * after you pick a backend that needs extra dependencies.
 */
function PostSetupRunner({ toolset, postSetupKey, onComplete }: PostSetupRunnerProps) {
  const { t } = useI18n()
  const copy = t.settings.toolsets
  const [running, setRunning] = useState(false)
  const [status, setStatus] = useState<ActionStatusResponse | null>(null)
  // Guard against overlapping polls / state updates after unmount.
  const activeRef = useRef(false)

  useEffect(() => {
    return () => {
      activeRef.current = false
    }
  }, [])

  const run = useCallback(async () => {
    setRunning(true)
    setStatus(null)
    activeRef.current = true

    try {
      const started = await runToolsetPostSetup(toolset, postSetupKey)

      // The spawn endpoint reports ok:false if it couldn't launch the action
      // (e.g. unknown key, server-side spawn failure). Don't poll a status
      // that will never exist — surface the failure and stop.
      if (!started.ok) {
        notifyError(new Error('spawn failed'), copy.postSetupFailed(postSetupKey))

        return
      }

      let last: ActionStatusResponse | null = null

      // Mirror command-center's runSystemAction poll loop: poll the action log
      // until it exits (or we hit the attempt ceiling), feeding the global
      // activity rail as we go.
      for (let attempt = 0; attempt < 150 && activeRef.current; attempt += 1) {
        await new Promise(resolve => window.setTimeout(resolve, 1200))

        if (!activeRef.current) {
          break
        }

        const polled = await getActionStatus(started.name, 300)
        last = polled
        setStatus(polled)
        upsertDesktopActionTask(polled)

        if (!polled.running) {
          break
        }
      }

      if (activeRef.current) {
        const ok = last?.exit_code === 0

        notify(
          ok
            ? {
                kind: 'success',
                title: copy.postSetupCompleteTitle,
                message: copy.postSetupCompleteMessage(postSetupKey)
              }
            : { kind: 'error', title: copy.postSetupErrorTitle, message: copy.postSetupErrorMessage(postSetupKey) }
        )
        onComplete?.()
      }
    } catch (err) {
      if (activeRef.current) {
        notifyError(err, copy.postSetupFailed(postSetupKey))
      }
    } finally {
      if (activeRef.current) {
        setRunning(false)
      }
    }
  }, [toolset, postSetupKey, onComplete, copy])

  return (
    <div className="grid gap-2 rounded-lg bg-background/55 p-2.5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="min-w-0">
          <p className="text-[0.72rem] text-muted-foreground">{copy.postSetupHint(postSetupKey)}</p>
        </div>
        <Button disabled={running} onClick={() => void run()} size="sm">
          {running ? <Loader2 className="size-3.5 animate-spin" /> : <Terminal className="size-3.5" />}
          {running ? copy.postSetupRunning : copy.postSetupRun}
        </Button>
      </div>

      {status && (status.lines.length > 0 || status.running) && (
        <pre className="max-h-48 overflow-y-auto rounded-md bg-background px-2.5 py-1.5 font-mono text-[0.7rem] leading-relaxed text-muted-foreground whitespace-pre-wrap">
          {status.lines.length > 0 ? status.lines.join('\n') : copy.postSetupStarting}
        </pre>
      )}
    </div>
  )
}

export function ToolsetConfigPanel({ toolset, onConfiguredChange }: ToolsetConfigPanelProps) {
  const { t } = useI18n()
  const copy = t.settings.toolsets
  const [cfg, setCfg] = useState<ToolsetConfig | null>(null)
  const [loading, setLoading] = useState(true)
  const [selecting, setSelecting] = useState<string | null>(null)
  const [activeProvider, setActiveProvider] = useState<string | null>(null)
  // Live per-key set/unset state, seeded from the endpoint then patched locally.
  const [envState, setEnvState] = useState<Record<string, boolean>>({})

  const refresh = useCallback(async () => {
    setLoading(true)

    try {
      const next = await getToolsetConfig(toolset)
      setCfg(next)
      const seeded: Record<string, boolean> = {}

      for (const provider of next.providers) {
        for (const ev of provider.env_vars) {
          seeded[ev.key] = ev.is_set
        }
      }

      setEnvState(seeded)
    } catch (err) {
      notifyError(err, copy.failedLoad)
    } finally {
      setLoading(false)
    }
  }, [toolset])

  useEffect(() => {
    void refresh()
  }, [refresh])

  const providers = useMemo(() => cfg?.providers ?? [], [cfg])

  // Default the expanded provider to the one actually active in config
  // (`is_active` / `cfg.active_provider`, mirroring the CLI picker), then the
  // first fully-configured provider, else the first provider. Without this the
  // panel highlighted the first keyless provider (e.g. Nous Portal) even when
  // the user had already selected another (e.g. DuckDuckGo).
  useEffect(() => {
    if (activeProvider || providers.length === 0) {
      return
    }

    const selected =
      providers.find(p => p.is_active) ??
      (cfg?.active_provider ? providers.find(p => p.name === cfg.active_provider) : undefined) ??
      providers.find(p => providerConfigured(p, envState)) ??
      providers[0]

    setActiveProvider(selected.name)
  }, [activeProvider, providers, envState, cfg])

  async function handleSelect(provider: ToolProvider) {
    setActiveProvider(provider.name)
    setSelecting(provider.name)

    try {
      await selectToolsetProvider(toolset, provider.name)
      notify({ kind: 'success', title: copy.selectedTitle, message: copy.selectedMessage(provider.name) })
      onConfiguredChange?.()
    } catch (err) {
      notifyError(err, copy.failedSelect(provider.name))
    } finally {
      setSelecting(null)
    }
  }

  function patchEnv(key: string, isSet: boolean) {
    setEnvState(c => ({ ...c, [key]: isSet }))
    onConfiguredChange?.()
  }

  const emptyMessage = useMemo(() => {
    if (loading || !cfg) {
      return null
    }

    if (!cfg.has_category) {
      return copy.noProviderOptions
    }

    if (providers.length === 0) {
      return copy.noProviders
    }

    return null
  }, [cfg, copy, loading, providers.length])

  if (loading) {
    return <PageLoader className="min-h-32" label={copy.loadingConfig} />
  }

  if (emptyMessage) {
    return <p className="px-1 py-3 text-xs text-muted-foreground">{emptyMessage}</p>
  }

  return (
    <div className="mt-3 grid gap-2">
      {providers.map(provider => {
        const isActive = activeProvider === provider.name
        const configured = providerConfigured(provider, envState)

        return (
          <div className="overflow-hidden rounded-xl bg-background/60" key={provider.name}>
            <button
              aria-pressed={isActive}
              className={cn(
                'flex w-full items-center justify-between gap-3 px-3 py-2.5 text-left transition hover:bg-accent/50',
                isActive && 'bg-accent/40'
              )}
              onClick={() => void handleSelect(provider)}
              type="button"
            >
              <span className="flex min-w-0 items-center gap-2">
                <span className="truncate text-sm font-medium">{provider.name}</span>
                {provider.badge && <Pill>{provider.badge}</Pill>}
                {configured && (
                  <Pill tone="primary">
                    <Check className="size-3" />
                    {copy.ready}
                  </Pill>
                )}
              </span>
              {selecting === provider.name && <Loader2 className="size-3.5 shrink-0 animate-spin" />}
            </button>

            {isActive && (
              <div className="grid gap-2 bg-muted/20 p-3">
                {provider.tag && <p className="text-[0.72rem] text-muted-foreground">{provider.tag}</p>}
                {provider.requires_nous_auth && (
                  <p className="text-[0.72rem] text-muted-foreground">
                    {copy.nousIncluded}
                  </p>
                )}
                {provider.env_vars.length === 0 ? (
                  <p className="text-[0.72rem] text-muted-foreground">{copy.noApiKeyRequired}</p>
                ) : (
                  provider.env_vars.map(ev => (
                    <EnvVarField
                      envVar={ev}
                      isSet={Boolean(envState[ev.key])}
                      key={ev.key}
                      onCleared={key => patchEnv(key, false)}
                      onSaved={key => patchEnv(key, true)}
                    />
                  ))
                )}
                {provider.post_setup && (
                  <PostSetupRunner
                    onComplete={() => void refresh()}
                    postSetupKey={provider.post_setup}
                    toolset={toolset}
                  />
                )}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}
