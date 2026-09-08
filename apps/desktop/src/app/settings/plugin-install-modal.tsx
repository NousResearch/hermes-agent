import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useLocation, useNavigate } from 'react-router'

import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { NEW_CHAT_ROUTE, SETTINGS_ROUTE } from '@/app/routes'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  preventCloseButtonAutoFocus
} from '@/components/ui/dialog'
import { Switch } from '@/components/ui/switch'
import { $pluginRecords, setPluginEnabled } from '@/contrib/plugins-store'
import { discoverRuntimePlugins } from '@/contrib/runtime-loader'
import { useI18n } from '@/i18n'
import { ExternalLink } from '@/lib/external-link'
import { AlertTriangle } from '@/lib/icons'
import { resolvePluginSourceLinks } from '@/lib/plugin-source-urls'
import { installAgentPlugin, loadAgentPlugins } from '@/store/agent-plugins'
import { notify } from '@/store/notifications'
import {
  $pluginInstallRequest,
  closePluginInstallRequest,
  type PluginInstallRequest
} from '@/store/plugin-install-request'
import { $activeGatewayProfile, $profileScope } from '@/store/profile'
import { $connection } from '@/store/session'

import {
  desktopHalfMayShareLocalRoot,
  findStandaloneDesktopEntry,
  findUnifiedDesktopEntry,
  settleUnifiedDesktopPluginId
} from './plugin-install-plan'

type ProbeResult = Awaited<ReturnType<NonNullable<NonNullable<Window['hermesDesktop']>['probePluginRepo']>>>

type ProbePhase = 'idle' | 'probing' | 'ready' | 'error'

export function PluginInstallModal() {
  const request = useStore($pluginInstallRequest)
  const { t } = useI18n()
  const m = t.settings.plugins.installModal
  const { requestGateway } = useGatewayRequest()
  const navigate = useNavigate()
  const location = useLocation()
  const onSettings = location.pathname.startsWith(SETTINGS_ROUTE)
  const connection = useStore($connection)
  const activeProfile = useStore($activeGatewayProfile)
  const profileScope = useStore($profileScope)

  const [phase, setPhase] = useState<ProbePhase>('idle')
  const [probe, setProbe] = useState<ProbeResult | null>(null)
  const [installAgent, setInstallAgent] = useState(true)
  const [installDesktop, setInstallDesktop] = useState(true)
  const [enableAgent, setEnableAgent] = useState(true)
  const [forceReinstall, setForceReinstall] = useState(false)
  const [installing, setInstalling] = useState(false)
  const [installError, setInstallError] = useState<string | null>(null)
  // An existing desktop-plugins/<name>/plugin.js from an earlier install. The
  // loader serves that copy, so the install keeps refreshing it (#100412).
  const [standaloneEntry, setStandaloneEntry] = useState<string | null>(null)
  const probeToken = useRef(0)

  const resetState = useCallback(() => {
    setPhase('idle')
    setProbe(null)
    setStandaloneEntry(null)
    setInstallAgent(true)
    setInstallDesktop(true)
    setEnableAgent(true)
    setForceReinstall(false)
    setInstalling(false)
    setInstallError(null)
  }, [])

  const applyLegacyHint = useCallback((payload: PluginInstallRequest, detected: ProbeResult) => {
    if (payload.legacyHint === 'agent') {
      setInstallAgent(Boolean(detected.agent))
      setInstallDesktop(false)
    } else if (payload.legacyHint === 'desktop') {
      setInstallAgent(false)
      setInstallDesktop(Boolean(detected.desktop))
    } else {
      setInstallAgent(Boolean(detected.agent))
      setInstallDesktop(Boolean(detected.desktop))
    }
  }, [])

  const runProbe = useCallback(
    async (payload: PluginInstallRequest) => {
      const token = ++probeToken.current
      setPhase('probing')
      setProbe(null)
      setInstallError(null)
      setEnableAgent(payload.enable ?? true)
      setForceReinstall(payload.force ?? false)

      const probeFn = window.hermesDesktop?.probePluginRepo

      if (!probeFn) {
        if (token !== probeToken.current) {
          return
        }

        setPhase('error')
        setProbe({
          ok: false,
          agent: false,
          desktop: false,
          warnings: [],
          error: m.probeUnavailable
        })

        return
      }

      const result = await probeFn({ identifier: payload.repo })

      if (token !== probeToken.current) {
        return
      }

      setProbe(result)

      if (!result.ok) {
        setPhase('error')

        return
      }

      const standalone =
        result.desktop && result.desktopName
          ? await findStandaloneDesktopEntry(window.hermesDesktop, result.desktopName)
          : null

      if (token !== probeToken.current) {
        return
      }

      setStandaloneEntry(standalone)
      applyLegacyHint(payload, result)
      setPhase('ready')
    },
    [applyLegacyHint, m.probeUnavailable]
  )

  useEffect(() => {
    if (request && onSettings) {
      navigate(NEW_CHAT_ROUTE)
    }
  }, [request, onSettings, navigate])

  useEffect(() => {
    if (!request) {
      resetState()

      return
    }

    void runProbe(request)
  }, [request, resetState, runProbe])

  const profileLabel = activeProfile || profileScope || 'default'

  const agentTargetHint =
    connection?.mode === 'remote' ? m.agentTargetRemote(profileLabel) : m.agentTargetLocal(profileLabel)

  // A hybrid repo installed into a local backend lands its desktop half at
  // plugins/<name>/desktop/ too; a second copy under desktop-plugins/ would
  // load the same plugin id twice (#100412). This is the cheap pre-check for
  // the caption — the install itself only skips the copy once that unified
  // entry is actually on disk. An existing standalone copy keeps today's
  // path: the loader serves it, so Force must keep refreshing it.
  const desktopMayShareLocalRoot = desktopHalfMayShareLocalRoot({
    connectionMode: connection?.mode,
    probeAgent: probe?.agent === true,
    probeDesktop: probe?.desktop === true,
    desktopSourceSubdir: probe?.desktopSourceSubdir ?? null,
    standaloneCopy: standaloneEntry !== null,
    installAgent,
    installDesktop
  })

  const desktopTargetHint = desktopMayShareLocalRoot ? m.desktopTargetUnified : m.desktopTarget
  // The unified half lives under the agent package's folder, not desktopName.
  const desktopTargetName = desktopMayShareLocalRoot ? (probe?.agentName ?? probe?.desktopName) : probe?.desktopName

  const sourceLinks = useMemo(() => (request ? resolvePluginSourceLinks(request.repo) : null), [request])

  const handleClose = () => {
    if (installing) {
      return
    }

    probeToken.current += 1
    closePluginInstallRequest()
  }

  const handleInstall = async () => {
    if (!request || !probe?.ok || installing) {
      return
    }

    if (!installAgent && !installDesktop) {
      setInstallError(m.selectComponent)

      return
    }

    setInstalling(true)
    setInstallError(null)

    const errors: string[] = []
    const successes: string[] = []
    // The backend names the install folder after the manifest `name`, which is
    // what both `pluginName` and the probe's `agentName` report.
    let agentPluginName: null | string = probe.agentName ?? null

    try {
      if (installAgent && probe.agent) {
        const result = await installAgentPlugin(requestGateway, {
          identifier: request.repo,
          force: forceReinstall,
          enable: enableAgent
        })

        if (result.ok) {
          agentPluginName = result.pluginName ?? agentPluginName
          successes.push(m.agentSuccess(result.pluginName ?? request.repo))

          if (result.missingEnv?.length) {
            notify({
              kind: 'warning',
              message: m.missingEnv(result.missingEnv.join(', '))
            })
          }

          for (const warning of result.warnings ?? []) {
            notify({ kind: 'warning', message: warning })
          }
        } else {
          errors.push(result.error || m.agentFailed)
        }
      }

      // Skip the desktop-plugins/ copy only on hard evidence: the unified
      // entry plugins/<name>/desktop/plugin.js exists in the root this app
      // scans. That covers an agent install that just landed it AND one that
      // failed as "already exists"; a clone failure, a backend on another
      // hermes home, a root-level plugin.js, or an existing standalone copy
      // (desktopMayShareLocalRoot is false then) all fall through to the copy
      // — a wrong guess costs a duplicate, never the loss of the desktop half.
      const unifiedEntry =
        installDesktop && probe.desktop && desktopMayShareLocalRoot && agentPluginName
          ? await findUnifiedDesktopEntry(window.hermesDesktop, agentPluginName)
          : null

      if (unifiedEntry) {
        // That root loads opt-in, but the user ticked "Desktop UI" — honour
        // it the way `enable` honours the agent half. A rescan is dropped
        // while another scan holds the lock (and watched roots have no poll
        // to catch up), so rescan-and-wait a few times; if the record never
        // shows, leave it opt-in and say where to turn it on.
        const outcome = await settleUnifiedDesktopPluginId(discoverRuntimePlugins, $pluginRecords, unifiedEntry)
        const desktopName = agentPluginName ?? request.repo

        if (!outcome) {
          successes.push(m.desktopUnified(desktopName))
        } else if ('error' in outcome) {
          // The loader already inventoried it as broken or shadowed; that is
          // final, not "not yet".
          errors.push(`${m.desktopFailed}: ${outcome.error}`)
        } else if ($pluginRecords.get()[outcome.id]?.status === 'loaded') {
          // Already running (an "already exists" retry of a plugin the user
          // enabled earlier) — nothing to change, and never risk its state.
          successes.push(m.desktopUnifiedEnabled(desktopName))
        } else {
          try {
            await setPluginEnabled(outcome.id, true)
            successes.push(m.desktopUnifiedEnabled(desktopName))
          } catch (error) {
            // setPluginEnabled persists the decision before it activates, so
            // a throw leaves a dangling "on". The record was disabled before
            // (no decision, or an explicit off); an explicit off is the
            // closest state the store can express, and renders the same.
            await setPluginEnabled(outcome.id, false).catch(() => undefined)
            errors.push(`${m.desktopFailed}: ${error instanceof Error ? error.message : String(error)}`)
          }
        }
      } else if (installDesktop && probe.desktop) {
        const installFn = window.hermesDesktop?.installDesktopPlugin

        if (!installFn) {
          errors.push(m.desktopUnavailable)
        } else {
          const result = await installFn({ identifier: request.repo, force: forceReinstall })

          if (result.ok) {
            successes.push(m.desktopSuccess(result.pluginName ?? request.repo))
            await discoverRuntimePlugins()
          } else {
            errors.push(result.error || m.desktopFailed)
          }
        }
      }

      await loadAgentPlugins(requestGateway)

      if (errors.length === 0) {
        for (const message of successes) {
          notify({ kind: 'success', message })
        }

        closePluginInstallRequest()
        navigate('/settings?tab=plugins')

        return
      }

      if (successes.length > 0) {
        for (const message of successes) {
          notify({ kind: 'success', message })
        }
      }

      setInstallError(errors.join('\n'))
    } finally {
      setInstalling(false)
    }
  }

  const open = request !== null && !onSettings
  const busy = phase === 'probing' || installing

  return (
    <Dialog
      onOpenChange={next => {
        if (!next) {
          handleClose()
        }
      }}
      open={open}
    >
      <DialogContent className="max-w-lg" onOpenAutoFocus={preventCloseButtonAutoFocus}>
        <DialogHeader>
          <DialogTitle>{m.title}</DialogTitle>
          <DialogDescription>{m.description}</DialogDescription>
        </DialogHeader>

        {request && (
          <div className="space-y-4">
            <div>
              <div className="mb-1 text-[length:var(--conversation-caption-font-size)] font-medium text-foreground">
                {m.repoLabel}
              </div>
              <div className="rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) px-3 py-2 font-mono text-[length:var(--conversation-caption-font-size)] break-all text-foreground">
                {request.repo}
              </div>
            </div>

            <div className="space-y-3 rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) px-3 py-2.5">
              <div className="space-y-2 text-[length:var(--conversation-caption-font-size)]">
                <div className="font-medium text-foreground">{m.securityHeading}</div>
                <p className="text-(--ui-text-secondary)">{m.securityIntro}</p>
              </div>

              {sourceLinks && (
                <div className="space-y-2 border-t border-(--ui-stroke-tertiary) pt-3">
                  <div className="font-medium text-foreground">{m.sourceHeading}</div>
                  {sourceLinks.browseUrl && (
                    <ExternalLink
                      className="text-[length:var(--conversation-caption-font-size)]"
                      href={sourceLinks.browseUrl}
                      showExternalIcon
                    >
                      {sourceLinks.subdir ? m.viewPluginFiles : m.viewRepository}
                    </ExternalLink>
                  )}
                  <div>
                    <div className="mb-1 text-(--ui-text-tertiary)">{m.gitCloneLabel}</div>
                    <div className="rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-bg-primary) px-2.5 py-1.5 font-mono break-all text-foreground">
                      {sourceLinks.gitUrl}
                    </div>
                  </div>
                </div>
              )}
            </div>

            {phase === 'probing' && (
              <p className="text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
                {m.probing}
              </p>
            )}

            {phase === 'error' && probe?.error && (
              <p className="rounded-lg border border-destructive/30 bg-destructive/10 px-3 py-2 text-[length:var(--conversation-caption-font-size)] text-destructive">
                {probe.error}
              </p>
            )}

            {phase === 'ready' && probe && (
              <div className="space-y-3">
                <div className="text-[length:var(--conversation-caption-font-size)] font-medium text-foreground">
                  {m.includesHeading}
                </div>

                {probe.agent && (
                  <label className="flex items-start gap-3 rounded-lg border border-(--ui-stroke-tertiary) px-3 py-2">
                    <Checkbox
                      checked={installAgent}
                      disabled={busy}
                      onCheckedChange={value => setInstallAgent(value === true)}
                    />
                    <span className="min-w-0">
                      <span className="block font-medium text-foreground">{m.agentLabel}</span>
                      <span className="block text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
                        {agentTargetHint}
                        {probe.agentName ? ` · ${probe.agentName}` : ''}
                      </span>
                    </span>
                  </label>
                )}

                {probe.desktop && (
                  <label className="flex items-start gap-3 rounded-lg border border-(--ui-stroke-tertiary) px-3 py-2">
                    <Checkbox
                      checked={installDesktop}
                      disabled={busy}
                      onCheckedChange={value => setInstallDesktop(value === true)}
                    />
                    <span className="min-w-0">
                      <span className="block font-medium text-foreground">{m.desktopLabel}</span>
                      <span className="block text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
                        {desktopTargetHint}
                        {desktopTargetName ? ` · ${desktopTargetName}` : ''}
                      </span>
                    </span>
                  </label>
                )}

                {probe.desktop && !probe.agent && (
                  <p className="text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
                    {m.desktopOnlyNote}
                  </p>
                )}

                {(probe.insecure || (probe.warnings?.length ?? 0) > 0) && (
                  <div className="flex items-start gap-2 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-[length:var(--conversation-caption-font-size)] text-foreground">
                    <AlertTriangle
                      aria-hidden
                      className="mt-0.5 size-3.5 shrink-0 text-amber-600 dark:text-amber-400"
                    />
                    <span>
                      {[...(probe.warnings ?? []), probe.insecure ? m.insecureWarning : ''].filter(Boolean).join(' ')}
                    </span>
                  </div>
                )}

                {probe.agent && (
                  <label className="flex items-center justify-between gap-3">
                    <span className="text-[length:var(--conversation-caption-font-size)] text-foreground">
                      {m.enableAgent}
                    </span>
                    <Switch checked={enableAgent} disabled={busy || !installAgent} onCheckedChange={setEnableAgent} />
                  </label>
                )}

                <label className="flex items-center justify-between gap-3">
                  <span className="text-[length:var(--conversation-caption-font-size)] text-foreground">
                    {m.forceReinstall}
                  </span>
                  <Switch checked={forceReinstall} disabled={busy} onCheckedChange={setForceReinstall} />
                </label>
              </div>
            )}

            {installError && (
              <p className="rounded-lg border border-destructive/30 bg-destructive/10 px-3 py-2 whitespace-pre-wrap text-[length:var(--conversation-caption-font-size)] text-destructive">
                {installError}
              </p>
            )}
          </div>
        )}

        <DialogFooter>
          <Button disabled={busy} onClick={handleClose} variant="outline">
            {t.common.cancel}
          </Button>
          <Button disabled={busy || phase !== 'ready' || !probe?.ok} onClick={() => void handleInstall()}>
            {installing ? m.installing : m.install}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
