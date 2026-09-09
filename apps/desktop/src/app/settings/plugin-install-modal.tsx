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
import { Input } from '@/components/ui/input'
import { Switch } from '@/components/ui/switch'
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
  openPluginInstallRequest,
  type PluginInstallRequest
} from '@/store/plugin-install-request'
import { $activeGatewayProfile, $profileScope } from '@/store/profile'
import { $connection } from '@/store/session'
import { runGatewayRestart } from '@/store/system-actions'

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

  const [repoInput, setRepoInput] = useState('')
  const [phase, setPhase] = useState<ProbePhase>('idle')
  const [probe, setProbe] = useState<ProbeResult | null>(null)
  const [installAgent, setInstallAgent] = useState(true)
  const [installDesktop, setInstallDesktop] = useState(true)
  const [enableAgent, setEnableAgent] = useState(true)
  const [forceReinstall, setForceReinstall] = useState(false)
  const [installing, setInstalling] = useState(false)
  const [installError, setInstallError] = useState<string | null>(null)
  const probeToken = useRef(0)

  const resetState = useCallback(() => {
    setRepoInput('')
    setPhase('idle')
    setProbe(null)
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
      // Reviewed catalog picks streamline the ceremony: enable defaults ON
      // (installing a reviewed entry to not use it is the rare case).
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

    if (request.repo) {
      void runProbe(request)
    }
  }, [request, resetState, runProbe])

  const profileLabel = request?.profile || activeProfile || profileScope || 'default'

  const agentTargetHint =
    connection?.mode === 'remote' ? m.agentTargetRemote(profileLabel) : m.agentTargetLocal(profileLabel)

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
    let agentInstalled = false

    try {
      if (installAgent && probe.agent) {
        const result = await installAgentPlugin(requestGateway, {
          identifier: request.repo,
          force: forceReinstall,
          enable: enableAgent,
          catalogName: request.catalogName,
          profile: request.profile
        })

        if (result.ok) {
          successes.push(m.agentSuccess(result.pluginName ?? request.repo))
          agentInstalled = true

          if (result.missingEnv?.length) {
            const firstVar = result.missingEnv[0]

            notify({
              kind: 'warning',
              message: m.missingEnv(result.missingEnv.join(', ')),
              // Deep-link straight to the credential card instead of leaving
              // the user to hunt through Settings → Tools & Keys by hand.
              action: {
                label: m.missingEnvAction,
                onClick: () => navigate(`/settings?tab=keys&key=${encodeURIComponent(firstVar)}`)
              }
            })
          }

          for (const warning of result.warnings ?? []) {
            notify({ kind: 'warning', message: warning })
          }
        } else {
          errors.push(result.error || m.agentFailed)
        }
      }

      if (installDesktop && probe.desktop) {
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

        // An enabled agent plugin only takes effect after a gateway restart —
        // offer the restart right here instead of a dim hint to run later.
        if (agentInstalled && enableAgent) {
          notify({
            kind: 'success',
            message: m.restartToApply,
            action: { label: m.restartNow, onClick: () => void runGatewayRestart() }
          })
        }

        closePluginInstallRequest()
        // Catalog picks come from Capabilities → Plugins; land back there.
        navigate(request.catalogName ? '/skills?tab=plugins' : '/settings?tab=plugins')

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
      <DialogContent className="max-w-lg" onOpenAutoFocus={request?.repo ? preventCloseButtonAutoFocus : undefined}>
        <DialogHeader>
          <DialogTitle>{m.title}</DialogTitle>
          <DialogDescription>{m.description}</DialogDescription>
        </DialogHeader>

        {request && !request.repo && (
          <form
            className="space-y-3"
            id="plugin-repository-form"
            onSubmit={event => {
              event.preventDefault()
              const repo = repoInput.trim()

              if (repo) {
                openPluginInstallRequest({ ...request, repo })
              }
            }}
          >
            <label className="block space-y-1">
              <span>{m.repoLabel}</span>
              <Input
                autoFocus
                onChange={event => setRepoInput(event.target.value)}
                placeholder={m.repoPlaceholder}
                spellCheck={false}
                value={repoInput}
              />
            </label>
          </form>
        )}

        {request?.repo && (
          <div className="space-y-4">
            <div>
              <div className="mb-1 text-[length:var(--conversation-caption-font-size)] font-medium text-foreground">
                {m.repoLabel}
              </div>
              <div className="rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) px-3 py-2 font-mono text-[length:var(--conversation-caption-font-size)] break-all text-foreground">
                {request.repo}
              </div>
              {request.catalogName && (
                <p className="mt-1 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
                  {m.catalogPinned(request.catalogName, request.sha?.slice(0, 8) ?? '')}
                </p>
              )}
            </div>

            <div className="space-y-3 rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) px-3 py-2.5">
              <div className="space-y-2 text-[length:var(--conversation-caption-font-size)]">
                <div className="font-medium text-foreground">
                  {request.catalogName ? m.reviewedHeading : m.securityHeading}
                </div>
                <p className="text-(--ui-text-secondary)">{request.catalogName ? m.reviewedIntro : m.securityIntro}</p>
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
                        {m.desktopTarget}
                        {probe.desktopName ? ` · ${probe.desktopName}` : ''}
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

                {!request.catalogName && (
                  <label className="flex items-center justify-between gap-3">
                    <span className="text-[length:var(--conversation-caption-font-size)] text-foreground">
                      {m.forceReinstall}
                    </span>
                    <Switch checked={forceReinstall} disabled={busy} onCheckedChange={setForceReinstall} />
                  </label>
                )}
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
          {request && !request.repo ? (
            <Button disabled={!repoInput.trim()} form="plugin-repository-form" type="submit">
              {m.reviewRepository}
            </Button>
          ) : (
            <Button disabled={busy || phase !== 'ready' || !probe?.ok} onClick={() => void handleInstall()}>
              {installing ? m.installing : m.install}
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
