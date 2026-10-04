import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useEffectEvent, useMemo, useRef, useState } from 'react'
import { useLocation, useNavigate } from 'react-router'

import { $apiRequestScope } from '@/api/client'
import { getMemoryStatus } from '@/api/system'
import { memoryDiscoveryKey } from '@/api/system'
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
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { discoverRuntimePlugins } from '@/contrib/runtime-loader'
import { useI18n } from '@/i18n'
import { ExternalLink } from '@/lib/external-link'
import { AlertTriangle } from '@/lib/icons'
import { resolvePluginSourceLinks } from '@/lib/plugin-source-urls'
import { queryClient } from '@/lib/query-client'
import { type AgentPluginLiveNow, COMMIT_SHA_RE, installAgentPlugin, loadAgentPlugins } from '@/store/agent-plugins'
import { activeGateway } from '@/store/gateway'
import { notify } from '@/store/notifications'
import {
  $pluginInstallRequest,
  closePluginInstallRequest,
  openPluginInstallRequest,
  type PluginInstallRequest
} from '@/store/plugin-install-request'
import { $activeGatewayProfile, $profiles, $profileScope, normalizeProfileKey, profileLabel } from '@/store/profile'
import { $connection } from '@/store/session'
import { setSettingsScope } from '@/store/settings-scope'

type ProbeResult = Awaited<ReturnType<NonNullable<NonNullable<Window['hermesDesktop']>['probePluginRepo']>>>

type ProbePhase = 'idle' | 'probing' | 'ready' | 'error'

type InstallModalCopy = ReturnType<typeof useI18n>['t']['settings']['plugins']['installModal']

const MEMORY_SETTINGS_ROUTE = '/settings?tab=config:memory&page=persistent'

/** What an agent-plugin install made usable, as toast fragments ("12 tools connected", ...). */
function installOutcome(m: InstallModalCopy, live: AgentPluginLiveNow, nextChat: boolean): string[] {
  const tools = live.mcpServers.reduce((n, server) => n + (server.connected ? server.tools.length : 0), 0)

  return [
    ...(tools > 0 ? [m.toolsConnected(tools)] : []),
    ...(live.skills.length > 0 ? [m.skillsReady(live.skills)] : []),
    ...(nextChat ? [m.nextChat] : [])
  ]
}

export function PluginInstallModal() {
  const request = useStore($pluginInstallRequest)
  const { t } = useI18n()
  const m = t.settings.plugins.installModal
  const { requestGateway } = useGatewayRequest()
  const requestScope = useStore($apiRequestScope)
  const navigate = useNavigate()
  const location = useLocation()
  const onSettings = location.pathname.startsWith(SETTINGS_ROUTE)
  const connection = useStore($connection)
  const profiles = useStore($profiles)

  const [repoInput, setRepoInput] = useState('')
  const [targetProfile, setTargetProfile] = useState('default')
  const [phase, setPhase] = useState<ProbePhase>('idle')
  const [probe, setProbe] = useState<ProbeResult | null>(null)
  const [installAgent, setInstallAgent] = useState(true)
  const [installDesktop, setInstallDesktop] = useState(true)
  const [enableAgent, setEnableAgent] = useState(true)
  const [forceReinstall, setForceReinstall] = useState(false)
  const [pinRef, setPinRef] = useState('')
  const [installing, setInstalling] = useState(false)
  const [installError, setInstallError] = useState<string | null>(null)
  const [installUncertain, setInstallUncertain] = useState(false)
  const [memoryResult, setMemoryResult] = useState<'discovered' | 'missing' | null>(null)
  const installPending = useRef(false)
  const probeToken = useRef(0)

  const resetState = useCallback(() => {
    probeToken.current += 1
    setMemoryResult(null)
    setRepoInput('')
    setPhase('idle')
    setProbe(null)
    setInstallAgent(true)
    setInstallDesktop(true)
    setEnableAgent(true)
    setForceReinstall(false)
    setPinRef('')
    // Replacing the review must not admit another install until the old one settles.
    setInstalling(installPending.current)
    setInstallError(null)
    setInstallUncertain(false)
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
      setInstallUncertain(false)
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
    if (request && !request.memory && onSettings) {
      navigate(NEW_CHAT_ROUTE)
    }
  }, [request, onSettings, navigate])

  // Navigation is presentation, not a new repository inspection.
  const openMemorySettings = useEffectEvent((profile: string) => {
    setSettingsScope(profile)
    navigate(MEMORY_SETTINGS_ROUTE)
  })

  useEffect(() => {
    resetState()

    if (!request) {
      return
    }

    if (request.memory) {
      const { owner } = request.memory
      const foreground = $apiRequestScope.get()

      // A slow catalog lookup must not take over a newly selected owner.
      if (
        foreground.connectionId === owner.connectionId &&
        normalizeProfileKey(foreground.profile) === normalizeProfileKey(owner.profile)
      ) {
        openMemorySettings(normalizeProfileKey(owner.profile))
      }
    }

    setTargetProfile(
      normalizeProfileKey(
        request.memory?.owner.profile ?? request.profile ?? $activeGatewayProfile.get() ?? $profileScope.get()
      )
    )

    if (request.repo) {
      void runProbe(request)
    }
  }, [request, resetState, runProbe])

  const targetProfileInfo = profiles.find(profile => normalizeProfileKey(profile.name) === targetProfile)
  const profileOptions = targetProfileInfo ? profiles : [...profiles, { name: targetProfile }]
  const targetProfileLabel = profileLabel(targetProfileInfo ?? { name: targetProfile })

  const agentTargetHint =
    connection?.mode === 'remote'
      ? m.agentTargetRemote(targetProfileLabel)
      : m.agentTargetLocal(
          targetProfileLabel,
          targetProfile === 'default' ? '~/.hermes/plugins/' : `~/.hermes/profiles/${targetProfile}/plugins/`
        )

  // A unified package installed into a local backend carries its own desktop
  // half; the app copies that half out of the package folder. Only a remote
  // backend (whose plugins/ folder this machine cannot read) or a desktop-only
  // repo needs a separate desktop clone.
  const desktopHalfFromPackage = Boolean(probe?.agent && installAgent && connection?.mode !== 'remote')

  const sourceLinks = useMemo(() => (request ? resolvePluginSourceLinks(request.repo) : null), [request])

  const handleClose = () => {
    if (installing) {
      return
    }

    probeToken.current += 1
    closePluginInstallRequest()
  }

  const handleInstall = async () => {
    if (!request || !probe?.ok || installPending.current || installing || installUncertain || memoryResult) {
      return
    }

    if (!installAgent && !installDesktop) {
      setInstallError(m.selectComponent)

      return
    }

    const memory = request.memory

    if (memory && !probe.agent) {
      setInstallError(m.selectComponent)

      return
    }

    const gateway = activeGateway()

    const ownerMatches =
      !memory ||
      ($apiRequestScope.get().connectionId === memory.owner.connectionId &&
        normalizeProfileKey($apiRequestScope.get().profile) === normalizeProfileKey(memory.owner.profile))

    if (memory && (!ownerMatches || !gateway)) {
      setInstallError(t.memoryDiscovery.ownerChanged)

      return
    }

    // Pin the socket before the first await. Never reconnect/retry a mutation
    // through the ambient requestGateway, which can change owners mid-install.
    const installRequest = memory && gateway ? gateway.request.bind(gateway) : requestGateway
    installPending.current = true
    setInstalling(true)
    setInstallError(null)
    setInstallUncertain(false)

    const errors: string[] = []
    const successes: string[] = []
    let agentInstalled = false
    let live: AgentPluginLiveNow = { mcpServers: [], skills: [] }

    try {
      if (installAgent && probe.agent) {
        const result = await installAgentPlugin(installRequest, {
          identifier: request.repo,
          force: forceReinstall,
          enable: memory ? true : enableAgent,
          catalogName: request.catalogName,
          ref: pinRefTrimmed || undefined,
          profile: memory ? memory.owner.profile : targetProfile
        })

        if (memory && result.ok) {
          try {
            const status = await getMemoryStatus(memory.owner)
            queryClient.setQueryData(memoryDiscoveryKey(memory.owner), status)

            if ($pluginInstallRequest.get() === request) {
              setMemoryResult(
                status.providers.some(provider => provider.name === memory.name && provider.status !== 'missing')
                  ? 'discovered'
                  : 'missing'
              )
            }
          } catch {
            if ($pluginInstallRequest.get() === request) {
              setMemoryResult('missing')
            }
          }

          return
        }

        if (memory && $pluginInstallRequest.get() !== request) {
          return
        }

        if (result.ok) {
          successes.push(
            [
              m.agentSuccess(result.pluginName ?? request.repo),
              ...installOutcome(m, result.live, result.nextChat)
            ].join(' · ')
          )
          agentInstalled = true
          live = result.live

          if (result.missingEnv?.length) {
            const firstVar = result.missingEnv[0]

            notify({
              kind: 'warning',
              message: m.missingEnv(result.pluginName ?? request.repo, result.missingEnv.join(', ')),
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
        } else if (result.timedOut) {
          // A client timeout does not cancel the backend install. Do not clone
          // the desktop half or offer a retry while the package may still be
          // installing. A read-only list refresh can show an already landed
          // package; the user can rescan later if the backend is still busy.
          setInstallUncertain(true)

          if (!memory) {
            void loadAgentPlugins(requestGateway, targetProfile)
          }

          return
        } else {
          errors.push(result.error || m.agentFailed)
        }
      }

      if (!memory && installDesktop && probe.desktop) {
        if (desktopHalfFromPackage) {
          // Unified package into a LOCAL backend: the desktop half ships inside
          // the package folder. Materialise it from there (one source of truth,
          // follows updates/uninstall) instead of cloning a second, standalone
          // copy under another folder name. This holds whether or not the agent
          // install above succeeded: a package already on disk answers "already
          // exists" without Force, and falling through to the clone would land
          // desktop-plugins/<git-name>/ beside the package copy (#100412). When
          // there is nothing to materialise, nothing was installed. The agent
          // error already says so.
          const touched = (await window.hermesDesktop?.reconcileDesktopPlugins?.()) ?? []

          if (agentInstalled || touched.length > 0) {
            successes.push(m.desktopSuccess(probe.agentName ?? request.repo))
          }

          if (touched.length > 0) {
            await discoverRuntimePlugins()
          }
        } else {
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
      }

      if (!memory) {
        await loadAgentPlugins(requestGateway, targetProfile)
      }

      if (errors.length === 0) {
        for (const message of successes) {
          notify({ kind: 'success', message })
        }

        // Open chats of the profile already have the plugin's MCP tools and skills (no click).
        if (agentInstalled && enableAgent) {
          for (const server of live.mcpServers.filter(s => !s.connected)) {
            notify({ kind: 'warning', message: m.serverNotConnected(server.name, server.error || '') })
          }
        }

        closePluginInstallRequest()
        // Land on the inventory (Capabilities → Plugins) — Git installs too;
        // `/settings?tab=plugins` is the plugin settings pages now.
        navigate('/capabilities?tab=plugins')

        return
      }

      if (successes.length > 0) {
        for (const message of successes) {
          notify({ kind: 'success', message })
        }
      }

      setInstallError(errors.join('\n'))
    } finally {
      installPending.current = false
      setInstalling(false)
    }
  }

  const open = request !== null && (!onSettings || Boolean(request.memory))

  const memoryOwnerMatches =
    !request?.memory ||
    (requestScope.connectionId === request.memory.owner.connectionId &&
      normalizeProfileKey(requestScope.profile) === normalizeProfileKey(request.memory.owner.profile))

  const busy = phase === 'probing' || installing
  const pinRefTrimmed = pinRef.trim().toLowerCase()
  const pinRefInvalid = pinRefTrimmed !== '' && !COMMIT_SHA_RE.test(pinRefTrimmed)

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
                  <div className="space-y-2 rounded-lg border border-(--ui-stroke-tertiary) px-3 py-2">
                    <label className="flex items-start gap-3">
                      <Checkbox
                        checked={installAgent}
                        disabled={busy || Boolean(request.memory)}
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
                    <label className="block space-y-1 pl-7">
                      <span className="text-[length:var(--conversation-caption-font-size)] text-foreground">
                        {m.profileLabel}
                      </span>
                      <Select
                        disabled={busy || !installAgent || Boolean(request.memory)}
                        onValueChange={setTargetProfile}
                        value={targetProfile}
                      >
                        <SelectTrigger aria-label={m.profileLabel} className="w-full">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {profileOptions.map(profile => (
                            <SelectItem key={profile.name} value={normalizeProfileKey(profile.name)}>
                              {profileLabel(profile)}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </label>
                  </div>
                )}

                {probe.desktop && !request.memory && (
                  <label className="flex items-start gap-3 rounded-lg border border-(--ui-stroke-tertiary) px-3 py-2">
                    <Checkbox
                      checked={installDesktop}
                      disabled={busy}
                      onCheckedChange={value => setInstallDesktop(value === true)}
                    />
                    <span className="min-w-0">
                      <span className="block font-medium text-foreground">{m.desktopLabel}</span>
                      <span className="block text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
                        {desktopHalfFromPackage ? m.desktopTargetFromPackage : m.desktopTarget}
                        {desktopHalfFromPackage ? '' : probe.desktopName ? ` · ${probe.desktopName}` : ''}
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
                      {[...new Set([...(probe.warnings ?? []), probe.insecure ? m.insecureWarning : ''])]
                        .filter(Boolean)
                        .join(' ')}
                    </span>
                  </div>
                )}

                {request.memory && <p className="text-sm text-muted-foreground">{t.memoryDiscovery.installConsent}</p>}
                {probe.agent && !request.memory && (
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

                {!request.catalogName && probe.agent && (
                  <label className="block space-y-1">
                    <span className="text-[length:var(--conversation-caption-font-size)] text-foreground">
                      {m.pinToCommit}
                    </span>
                    <Input
                      aria-invalid={pinRefInvalid || undefined}
                      aria-label={m.pinToCommit}
                      disabled={busy || !installAgent}
                      onChange={event => setPinRef(event.target.value)}
                      placeholder={m.pinToCommitPlaceholder}
                      spellCheck={false}
                      value={pinRef}
                    />
                    <span
                      className={`block text-[length:var(--conversation-caption-font-size)] ${pinRefInvalid ? 'text-destructive' : 'text-(--ui-text-tertiary)'}`}
                    >
                      {pinRefInvalid ? m.pinToCommitInvalid : m.pinToCommitHint}
                    </span>
                  </label>
                )}
              </div>
            )}

            {memoryResult && (
              <p role="status">
                {memoryResult === 'discovered' ? t.memoryDiscovery.installedNotice : t.memoryDiscovery.notDiscovered}
              </p>
            )}
            {!memoryOwnerMatches && !installError && <p role="status">{t.memoryDiscovery.ownerChanged}</p>}
            {installError && (
              <p className="rounded-lg border border-destructive/30 bg-destructive/10 px-3 py-2 whitespace-pre-wrap text-[length:var(--conversation-caption-font-size)] text-destructive">
                {installError}
              </p>
            )}
            {installUncertain && (
              <p
                className="text-[length:var(--conversation-caption-font-size)] text-(--ui-text-secondary)"
                role="status"
              >
                {m.installUncertain}
              </p>
            )}
          </div>
        )}

        <DialogFooter>
          <Button disabled={busy} onClick={handleClose} variant="outline">
            {t.common.cancel}
          </Button>
          {memoryResult === 'discovered' && memoryOwnerMatches ? (
            <Button
              onClick={() => {
                closePluginInstallRequest()
                navigate(MEMORY_SETTINGS_ROUTE)
              }}
            >
              {t.memoryDiscovery.backToMemory}
            </Button>
          ) : request && !request.repo ? (
            <Button disabled={!repoInput.trim()} form="plugin-repository-form" type="submit">
              {m.reviewRepository}
            </Button>
          ) : (
            <Button
              disabled={
                busy || Boolean(memoryResult) || installUncertain || phase !== 'ready' || !probe?.ok || pinRefInvalid
              }
              onClick={() => void handleInstall()}
            >
              {installing ? m.installing : m.install}
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
