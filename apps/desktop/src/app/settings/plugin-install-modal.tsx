import { useStore } from '@nanostores/react'
import { Puzzle } from 'lucide-react'
import { type ReactNode, useEffect, useState } from 'react'
import { useLocation, useNavigate } from 'react-router'

import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { NEW_CHAT_ROUTE, SETTINGS_ROUTE } from '@/app/routes'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  preventCloseButtonAutoFocus
} from '@/components/ui/dialog'
import { DisclosureCaret } from '@/components/ui/disclosure-caret'
import { Input } from '@/components/ui/input'
import { useI18n } from '@/i18n'
import { AlertTriangle, Loader2 } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { COMMIT_SHA_RE } from '@/store/agent-plugins'
import {
  $pluginInstallRequest,
  closePluginInstallRequest,
  openPluginInstallRequest,
  type PluginInstallRequest
} from '@/store/plugin-install-request'
import { $activeGatewayProfile, $profiles, $profileScope, normalizeProfileKey, profileLabel } from '@/store/profile'
import { $connection } from '@/store/session'

import { PluginInstallDetails } from './plugin-install-details'
import { type InstallModalCopy, type ProbeResult, runPluginInstall } from './plugin-install-run'
import { usePluginProbe } from './use-plugin-probe'

const CAPTION = 'text-[length:var(--conversation-caption-font-size)]'

/** Close the dialog and land on the inventory (Capabilities → Plugins), unless the caller keeps the user in place. */
function finishInstall(request: PluginInstallRequest, navigate: (to: string) => void): void {
  closePluginInstallRequest()

  if (!request.stayAfterInstall) {
    navigate('/capabilities?tab=plugins')
  }
}

/** `owner/repo#sub/dir` → `dir`; `https://host/owner/repo.git` → `repo`. */
function repoName(repo: string): string {
  const [base, subdir] = repo.split('#')

  return (
    (subdir || base)
      .replace(/\/+$/, '')
      .split('/')
      .pop()
      ?.replace(/\.git$/, '') || repo
  )
}

function pluginTitle(request: PluginInstallRequest, probe: null | ProbeResult): string {
  return request.title || request.catalogName || probe?.agentName || probe?.desktopName || repoName(request.repo)
}

/** The inspection's one-line reason, when it could not read the repository. */
function probeFailure(probe: null | ProbeResult, fallback: string): null | string {
  return probe && !probe.ok ? probe.error || fallback : null
}

function ProbeWarnings({ copy, probe }: { copy: InstallModalCopy; probe: null | ProbeResult }) {
  const warnings = [...new Set([...(probe?.warnings ?? []), probe?.insecure ? copy.insecureWarning : ''])].filter(
    Boolean
  )

  if (!probe?.ok || warnings.length === 0) {
    return null
  }

  return (
    <div
      className={cn(
        CAPTION,
        'flex items-start gap-2 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-foreground'
      )}
    >
      <AlertTriangle aria-hidden className="mt-0.5 size-3.5 shrink-0 text-amber-600 dark:text-amber-400" />
      <span>{warnings.join(' ')}</span>
    </div>
  )
}

/** "Install from Git" with no repository yet: one field, then the same dialog inspects what was pasted. */
function RepositoryEntry({ request }: { request: PluginInstallRequest }) {
  const { t } = useI18n()
  const copy = t.settings.plugins.installModal
  const [repo, setRepo] = useState('')

  return (
    <form
      className="space-y-3"
      id="plugin-repository-form"
      onSubmit={event => {
        event.preventDefault()

        if (repo.trim()) {
          openPluginInstallRequest({ ...request, repo: repo.trim() })
        }
      }}
    >
      <label className="block space-y-1">
        <span>{copy.repoLabel}</span>
        <Input
          autoFocus
          onChange={event => setRepo(event.target.value)}
          placeholder={copy.repoPlaceholder}
          spellCheck={false}
          value={repo}
        />
      </label>
      <DialogFooter>
        <Button onClick={closePluginInstallRequest} type="button" variant="outline">
          {t.common.cancel}
        </Button>
        <Button disabled={!repo.trim()} type="submit">
          {copy.reviewRepository}
        </Button>
      </DialogFooter>
    </form>
  )
}

/**
 * THE plugin install dialog, for every door: a reviewed catalog pick (Plugins tab, `hermes://` link,
 * the setup chat's plugins card) and a raw git URL. A catalog entry shows its name, one line, "Verified"
 * and Install; its particulars wait behind Advanced. A git URL is marked Unverified with the same
 * particulars open, because there they are the security check. Install always turns the plugin on
 * (a deep link's explicit `enable=0` aside) and shows one loader until it is done.
 */
export function PluginInstallModal() {
  const request = useStore($pluginInstallRequest)
  const { t } = useI18n()
  const m = t.settings.plugins.installModal
  const navigate = useNavigate()
  const onSettings = useLocation().pathname.startsWith(SETTINGS_ROUTE)
  // Lifted so the dialog cannot be dismissed mid-install.
  const [installing, setInstalling] = useState(false)

  useEffect(() => {
    if (request && onSettings) {
      navigate(NEW_CHAT_ROUTE)
    }
  }, [request, onSettings, navigate])

  return (
    <Dialog
      onOpenChange={next => {
        if (!next && !installing) {
          closePluginInstallRequest()
        }
      }}
      open={request !== null && !onSettings}
    >
      <DialogContent className="max-w-lg" onOpenAutoFocus={request?.repo ? preventCloseButtonAutoFocus : undefined}>
        {request && !request.repo && (
          <>
            <DialogHeader>
              <DialogTitle>{m.title}</DialogTitle>
              <DialogDescription>{m.description}</DialogDescription>
            </DialogHeader>
            <RepositoryEntry request={request} />
          </>
        )}
        {request?.repo && (
          <InstallView
            installing={installing}
            key={request.repo}
            onInstallingChange={setInstalling}
            request={request}
          />
        )}
      </DialogContent>
    </Dialog>
  )
}

/** Name, Verified/Unverified, the one line on what it does, then who reviewed it and where it lands. */
function InstallHeader({
  line,
  probe,
  request
}: {
  line: string
  probe: null | ProbeResult
  request: PluginInstallRequest
}) {
  const { t } = useI18n()
  const m = t.settings.plugins.installModal
  const verified = Boolean(request.catalogName)

  return (
    <DialogHeader>
      <div className="flex items-start gap-3">
        <span className="grid size-9 shrink-0 place-items-center rounded-lg bg-(--ui-bg-quinary) text-(--ui-text-secondary)">
          <Puzzle className="size-4" />
        </span>
        <div className="min-w-0 space-y-1">
          <div className="flex flex-wrap items-center gap-2">
            <DialogTitle>{pluginTitle(request, probe)}</DialogTitle>
            <Badge variant={verified ? 'success' : 'warn'}>{verified ? m.verified : m.unverified}</Badge>
          </div>
          <DialogDescription>{request.summary || line}</DialogDescription>
          {request.summary && <p className={cn(CAPTION, 'text-(--ui-text-secondary)')}>{line}</p>}
        </div>
      </div>
    </DialogHeader>
  )
}

function Disclosure({
  children,
  label,
  onToggle,
  open
}: {
  children: ReactNode
  label: string
  onToggle: () => void
  open: boolean
}) {
  return (
    <div>
      <button
        aria-expanded={open}
        className={cn(CAPTION, 'flex items-center gap-1.5 font-medium text-(--ui-text-secondary)')}
        onClick={onToggle}
        type="button"
      >
        <DisclosureCaret open={open} />
        {label}
      </button>
      {open && <div className="pt-2">{children}</div>}
    </div>
  )
}

interface InstallFooterProps {
  canInstall: boolean
  failed: boolean
  installing: boolean
  onInstall: () => void
  onRetry: () => void
}

/** Cancel, then Install (one loader while it runs) or, after a failure, Retry. */
function InstallFooter({ canInstall, failed, installing, onInstall, onRetry }: InstallFooterProps) {
  const { t } = useI18n()
  const m = t.settings.plugins.installModal

  return (
    <DialogFooter>
      <Button disabled={installing} onClick={closePluginInstallRequest} variant="outline">
        {t.common.cancel}
      </Button>
      {failed ? (
        <Button onClick={onRetry}>{t.common.retry}</Button>
      ) : (
        <Button disabled={!canInstall} onClick={onInstall}>
          {installing && <Loader2 className="animate-spin" />}
          {installing ? m.installing : m.install}
        </Button>
      )}
    </DialogFooter>
  )
}

interface InstallViewProps {
  request: PluginInstallRequest
  installing: boolean
  onInstallingChange: (installing: boolean) => void
}

function InstallView({ request, installing, onInstallingChange }: InstallViewProps) {
  const { t } = useI18n()
  const m = t.settings.plugins.installModal
  const { requestGateway } = useGatewayRequest()
  const navigate = useNavigate()
  const remote = useStore($connection)?.mode === 'remote'
  const activeProfile = useStore($activeGatewayProfile)
  const profiles = useStore($profiles)
  const profileScope = useStore($profileScope)
  const verified = Boolean(request.catalogName)
  const enable = request.enable ?? true
  const { probe, start, cancel, settled } = usePluginProbe(m.probeUnavailable)

  const [targetProfile, setTargetProfile] = useState(() =>
    normalizeProfileKey(request.profile || activeProfile || profileScope)
  )

  const [advancedOpen, setAdvancedOpen] = useState(!verified)
  const [force, setForce] = useState(request.force ?? false)
  const [pinRef, setPinRef] = useState('')
  const [installError, setInstallError] = useState<null | string>(null)
  const [installUncertain, setInstallUncertain] = useState(false)

  useEffect(() => {
    start(request.repo)

    return cancel
  }, [cancel, request.repo, start])

  const targetInfo = profiles.find(profile => normalizeProfileKey(profile.name) === targetProfile)
  const targetLabel = profileLabel(targetInfo ?? { name: targetProfile })
  const pin = pinRef.trim().toLowerCase()
  const pinInvalid = pin !== '' && !COMMIT_SHA_RE.test(pin)
  // Hidden while a retry runs, so the loader is the only thing on screen.
  const failure = installing ? null : (installError ?? probeFailure(probe, m.probeUnavailable))
  const line = `${verified ? m.reviewedByNous : m.securityIntro} ${(enable ? m.installsAndEnables : m.installsOnly)(targetLabel)}`

  const install = async () => {
    onInstallingChange(true)
    setInstallError(null)

    try {
      // A catalog pick may click Install while the inspection is still running: wait for it under the loader.
      const detected = await settled()

      if (!detected?.ok) {
        return
      }

      const outcome = await runPluginInstall(
        { request, probe: detected, profile: targetProfile, enable, force, ref: pin || undefined, remote },
        { copy: m, navigate, requestGateway }
      )

      if (outcome.status === 'done') {
        finishInstall(request, navigate)
      } else if (outcome.status === 'uncertain') {
        setInstallUncertain(true)
      } else {
        setInstallError(outcome.error)
      }
    } finally {
      onInstallingChange(false)
    }
  }

  const retry = () => {
    if (installError) {
      void install()
    } else {
      start(request.repo)
    }
  }

  // An unverified repository installs only once its contents are on screen; a reviewed pick may start at once.
  const canInstall = !installing && !installUncertain && !pinInvalid && (verified || Boolean(probe?.ok))

  return (
    <>
      <InstallHeader line={line} probe={probe} request={request} />

      <div className="space-y-3">
        <ProbeWarnings copy={m} probe={probe} />

        <Disclosure label={m.advanced} onToggle={() => setAdvancedOpen(open => !open)} open={advancedOpen}>
          <PluginInstallDetails
            busy={installing}
            copy={m}
            force={force}
            onForceChange={setForce}
            pin={{ value: pinRef, invalid: pinInvalid, onChange: setPinRef }}
            probe={probe}
            profile={{
              value: targetProfile,
              label: targetLabel,
              options: targetInfo ? profiles : [...profiles, { name: targetProfile }],
              onChange: setTargetProfile
            }}
            remote={remote}
            request={request}
          />
        </Disclosure>

        {failure && (
          <p className={cn(CAPTION, 'text-destructive')} role="alert">
            {failure}
          </p>
        )}
        {installUncertain && (
          <p className={cn(CAPTION, 'text-(--ui-text-secondary)')} role="status">
            {m.installUncertain}
          </p>
        )}
      </div>

      <InstallFooter
        canInstall={canInstall}
        failed={Boolean(failure)}
        installing={installing}
        onInstall={() => void install()}
        onRetry={retry}
      />
    </>
  )
}
