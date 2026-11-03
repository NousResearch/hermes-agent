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
import { lookupPluginCatalogEntry } from '@/lib/plugin-catalog'
import { cn } from '@/lib/utils'
import { COMMIT_SHA_RE, inspectAgentPlugin } from '@/store/agent-plugins'
import {
  $pluginInstallRequest,
  closePluginInstallRequest,
  openPluginInstallRequest,
  type PluginInstallRequest
} from '@/store/plugin-install-request'
import { $activeGatewayProfile, $profiles, $profileScope, normalizeProfileKey, profileLabel } from '@/store/profile'
import { $connection } from '@/store/session'

import { PluginInstallDetails } from './plugin-install-details'
import {
  installComponents,
  type InstallModalCopy,
  type InstallOutcome,
  type ProbeResult,
  reviewedCommit,
  runPluginInstall
} from './plugin-install-run'
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

/** The inspection's one-line reason, when it could not read the repository. An environment that cannot
 *  inspect at all reads `fallback`, not the transport's wording. */
function probeFailure(probe: null | ProbeResult, fallback: string): null | string {
  return probe && !probe.ok ? (!probe.unavailable && probe.error) || fallback : null
}

/** A repository read fine but nothing in it can be installed from here (a desktop-only plugin in the
 *  browser-hosted app): no retry changes that. */
function nothingInstallable(request: PluginInstallRequest, probe: null | ProbeResult): boolean {
  if (!probe?.ok) {
    return false
  }

  const components = installComponents(request, probe)

  return !components.agent && !components.desktop
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

/** The catalog tier as the badge reads it; an unknown tier keeps the generic Verified. */
function catalogTierLabel(t: ReturnType<typeof useI18n>['t'], tier: string): string {
  const labels = t.catalog.tierLabels

  return tier in labels ? labels[tier as keyof typeof labels] : t.settings.plugins.installModal.verified
}

/** Name, Official/Verified/Unverified, the one line on what it does, then who reviewed it and where it lands. */
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
            <Badge variant={!verified ? 'warn' : request.tier === 'community' ? 'muted' : 'success'}>
              {!verified ? m.unverified : request.tier ? catalogTierLabel(t, request.tier) : m.verified}
            </Badge>
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
  /** An older backend cannot hold the install to the reviewed commit: Install becomes "Install anyway". */
  confirmUnreviewed: boolean
  failed: boolean
  installing: boolean
  onInstall: () => void
  onInstallAnyway: () => void
  onRetry: () => void
  retryable: boolean
}

/** Cancel, then Install (one loader while it runs) or, after a failure, Retry — unless retrying cannot help. */
function InstallFooter({
  canInstall,
  confirmUnreviewed,
  failed,
  installing,
  onInstall,
  onInstallAnyway,
  onRetry,
  retryable
}: InstallFooterProps) {
  const { t } = useI18n()
  const m = t.settings.plugins.installModal

  return (
    <DialogFooter>
      <Button disabled={installing} onClick={closePluginInstallRequest} variant="outline">
        {t.common.cancel}
      </Button>
      {confirmUnreviewed && !installing ? (
        <Button onClick={onInstallAnyway}>{m.installAnyway}</Button>
      ) : failed ? (
        retryable && <Button onClick={onRetry}>{t.common.retry}</Button>
      ) : (
        <Button disabled={!canInstall} onClick={onInstall}>
          {installing && <Loader2 className="animate-spin" />}
          {installing ? m.installing : m.install}
        </Button>
      )}
    </DialogFooter>
  )
}

interface SourceMovedNoticeProps {
  copy: InstallModalCopy
  /** A failure on screen replaces the notice. */
  failure: null | string
  probe: null | ProbeResult
  request: PluginInstallRequest
  shown: boolean
}

/** The source moved past the commit the user reviewed: name the commit the fresh review holds the install to. */
function SourceMovedNotice({ copy, failure, probe, request, shown }: SourceMovedNoticeProps) {
  if (!shown || failure) {
    return null
  }

  return (
    <p className={cn(CAPTION, 'text-(--ui-text-secondary)')} role="status">
      {copy.sourceMoved(probe?.ok ? (reviewedCommit({ probe, request })?.slice(0, 12) ?? '') : '')}
    </p>
  )
}

/** Read the source again after a refusal. A catalog pick whose pin the backend did not read (native Desktop
 *  inspects the branch tip in Electron) also takes the catalog's current pin, or every Install would resend
 *  the one the catalog left; `repin` is that lookup, which Install waits on. */
function useReviewAgain(request: PluginInstallRequest, start: (repo: string) => void) {
  const [repin, setRepin] = useState<Promise<void>>(() => Promise.resolve())
  // Set once the source is read again: Install then waits until the new inspection AND pin are on screen.
  const [rereview, setRereview] = useState<'idle' | 'pending' | 'shown'>('idle')

  const reviewAgain = (current: ProbeResult) => {
    start(request.repo)
    const name = request.catalogName

    if (name && !current.catalogPin) {
      setRereview('pending')
      setRepin(
        lookupPluginCatalogEntry(name)
          .then(found => {
            const open = $pluginInstallRequest.get()

            if (found.ok && open?.catalogName === name) {
              openPluginInstallRequest({ ...open, sha: found.entry.sha })
            }
          })
          .finally(() => setRereview('shown'))
      )
    } else {
      setRereview('shown')
    }
  }

  return { repin, rereview, reviewAgain }
}

interface InstallNoticesProps {
  confirmUnreviewed: boolean
  copy: InstallModalCopy
  /** A failure on screen replaces the partial-install notice. */
  failure: null | string
  partial: null | string
  uncertain: boolean
}

/** What the last Install left: only the agent half in, an older backend's missing guarantee, or no answer. */
function InstallNotices({ confirmUnreviewed, copy, failure, partial, uncertain }: InstallNoticesProps) {
  return (
    <>
      {partial && !failure && (
        <p className={cn(CAPTION, 'text-destructive')} role="alert">
          {partial}
        </p>
      )}
      {confirmUnreviewed && (
        <p className={cn(CAPTION, 'text-foreground')} role="alert">
          {copy.legacyBackendConfirm}
        </p>
      )}
      {uncertain && (
        <p className={cn(CAPTION, 'text-(--ui-text-secondary)')} role="status">
          {copy.installUncertain}
        </p>
      )}
    </>
  )
}

interface InstallGate {
  force: boolean
  installing: boolean
  partial: null | string
  pinInvalid: boolean
  probe: null | ProbeResult
  rereview: 'idle' | 'pending' | 'shown'
  uncertain: boolean
  verified: boolean
}

/** An unverified repository installs only once its contents are on screen; a reviewed pick may start at once,
 *  except after the source moved: then nothing installs until the new inspection and pin are shown.
 *  After a partial install only Force (a replacement of both halves) may proceed. */
function installEnabled(gate: InstallGate): boolean {
  const { force, installing, partial, pinInvalid, probe, rereview, uncertain, verified } = gate
  const reviewed = rereview === 'idle' ? verified || Boolean(probe?.ok) : rereview === 'shown' && Boolean(probe?.ok)

  return !installing && !uncertain && !pinInvalid && (!partial || force) && reviewed
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

  const [targetProfile, setTargetProfile] = useState(() =>
    normalizeProfileKey(request.profile || activeProfile || profileScope)
  )

  // Without Electron's inspection (the browser-hosted app) the backend clones and reads the repository.
  const { probe, start, cancel, settled } = usePluginProbe(m.probeUnavailable, repo =>
    inspectAgentPlugin(requestGateway, { identifier: repo, catalogName: request.catalogName, profile: targetProfile })
  )

  const [advancedOpen, setAdvancedOpen] = useState(!verified)
  const [force, setForce] = useState(request.force ?? false)
  const [pinRef, setPinRef] = useState('')
  const [installError, setInstallError] = useState<null | string>(null)
  const [installUncertain, setInstallUncertain] = useState(false)
  // The source moved past the commit on screen: the dialog read it again and shows the new one for review.
  const [sourceMoved, setSourceMoved] = useState(false)
  // The backend predates `inspected_sha` and installed nothing: Install waits for an explicit "Install anyway".
  const [confirmUnreviewed, setConfirmUnreviewed] = useState(false)
  // The agent half installed and the desktop half could not follow it: only a replacement of both proceeds.
  const [partial, setPartial] = useState<null | string>(null)
  const { repin, rereview, reviewAgain } = useReviewAgain(request, start)

  useEffect(() => {
    start(request.repo)

    return cancel
  }, [cancel, request.repo, start])

  const targetInfo = profiles.find(profile => normalizeProfileKey(profile.name) === targetProfile)
  const targetLabel = profileLabel(targetInfo ?? { name: targetProfile })
  const pin = pinRef.trim().toLowerCase()
  const pinInvalid = pin !== '' && !COMMIT_SHA_RE.test(pin)
  const nothing = nothingInstallable(request, probe)

  // Hidden while a retry runs, so the loader is the only thing on screen.
  const failure = installing
    ? null
    : (installError ?? probeFailure(probe, m.probeUnavailable) ?? (nothing ? m.nothingToInstall : null))

  // Retrying reruns an install or a read that failed; it cannot add a capability this environment lacks.
  const retryable = Boolean(installError) || !(probe?.unavailable || nothing)
  const line = `${verified ? m.reviewedByNous : m.securityIntro} ${(enable ? m.installsAndEnables : m.installsOnly)(targetLabel)}`

  const install = async (withoutReviewedCommit = false) => {
    onInstallingChange(true)
    setInstallError(null)
    setSourceMoved(false)
    setConfirmUnreviewed(false)
    // The commit the user clicked on: a lookup finishing after the click never changes it.
    const chosen = $pluginInstallRequest.get() ?? request

    try {
      // A catalog pick may click Install while the inspection is still running: wait for it under the loader.
      const [detected] = await Promise.all([settled(), repin])

      if (!detected?.ok) {
        return
      }

      const outcome = await runPluginInstall(
        {
          request: chosen,
          probe: detected,
          profile: targetProfile,
          enable,
          force,
          ref: pin || undefined,
          remote,
          withoutReviewedCommit
        },
        { copy: m, navigate, requestGateway }
      )

      const settle: Record<InstallOutcome['status'], () => void> = {
        done: () => finishInstall(request, navigate),
        uncertain: () => setInstallUncertain(true),
        // Never resend the commit the source left: drop that inspection and read the source again, so
        // Install waits on the new commit being on screen.
        stale: () => {
          setSourceMoved(true)
          reviewAgain(detected)
        },
        // Never install the desktop half beside it at another commit: show the source as it is now, and
        // Install waits for Force reinstall, which replaces both halves with that version.
        partial: () => {
          setPartial((outcome as Extract<InstallOutcome, { status: 'partial' }>).error)
          setAdvancedOpen(true)
          reviewAgain(detected)
        },
        needsConfirm: () => setConfirmUnreviewed(true),
        failed: () => setInstallError((outcome as Extract<InstallOutcome, { status: 'failed' }>).error)
      }

      settle[outcome.status]()
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

  const canInstall = installEnabled({
    force,
    installing,
    partial,
    pinInvalid,
    probe,
    rereview,
    uncertain: installUncertain,
    verified
  })

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
            offerForce={Boolean(partial)}
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
        <SourceMovedNotice copy={m} failure={failure} probe={probe} request={request} shown={sourceMoved} />
        <InstallNotices
          confirmUnreviewed={confirmUnreviewed}
          copy={m}
          failure={failure}
          partial={partial}
          uncertain={installUncertain}
        />
      </div>

      <InstallFooter
        canInstall={canInstall}
        confirmUnreviewed={confirmUnreviewed}
        failed={Boolean(failure)}
        installing={installing}
        onInstall={() => void install()}
        onInstallAnyway={() => void install(true)}
        onRetry={retry}
        retryable={retryable}
      />
    </>
  )
}
