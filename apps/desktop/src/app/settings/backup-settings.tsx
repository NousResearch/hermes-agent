import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useLocation, useNavigate } from 'react-router'

import { Button } from '@/components/ui/button'
import type { DesktopCloudAgent, DesktopCloudOrg } from '@/global'
import { useI18n } from '@/i18n'
import type { BackupTransferTranslations } from '@/i18n/types_backup_transfer'
import {
  cloudEndpoint,
  createUndoLedger,
  transferAgent,
  type TransferEndpoint,
  transferErrorMessage,
  type TransferProgress,
  type UndoPoint,
  undoTransfer
} from '@/lib/agent-transfer'
import { openExternalLink } from '@/lib/external-link'
import {
  backup,
  cloudLogin,
  cloudStatus,
  connectCloudAgent,
  discoverCloudAgents,
  PORTAL_AGENTS_URL,
  transferRequest as request
} from '@/lib/hermes-cloud-agents'
import { transferAvailable } from '@/lib/hermes-cloud-host'
import { hostHas } from '@/lib/host-capabilities'
import { Archive, CloudUpload } from '@/lib/icons'
import { $activeConnectionId } from '@/store/connections'
import { notifyError } from '@/store/notifications'
import { ensureGatewayAgent } from '@/store/profile'
import { $settingsScopeProfile } from '@/store/settings-scope'

import { CLOUD_AGENT_PARAM } from './backup-palette'
import { AgentList, ConfirmPanel, OrgPicker, Outcome, ProgressSteps } from './backup-transfer-views'
import { SettingsContent, SettingsSection } from './primitives'
import { WebappOauthPasteField } from './webapp-oauth-paste-field'

type Copy = BackupTransferTranslations

const PROVIDER_ACCOUNTS_ROUTE = '/settings?tab=providers'

type Phase =
  | { kind: 'signin' }
  | { kind: 'agents' }
  | { kind: 'confirm'; agent: DesktopCloudAgent }
  | { kind: 'working'; agent: DesktopCloudAgent; progress: TransferProgress }
  | {
      kind: 'done'
      agent: DesktopCloudAgent
      botsOff: string[]
      gatewayDown: boolean
      target: TransferEndpoint
      undoArchive: string
    }
  | { kind: 'undoing'; agent: DesktopCloudAgent; line?: string }
  | { kind: 'undone'; agent: DesktopCloudAgent }
  | { kind: 'error'; message: string; retry: () => void; undo?: () => void }

/** `hermes backup` of the whole Hermes home, saved to a file the user picks. */
function BackupCard({ copy, owner }: { copy: Copy; owner: TransferEndpoint }) {
  const [busy, setBusy] = useState(false)
  const [lastLine, setLastLine] = useState('')
  const [result, setResult] = useState<{ archive: string; saved: boolean } | null>(null)

  const run = async () => {
    setBusy(true)
    setResult(null)

    try {
      setResult(await backup(owner, setLastLine))
    } catch (error) {
      notifyError(error, copy.backupFailed)
    } finally {
      setBusy(false)
      setLastLine('')
    }
  }

  return (
    <SettingsSection icon={Archive} title={copy.backupTitle}>
      <div className="flex items-start gap-3">
        <p className="min-w-0 flex-1 text-xs text-muted-foreground">{copy.backupHelp}</p>
        <Button disabled={busy} onClick={() => void run()} variant="secondary">
          {busy ? copy.backupRunning : copy.backup}
        </Button>
      </div>
      {busy && lastLine && <p className="mt-2 truncate font-mono text-xs text-muted-foreground">{lastLine}</p>}
      {result && (
        <p className="mt-2 text-xs text-muted-foreground">
          {result.saved ? copy.backupSaved : copy.backupKept(result.archive)}
        </p>
      )}
    </SettingsSection>
  )
}

/** Nous sign-in and agent discovery, with the team picker a multi-team account needs. */
function useCloudAgents(onSignedOut: () => void, onError: (message: string, retry: () => void) => void) {
  const [agents, setAgents] = useState<DesktopCloudAgent[] | null>(null)
  const [org, setOrg] = useState<string | undefined>()
  const [orgs, setOrgs] = useState<DesktopCloudOrg[] | null>(null)

  const discover = useCallback(
    async (orgId?: string) => {
      try {
        if (!(await cloudStatus()).signedIn) {
          onSignedOut()

          return
        }

        const result = await discoverCloudAgents(orgId)

        if (result.needsOrgSelection) {
          setOrgs(result.orgs)
        } else {
          setOrgs(null)
          setAgents(result.agents)
          setOrg(result.org?.slug ?? result.org?.id ?? orgId)
        }
      } catch (error) {
        onError(transferErrorMessage(error), () => void discover(orgId))
      }
    },
    [onError, onSignedOut]
  )

  useEffect(() => {
    void discover()
  }, [discover])

  return { agents, discover, org, orgs }
}

interface TransferCardProps {
  copy: Copy
  hint: string
  owner: TransferEndpoint
}

/** The done line, plus which bots came over switched off (when any did); a gateway that did not
 *  start again replaces the done line: the files arrived but the agent is not running. */
function doneBody(copy: Copy, agent: string, botsOff: string[], gatewayDown: boolean, locale: string): string {
  const done = gatewayDown ? copy.gatewayDown(agent) : copy.doneHelp(agent)

  if (!botsOff.length) {
    return done
  }

  const bots = new Intl.ListFormat(locale, { style: 'long', type: 'conjunction' }).format(botsOff)

  return `${done} ${copy.botsOff(bots, agent)}`
}

function TransferCard({ copy, hint, owner }: TransferCardProps) {
  const navigate = useNavigate()
  const { locale } = useI18n()
  const [phase, setPhase] = useState<Phase>({ kind: 'agents' })
  const abort = useRef<AbortController | null>(null)
  const undoPoints = useRef(createUndoLedger())
  const signedOut = useCallback(() => setPhase({ kind: 'signin' }), [])
  const failed = useCallback((message: string, retry: () => void) => setPhase({ kind: 'error', message, retry }), [])
  const cloud = useCloudAgents(signedOut, failed)
  const openPortal = () => openExternalLink(PORTAL_AGENTS_URL)

  useEffect(() => () => abort.current?.abort(), [])

  const backToAgents = () => {
    setPhase({ kind: 'agents' })
    void cloud.discover(cloud.org)
  }

  const run = async (agent: DesktopCloudAgent) => {
    const controller = new AbortController()
    abort.current = controller

    const show = (progress: TransferProgress) =>
      !controller.signal.aborted && setPhase({ agent, kind: 'working', progress })

    try {
      show({ phase: 'backup' })
      const connected = cloudEndpoint(await connectCloudAgent(agent, cloud.org))

      const result = await transferAgent(request, owner, connected, show, {
        onUndoArchive: archive => undoPoints.current.record(agent.id, { archive, target: connected }),
        signal: controller.signal
      })

      const point = undoPoints.current.get(agent.id) ?? { archive: result.undoArchive, target: connected }
      setPhase({
        agent,
        botsOff: result.messagingOff,
        gatewayDown: result.gatewayDown,
        kind: 'done',
        target: point.target,
        undoArchive: point.archive
      })
    } catch (error) {
      if (controller.signal.aborted) {
        return
      }

      const point = undoPoints.current.get(agent.id)

      setPhase({
        kind: 'error',
        message: point
          ? `${transferErrorMessage(error)}\n\n${copy.partialHelp(agent.name)}`
          : transferErrorMessage(error),
        retry: () => void run(agent),
        ...(point ? { undo: () => void revert(agent, point) } : {})
      })
    }
  }

  const revert = async (agent: DesktopCloudAgent, point: UndoPoint) => {
    setPhase({ agent, kind: 'undoing' })

    try {
      await undoTransfer(request, point.target, point.archive, line => setPhase({ agent, kind: 'undoing', line }))
      undoPoints.current.clear(agent.id)
      setPhase({ agent, kind: 'undone' })
    } catch (error) {
      setPhase({
        kind: 'error',
        message: transferErrorMessage(error),
        retry: () => void revert(agent, point)
      })
    }
  }

  // Only offered before the restore is requested, so nothing on the cloud agent has changed.
  const cancel = () => {
    abort.current?.abort()
    backToAgents()
  }

  const signIn = () =>
    hostHas('cloudPortalLogin') ? void cloudLogin().then(backToAgents) : navigate(PROVIDER_ACCOUNTS_ROUTE)

  const views: Record<Phase['kind'], () => React.ReactNode> = {
    signin: () => (
      <div className="flex flex-col items-start gap-2">
        <p className="text-sm font-medium">{copy.signInTitle}</p>
        <p className="text-xs text-muted-foreground">
          {hostHas('cloudPortalLogin') ? copy.signInHelp : copy.signInWebappHelp}
        </p>
        <Button onClick={signIn}>{hostHas('cloudPortalLogin') ? copy.signIn : copy.openProviders}</Button>
      </div>
    ),
    agents: () =>
      cloud.orgs ? (
        <OrgPicker copy={copy} onPick={org => void cloud.discover(org)} orgs={cloud.orgs} />
      ) : (
        <AgentList
          agents={cloud.agents}
          copy={copy}
          hint={hint}
          onOpenPortal={openPortal}
          onPick={agent => setPhase({ agent, kind: 'confirm' })}
        />
      ),
    confirm: () =>
      phase.kind === 'confirm' && (
        <ConfirmPanel
          agent={phase.agent.name}
          copy={copy}
          onCancel={() => setPhase({ kind: 'agents' })}
          onReplace={() => {
            // A new flow; a Retry keeps the flow's first undo point.
            undoPoints.current.clear(phase.agent.id)
            void run(phase.agent)
          }}
        />
      ),
    working: () =>
      phase.kind === 'working' && (
        <ProgressSteps
          agent={phase.agent.name}
          copy={copy}
          fraction={phase.progress.fraction}
          line={phase.progress.line}
          onCancel={phase.progress.phase === 'restore' ? undefined : cancel}
          phase={phase.progress.phase}
        />
      ),
    done: () =>
      phase.kind === 'done' && (
        <Outcome
          actions={[
            {
              label: copy.undo,
              onClick: () => void revert(phase.agent, { archive: phase.undoArchive, target: phase.target })
            },
            {
              label: copy.openAgent,
              onClick: () =>
                void ensureGatewayAgent(phase.target.connectionId, phase.target.profile).then(() => navigate('/')),
              primary: true
            }
          ]}
          body={doneBody(copy, phase.agent.name, phase.botsOff, phase.gatewayDown, locale)}
          title={copy.doneTitle}
        />
      ),
    undoing: () =>
      phase.kind === 'undoing' && <Outcome actions={[]} body={phase.line ?? copy.undoing} title={copy.undoing} />,
    undone: () =>
      phase.kind === 'undone' && (
        <Outcome
          actions={[{ label: copy.startAgain, onClick: backToAgents, primary: true }]}
          body={copy.undoneHelp(phase.agent.name)}
          title={copy.undoneTitle}
        />
      ),
    error: () =>
      phase.kind === 'error' && (
        <Outcome
          actions={[
            { label: copy.cancel, onClick: backToAgents },
            ...(phase.undo ? [{ label: copy.undo, onClick: phase.undo }] : []),
            { label: copy.retry, onClick: phase.retry, primary: true }
          ]}
          body={phase.message}
          title={copy.errorTitle}
          tone="error"
        />
      )
  }

  return (
    <SettingsSection icon={CloudUpload} title={copy.transferTitle}>
      <p className="mb-3 text-xs text-muted-foreground">{copy.transferHelp}</p>
      {views[phase.kind]()}
      {phase.kind === 'working' && <WebappOauthPasteField />}
    </SettingsSection>
  )
}

/**
 * Settings ▸ Backup & Transfer. Acts on the profile Settings is editing on the active connection;
 * the mount site keys it by that owner, so a scope switch remounts it and a flow never splits.
 */
export function BackupSettings() {
  const { t } = useI18n()
  const copy = t.settings.backup
  const profile = useStore($settingsScopeProfile)
  const connectionId = useStore($activeConnectionId)
  const owner = useMemo(() => ({ connectionId, profile }), [connectionId, profile])
  const { search } = useLocation()
  const hint = useMemo(() => (new URLSearchParams(search).get(CLOUD_AGENT_PARAM) ?? '').trim(), [search])
  const transfer = transferAvailable()

  return (
    <SettingsContent>
      {transfer && <p className="mb-4 pt-2 text-xs text-muted-foreground">{copy.intro}</p>}
      <BackupCard copy={copy} owner={owner} />
      {transfer && <TransferCard copy={copy} hint={hint} owner={owner} />}
    </SettingsContent>
  )
}
