import type { ToolCallMessagePartProps } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import { useMemo, useRef, useState } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import { CatalogAdvancedDialog } from '@/components/assistant-ui/catalog-advanced-dialog'
import { connectionRequestOwnsPart, useConnectorFocusHandoff } from '@/components/assistant-ui/connector-tool'
import { ToolFallback } from '@/components/assistant-ui/tool/fallback'
import { WIDGET_SHELL_CLASS } from '@/components/chat/widget-shell'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { useI18n } from '@/i18n'
import { Book, Loader2, Plug } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { type AgentPluginInstallResult, installAgentPlugin } from '@/store/agent-plugins'
import {
  type CatalogEntry,
  connectionOpOf,
  connectionOwnerFor,
  type ConnectionRequest,
  type ConnectionTarget,
  continueConnectionRequest,
  respondToConnectionRequest,
  toolConnectionRequest
} from '@/store/connection-request'
import { requestGatewayForAgent } from '@/store/gateway'
import { notifyError } from '@/store/notifications'
import { profileLabel } from '@/store/profile'

type CatalogCopy = ReturnType<typeof useI18n>['t']['assistant']['catalogInstall']
type CatalogTarget = ConnectionTarget & { catalog: CatalogEntry; kind: 'plugin' | 'skill' }

const SHELL_CLASS = `${WIDGET_SHELL_CLASS} text-[length:var(--conversation-text-font-size)] text-(--ui-text-primary)`
const CAPTION = 'text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height)'
const PILL = 'inline-flex items-center rounded-full px-1.5 py-0.5 text-[0.62rem] font-medium leading-[0.93rem]'

const KIND_GLYPH = { plugin: Plug, skill: Book } as const

function platformName(platform: string): string {
  switch (platform.toLowerCase()) {
    case 'darwin':

    case 'macos':
      return 'macOS'

    case 'linux':
      return 'Linux'

    case 'windows':
      return 'Windows'

    default:
      return platform
  }
}

const isCatalogTarget = (target: ConnectionTarget): target is CatalogTarget => Boolean(target.catalog)

/** The row a successful settled Try again draws: the backend's install result for that retry, in the
 *  fields the runner's installed row carries (`tools/connectors/catalog.py::_installed_row`). */
function retriedRow(target: CatalogTarget, result: AgentPluginInstallResult): CatalogTarget {
  const servers = result.live.mcpServers

  return {
    ...target,
    catalog: {
      ...target.catalog,
      alreadyInstalled: false,
      enabled: result.enabled ?? null,
      missingEnv: result.missingEnv ?? [],
      serverErrors: servers.filter(server => !server.connected).map(({ error, name }) => ({ error: error ?? '', name })),
      skill: result.skillIds?.[0] ?? null
    },
    detail: '',
    state: 'connected',
    tools: servers.filter(server => server.connected).flatMap(server => server.tools)
  }
}

/** `manage_catalog`: the host's catalog-install card. The model named ids; every word on a row is the
 *  host's resolution of that id. The card lives on the tool row that opened the operation only. */
export function CatalogInstallTool(props: ToolCallMessagePartProps) {
  const { t } = useI18n()
  const view = useSessionView()
  const runtimeId = useStore(view.$runtimeId)
  const opId = connectionOpOf(props.args)
  const running = props.result === undefined
  // Only an install opens an operation; a search row that reuses an install's call id draws no card.
  const opens = opId !== null || props.args.action === 'install'

  const $request = useMemo(
    () => toolConnectionRequest(runtimeId, props.toolCallId, opId, running),
    [opId, props.toolCallId, running, runtimeId]
  )

  const request = useStore($request)

  if (opens && request && connectionRequestOwnsPart(props, request)) {
    return <CatalogInstallCard request={request} />
  }

  // `tool.start` arrives before `connection.request`; a finished call with no card is plain history.
  if (props.result !== undefined || props.status?.type !== 'running') {
    return <ToolFallback {...props} />
  }

  return (
    <div className={cn(SHELL_CLASS, 'my-1.5 flex max-w-lg items-center gap-2')} data-slot="connector-card">
      <Loader2 aria-hidden className="size-4 animate-spin text-(--ui-text-tertiary) motion-reduce:animate-none" />
      <span className="text-(--ui-text-tertiary)">{t.assistant.catalogInstall.preparing}</span>
    </div>
  )
}

export function CatalogInstallCard({ request }: { request: ConnectionRequest }) {
  const { t } = useI18n()
  const cardRef = useRef<HTMLDivElement | null>(null)
  const rows = request.targets.filter(isCatalogTarget)
  const unresolved = rows.some(target => !target.catalog.resolved)
  const profile = rows[0]?.catalog.targetProfile

  useConnectorFocusHandoff(request.targets, cardRef)

  return (
    <div className="my-2 grid min-w-0 max-w-lg gap-4.5" data-catalog-card data-connector-offer ref={cardRef}>
      {rows.map(target => (
        <CatalogRow key={target.name} request={request} target={target} />
      ))}
      <div className="flex min-w-0 flex-wrap items-center gap-x-2.5 gap-y-1 px-3.5">
        {unresolved && !request.settled ? (
          <Button onClick={() => void continueConnectionRequest(request)} size="xs" variant="textStrong">
            {t.common.continue}
          </Button>
        ) : null}
        {profile ? (
          <span className={cn(CAPTION, 'text-(--ui-text-tertiary)')}>
            {t.assistant.catalogInstall.targetProfile(profileLabel({ name: profile }))}
          </span>
        ) : null}
      </div>
    </div>
  )
}

interface CatalogRowProps {
  request: ConnectionRequest
  target: CatalogTarget
}

/** One catalog item: what it is, what the agent can do with it, and one decision. */
export function CatalogRow({ request, target }: CatalogRowProps) {
  const { t } = useI18n()
  const copy = t.assistant.catalogInstall
  const { catalog } = target
  const [advancedOpen, setAdvancedOpen] = useState(false)
  // The operation and seq an answer was sent at; the verbs stay held until a newer frame of that
  // operation answers it, so a second click cannot send the answer twice. Keyed by op so a seq from
  // another operation can never hold this row.
  const [sentAt, setSentAt] = useState<null | { opId: string; seq: number }>(null)
  // Try again on a failed plugin row after the operation settled: a fresh host install (which enables a
  // plugin already on disk). The settled operation is frozen, so the row draws that install's result.
  const [retry, setRetry] = useState<null | { row: CatalogTarget; status: 'done' } | { status: 'running' }>(null)
  const sending = (sentAt?.opId === request.opId && request.seq <= sentAt.seq) || retry?.status === 'running'
  const Glyph = KIND_GLYPH[target.kind]

  const answer = async (status: 'approved' | 'skipped', env: Record<string, string> | null = null) => {
    setSentAt({ opId: request.opId, seq: request.seq })

    try {
      const sent = await respondToConnectionRequest(request, { targets: [{ env, name: target.name, status }] })

      if (!sent) {
        setSentAt(null)
      }
    } catch (error) {
      notifyError(error, copy.sendFailed)
      setSentAt(null)
    }
  }

  const retrySettled = async () => {
    setRetry({ status: 'running' })
    const owner = request.sessionId ? await connectionOwnerFor(request.sessionId, 'plugins.manage') : null

    if (!owner) {
      setRetry({ row: { ...target, detail: copy.sendFailed }, status: 'done' })

      return
    }

    // The choices the user approved on this row, so the retry installs what they asked for.
    const approved = catalog.approved

    const result = await installAgentPlugin(
      (method, params, timeoutMs) => requestGatewayForAgent(owner.connectionId, owner.profile, method, params, timeoutMs),
      {
        catalogName: target.name,
        enable: approved?.enable,
        force: approved?.force,
        identifier: '',
        profile: catalog.targetProfile,
        ref: approved?.ref ?? undefined
      }
    )

    const row = result.ok ? retriedRow(target, result) : { ...target, detail: result.error || target.detail }
    setRetry({ row, status: 'done' })
  }

  const shown = retry?.status === 'done' ? retry.row : target

  const onRetry = !request.settled
    ? () => void answer('approved')
    : target.kind === 'plugin'
      ? () => void retrySettled()
      : undefined

  return (
    <div className={cn(SHELL_CLASS, 'grid min-w-0 gap-1.5')} data-connector-row={target.name} tabIndex={-1}>
      <div className="flex min-w-0 items-start gap-3">
        <span
          aria-hidden
          className="grid size-10 shrink-0 place-items-center rounded-xl bg-(--ui-bg-quaternary) text-(--ui-text-secondary)"
        >
          <Glyph className="size-5" stroke={1.75} />
        </span>
        <div className="grid min-w-0 flex-1 gap-0.5">
          <div className="flex min-w-0 flex-wrap items-baseline gap-x-1.5 gap-y-1">
            <span className="font-medium leading-4.5 wrap-anywhere">{catalog.display}</span>
            <span className={cn(PILL, 'bg-(--ui-bg-quaternary) text-muted-foreground')}>{copy.kind[target.kind]}</span>
            {catalog.tier ? (
              <span className={cn(PILL, 'bg-(--ui-bg-quaternary) text-muted-foreground')}>
                {copy.tier[catalog.tier]}
              </span>
            ) : null}
            {catalog.platforms.map(platform => (
              <span className={cn(PILL, 'bg-primary/8 text-primary')} key={platform}>
                {platformName(platform)}
              </span>
            ))}
          </div>
          {catalog.description ? (
            <p className="leading-4.5 text-(--ui-text-secondary) wrap-anywhere">{catalog.description}</p>
          ) : null}
        </div>
      </div>

      <div className="min-w-0 pl-13">
        <RowOutcome
          copy={copy}
          onAdvanced={() => setAdvancedOpen(true)}
          onInstall={() => void answer('approved')}
          onRetry={onRetry}
          onSkip={() => void answer('skipped')}
          retryLabel={t.connectors.retry}
          sending={sending}
          settled={request.settled}
          skippedLabel={t.connectors.skipped}
          target={shown}
          toolCount={t.assistant.mcpSetup.toolCount}
        />
      </div>

      <CatalogAdvancedDialog
        entry={catalog}
        fields={target.requiredEnv}
        kind={target.kind}
        onCancel={() => setAdvancedOpen(false)}
        onInstall={env => {
          setAdvancedOpen(false)
          void answer('approved', env)
        }}
        open={advancedOpen}
      />
    </div>
  )
}

interface RowOutcomeProps {
  copy: CatalogCopy
  onAdvanced: () => void
  onInstall: () => void
  /** Try again on a failed row; absent when nothing can run it again. */
  onRetry?: () => void
  onSkip: () => void
  retryLabel: string
  sending: boolean
  skippedLabel: string
  settled: boolean
  target: CatalogTarget
  toolCount: (count: number) => string
}

/** What an installed row still needs, worded from the backend's facts. */
function installedNotes(catalog: CatalogEntry, copy: CatalogCopy): string[] {
  return [
    ...catalog.serverErrors.map(({ error, name }) => copy.serverNotConnected(name, error)),
    ...(catalog.enabled === false ? [copy.notEnabled] : []),
    ...(catalog.missingEnv.length > 0 ? [copy.missingEnv(catalog.missingEnv.join(', '))] : []),
    ...(catalog.alreadyInstalled ? [copy.alreadyInstalled] : [])
  ]
}

/** The third line: the verbs while the row waits on the user, else what happened to it. */
function RowOutcome({
  copy,
  onAdvanced,
  onInstall,
  onRetry,
  onSkip,
  retryLabel,
  sending,
  settled,
  skippedLabel,
  target,
  toolCount
}: RowOutcomeProps) {
  const [namesOpen, setNamesOpen] = useState(false)

  if (target.state === 'connected') {
    const { skill } = target.catalog
    const notes = installedNotes(target.catalog, copy)

    const parts = [
      copy.installed,
      target.tools.length > 0 ? toolCount(target.tools.length) : '',
      skill ? copy.skill(skill) : ''
    ]

    return (
      <div className="grid min-w-0 gap-1">
        <p className={cn(CAPTION, 'text-emerald-600 dark:text-emerald-400')} role="status">
          {parts.filter(Boolean).join(' · ')}
          {target.tools.length > 0 ? (
            <>
              {' · '}
              <button
                aria-expanded={namesOpen}
                className="underline-offset-2 hover:underline"
                onClick={() => setNamesOpen(open => !open)}
                type="button"
              >
                {namesOpen ? copy.hideNames : copy.showNames}
              </button>
            </>
          ) : null}
        </p>
        {notes.length > 0 ? (
          <p className={cn(CAPTION, 'text-(--ui-text-tertiary) wrap-anywhere')}>{notes.join(' · ')}</p>
        ) : null}
        {namesOpen ? (
          <ul className="flex min-w-0 flex-wrap gap-x-3 gap-y-0.5 font-mono text-[0.6875rem] leading-4 text-(--ui-text-secondary)">
            {target.tools.map(tool => (
              <li className="min-w-0 break-all" key={tool}>
                {tool}
              </li>
            ))}
          </ul>
        ) : null}
      </div>
    )
  }

  if (target.state === 'failed' || target.state === 'expired') {
    return (
      <div className="flex min-w-0 flex-wrap items-center gap-x-2.5 gap-y-1">
        <p className={cn(CAPTION, 'min-w-0 text-destructive wrap-anywhere')} role="status">
          {target.detail ? `${copy.failed} · ${target.detail}` : copy.failed}
        </p>
        {onRetry ? (
          <Button
            className="h-6 px-1.5 text-(--ui-text-tertiary)"
            disabled={sending}
            loading={sending}
            onClick={onRetry}
            size="xs"
            variant="ghost"
          >
            {retryLabel}
          </Button>
        ) : null}
      </div>
    )
  }

  if (target.state === 'skipped') {
    return (
      <p className={cn(CAPTION, 'text-(--ui-text-tertiary)')} role="status">
        {skippedLabel}
      </p>
    )
  }

  if (settled) {
    return <p className={cn(CAPTION, 'text-(--ui-text-tertiary)')}>{copy.notInstalled}</p>
  }

  if (target.state === 'initiated' || sending) {
    return (
      <div className="grid min-w-0 gap-1.5" role="status">
        <p className={cn(CAPTION, 'text-(--ui-text-tertiary)')}>{copy.installing}</p>
        <Progress animated aria-label={copy.installing} className="h-0.5 bg-primary/15" indeterminate />
        {target.state === 'initiated' && target.catalog.phase ? (
          <p className={cn(CAPTION, 'text-(--ui-text-quaternary)')}>{copy.phase[target.catalog.phase]}</p>
        ) : null}
      </div>
    )
  }

  return (
    <div className="flex min-w-0 flex-wrap items-center gap-x-2.5 gap-y-1">
      <span className="inline-flex h-6 items-stretch overflow-hidden rounded-md border border-primary/25 bg-primary/10 text-primary">
        <Button
          className="h-full rounded-none px-2 text-xs font-medium text-primary hover:bg-primary/15 hover:text-primary"
          onClick={onInstall}
          size="xs"
          variant="ghost"
        >
          {copy.install}
        </Button>
      </span>
      <Button className="h-6 px-1.5 text-(--ui-text-tertiary)" onClick={onAdvanced} size="xs" variant="ghost">
        {copy.advanced}
      </Button>
      <Button className="h-6 px-1.5 text-(--ui-text-tertiary)" onClick={onSkip} size="xs" variant="ghost">
        {copy.skip}
      </Button>
    </div>
  )
}
