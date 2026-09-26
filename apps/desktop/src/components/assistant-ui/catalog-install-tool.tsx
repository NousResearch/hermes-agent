import type { ToolCallMessagePartProps } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import { useMemo, useRef, useState } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import { CatalogAdvancedDialog } from '@/components/assistant-ui/catalog-advanced-dialog'
import {
  connectionRequestOwnsPart,
  CONNECTOR_CARD_PHASES,
  useConnectorFocusHandoff
} from '@/components/assistant-ui/connector-tool'
import { ToolFallback } from '@/components/assistant-ui/tool/fallback'
import { WIDGET_SHELL_CLASS } from '@/components/chat/widget-shell'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { useI18n } from '@/i18n'
import { Book, Loader2, Plug } from '@/lib/icons'
import { cn } from '@/lib/utils'
import {
  type CatalogEntry,
  type ConnectionRequest,
  type ConnectionTarget,
  continueConnectionRequest,
  respondToConnectionRequest,
  sessionConnectionRequest
} from '@/store/connection-request'
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

/** `manage_catalog`: the host's catalog-install card. The model named ids; every word on a row is the
 *  host's resolution of that id. The card lives on the tool row that opened the operation only. */
export function CatalogInstallTool(props: ToolCallMessagePartProps) {
  const { t } = useI18n()
  const view = useSessionView()
  const runtimeId = useStore(view.$runtimeId)
  const $request = useMemo(() => sessionConnectionRequest(runtimeId), [runtimeId])
  const request = useStore($request)

  if (request && connectionRequestOwnsPart(props, request)) {
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
  const unresolved = rows.some(target => !CONNECTOR_CARD_PHASES[target.state].resolved)
  const profile = rows[0]?.catalog.targetProfile ?? 'default'

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
        <span className={cn(CAPTION, 'text-(--ui-text-tertiary)')}>
          {t.assistant.catalogInstall.targetProfile(profileLabel({ name: profile }))}
        </span>
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
  // The operation's seq when an answer was sent; the verbs stay held until a newer frame answers it,
  // so a second click cannot send the answer twice.
  const [sentAtSeq, setSentAtSeq] = useState<null | number>(null)
  const sending = sentAtSeq !== null && request.seq <= sentAtSeq
  const Glyph = KIND_GLYPH[target.kind]

  const answer = async (status: 'approved' | 'skipped', env: Record<string, string> | null = null) => {
    setSentAtSeq(request.seq)

    try {
      const sent = await respondToConnectionRequest(request, { targets: [{ env, name: target.name, status }] })

      if (!sent) {
        setSentAtSeq(null)
      }
    } catch (error) {
      notifyError(error, copy.sendFailed)
      setSentAtSeq(null)
    }
  }

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
          onSkip={() => void answer('skipped')}
          retryLabel={t.connectors.retry}
          sending={sending}
          settled={request.settled}
          skippedLabel={t.connectors.skipped}
          target={target}
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
  onSkip: () => void
  retryLabel: string
  sending: boolean
  skippedLabel: string
  settled: boolean
  target: CatalogTarget
  toolCount: (count: number) => string
}

/** The third line: the verbs while the row waits on the user, else what happened to it. */
function RowOutcome({
  copy,
  onAdvanced,
  onInstall,
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
        {settled ? null : (
          <Button
            className="h-6 px-1.5 text-(--ui-text-tertiary)"
            disabled={sending}
            loading={sending}
            onClick={onInstall}
            size="xs"
            variant="ghost"
          >
            {retryLabel}
          </Button>
        )}
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
        <p className={cn(CAPTION, 'text-(--ui-text-tertiary) wrap-anywhere')}>
          {target.state === 'initiated' && target.detail ? target.detail : copy.installing}
        </p>
        <Progress animated aria-label={copy.installing} className="h-0.5 bg-primary/15" indeterminate />
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
