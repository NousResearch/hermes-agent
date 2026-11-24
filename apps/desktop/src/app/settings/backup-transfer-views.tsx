import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import type { DesktopCloudAgent, DesktopCloudOrg } from '@/global'
import type { BackupTransferTranslations } from '@/i18n/types_backup_transfer'
import type { TransferPhase } from '@/lib/agent-transfer'
import { Check, Cloud, Loader2, Plus } from '@/lib/icons'
import { cn } from '@/lib/utils'

import { Pill } from './primitives'

type Copy = BackupTransferTranslations

const ROW = 'flex items-center gap-3 rounded-lg border border-(--ui-stroke-tertiary) px-4 py-3 text-start text-sm'

export const agentReady = (agent: DesktopCloudAgent) =>
  Boolean(agent.dashboardUrl) && agent.dashboardGatewayState !== 'down'

interface AgentRowProps {
  agent: DesktopCloudAgent
  copy: Copy
  onPick: () => void
  preselected: boolean
}

function AgentRow({ agent, copy, onPick, preselected }: AgentRowProps) {
  return (
    <button
      aria-pressed={preselected}
      className={cn(
        ROW,
        'transition-colors hover:bg-(--chrome-action-hover) disabled:cursor-not-allowed disabled:opacity-50',
        preselected && 'border-(--ui-accent)'
      )}
      disabled={!agentReady(agent)}
      onClick={onPick}
      type="button"
    >
      <Cloud className="size-4 shrink-0 text-muted-foreground" />
      <span className="flex-1 truncate font-medium">{agent.name}</span>
      {!agent.dashboardUrl && <Pill>{copy.provisioning}</Pill>}
      {agent.dashboardUrl && agent.dashboardGatewayState === 'down' && <Pill tone="warn">{copy.down}</Pill>}
    </button>
  )
}

/** Creating a cloud agent from here waits on NousResearch/hermes-portal#2160; until then the row
 *  opens the portal, where one click does it. */
function NewAgentRow({ copy, onOpenPortal }: { copy: Copy; onOpenPortal: () => void }) {
  return (
    <button
      className={cn(ROW, 'border-dashed transition-colors hover:bg-(--chrome-action-hover)')}
      onClick={onOpenPortal}
      type="button"
    >
      <Plus className="size-4 shrink-0 text-muted-foreground" />
      <span className="min-w-0 flex-1">
        <span className="block font-medium">{copy.newAgentTitle}</span>
        <span className="block text-xs text-muted-foreground">{copy.newAgentHelp}</span>
      </span>
    </button>
  )
}

export interface AgentListProps {
  agents: DesktopCloudAgent[] | null
  copy: Copy
  hint: string
  onOpenPortal: () => void
  onPick: (agent: DesktopCloudAgent) => void
}

export function AgentList({ agents, copy, hint, onOpenPortal, onPick }: AgentListProps) {
  if (agents === null) {
    return <Loader2 className="my-4 size-4 animate-spin text-muted-foreground" />
  }

  if (agents.length === 0) {
    return (
      <div className="flex flex-col items-start gap-2 py-2">
        <p className="text-sm font-medium">{copy.noAgentsTitle}</p>
        <p className="text-xs text-muted-foreground">{copy.noAgentsHelp}</p>
        <Button onClick={onOpenPortal} variant="secondary">
          {copy.openPortal}
        </Button>
      </div>
    )
  }

  return (
    <div className="grid gap-2">
      <p className="text-xs text-muted-foreground">{copy.chooseAgent}</p>
      {agents.map(agent => (
        <AgentRow
          agent={agent}
          copy={copy}
          key={agent.id}
          onPick={() => onPick(agent)}
          preselected={Boolean(hint) && agent.name === hint}
        />
      ))}
      <NewAgentRow copy={copy} onOpenPortal={onOpenPortal} />
    </div>
  )
}

export function OrgPicker({
  copy,
  onPick,
  orgs
}: {
  copy: Copy
  onPick: (org: string) => void
  orgs: DesktopCloudOrg[]
}) {
  return (
    <div className="grid gap-2">
      <p className="text-xs text-muted-foreground">{copy.pickOrg}</p>
      <div className="flex flex-wrap gap-2">
        {orgs.map(org => (
          <Button key={org.id} onClick={() => onPick(org.slug ?? org.id)} variant="secondary">
            {org.name}
          </Button>
        ))}
      </div>
    </div>
  )
}

function PlainList({ items, title }: { items: string[]; title: string }) {
  return (
    <section>
      <h3 className="mb-1 text-xs font-medium text-muted-foreground">{title}</h3>
      <ul className="list-disc ps-5 text-sm">
        {items.map(item => (
          <li key={item}>{item}</li>
        ))}
      </ul>
    </section>
  )
}

export interface ConfirmPanelProps {
  agent: string
  copy: Copy
  onCancel: () => void
  onReplace: () => void
}

export function ConfirmPanel({ agent, copy, onCancel, onReplace }: ConfirmPanelProps) {
  const [understood, setUnderstood] = useState(false)

  return (
    <div className="flex flex-col gap-4 rounded-lg border border-(--ui-stroke-tertiary) p-4">
      <h3 className="text-sm font-medium">{copy.confirmTitle(agent)}</h3>
      <div className="grid gap-4 sm:grid-cols-2">
        <PlainList items={copy.replaces} title={copy.replacesTitle} />
        <PlainList items={copy.keeps} title={copy.keepsTitle} />
      </div>
      <p className="text-xs text-muted-foreground">{copy.botNote(agent)}</p>
      <p className="text-xs text-muted-foreground">{copy.undoNote(agent)}</p>
      <label className="flex items-center gap-2 text-sm font-medium">
        <Checkbox checked={understood} onCheckedChange={value => setUnderstood(value === true)} />
        {copy.acknowledge(agent)}
      </label>
      <div className="flex justify-end gap-2">
        <Button onClick={onCancel} variant="text">
          {copy.cancel}
        </Button>
        <Button disabled={!understood} onClick={onReplace} variant="destructive">
          {copy.replace(agent)}
        </Button>
      </div>
    </div>
  )
}

const STEP_ORDER: TransferPhase[] = ['backup', 'upload', 'restore']

export interface ProgressStepsProps {
  agent: string
  copy: Copy
  fraction?: number
  line?: string
  onCancel?: () => void
  phase: TransferPhase
}

export function ProgressSteps({ agent, copy, fraction, line, onCancel, phase }: ProgressStepsProps) {
  const current = STEP_ORDER.indexOf(phase)

  const labels: Record<string, string> = {
    backup: copy.stepBackup,
    restore: copy.stepRestore(agent),
    upload: copy.stepUpload
  }

  return (
    <div aria-live="polite" className="flex flex-col gap-3 rounded-lg border border-(--ui-stroke-tertiary) p-4">
      <ol className="grid gap-2 text-sm">
        {STEP_ORDER.map((step, index) => (
          <li className={cn('flex items-center gap-2', index > current && 'text-muted-foreground')} key={step}>
            {index < current ? (
              <Check className="size-4 text-primary" />
            ) : index === current ? (
              <Loader2 className="size-4 animate-spin" />
            ) : (
              <span className="inline-block size-4 text-center text-xs">{index + 1}</span>
            )}
            <span>{labels[step]}</span>
            {index === current && step === 'upload' && fraction !== undefined && (
              <span className="tabular-nums text-muted-foreground">{Math.round(fraction * 100)}%</span>
            )}
          </li>
        ))}
      </ol>
      {line && <p className="truncate font-mono text-xs text-muted-foreground">{line}</p>}
      {onCancel && (
        <div className="flex justify-end">
          <Button onClick={onCancel} variant="text">
            {copy.cancel}
          </Button>
        </div>
      )}
    </div>
  )
}

export interface OutcomeProps {
  actions: { label: string; onClick: () => void; primary?: boolean }[]
  body: string
  title: string
  tone?: 'error'
}

/** Done, Undone and error screens: a title, one line and their buttons. */
export function Outcome({ actions, body, title, tone }: OutcomeProps) {
  return (
    <div className="flex flex-col gap-3 rounded-lg border border-(--ui-stroke-tertiary) p-4">
      <h3 className="text-sm font-medium">{title}</h3>
      <p className={cn('text-sm whitespace-pre-wrap', tone === 'error' ? 'text-destructive' : 'text-muted-foreground')}>
        {body}
      </p>
      <div className="flex justify-end gap-2">
        {actions.map(action => (
          <Button key={action.label} onClick={action.onClick} variant={action.primary ? 'default' : 'secondary'}>
            {action.label}
          </Button>
        ))}
      </div>
    </div>
  )
}
