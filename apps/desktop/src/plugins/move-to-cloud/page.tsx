import {
  Button,
  Checkbox,
  cn,
  Codicon,
  ErrorState,
  host,
  Loader,
  move,
  PanelEmpty,
  PanelPill,
  SegmentedControl,
  usePluginI18n
} from '@hermes/plugin-sdk'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { bindCopy, type MoveCopy } from './i18n'
import {
  apply,
  buildAndRead,
  type BundleInventory,
  cleanupLocal,
  cloudEndpoint,
  inventory,
  type MoveEndpoint,
  moveOutcome,
  type MovePreview,
  type MoveRequestFn,
  type MoveState,
  type Part,
  PARTS,
  type Progress,
  refusalMessage,
  undo,
  upload
} from './transfer'

const PLUGIN_ID = 'move-to-cloud'
const PORTAL_AGENTS = 'https://portal.nousresearch.com/agents'

type CloudAgent =
  Awaited<ReturnType<typeof move.discoverCloudAgents>> extends infer R
    ? R extends { agents: infer A }
      ? A extends (infer X)[]
        ? X
        : never
      : never
    : never

type Phase =
  | { kind: 'signin' }
  | { kind: 'agents' }
  | { kind: 'working'; progress: Progress }
  | { kind: 'confirm'; state: MoveState; target: MoveEndpoint }
  | { kind: 'done'; state: MoveState; target: MoveEndpoint }
  | { kind: 'undone' }
  | { kind: 'undoFailed'; problems: string[]; target: MoveEndpoint; moveId: string }
  | { kind: 'error'; message: string; retry: () => void }

/** A bundle built this session, kept until it lands so a retry resumes its digest-keyed upload
 *  instead of packing a new one (a fresh ZIP has a new digest, so the receiver starts over). */
interface BuiltBundle {
  source: MoveEndpoint
  includeHistory: boolean
  inventory: BundleInventory
  bytes: Uint8Array
  sha256: string
}

const sameEndpoint = (a: MoveEndpoint, b: MoveEndpoint) => a.connectionId === b.connectionId && a.profile === b.profile

const request: MoveRequestFn = req => move.request(req)

async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', bytes as Uint8Array<ArrayBuffer>)

  return [...new Uint8Array(digest)].map(b => b.toString(16).padStart(2, '0')).join('')
}

function useCopy(): MoveCopy {
  const t = usePluginI18n(PLUGIN_ID)

  return useMemo(() => bindCopy(t), [t])
}

function Row({ icon, children, muted }: { icon: string; children: React.ReactNode; muted?: boolean }) {
  return (
    <li className={cn('flex items-start gap-3 py-1.5 text-sm', muted && 'text-(--ui-text-secondary)')}>
      <Codicon className="mt-0.5 shrink-0" name={icon} size="0.95rem" />
      <span className="min-w-0">{children}</span>
    </li>
  )
}

function Inventory({ copy, inv }: { copy: MoveCopy; inv: BundleInventory }) {
  const mem = inv.memory_entries
  const skipped = inv.skills_skipped.length + inv.sessions_skipped
  const localMcp = inv.mcp_servers.filter(m => !m.portable).map(m => m.name)
  const redactions = Object.keys(inv.redacted || {}).length

  return (
    <div className="grid gap-6 sm:grid-cols-2">
      <section>
        <h3 className="mb-2 text-xs font-medium text-(--ui-text-secondary)">{copy.whatMoves}</h3>
        <ul>
          <Row icon="tools">{copy.skills(inv.skills.length)}</Row>
          <Row icon="note">{copy.memory(mem['MEMORY.md'] ?? 0, mem['USER.md'] ?? 0)}</Row>
          {inv.has_soul && <Row icon="person">{copy.persona}</Row>}
          {inv.cron_jobs.length > 0 && <Row icon="watch">{copy.jobs(inv.cron_jobs.length)}</Row>}
          {inv.sessions > 0 && <Row icon="comment-discussion">{copy.conversations(inv.sessions)}</Row>}
        </ul>
      </section>
      <section>
        <h3 className="mb-2 text-xs font-medium text-(--ui-text-secondary)">{copy.staysTitle}</h3>
        <ul>
          <Row icon="key" muted>
            {copy.keysStay}
          </Row>
          {skipped > 0 && (
            <Row icon="circle-slash" muted>
              {copy.tooLarge(skipped)}
            </Row>
          )}
          {localMcp.length > 0 && (
            <Row icon="plug" muted>
              {copy.localMcp(localMcp.join(', '))}
            </Row>
          )}
          {redactions > 0 && (
            <Row icon="shield" muted>
              {copy.redacted(redactions)}
            </Row>
          )}
        </ul>
      </section>
    </div>
  )
}

function ProgressBar({ copy, progress, onCancel }: { copy: MoveCopy; progress: Progress; onCancel: () => void }) {
  const label = {
    apply: copy.applying,
    build: copy.building,
    finish: copy.finishing,
    read: copy.reading,
    upload: copy.uploading
  }[progress.phase]

  const fraction = progress.total ? progress.done / progress.total : 0
  const determinate = progress.phase === 'read' || progress.phase === 'upload'

  return (
    <div aria-live="polite" className="flex flex-col gap-3">
      <div className="flex items-center justify-between text-sm">
        <span>{label}</span>
        {determinate && <span className="tabular-nums text-(--ui-text-secondary)">{Math.round(fraction * 100)}%</span>}
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-(--ui-bg-tertiary)" role="progressbar">
        <div
          className={cn(
            'h-full rounded-full bg-(--ui-accent) transition-[width]',
            !determinate && 'w-1/3 animate-pulse'
          )}
          style={determinate ? { width: `${Math.max(2, fraction * 100)}%` } : undefined}
        />
      </div>
      <div className="flex items-center justify-between">
        <p className="text-xs text-(--ui-text-secondary)">{copy.resumeHint}</p>
        <Button onClick={onCancel} variant="text">
          {copy.cancel}
        </Button>
      </div>
    </div>
  )
}

function KeysChecklist({ copy, keys }: { copy: MoveCopy; keys: BundleInventory['keys_to_reenter'] }) {
  if (!keys.length) {
    return null
  }

  const bots = keys.some(k => k.category === 'messaging')

  return (
    <section className="rounded-lg border border-(--ui-stroke-tertiary) p-4">
      <h3 className="text-sm font-medium">{copy.keysTitle}</h3>
      <p className="mt-1 text-xs text-(--ui-text-secondary)">{copy.keysHelp}</p>
      <ul className="mt-3 grid gap-1 sm:grid-cols-2">
        {keys.map(key => (
          <li className="flex items-center gap-2 text-sm" key={key.key}>
            <code className="truncate text-xs">{key.key}</code>
            <PanelPill tone={key.category === 'messaging' ? 'warn' : 'muted'}>{key.category}</PanelPill>
          </li>
        ))}
      </ul>
      {bots && <p className="mt-3 text-xs text-(--ui-text-secondary)">{copy.botWarning}</p>}
    </section>
  )
}

interface PickProps {
  copy: MoveCopy
  agents: CloudAgent[] | null
  selected: CloudAgent | null
  onSelect: (agent: CloudAgent) => void
  includeHistory: boolean
  onIncludeHistory: (value: boolean) => void
  inv: BundleInventory | null
  invError: string
  onStart: () => void
  openExternal: (url: string) => Promise<boolean>
}

function AgentRow({
  agent,
  copy,
  onSelect,
  selected
}: {
  agent: CloudAgent
  copy: MoveCopy
  onSelect: () => void
  selected: boolean
}) {
  const ready = Boolean(agent.dashboardUrl) && agent.dashboardGatewayState !== 'down'

  return (
    <button
      aria-pressed={selected}
      className={cn(
        'flex items-center gap-3 rounded-lg border border-(--ui-stroke-tertiary) px-4 py-3 text-start transition-colors',
        'hover:bg-(--chrome-action-hover) disabled:cursor-not-allowed disabled:opacity-50',
        selected && 'border-(--ui-accent) bg-(--chrome-action-hover)'
      )}
      disabled={!ready}
      onClick={onSelect}
      type="button"
    >
      <Codicon name="cloud" />
      <span className="flex-1 truncate text-sm font-medium">{agent.name}</span>
      {!agent.dashboardUrl && <PanelPill>{copy.provisioning}</PanelPill>}
      {agent.dashboardUrl && agent.dashboardGatewayState === 'down' && <PanelPill tone="warn">{copy.down}</PanelPill>}
    </button>
  )
}

function PickView(props: PickProps) {
  const { copy, agents, selected, inv, invError } = props

  return (
    <>
      <section>
        <h2 className="mb-3 text-sm font-medium">{copy.chooseAgent}</h2>
        {agents === null && <Loader className="my-6" />}
        {agents?.length === 0 && (
          <PanelEmpty
            action={
              <Button onClick={() => void props.openExternal(PORTAL_AGENTS)} variant="secondary">
                {copy.openPortal}
              </Button>
            }
            description={copy.noAgentsHelp}
            icon="cloud"
            title={copy.noAgentsTitle}
          />
        )}
        <div className="grid gap-2">
          {agents?.map(agent => (
            <AgentRow
              agent={agent}
              copy={copy}
              key={agent.id}
              onSelect={() => props.onSelect(agent)}
              selected={selected?.id === agent.id}
            />
          ))}
          {agents && agents.length > 0 && <NewAgentCard copy={copy} openExternal={props.openExternal} />}
        </div>
      </section>
      <section className="flex flex-col gap-4">
        <label className="flex items-center gap-2 text-sm">
          <Checkbox checked={props.includeHistory} onCheckedChange={value => props.onIncludeHistory(value === true)} />
          {copy.includeHistory}
        </label>
        {!inv && !invError && <Loader className="my-4" label={copy.scanning} />}
        {invError && <p className="text-sm text-destructive">{invError}</p>}
        {inv && <Inventory copy={copy} inv={inv} />}
      </section>
      <div className="flex justify-end">
        <Button disabled={!selected || !inv} onClick={props.onStart}>
          {copy.start}
          <Codicon name="arrow-right" />
        </Button>
      </div>
    </>
  )
}

function ConfirmView({
  copy,
  preview,
  onCancel,
  onApply
}: {
  copy: MoveCopy
  preview: MovePreview
  onCancel: () => void
  onApply: (replace: Part[]) => void
}) {
  const [mode, setMode] = useState<'merge' | 'replace'>('merge')
  const [parts, setParts] = useState<Set<Part>>(() => new Set(PARTS))
  const [acknowledged, setAcknowledged] = useState(false)
  const target = preview.target
  const chosen = PARTS.filter(part => parts.has(part))

  const loses: Record<Part, string> = {
    memory: copy.replaceMemory,
    soul: copy.replaceSoul,
    skills: copy.replaceSkills(preview.skills.keep_existing.length),
    cron: copy.replaceCron(target?.cron_jobs ?? 0),
    sessions: copy.replaceSessions(target?.conversations ?? 0),
    config: copy.replaceConfig
  }

  const toggle = (part: Part, on: boolean) => {
    setParts(current => {
      const next = new Set(current)
      on ? next.add(part) : next.delete(part)

      return next
    })
    setAcknowledged(false)
  }

  return (
    <section className="flex flex-col gap-4">
      <h2 className="text-lg font-medium">{copy.confirmTitle}</h2>
      <ul>
        <Row icon="tools">{copy.newSkills(preview.skills.add.length)}</Row>
        {preview.cron.add.length > 0 && <Row icon="watch">{copy.jobs(preview.cron.add.length)}</Row>}
        {preview.sessions > 0 && <Row icon="comment-discussion">{copy.conversations(preview.sessions)}</Row>}
      </ul>
      <SegmentedControl
        onChange={value => {
          setMode(value)
          setAcknowledged(false)
        }}
        options={[
          { id: 'merge', label: copy.modeMerge },
          { id: 'replace', label: copy.modeReplace }
        ]}
        value={mode}
      />
      {mode === 'merge' ? (
        <p className="text-sm text-(--ui-text-secondary)">{copy.confirmHelp}</p>
      ) : (
        <div className="flex flex-col gap-3 rounded-lg border border-(--ui-stroke-tertiary) p-4">
          <p className="text-sm">{copy.replaceHelp}</p>
          <ul className="grid gap-2">
            {PARTS.map(part => (
              <li key={part}>
                <label className="flex items-start gap-2 text-sm">
                  <Checkbox checked={parts.has(part)} onCheckedChange={value => toggle(part, value === true)} />
                  <span>{loses[part]}</span>
                </label>
              </li>
            ))}
          </ul>
          <p className="text-xs text-(--ui-text-secondary)">{copy.replaceKeeps}</p>
          <label className="flex items-start gap-2 text-sm font-medium">
            <Checkbox
              checked={acknowledged}
              disabled={chosen.length === 0}
              onCheckedChange={value => setAcknowledged(value === true)}
            />
            <span>{copy.replaceAcknowledge}</span>
          </label>
        </div>
      )}
      <div className="flex justify-end gap-2">
        <Button onClick={onCancel} variant="text">
          {copy.cancel}
        </Button>
        {mode === 'merge' ? (
          <Button onClick={() => onApply([])}>{copy.merge}</Button>
        ) : (
          <Button disabled={!acknowledged || chosen.length === 0} onClick={() => onApply(chosen)} variant="destructive">
            {copy.replace}
          </Button>
        )}
      </div>
    </section>
  )
}

/** `hermes backup` of the whole Hermes home, saved to a file the user picks. */
function BackupCard({ copy }: { copy: MoveCopy }) {
  const [busy, setBusy] = useState(false)
  const [lastLine, setLastLine] = useState('')
  const [result, setResult] = useState<'' | 'saved' | 'unsaved'>('')
  const [archive, setArchive] = useState('')

  const run = async () => {
    setBusy(true)
    setResult('')

    try {
      const done = await move.backup(move.owner(), lines => setLastLine(lines.at(-1) ?? ''))
      setArchive(done.archive)
      setResult(done.saved ? 'saved' : 'unsaved')
    } catch (error) {
      host.notifyError(error, copy.backupFailed)
    } finally {
      setBusy(false)
      setLastLine('')
    }
  }

  return (
    <section className="flex flex-col gap-3 rounded-lg border border-(--ui-stroke-tertiary) p-5">
      <div className="flex items-start gap-3">
        <Codicon className="mt-0.5" name="archive" />
        <div className="min-w-0 flex-1">
          <h2 className="text-sm font-medium">{copy.backupTitle}</h2>
          <p className="mt-1 text-xs text-(--ui-text-secondary)">{copy.backupHelp}</p>
        </div>
        <Button disabled={busy} onClick={() => void run()} variant="secondary">
          {busy ? copy.backupRunning : copy.backup}
        </Button>
      </div>
      {busy && lastLine && <p className="truncate font-mono text-xs text-(--ui-text-tertiary)">{lastLine}</p>}
      {result === 'saved' && <p className="text-xs text-(--ui-text-secondary)">{copy.backupSaved}</p>}
      {result === 'unsaved' && <p className="text-xs text-(--ui-text-secondary)">{copy.backupKept(archive)}</p>}
    </section>
  )
}

/** Creating a cloud agent from here waits on Nous Portal letting this client create agents
 *  (NousResearch/hermes-portal#2160); until then this links to the portal, where one click does it. */
function NewAgentCard({ copy, openExternal }: { copy: MoveCopy; openExternal: (url: string) => Promise<boolean> }) {
  return (
    <div className="flex items-center gap-3 rounded-lg border border-dashed border-(--ui-stroke-tertiary) px-4 py-3">
      <Codicon name="add" />
      <div className="min-w-0 flex-1">
        <div className="text-sm font-medium">{copy.newAgentTitle}</div>
        <p className="text-xs text-(--ui-text-secondary)">{copy.newAgentHelp}</p>
      </div>
      <Button onClick={() => void openExternal(PORTAL_AGENTS)} variant="text">
        {copy.openPortal}
      </Button>
    </div>
  )
}

interface DoneProps {
  copy: MoveCopy
  state: MoveState
  keys: BundleInventory['keys_to_reenter']
  onUndo: () => void
  onRetry: () => void
  onOpen: () => void
}

function DoneView({ copy, state, keys, onUndo, onRetry, onOpen }: DoneProps) {
  const { complete: applied, steps: failed } = moveOutcome(state)

  return (
    <section className="flex flex-col gap-5">
      <div>
        <h2 className="text-lg font-medium">{applied ? copy.doneTitle : copy.partialTitle}</h2>
        <p className="mt-1 text-sm text-(--ui-text-secondary)">{applied ? copy.doneHelp : copy.partialHelp}</p>
        {failed.map(({ step, failed: items, errors }) => (
          <div className="mt-2 text-sm text-destructive" key={step}>
            <p>{items ? copy.failedItems(step, items) : copy.failedStep(step, errors[0] ?? '')}</p>
            {items > 0 &&
              errors.map(error => (
                <p className="text-xs" key={error}>
                  {error}
                </p>
              ))}
          </div>
        ))}
      </div>
      <KeysChecklist copy={copy} keys={keys} />
      <div className="flex flex-wrap justify-end gap-2">
        <Button onClick={onUndo} variant="text">
          {copy.undo}
        </Button>
        {!applied && (
          <Button onClick={onRetry} variant="secondary">
            {copy.retry}
          </Button>
        )}
        <Button onClick={onOpen}>{copy.openAgent}</Button>
      </div>
    </section>
  )
}

function UndoFailedView({ copy, problems, onRetry }: { copy: MoveCopy; problems: string[]; onRetry: () => void }) {
  return (
    <section className="flex flex-col gap-3">
      <h2 className="text-lg font-medium">{copy.undoProblemsTitle}</h2>
      <p className="text-sm text-(--ui-text-secondary)">{copy.undoProblemsHelp}</p>
      <ul className="text-sm text-destructive">
        {problems.map(problem => (
          <li key={problem}>{problem}</li>
        ))}
      </ul>
      <div className="flex justify-end">
        <Button onClick={onRetry} variant="secondary">
          {copy.retryUndo}
        </Button>
      </div>
    </section>
  )
}

function OrgPicker({ copy, orgs, onPick }: { copy: MoveCopy; orgs: Org[]; onPick: (org: string) => void }) {
  return (
    <section>
      <h2 className="mb-3 text-sm font-medium">{copy.pickOrg}</h2>
      <div className="flex flex-wrap gap-2">
        {orgs.map(o => (
          <Button key={o.id} onClick={() => onPick(o.slug ?? o.id)} variant="secondary">
            {o.name}
          </Button>
        ))}
      </div>
    </section>
  )
}

type Org = { id: string; slug: null | string; name: string }

/** `hermes://open/move-to-cloud?agent=<name>` (the portal's agent page) names the agent to
 *  preselect. A hint only: an unknown or missing name leaves the choice to the user. */
export function agentHint(hash: string): string {
  const query = hash.split('?')[1] ?? ''

  return (new URLSearchParams(query).get('agent') ?? '').trim()
}

/** Cloud sign-in and agent discovery, including the team picker a multi-team account needs. */
function useCloudAgents(
  enabled: boolean,
  onSignedOut: () => void,
  onSignedIn: () => void,
  onError: (message: string, retry: () => void) => void
) {
  const [agents, setAgents] = useState<CloudAgent[] | null>(null)
  const [org, setOrg] = useState<string | undefined>()
  const [orgs, setOrgs] = useState<Org[] | null>(null)

  const discover = useCallback(
    async (orgId?: string) => {
      try {
        if (!(await move.cloudStatus()).signedIn) {
          onSignedOut()

          return
        }

        const result = await move.discoverCloudAgents(orgId)

        if ('needsOrgSelection' in result && result.needsOrgSelection) {
          setOrgs(result.orgs)
        } else if ('agents' in result) {
          setOrgs(null)
          setAgents(result.agents)
          setOrg(result.org?.slug ?? result.org?.id ?? orgId)
        }

        onSignedIn()
      } catch (error) {
        onError(refusalMessage(error), () => void discover(orgId))
      }
    },
    [onError, onSignedIn, onSignedOut]
  )

  useEffect(() => {
    if (enabled) {
      void discover()
    }
  }, [discover, enabled])

  return { agents, org, orgs, discover }
}

/** The dry-run inventory, refreshed when the history switch changes. */
function useInventory(enabled: boolean, includeHistory: boolean) {
  const [inv, setInv] = useState<BundleInventory | null>(null)
  const [invError, setInvError] = useState('')

  useEffect(() => {
    if (!enabled) {
      return
    }

    let live = true
    setInv(null)
    setInvError('')
    inventory(request, move.owner(), includeHistory).then(
      value => live && setInv(value),
      error => live && setInvError(refusalMessage(error))
    )

    return () => {
      live = false
    }
  }, [enabled, includeHistory])

  return { inv, invError, setInv }
}

/** The source bundle for a move: the one this session already built when it still matches (same
 *  owner, same history choice), else a fresh build. The local ZIP is deleted once read: the bytes
 *  live in memory for the session. Cross-restart resume would need the bundle persisted: not done. */
function useSourceBundle() {
  const kept = useRef<BuiltBundle | null>(null)

  const get = async (
    source: MoveEndpoint,
    includeHistory: boolean,
    onProgress: (p: Progress) => void,
    signal: AbortSignal
  ): Promise<BuiltBundle> => {
    const prior = kept.current

    if (prior && sameEndpoint(prior.source, source) && prior.includeHistory === includeHistory) {
      return prior
    }

    let name: null | string = null

    try {
      const built = await buildAndRead(request, source, includeHistory, onProgress, signal)
      name = built.inventory.name
      kept.current = { ...built, includeHistory, sha256: await sha256Hex(built.bytes), source }

      return kept.current
    } finally {
      void cleanupLocal(request, source, name)
    }
  }

  return { get, landed: () => (kept.current = null) }
}

export function MoveToCloudPage({ openExternal }: { openExternal: (url: string) => Promise<boolean> }) {
  const copy = useCopy()
  const transfer = move.transferAvailable()
  const [phase, setPhase] = useState<Phase>({ kind: 'agents' })
  const [selected, setSelected] = useState<CloudAgent | null>(null)
  const [includeHistory, setIncludeHistory] = useState(true)
  const abort = useRef<AbortController | null>(null)
  const signedOut = useCallback(() => setPhase({ kind: 'signin' }), [])

  // Discovery answered: leave the sign-in screen, or the error screen whose Retry re-ran it.
  const signedIn = useCallback(
    () => setPhase(current => (current.kind === 'signin' || current.kind === 'error' ? { kind: 'agents' } : current)),
    []
  )

  const failed = useCallback((message: string, retry: () => void) => setPhase({ kind: 'error', message, retry }), [])

  const cloud = useCloudAgents(transfer, signedOut, signedIn, failed)
  const { inv, invError, setInv } = useInventory(transfer, includeHistory)
  const bundle = useSourceBundle()

  useEffect(() => () => abort.current?.abort(), [])

  const hint = useMemo(() => agentHint(typeof window === 'undefined' ? '' : window.location.hash), [])

  useEffect(() => {
    const match = hint && cloud.agents?.find(agent => agent.name === hint && agent.dashboardUrl)

    if (match) {
      setSelected(current => current ?? match)
    }
  }, [cloud.agents, hint])

  // The owner is read once, at the first start: a profile switch mid-move (or before its retry)
  // must not split one move across two profiles.
  const run = async (agent: CloudAgent, source: MoveEndpoint = move.owner()) => {
    const controller = new AbortController()
    abort.current = controller
    const onProgress = (progress: Progress) => setPhase({ kind: 'working', progress })

    try {
      onProgress({ phase: 'build', done: 0, total: 1 })
      const target = cloudEndpoint(await move.connectCloudAgent(agent, cloud.org))
      const built = await bundle.get(source, includeHistory, onProgress, controller.signal)
      setInv(built.inventory)
      const state = await upload(request, target, built.bytes, built.sha256, onProgress, controller.signal)
      bundle.landed()
      setPhase({ kind: state.status === 'applied' ? 'done' : 'confirm', state, target })
    } catch (error) {
      setPhase(
        controller.signal.aborted
          ? { kind: 'agents' }
          : { kind: 'error', message: refusalMessage(error), retry: () => void run(agent, source) }
      )
    }
  }

  const merge = async (target: MoveEndpoint, moveId: string, replace: Part[] = []) => {
    setPhase({ kind: 'working', progress: { phase: 'apply', done: 0, total: 1 } })

    try {
      setPhase({ kind: 'done', state: await apply(request, target, moveId, replace), target })
    } catch (error) {
      setPhase({ kind: 'error', message: refusalMessage(error), retry: () => void merge(target, moveId, replace) })
    }
  }

  const revert = async (target: MoveEndpoint, moveId: string) => {
    try {
      const { undoProblems } = moveOutcome(await undo(request, target, moveId))
      setPhase(
        undoProblems.length ? { kind: 'undoFailed', problems: undoProblems, target, moveId } : { kind: 'undone' }
      )
    } catch (error) {
      host.notifyError(error, copy.errorTitle)
    }
  }

  const views: Record<Phase['kind'], () => React.ReactNode> = {
    signin: () => (
      <PanelEmpty
        action={<Button onClick={() => void move.cloudLogin().then(() => cloud.discover())}>{copy.signIn}</Button>}
        description={copy.signInHelp}
        icon="cloud"
        title={copy.signInTitle}
      />
    ),
    error: () => (
      <ErrorState description={phase.kind === 'error' ? phase.message : ''} title={copy.errorTitle}>
        <Button onClick={() => phase.kind === 'error' && phase.retry()} variant="secondary">
          {copy.tryAgain}
        </Button>
      </ErrorState>
    ),
    undone: () => <PanelEmpty description={copy.undone} icon="discard" title={copy.title} />,
    undoFailed: () =>
      phase.kind === 'undoFailed' && (
        <UndoFailedView copy={copy} onRetry={() => void revert(phase.target, phase.moveId)} problems={phase.problems} />
      ),
    agents: () =>
      cloud.orgs ? (
        <OrgPicker copy={copy} onPick={org => void cloud.discover(org)} orgs={cloud.orgs} />
      ) : (
        <PickView
          agents={cloud.agents}
          copy={copy}
          includeHistory={includeHistory}
          inv={inv}
          invError={invError}
          onIncludeHistory={setIncludeHistory}
          onSelect={setSelected}
          onStart={() => selected && void run(selected)}
          openExternal={openExternal}
          selected={selected}
        />
      ),
    working: () =>
      phase.kind === 'working' && (
        <ProgressBar copy={copy} onCancel={() => abort.current?.abort()} progress={phase.progress} />
      ),
    confirm: () =>
      phase.kind === 'confirm' &&
      phase.state.preview && (
        <ConfirmView
          copy={copy}
          onApply={replace => void merge(phase.target, phase.state.id, replace)}
          onCancel={() => setPhase({ kind: 'agents' })}
          preview={phase.state.preview}
        />
      ),
    done: () =>
      phase.kind === 'done' && (
        <DoneView
          copy={copy}
          keys={inv?.keys_to_reenter ?? []}
          onOpen={() =>
            void host.ensureAgent(phase.target.connectionId, phase.target.profile).then(() => host.navigate('/'))
          }
          onRetry={() => void merge(phase.target, phase.state.id)}
          onUndo={() => void revert(phase.target, phase.state.id)}
          state={phase.state}
        />
      )
  }

  return (
    <section
      aria-label={copy.title}
      className="mx-auto flex h-full w-full max-w-3xl flex-col gap-8 overflow-y-auto px-8 pb-12 pt-12"
    >
      <header>
        <h1 className="text-3xl font-semibold tracking-tight">{transfer ? copy.title : copy.backupOnlyTitle}</h1>
        {transfer && <p className="mt-2 max-w-prose text-sm text-(--ui-text-secondary)">{copy.subtitle}</p>}
      </header>
      {transfer && views[phase.kind]()}
      <BackupCard copy={copy} />
    </section>
  )
}
