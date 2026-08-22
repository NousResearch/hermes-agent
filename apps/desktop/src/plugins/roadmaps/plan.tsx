/**
 * Roadmaps plugin — Plan view.
 *
 * Version timeline (newest first, active marked) fed by `GET /roadmaps/{id}/plans`,
 * plan governance (proposed → validate, validated → activate) via versioned
 * POSTs, a "New plan from JSON" form (parse → preview → `POST /roadmaps/{id}/plans`),
 * and the versioned planning rules (`GET /planning-rules`) with a copy button.
 * No Vision session is wired here: the roadmaps REST API has no session door,
 * so drafts enter as structured JSON pasted by the operator.
 */

import { Badge, Button, cn, Codicon, CopyButton, EmptyState, Textarea, useQueryClient } from '@hermes/plugin-sdk'
import { useCallback, useMemo, useState } from 'react'

import { activatePlan, createPlan, roadmapPlansKey, roadmapSnapshotKey, usePlanningRules, useRoadmapPlans, validatePlan } from './api'
import { ID } from './config'
import {
  extractPlanJsonBlock,
  formatDate,
  type MutationError,
  mutationErrorCopy,
  planPreviewFromJson,
  plural,
  rpcError,
  validatePlanPayload
} from './data'
import type { PlanMeta, Scope, SnapshotResponse } from './types'
import { SectionTitle } from './ui'

function VersionRow({
  v,
  active,
  busy,
  onValidate,
  onActivate
}: {
  v: PlanMeta
  active: number | null
  busy: 'validate' | 'activate' | null
  onValidate: (version: number) => void
  onActivate: (version: number) => void
}) {
  const isActive = v.version === active
  const canValidate = v.state === 'proposed'
  const canActivate = v.state === 'validated' && !isActive

  return (
    <div className="relative flex gap-3 px-0.5 py-1.5">
      <span className={cn('relative z-10 mt-1.5 size-2 shrink-0 rounded-full', isActive ? 'bg-(--ui-accent)' : 'bg-(--ui-stroke-secondary)')} />
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-2">
          <span className={cn('font-mono text-xs', isActive ? 'font-semibold text-foreground' : 'text-(--ui-text-secondary)')}>{`v${v.version}`}</span>
          {isActive ? (
            <Badge size="xs" variant="outline">
              Active
            </Badge>
          ) : null}
          <span className="font-mono text-[0.625rem] uppercase text-(--ui-text-tertiary)">{v.state}</span>
          {v.created_at ? <span className="ml-auto text-[0.625rem] tabular-nums text-(--ui-text-quaternary)">{formatDate(v.created_at)}</span> : null}
        </div>
        {v.source ? (
          <div className="mt-0.5 flex min-w-0 items-center gap-1 text-[0.625rem] text-(--ui-text-tertiary)">
            <span className="shrink-0 font-medium uppercase tracking-wide">source</span>
            <span className="truncate">{v.source}</span>
          </div>
        ) : null}
        {v.reason ? <div className="mt-0.5 line-clamp-2 text-[0.625rem] text-(--ui-text-tertiary)">{v.reason}</div> : null}
        {canValidate || canActivate ? (
          <div className="mt-1 flex items-center gap-1.5">
            {canValidate ? (
              <Button
                className="gap-1"
                disabled={busy !== null}
                onClick={() => onValidate(v.version)}
                size="xs"
                type="button"
                variant="secondary"
              >
                <Codicon name="check" size="0.7rem" />
                {busy === 'validate' ? 'Validating…' : 'Validate'}
              </Button>
            ) : (
              <Button
                className="gap-1"
                disabled={busy !== null}
                onClick={() => onActivate(v.version)}
                size="xs"
                type="button"
                variant="secondary"
              >
                <Codicon name="play" size="0.7rem" />
                {busy === 'activate' ? 'Activating…' : 'Activate'}
              </Button>
            )}
            <span className="text-[0.625rem] text-(--ui-text-quaternary)">
              {canValidate ? 'Validates this proposed version.' : 'Supersedes the currently active version.'}
            </span>
          </div>
        ) : null}
      </div>
    </div>
  )
}

/** Compact preview of a pasted JSON plan draft. */
function PlanDraftCard({
  draft,
  onSave,
  saveBusy
}: {
  draft: ReturnType<typeof planPreviewFromJson>
  onSave: () => void
  saveBusy: boolean
}) {
  if (!draft) {
    return <div className="px-0.5 text-[0.625rem] text-(--ui-text-tertiary)">Paste a complete ```json plan draft to enable Save.</div>
  }

  return (
    <div className="flex flex-col gap-1 rounded-[3px] border border-(--ui-stroke-tertiary) px-2 py-1.5">
      <SectionTitle
        right={
          <Button className="gap-1" disabled={saveBusy} onClick={onSave} size="xs" type="button" variant="secondary">
            <Codicon name="pass-filled" size="0.7rem" />
            {saveBusy ? 'Saving…' : 'Save plan'}
          </Button>
        }
      >
        Plan draft
      </SectionTitle>
      <div className="truncate text-xs font-medium">{draft.title || 'Untitled plan'}</div>
      {draft.kinds.length > 0 ? (
        <div className="flex flex-wrap gap-1">
          {draft.kinds.map((k) => (
            <Badge key={k} size="xs" variant="outline">
              {k}
            </Badge>
          ))}
        </div>
      ) : null}
      <div className="flex flex-wrap items-center gap-1.5 text-[0.625rem] text-(--ui-text-quaternary)">
        {`${plural(draft.counts.nodes, 'node')} · ${plural(draft.counts.relations, 'relation')} · ${plural(draft.counts.todos, 'todo')}`}
      </div>
    </div>
  )
}

export function PlanView({
  snapshot,
  scope,
  actor,
  onMutated
}: {
  snapshot: SnapshotResponse | null | undefined
  scope: Scope | null
  actor: string
  onMutated: () => void
}) {
  const queryClient = useQueryClient()
  const profile = scope?.profile ?? ''
  const projectId = scope?.projectId ?? ''
  const roadmapId = scope?.roadmapId ?? ''
  const enabled = Boolean(scope)

  const plansQuery = useRoadmapPlans(profile, projectId, roadmapId, enabled)
  const rulesQuery = usePlanningRules(enabled)

  const [draftText, setDraftText] = useState('')
  const [saveBusy, setSaveBusy] = useState(false)
  const [busy, setBusy] = useState<'validate' | 'activate' | null>(null)
  const [error, setError] = useState<MutationError | null>(null)

  const active = snapshot?.roadmap?.active_version ?? null
  const versions = plansQuery.data?.plans ?? []

  const preview = useMemo(() => {
    if (!draftText.trim()) {return null}
    const block = extractPlanJsonBlock(draftText)

    return block ? planPreviewFromJson(block) : null
  }, [draftText])

  const refresh = useCallback(async () => {
    await queryClient.invalidateQueries({ queryKey: roadmapPlansKey(profile, projectId, roadmapId) })
    await queryClient.invalidateQueries({ queryKey: roadmapSnapshotKey(profile, projectId, roadmapId) })
    await queryClient.invalidateQueries({ queryKey: [ID, 'list', profile, projectId] })
    onMutated()
  }, [onMutated, profile, projectId, queryClient, roadmapId])

  const savePlan = useCallback(async () => {
    if (!scope || !preview || saveBusy) {return}
    setSaveBusy(true)
    setError(null)

    try {
      const payload = validatePlanPayload({ nodes: preview.nodes, relations: preview.relations, todos: preview.todos })
      await createPlan(profile, projectId, roadmapId, {
        actor: actor.trim() || 'user',
        nodes: payload.nodes,
        relations: payload.relations,
        todos: payload.todos,
        source: 'desktop',
        reason: 'Plan created from a pasted JSON draft.'
      })
      await refresh()
      setDraftText('')
    } catch (err) {
      setError({ code: rpcError(err).code, hint: (err as { hint?: string })?.hint })
    } finally {
      setSaveBusy(false)
    }
  }, [actor, preview, profile, projectId, refresh, roadmapId, saveBusy, scope])

  const runTransition = useCallback(
    async (kind: 'validate' | 'activate', version: number) => {
      if (!scope || busy !== null) {return}
      const expected = snapshot?.roadmap?.active_version ?? 0
      setBusy(kind)
      setError(null)

      try {
        if (kind === 'validate') {
          await validatePlan(profile, projectId, roadmapId, version, { actor: actor.trim() || 'user', expected_version: expected })
        } else {
          await activatePlan(profile, projectId, roadmapId, version, { actor: actor.trim() || 'user', expected_version: expected })
        }

        await refresh()
      } catch (err) {
        setError({ code: rpcError(err).code })
      } finally {
        setBusy(null)
      }
    },
    [actor, busy, profile, projectId, refresh, roadmapId, scope, snapshot]
  )

  const ec = mutationErrorCopy(error)
  const rules = rulesQuery.data

  return (
    <div className="flex flex-col gap-1.5">
      {/* Planning rules — the versioned Vision contract, read-only. */}
      {rules ? (
        <div className="flex items-center gap-1.5 rounded-[3px] bg-(--ui-bg-quaternary) px-2 py-1">
          <span className="inline-flex items-center gap-1 text-[0.625rem] font-medium uppercase tracking-wide text-(--ui-text-tertiary)">
            <Codicon name="info" size="0.7rem" />
            {`Planning rules · v${rules.version}`}
          </span>
          <span className="min-w-0 flex-1 truncate text-[0.625rem] text-(--ui-text-tertiary)">{rules.rules.prompt}</span>
          <CopyButton
            appearance="icon"
            buttonSize="icon-xs"
            buttonVariant="ghost"
            label="Copy planning rules"
            text={rules.rules.prompt}
            title="Copy planning rules"
          />
        </div>
      ) : null}

      <div className="flex items-center justify-between gap-2 px-0.5">
        <SectionTitle right={<span className="tabular-nums text-(--ui-text-quaternary)">{plural(versions.length, 'version')}</span>}>
          Plan history
        </SectionTitle>
      </div>

      {/* New plan from JSON — paste → preview → save. */}
      <div className="flex flex-col gap-1 px-0.5">
        <Textarea
          aria-label="Plan draft JSON"
          className="min-h-16 text-xs"
          onChange={(ev) => setDraftText(ev.target.value)}
          placeholder={'Paste a plan draft as ```json … ``` (nodes / relations / todos).'}
          spellCheck={false}
          value={draftText}
        />
        <PlanDraftCard draft={preview} onSave={() => void savePlan()} saveBusy={saveBusy} />
      </div>

      {error && ec ? (
        <div className="flex items-start gap-1.5 rounded-[3px] bg-destructive/10 px-2 py-1 text-xs text-destructive">
          <Codicon className="mt-px shrink-0" name="error" size="0.75rem" />
          <span>{`${ec.hint}${ec.code != null ? ` (code ${ec.code})` : ''}`}</span>
        </div>
      ) : null}

      {plansQuery.isError ? (
        <EmptyState
          description="The plan versions could not be loaded. Reload the roadmap, then retry."
          title="Plan history unavailable"
        />
      ) : versions.length === 0 ? (
        <EmptyState
          description="Paste a structured plan draft above — the first published version lands here once saved."
          title="No versions yet"
        />
      ) : (
        <div className="relative mt-1 flex flex-col">
          <span className="absolute bottom-2 left-[3px] top-2 w-px bg-(--ui-stroke-tertiary)" />
          {versions.map((v) => (
            <VersionRow
              active={active}
              busy={busy}
              key={String(v.version)}
              onActivate={(version) => void runTransition('activate', version)}
              onValidate={(version) => void runTransition('validate', version)}
              v={v}
            />
          ))}
        </div>
      )}
    </div>
  )
}
