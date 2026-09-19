/**
 * Collective Wisdom — "Team Skills" page. Pure SDK consumer over the plugin's own
 * `/api/plugins/wisdom/*` router (`plugins/wisdom/dashboard/plugin_api.py`): browse the
 * team catalog, see installed versions, resolve update-policy conflicts, share qualifying
 * local skills, install/update/remove.
 *
 * Consent is two-step by construction: `/plan` (or `/share/prepare`) returns the exact
 * version, content hash and Gateway verdict; the ConfirmDialog shows them; `/install`
 * (or `/share`) echoes the hash back and the server refuses if the package changed in
 * between. Ships ON: it renders nothing but an explanatory empty state unless the Nous
 * token carries Wisdom scopes.
 */

import {
  Badge,
  Button,
  Codicon,
  ConfirmDialog,
  EmptyState,
  type HermesPlugin,
  host,
  type PluginContext,
  type RouteContribution,
  ROUTES_AREA,
  SIDEBAR_NAV_AREA,
  type SidebarNavContribution,
  STATUSBAR_AREAS,
  Tip,
  useMutation,
  useQuery,
  useQueryClient
} from '@hermes/plugin-sdk'
import { useState } from 'react'

interface Skill { id: string; slug: string | null; version: number | null; installs: number; description: string | null; security: string | null }
interface Installed { slug: string; version: number; path: string }
interface Update {
  skill_id: string
  slug: string
  installed: number
  latest: number
  required: boolean
  mode: 'MANUAL' | 'AUTO_WITH_NOTICE' | 'REQUIRED'
  modified: boolean
  action: 'auto' | 'conflict' | 'manual' | 'deferred'
}
interface Notice { skill_id: string; version: number | null; kind: string; installed: number | null }
interface Candidate { skill: string; reason: 'high_usage' | 'refinement'; evidence: Record<string, number> }
interface Overview {
  entitled: boolean
  skills: Skill[]
  status: { installed?: Record<string, Installed>; updates?: Update[]; notices?: Notice[] }
  candidates?: Candidate[]
}
interface Plan { skill_id: string; slug: string; version: number; content_hash: string; security: string; author: string | null; explanation: string | null; update_mode: string | null; target: string }
interface SharePlan { skill_name: string; slug: string; description: string; content_hash: string; files: Array<{ path: string; bytes: number }> }

let rest: PluginContext['rest'] | null = null
const KEY = ['wisdom', 'overview'] as const
const fetchOverview = () => rest!<Overview>('/overview')

const ACTION_LABEL: Record<Update['action'], string> = {
  auto: 'applies automatically',
  conflict: 'you edited your copy',
  manual: 'review to update',
  deferred: 'kept your copy'
}

function describeCandidate(c: Candidate): string {
  if (c.reason === 'high_usage') {return `used on ${c.evidence.consecutive_business_days} consecutive business days`}

  return `refined ${c.evidence.edits} times, stable for ${c.evidence.stable_days} days and still in use`
}

function PlanDialog({ plan, onClose }: { plan: Plan; onClose: () => void }) {
  const qc = useQueryClient()

  return (
    <ConfirmDialog
      confirmLabel="Install"
      description={
        <span className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
          <span className="text-muted-foreground">security</span><span>{plan.security}</span>
          <span className="text-muted-foreground">hash</span><span className="font-mono break-all">{plan.content_hash}</span>
          {plan.author && (<><span className="text-muted-foreground">publisher</span><span>{plan.author}</span></>)}
          {plan.update_mode && (<><span className="text-muted-foreground">update policy</span><span>{plan.update_mode}</span></>)}
          <span className="text-muted-foreground">target</span><span className="font-mono break-all">{plan.target}</span>
        </span>
      }
      onClose={onClose}
      onConfirm={async () => {
        await rest!('/install', { method: 'POST', body: { skill_id: plan.skill_id, version: plan.version, content_hash: plan.content_hash } })
        await qc.invalidateQueries({ queryKey: KEY })
        host.notify({ kind: 'info', message: `Installed ${plan.slug} v${plan.version}` })
      }}
      open
      title={`Install ${plan.slug} v${plan.version}`}
    />
  )
}

function ShareDialog({ plan, onClose }: { plan: SharePlan; onClose: () => void }) {
  const qc = useQueryClient()

  return (
    <ConfirmDialog
      confirmLabel="Share"
      description={
        <span className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
          <span className="text-muted-foreground">description</span><span>{plan.description}</span>
          <span className="text-muted-foreground">files</span>
          <span className="font-mono">{plan.files.map(f => `${f.path} (${f.bytes} B)`).join(', ')}</span>
          <span className="text-muted-foreground">hash</span><span className="font-mono break-all">{plan.content_hash}</span>
          <span className="col-span-2 text-muted-foreground">Publishes only if the Gateway's security and professionalism checks both pass; otherwise the draft is withdrawn and the verdict shown.</span>
        </span>
      }
      onClose={onClose}
      onConfirm={async () => {
        await rest!('/share', { method: 'POST', body: { skill_name: plan.skill_name, description: plan.description, content_hash: plan.content_hash } })
        await qc.invalidateQueries({ queryKey: KEY })
        host.notify({ kind: 'info', message: `Shared ${plan.slug} with your team` })
      }}
      open
      title={`Share ${plan.skill_name} as ${plan.slug}`}
    />
  )
}

function TeamSkillsPage() {
  const qc = useQueryClient()
  const { data, error, isLoading } = useQuery({ queryKey: KEY, queryFn: fetchOverview, refetchInterval: 120_000 })
  const [plan, setPlan] = useState<Plan | null>(null)
  const [share, setShare] = useState<SharePlan | null>(null)
  const refresh = () => void qc.invalidateQueries({ queryKey: KEY })

  const planFor = useMutation({
    mutationFn: (body: { skill_id: string; version?: number }) => rest!<Plan>('/plan', { method: 'POST', body }),
    onSuccess: setPlan,
    onError: e => host.notifyError(e, 'Could not prepare install')
  })

  const remove = useMutation({
    mutationFn: (skill_id: string) => rest!('/uninstall', { method: 'POST', body: { skill_id } }),
    onSuccess: refresh,
    onError: e => host.notifyError(e, 'Could not remove skill')
  })

  const keep = useMutation({
    mutationFn: (u: Update) => rest!('/update/keep', { method: 'POST', body: { skill_id: u.skill_id, version: u.latest } }),
    onSuccess: refresh,
    onError: e => host.notifyError(e, 'Could not record your choice')
  })

  const prepareShare = useMutation({
    mutationFn: (skill_name: string) => rest!<SharePlan>('/share/prepare', { method: 'POST', body: { skill_name } }),
    onSuccess: setShare,
    onError: e => host.notifyError(e, 'Could not prepare the skill for sharing')
  })

  const notNow = useMutation({
    mutationFn: (skill: string) => rest!('/candidates/not-now', { method: 'POST', body: { skill } }),
    onSuccess: refresh,
    onError: e => host.notifyError(e, 'Could not snooze the suggestion')
  })

  if (isLoading) {return <div className="p-6 text-sm text-muted-foreground">Loading team skills…</div>}

  if (error) {return <EmptyState description={String((error as Error).message)} title="Collective Wisdom unavailable" />}

  if (!data?.entitled) {
    return <EmptyState description="Sign in to Nous with a team that has Collective Wisdom enabled (hermes login) to browse and share skills." title="Collective Wisdom" />
  }

  const installed = data.status.installed ?? {}
  const pending = (data.status.updates ?? []).filter(u => u.action !== 'auto')
  const updates = new Map(pending.map(u => [u.skill_id, u]))
  const candidates = data.candidates ?? []

  return (
    <div className="flex h-full min-h-0 flex-col gap-4 overflow-y-auto p-6">
      <header className="flex items-baseline justify-between">
        <h1 className="text-lg font-semibold">Team Skills</h1>
        <span className="text-xs text-muted-foreground">{data.skills.length} shared · {Object.keys(installed).length} installed</span>
      </header>

      {pending.length > 0 && (
        <section aria-label="Updates" className="rounded-md border border-(--ui-stroke-secondary) p-3">
          <h2 className="mb-2 text-sm font-medium">Updates needing your decision</h2>
          <ul className="flex flex-col gap-2">
            {pending.map(u => (
              <li className="flex items-center gap-3 text-sm" key={u.skill_id}>
                <span className="min-w-0 flex-1 truncate">
                  {u.slug} v{u.installed} → v{u.latest}
                  <Badge className="ml-2" variant="outline">{u.mode}</Badge>
                  <Badge className="ml-1" variant={u.action === 'conflict' ? 'destructive' : 'muted'}>{ACTION_LABEL[u.action]}</Badge>
                </span>
                <Button disabled={planFor.isPending} onClick={() => planFor.mutate({ skill_id: u.skill_id, version: u.latest })} size="sm">
                  {u.modified ? 'Replace (keep a copy of my edits)' : 'Update'}
                </Button>
                {u.action === 'conflict' && (
                  <Button disabled={keep.isPending} onClick={() => keep.mutate(u)} size="sm" variant="ghost">Keep mine</Button>
                )}
              </li>
            ))}
          </ul>
        </section>
      )}

      {candidates.length > 0 && (
        <section aria-label="Share candidates" className="rounded-md border border-(--ui-stroke-secondary) p-3">
          <h2 className="mb-2 text-sm font-medium">Worth sharing with your team</h2>
          <ul className="flex flex-col gap-2">
            {candidates.map(c => (
              <li className="flex items-center gap-3 text-sm" key={c.skill}>
                <span className="min-w-0 flex-1 truncate"><span className="font-medium">{c.skill}</span> · {describeCandidate(c)}</span>
                <Button disabled={prepareShare.isPending} onClick={() => prepareShare.mutate(c.skill)} size="sm">Share…</Button>
                <Button disabled={notNow.isPending} onClick={() => notNow.mutate(c.skill)} size="sm" variant="ghost">Not now</Button>
              </li>
            ))}
          </ul>
        </section>
      )}

      {data.skills.length === 0 && <EmptyState description="Share a skill with `hermes wisdom share <name> --description …`." title="Nothing shared yet" />}
      <ul className="flex flex-col divide-y divide-(--ui-stroke-secondary)">
        {data.skills.map(s => {
          const local = installed[s.id]
          const upd = updates.get(s.id)

          return (
            <li className="flex items-center gap-3 py-3" key={s.id}>
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <span className="truncate font-medium">{s.slug ?? s.id}</span>
                  <Badge variant="outline">v{s.version}</Badge>
                  {s.security && <Badge variant={s.security === 'pass' ? 'success' : 'destructive'}>{s.security}</Badge>}
                  {local && <Badge variant={upd ? 'default' : 'muted'}>{upd ? `update v${local.version} → v${upd.latest}` : `installed v${local.version}`}</Badge>}
                </div>
                {s.description && <p className="truncate text-xs text-muted-foreground">{s.description}</p>}
              </div>
              <span className="text-xs tabular-nums text-muted-foreground">{s.installs} installs</span>
              <span className="grid w-32 grid-cols-[1fr_2rem] items-center justify-items-end gap-1">
              {(!local || upd) && (
                <Button disabled={planFor.isPending} onClick={() => planFor.mutate({ skill_id: s.id })} size="sm">
                  {upd ? 'Update' : 'Install'}
                </Button>
              )}
              {local && !upd && <span />}
              {local ? (
                <Tip label="Remove from this profile">
                  <Button disabled={remove.isPending} onClick={() => remove.mutate(s.id)} size="sm" variant="ghost">
                    <Codicon name="trash" size="0.8rem" />
                  </Button>
                </Tip>
              ) : <span />}
              </span>
            </li>
          )
        })}
      </ul>
      {plan && <PlanDialog onClose={() => setPlan(null)} plan={plan} />}
      {share && <ShareDialog onClose={() => setShare(null)} plan={share} />}
    </div>
  )
}

/** Sidebar-adjacent pill: pending team updates/notices/candidates, one glance from anywhere. */
function WisdomCount() {
  const { data } = useQuery({ queryKey: KEY, queryFn: fetchOverview, refetchInterval: 300_000 })
  const n = (data?.status.updates?.filter(u => u.action !== 'auto').length ?? 0) + (data?.status.notices?.length ?? 0) + (data?.candidates?.length ?? 0)

  if (!data?.entitled || n === 0) {return null}

  return (
    <Tip label={`${n} team skill item${n === 1 ? '' : 's'} waiting`}>
      <button
        className="inline-flex h-full items-center gap-1 px-1.5 text-[0.6875rem] tabular-nums text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground"
        onClick={() => host.navigate('/wisdom')}
        type="button"
      >
        <Codicon name="organization" size="0.7rem" />
        <span>{n}</span>
      </button>
    </Tip>
  )
}

const plugin: HermesPlugin = {
  id: 'wisdom',
  name: 'Collective Wisdom',
  description: 'Team Skills page: browse, install and update skills your Nous team shares, resolve update conflicts, share qualifying local skills; status-bar count of pending items.',
  defaultEnabled: true,
  register(ctx) {
    rest = ctx.rest
    ctx.onDispose(() => { rest = null })
    ctx.registerMany([
      { id: 'page', area: ROUTES_AREA, data: { path: '/wisdom' } satisfies RouteContribution, render: () => <TeamSkillsPage /> },
      { id: 'nav', area: SIDEBAR_NAV_AREA, order: 55, data: { codicon: 'organization', label: 'Team Skills', path: '/wisdom' } satisfies SidebarNavContribution },
      { id: 'count', area: STATUSBAR_AREAS.right, order: 81, render: () => <WisdomCount /> }
    ])
  }
}

export default plugin
