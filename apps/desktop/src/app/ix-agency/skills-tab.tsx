import { type ReactNode, useCallback, useEffect, useState } from 'react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import type { IxAgencySkillTemplate, IxAgencyUserSkill } from '@/global'
import { normalize } from '@/lib/text'
import { notify, notifyError } from '@/store/notifications'

import { DetailColumn, ListColumn, MasterDetail } from '../master-detail'
import { PanelEmpty, PanelListRow, PanelMeta } from '../overlays/panel'

import { $ixPendingSkill } from './copilot-store'
import skillsData from './data/skills.json'
import {
  groupSkillsByMcp,
  groupSkillsByPod,
  type IxPod,
  MCP_TILES,
  POD_LABELS,
  skillMcpIds,
  skillPods,
  tileWiring
} from './skill-mcps'
import type { IxSkillItem } from './types'

const POD_DOT: Record<IxPod, string> = {
  content: 'bg-purple-400',
  growth: 'bg-amber-400',
  product: 'bg-sky-400',
  engineering: 'bg-emerald-500'
}

// Bundled snapshot of the Intelliverse portal's admin-skills catalog
// (scripts/generate-ix-agency-data.mjs regenerates from the sibling checkout).
const SKILLS: IxSkillItem[] = Array.isArray(skillsData.items) ? (skillsData.items as IxSkillItem[]) : []

// Selection: an org playbook, one of MY drafts, or the new-skill flow.
type Selection = { id: string; kind: 'org' | 'user' } | { kind: 'new' } | null

type Draft = { content: string; description: string; id: null | string; title: string }

const EMPTY_DRAFT: Draft = { content: '', description: '', id: null, title: '' }

function ixApi() {
  return window.hermesDesktop?.ixAgency ?? null
}

function sectionLabel(text: ReactNode) {
  return (
    <div className="flex items-center gap-1.5 px-2 pb-1 pt-2 text-[0.62rem] font-semibold uppercase tracking-wide text-muted-foreground/50">
      {text}
    </div>
  )
}

/** Chips for the MCP tiles a skill's text references, with how each one is
 *  wired into Hermes (direct entry vs via the admin-mcp gateway). */
function McpChips({ skill }: { skill: { content?: string; description?: string; id?: string; title?: string } }) {
  const ids = skillMcpIds(skill)

  if (!ids.length) {
    return (
      <p className="text-[0.68rem] text-muted-foreground/60">
        No MCP referenced yet — name the tools this skill calls (e.g. notifuse, chatwoot, grafana) so it groups by
        tool.
      </p>
    )
  }

  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {ids.map(id => {
        const tile = MCP_TILES.find(t => t.id === id)

        if (!tile) {
          return null
        }

        return (
          <Badge key={id} variant="muted">
            <span
              aria-hidden
              className={tileWiring(tile) === 'direct' ? 'size-1.5 rounded-full bg-emerald-500' : 'size-1.5 rounded-full bg-sky-400'}
            />
            {tile.label}
            <span className="text-muted-foreground/60">{tileWiring(tile) === 'direct' ? 'direct' : 'gateway'}</span>
          </Badge>
        )
      })}
      {ids.length > 1 && <Badge variant="warn">cross-MCP ×{ids.length}</Badge>}
      {skillPods(skill).map(pod => (
        <Badge key={pod} variant="muted">
          <span aria-hidden className={`size-1.5 rounded-full ${POD_DOT[pod]}`} />
          {POD_LABELS[pod]}
        </Badge>
      ))}
    </div>
  )
}

/** Adapt a user draft to the copilot's pending-skill shape ("Run natively"). */
function asPendingSkill(skill: IxAgencyUserSkill): IxSkillItem {
  return {
    id: skill.id,
    title: skill.title,
    description: skill.description,
    persona: 'my skill (this machine)',
    rank: null,
    superAdminOnly: false,
    content: skill.content,
    starterPrompts: [],
    tiers: [],
    bundles: [],
    appIds: []
  }
}

// One row in the list column — an org playbook or a user draft, with the
// text fields groupSkillsByMcp scans at top level.
type SkillEntry = {
  content?: string
  description: string
  id: string
  kind: 'org' | 'user'
  meta?: string
  title: string
}

export function SkillsTab({ onRunNatively, query }: { onRunNatively?: () => void; query: string }) {
  const [selection, setSelection] = useState<Selection>(null)
  const [userSkills, setUserSkills] = useState<IxAgencyUserSkill[]>([])
  const [templates, setTemplates] = useState<IxAgencySkillTemplate[]>([])
  const [draft, setDraft] = useState<Draft>(EMPTY_DRAFT)
  const [busy, setBusy] = useState(false)
  const [grouping, setGrouping] = useState<'flat' | 'pod' | 'tool'>('flat')

  const refresh = useCallback(async () => {
    const api = ixApi()

    if (!api?.skillsList) {
      return
    }

    try {
      const result = await api.skillsList()

      setUserSkills(result.skills)
      setTemplates(result.templates)
    } catch {
      // Skills live on disk; a failed read just leaves the list empty.
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  const q = normalize(query)

  const orgSkills = SKILLS.filter(
    skill => !q || normalize(`${skill.id} ${skill.title} ${skill.description}`).includes(q)
  )

  const mySkills = userSkills.filter(
    skill => !q || normalize(`${skill.id} ${skill.title} ${skill.description}`).includes(q)
  )

  const selectedOrg = selection?.kind === 'org' ? (SKILLS.find(s => s.id === selection.id) ?? null) : null
  const selectedUser = selection?.kind === 'user' ? (userSkills.find(s => s.id === selection.id) ?? null) : null

  // "By tool" view: every skill (mine + org) bucketed under each MCP tile its
  // text references — cross-MCP skills intentionally appear in several groups.
  const entries: SkillEntry[] = [
    ...mySkills.map(s => ({
      kind: 'user' as const,
      id: s.id,
      title: s.title,
      description: s.description,
      content: s.content,
      meta: s.publishedId ? 'published' : 'draft'
    })),
    ...orgSkills.map(s => ({
      kind: 'org' as const,
      id: s.id,
      title: s.title,
      description: s.description,
      content: s.content,
      meta: s.rank ? `#${s.rank}` : undefined
    }))
  ]

  const grouped = groupSkillsByMcp(entries)
  const podGrouped = groupSkillsByPod(entries)

  const selectEntry = (entry: SkillEntry) => {
    if (entry.kind === 'user') {
      const skill = userSkills.find(s => s.id === entry.id)

      if (skill) {
        openUserSkill(skill)
      }

      return
    }

    setSelection({ kind: 'org', id: entry.id })
  }

  const openUserSkill = (skill: IxAgencyUserSkill) => {
    setSelection({ kind: 'user', id: skill.id })
    setDraft({ id: skill.id, title: skill.title, description: skill.description, content: skill.content })
  }

  const startFromTemplate = (template: IxAgencySkillTemplate) => {
    setDraft({
      id: null,
      title: template.id === 'blank' ? '' : template.title,
      description: template.id === 'blank' ? '' : template.description,
      content: template.content
    })
  }

  const save = async () => {
    const api = ixApi()

    if (!api?.skillsSave || busy) {
      return
    }

    setBusy(true)

    try {
      const saved = await api.skillsSave(draft)

      notify({ message: `Skill saved — Hermes and the Copilot tab see it now`, detail: `skills/ix-user/${saved.id}/SKILL.md` })
      await refresh()
      openUserSkill(saved)
    } catch (error) {
      notifyError(error, 'Could not save the skill')
    } finally {
      setBusy(false)
    }
  }

  const publish = async () => {
    const api = ixApi()

    if (!api?.skillsPublish || !selectedUser || busy) {
      return
    }

    setBusy(true)

    try {
      const published = await api.skillsPublish(selectedUser.id)

      notify({
        message: selectedUser.publishedId ? 'Skill updated on the portal' : 'Skill published globally',
        detail: `${published.publishedId} — it now shows on the web admin Skills.md page`
      })
      await refresh()
      openUserSkill(published)
    } catch (error) {
      notifyError(error, 'Publish failed')
    } finally {
      setBusy(false)
    }
  }

  const remove = async () => {
    const api = ixApi()

    if (!api?.skillsDelete || !selectedUser || busy) {
      return
    }

    setBusy(true)

    try {
      await api.skillsDelete(selectedUser.id)
      notify({
        message: 'Skill deleted from this machine',
        detail: selectedUser.publishedId ? 'The published portal copy is untouched.' : undefined
      })
      setSelection(null)
      setDraft(EMPTY_DRAFT)
      await refresh()
    } catch (error) {
      notifyError(error, 'Delete failed')
    } finally {
      setBusy(false)
    }
  }

  const runNatively = (skill: IxSkillItem) => {
    $ixPendingSkill.set(skill)
    onRunNatively?.()
  }

  const editorValid = draft.title.trim().length > 0 && draft.content.trim().length > 0
  const showEditor = selection?.kind === 'new' || selection?.kind === 'user'

  return (
    <MasterDetail split="wide">
      <ListColumn
        header={
          <div className="mb-1 space-y-1">
            <Button
              className="w-full justify-center"
              onClick={() => {
                setSelection({ kind: 'new' })
                setDraft(EMPTY_DRAFT)
              }}
              size="sm"
              variant="outline"
            >
              <Codicon name="add" size="0.8125rem" />
              New skill
            </Button>
            <div className="flex gap-1">
              <Button
                className="flex-1 justify-center text-[0.68rem]"
                onClick={() => setGrouping('flat')}
                size="xs"
                variant={grouping === 'flat' ? 'secondary' : 'ghost'}
              >
                All
              </Button>
              <Button
                className="flex-1 justify-center text-[0.68rem]"
                onClick={() => setGrouping('tool')}
                size="xs"
                variant={grouping === 'tool' ? 'secondary' : 'ghost'}
              >
                <Codicon name="plug" size="0.75rem" />
                By tool
              </Button>
              <Button
                className="flex-1 justify-center text-[0.68rem]"
                onClick={() => setGrouping('pod')}
                size="xs"
                variant={grouping === 'pod' ? 'secondary' : 'ghost'}
              >
                <Codicon name="organization" size="0.75rem" />
                By pod
              </Button>
            </div>
          </div>
        }
      >
        {grouping === 'pod' ? (
          <>
            {podGrouped.pods.map(({ pod, label, skills }) => (
              <div key={`pod:${pod}`}>
                {sectionLabel(
                  <>
                    <span aria-hidden className={`size-1.5 shrink-0 rounded-full ${POD_DOT[pod]}`} />
                    <span className="min-w-0 truncate">{label}</span>
                    <span className="text-muted-foreground/40">{skills.length}</span>
                  </>
                )}
                {skills.map(entry => (
                  <PanelListRow
                    active={selection?.kind === entry.kind && selection.id === entry.id}
                    icon={entry.kind === 'user' ? 'edit' : 'sparkle'}
                    key={`pod:${pod}:${entry.kind}:${entry.id}`}
                    meta={entry.meta}
                    onSelect={() => selectEntry(entry)}
                    title={entry.title}
                  />
                ))}
              </div>
            ))}
            {podGrouped.ungrouped.length > 0 && (
              <div>
                {sectionLabel('No MCP referenced')}
                {podGrouped.ungrouped.map(entry => (
                  <PanelListRow
                    active={selection?.kind === entry.kind && selection.id === entry.id}
                    icon={entry.kind === 'user' ? 'edit' : 'sparkle'}
                    key={`nopod:${entry.kind}:${entry.id}`}
                    meta={entry.meta}
                    onSelect={() => selectEntry(entry)}
                    title={entry.title}
                  />
                ))}
              </div>
            )}
          </>
        ) : grouping === 'tool' ? (
          <>
            {grouped.groups.map(({ tile, skills }) => (
              <div key={`mcp:${tile.id}`}>
                {sectionLabel(
                  <>
                    <span
                      aria-hidden
                      className={
                        tileWiring(tile) === 'direct'
                          ? 'size-1.5 shrink-0 rounded-full bg-emerald-500'
                          : 'size-1.5 shrink-0 rounded-full bg-sky-400'
                      }
                    />
                    <span className="min-w-0 truncate">{tile.label}</span>
                    <span className="text-muted-foreground/40">
                      {tileWiring(tile)} · {skills.length}
                    </span>
                  </>
                )}
                {skills.map(entry => (
                  <PanelListRow
                    active={selection?.kind === entry.kind && selection.id === entry.id}
                    icon={entry.kind === 'user' ? 'edit' : 'sparkle'}
                    key={`mcp:${tile.id}:${entry.kind}:${entry.id}`}
                    meta={entry.meta}
                    onSelect={() => selectEntry(entry)}
                    title={entry.title}
                  />
                ))}
              </div>
            ))}
            {grouped.ungrouped.length > 0 && (
              <div>
                {sectionLabel('No MCP referenced')}
                {grouped.ungrouped.map(entry => (
                  <PanelListRow
                    active={selection?.kind === entry.kind && selection.id === entry.id}
                    icon={entry.kind === 'user' ? 'edit' : 'sparkle'}
                    key={`nomcp:${entry.kind}:${entry.id}`}
                    meta={entry.meta}
                    onSelect={() => selectEntry(entry)}
                    title={entry.title}
                  />
                ))}
              </div>
            )}
          </>
        ) : (
          <>
            {mySkills.length > 0 && sectionLabel('My skills — this machine')}
            {mySkills.map(skill => (
              <PanelListRow
                active={selection?.kind === 'user' && selection.id === skill.id}
                icon="edit"
                key={`user:${skill.id}`}
                meta={skill.publishedId ? 'published' : 'draft'}
                onSelect={() => openUserSkill(skill)}
                title={skill.title}
              />
            ))}
            {sectionLabel('Org skills — published')}
            {orgSkills.map(skill => (
              <PanelListRow
                active={selection?.kind === 'org' && selection.id === skill.id}
                icon="sparkle"
                key={`org:${skill.id}`}
                meta={skill.rank ? `#${skill.rank}` : undefined}
                onSelect={() => setSelection({ kind: 'org', id: skill.id })}
                title={skill.title}
              />
            ))}
          </>
        )}
      </ListColumn>
      <DetailColumn
        actionBar={
          showEditor ? (
            <>
              <Button disabled={!editorValid || busy} onClick={() => void save()} size="sm">
                <Codicon name="save" size="0.8125rem" />
                {selection?.kind === 'new' ? 'Save to my skills' : 'Save'}
              </Button>
              {selectedUser && (
                <>
                  <Button
                    disabled={busy}
                    onClick={() => runNatively(asPendingSkill({ ...selectedUser, ...draft, id: selectedUser.id }))}
                    size="sm"
                    variant="outline"
                  >
                    <Codicon name="sparkle" size="0.8125rem" />
                    Run natively
                  </Button>
                  <Button disabled={busy} onClick={() => void publish()} size="sm" variant="outline">
                    <Codicon name="cloud-upload" size="0.8125rem" />
                    {selectedUser.publishedId ? 'Update on portal' : 'Publish globally'}
                  </Button>
                  <Button className="ml-auto" disabled={busy} onClick={() => void remove()} size="sm" variant="ghost">
                    <Codicon name="trash" size="0.8125rem" />
                    Delete
                  </Button>
                </>
              )}
            </>
          ) : undefined
        }
      >
        {selection?.kind === 'new' && !draft.content ? (
          <div className="space-y-4">
            <h3 className="text-sm font-semibold text-foreground">Start from a template</h3>
            <p className="text-xs leading-relaxed text-muted-foreground">
              A skill is a SKILL.md playbook: what it does, when to use it, which MCP tools to call, and the output
              shape. Pick a starting point — you can rewrite everything.
            </p>
            <div className="space-y-2">
              {templates.map(template => (
                <button
                  className="w-full rounded-md border border-(--ui-border) px-3 py-2 text-left hover:bg-(--ui-row-active-background)"
                  key={template.id}
                  onClick={() => startFromTemplate(template)}
                  type="button"
                >
                  <div className="text-xs font-semibold text-foreground">{template.title}</div>
                  <div className="mt-0.5 text-[0.68rem] leading-relaxed text-muted-foreground">
                    {template.description}
                  </div>
                </button>
              ))}
            </div>
          </div>
        ) : showEditor ? (
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <h3 className="min-w-0 flex-1 truncate text-sm font-semibold text-foreground">
                {selection?.kind === 'new' ? 'New skill' : draft.title || selectedUser?.title}
              </h3>
              {selectedUser?.publishedId ? <Badge>published</Badge> : <Badge variant="muted">user-level</Badge>}
            </div>
            <label className="block space-y-1">
              <span className="text-[0.68rem] font-medium text-muted-foreground">Title</span>
              <Input
                onChange={e => setDraft(d => ({ ...d, title: e.target.value }))}
                placeholder="Weekly notifuse report"
                value={draft.title}
              />
            </label>
            <label className="block space-y-1">
              <span className="text-[0.68rem] font-medium text-muted-foreground">
                What it does (one line — this is what teammates see in the catalog)
              </span>
              <Input
                onChange={e => setDraft(d => ({ ...d, description: e.target.value }))}
                placeholder="Pulls last week's campaign numbers into a Monday summary."
                value={draft.description}
              />
            </label>
            <label className="block space-y-1">
              <span className="text-[0.68rem] font-medium text-muted-foreground">SKILL.md content</span>
              <Textarea
                className="min-h-64 font-mono text-[0.72rem] leading-relaxed"
                onChange={e => setDraft(d => ({ ...d, content: e.target.value }))}
                value={draft.content}
              />
            </label>
            <div className="space-y-1">
              <span className="text-[0.68rem] font-medium text-muted-foreground">Wired MCPs (detected live from the text)</span>
              <McpChips skill={{ title: draft.title, description: draft.description, content: draft.content }} />
            </div>
            {selectedUser && (
              <PanelMeta
                rows={[
                  {
                    label: 'Lives at',
                    value: (
                      <code className="font-mono text-[0.68rem]">~/.hermes/skills/ix-user/{selectedUser.id}/SKILL.md</code>
                    )
                  },
                  {
                    label: 'Scope',
                    value: selectedUser.publishedId ? (
                      <span>
                        global — <code className="font-mono text-[0.68rem]">{selectedUser.publishedId}</code> on the
                        portal Skills.md page
                      </span>
                    ) : (
                      'user-level (this machine only) until you publish'
                    )
                  }
                ]}
              />
            )}
            <p className="text-[0.68rem] leading-relaxed text-muted-foreground/70">
              Saving makes the skill available immediately to local Hermes and the Copilot tab. Publishing pushes it
              to the admin portal&apos;s team catalog (requires the signed-in OTP session) where the whole org sees it.
            </p>
          </div>
        ) : selectedOrg ? (
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <h3 className="min-w-0 flex-1 truncate text-sm font-semibold text-foreground">{selectedOrg.title}</h3>
              {selectedOrg.superAdminOnly && <Badge variant="warn">super-admin</Badge>}
              {selectedOrg.content && onRunNatively && (
                <Button onClick={() => runNatively(selectedOrg)} size="sm">
                  <Codicon name="sparkle" size="0.8125rem" />
                  Run natively
                </Button>
              )}
            </div>
            <p className="text-xs leading-relaxed text-muted-foreground">{selectedOrg.description}</p>
            <div className="space-y-1">
              <span className="text-[0.68rem] font-medium text-muted-foreground">Wired MCPs this playbook spans</span>
              <McpChips skill={selectedOrg} />
            </div>
            <PanelMeta
              rows={[
                { label: 'Skill id', value: <code className="font-mono text-[0.68rem]">{selectedOrg.id}</code> },
                { label: 'Audience', value: selectedOrg.persona || 'all tiers' }
              ]}
            />
            <p className="text-[0.68rem] leading-relaxed text-muted-foreground/70">
              Playbooks run in the IX Agency admin portal copilot — or natively in the Copilot tab (&quot;Run
              natively&quot; injects the full playbook into a new conversation). Local agent skills (the SKILL.md
              folders under skills/) live in the Capabilities page.
            </p>
          </div>
        ) : (
          <PanelEmpty
            description="Browse the org's published playbooks, or hit New skill to write your own — it lands user-level first (this machine), then one click publishes it to the whole org's Skills.md catalog."
            icon="sparkle"
            title="Skills"
          />
        )}
      </DetailColumn>
    </MasterDetail>
  )
}
