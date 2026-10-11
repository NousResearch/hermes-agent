/**
 * Orchestration settings — the dashboard's dispatcher-knobs panel, flat-styled:
 * orchestrator profile, default assignee, auto-decompose, and the profile
 * descriptions the decomposer routes by (save / auto-generate per profile).
 */

import {
  Button,
  Codicon,
  host,
  Input,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  Switch,
  useMutation,
  useQuery,
  useQueryClient,
  useValue
} from '@hermes/plugin-sdk'
import { useState } from 'react'

import {
  $boardSlug,
  autoDescribeProfile,
  boardsKey,
  fetchBoards,
  fetchOrchestration,
  fetchProfiles,
  orchestrationKey,
  profilesKey,
  saveOrchestration,
  saveProfileDescription,
  useKanbanScope
} from './api'
import type { KanbanProfile } from './types'
import { errText, FIELD_LABEL, useKanban } from './ui'

const DEFAULT_SENTINEL = '__default__'

function ProfilePicker({
  label,
  onSave,
  profiles,
  value
}: {
  label: string
  onSave: (name: string) => void
  profiles: KanbanProfile[]
  value: string
}) {
  const k = useKanban()

  return (
    <label className="flex min-w-0 flex-col gap-1">
      <span className={FIELD_LABEL}>{label}</span>
      <Select onValueChange={name => onSave(name === DEFAULT_SENTINEL ? '' : name)} value={value || DEFAULT_SENTINEL}>
        <SelectTrigger className="w-44">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value={DEFAULT_SENTINEL}>{k.defaultParen}</SelectItem>
          {profiles.map(profile => (
            <SelectItem key={profile.name} value={profile.name}>
              {profile.name}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </label>
  )
}

function ProfileDescriptionRow({ profile }: { profile: KanbanProfile }) {
  const k = useKanban()
  const qc = useQueryClient()
  const scope = useKanbanScope()
  const [draft, setDraft] = useState(profile.description)
  const invalidate = () => void qc.invalidateQueries({ queryKey: profilesKey(scope) })

  const save = useMutation({
    mutationFn: () => saveProfileDescription(profile.name, draft.trim()),
    onError: err => host.notify({ kind: 'error', message: errText(err) }),
    onSuccess: invalidate
  })

  const auto = useMutation({
    mutationFn: () => autoDescribeProfile(profile.name),
    onError: err => host.notify({ kind: 'error', message: errText(err) }),
    onSuccess: result => {
      if (result.ok) {
        setDraft(result.description ?? '')
        invalidate()
      } else {
        host.notify({ kind: 'warning', message: result.reason || 'Auto-describe failed' })
      }
    }
  })

  return (
    <div className="flex items-center gap-2">
      <span className="w-24 shrink-0 truncate text-[0.75rem] font-medium text-(--ui-text-secondary)">
        {profile.name}
        {profile.is_default && (
          <span className="ml-1 text-[0.625rem] text-(--ui-text-quaternary)">{k.defaultParen}</span>
        )}
      </span>
      <Input
        className="h-7 flex-1 text-[0.71rem]"
        onChange={event => setDraft(event.target.value)}
        placeholder={k.profileGoodAt}
        value={draft}
      />
      <Button
        disabled={save.isPending || draft.trim() === profile.description}
        onClick={() => save.mutate()}
        size="xs"
        variant="outline"
      >
        {k.save}
      </Button>
      {/* Overlay the spinner so the button keeps its "Auto" width — the aux
          model can take a few seconds and a text swap would jump the row. */}
      <Button className="relative" disabled={auto.isPending} onClick={() => auto.mutate()} size="xs" variant="ghost">
        <span className={auto.isPending ? 'invisible' : ''}>{k.auto}</span>
        {auto.isPending && (
          <span className="absolute inset-0 grid place-items-center">
            <Codicon className="animate-spin [animation-duration:1.2s]" name="loading" size="0.75rem" />
          </span>
        )}
      </Button>
    </div>
  )
}

export function OrchestrationPanel() {
  const k = useKanban()
  const qc = useQueryClient()
  const scope = useKanbanScope()
  const slug = useValue($boardSlug)
  const { data: boards } = useQuery({ queryKey: boardsKey(scope), queryFn: fetchBoards, staleTime: 30_000 })
  const [globalMode, setGlobalMode] = useState(false)

  // The panel scopes to the EFFECTIVE board in view: an explicit selection, else
  // the machine's current board. No board yet (no boards data) degrades to the
  // global scope. The scope switch lets the operator opt back out to global.
  const effectiveBoard = slug || boards?.current || ''
  const boardMode = !globalMode && Boolean(effectiveBoard)
  const querySlug = boardMode ? effectiveBoard : ''
  const key = orchestrationKey(scope, querySlug)
  const { data: settings } = useQuery({ queryKey: key, queryFn: () => fetchOrchestration(querySlug) })
  const { data: roster } = useQuery({ queryKey: profilesKey(scope), queryFn: fetchProfiles, staleTime: 60_000 })

  const save = useMutation({
    mutationFn: (patch: Record<string, unknown>) => saveOrchestration(querySlug, patch),
    onError: err => host.notify({ kind: 'error', message: errText(err) }),
    onSuccess: () => void qc.invalidateQueries({ queryKey: key })
  })

  if (!settings || !roster) {
    return null
  }

  // With a board in view the two profile knobs are board-first: the picker value
  // is the effective (board -> global) value and the tag says which one won.
  const boardOrchestrator = settings.board_orchestrator_profile ?? ''
  const boardDefault = settings.board_default_assignee ?? ''
  const hasBoardOverride = boardMode && Boolean(boardOrchestrator || boardDefault)
  const scopeTag = (overridden: boolean) => (overridden ? k.boardOverrideLabel : k.boardInheritLabel)

  const orchestratorLabel = boardMode
    ? `${k.orchestratorProfile} · ${scopeTag(Boolean(boardOrchestrator))}`
    : k.orchestratorProfile

  const assigneeLabel = boardMode
    ? `${k.defaultAssignee} · ${scopeTag(Boolean(boardDefault))}`
    : k.defaultAssignee

  return (
    <div className="flex flex-col gap-4 border-t border-(--ui-stroke-tertiary) px-4 py-3">
      {effectiveBoard && (
        <div className="flex items-center gap-1">
          <Button onClick={() => setGlobalMode(false)} size="xs" variant={boardMode ? 'outline' : 'ghost'}>
            {k.boardSettingsLabel}
          </Button>
          <Button onClick={() => setGlobalMode(true)} size="xs" variant={globalMode ? 'outline' : 'ghost'}>
            {k.globalScopeLabel}
          </Button>
        </div>
      )}
      {boardMode && (
        <div className="flex items-center justify-between gap-2">
          <span className={FIELD_LABEL}>{k.boardSettingsLabel}</span>
          {hasBoardOverride && (
            <Button
              onClick={() => save.mutate({ orchestrator_profile: '', default_assignee: '' })}
              size="xs"
              variant="ghost"
            >
              {k.clearBoardOverride}
            </Button>
          )}
        </div>
      )}
      <div className="flex flex-wrap items-end gap-4">
        <ProfilePicker
          label={orchestratorLabel}
          onSave={name => save.mutate({ orchestrator_profile: name })}
          profiles={roster.profiles}
          value={settings.orchestrator_profile}
        />
        <ProfilePicker
          label={assigneeLabel}
          onSave={name => save.mutate({ default_assignee: name })}
          profiles={roster.profiles}
          value={settings.default_assignee}
        />
        {/* auto_decompose / auto_promote_children stay global in v1, so the
            switch is only offered on the global scope. */}
        {!boardMode && (
          <label className="flex cursor-pointer items-center gap-2 pb-1.5 text-[0.75rem] text-(--ui-text-secondary)">
            <Switch
              aria-label={k.autoDecompose}
              checked={settings.auto_decompose}
              onCheckedChange={checked => save.mutate({ auto_decompose: checked })}
              size="xs"
            />
            {k.autoDecompose}
          </label>
        )}
      </div>

      <div className="flex flex-col gap-1.5">
        <span className={FIELD_LABEL}>{k.profileDescriptions}</span>
        <p className="text-[0.6875rem] text-(--ui-text-quaternary)">{k.profileDescriptionsHint}</p>
        {roster.profiles.map(profile => (
          <ProfileDescriptionRow key={`${profile.name}:${profile.description}`} profile={profile} />
        ))}
      </div>
    </div>
  )
}
