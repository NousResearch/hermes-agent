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
  useQueryClient
} from '@hermes/plugin-sdk'
import { useState } from 'react'

import {
  autoDescribeProfile,
  fetchOrchestration,
  fetchProfiles,
  ORCHESTRATION_KEY,
  PROFILES_KEY,
  saveOrchestration,
  saveProfileDescription
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
  const [draft, setDraft] = useState(profile.description)
  const invalidate = () => void qc.invalidateQueries({ queryKey: PROFILES_KEY })

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
        {profile.is_default && <span className="ml-1 text-[0.625rem] text-(--ui-text-quaternary)">{k.defaultParen}</span>}
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
  const { data: settings } = useQuery({ queryKey: ORCHESTRATION_KEY, queryFn: fetchOrchestration })
  const { data: roster } = useQuery({ queryKey: PROFILES_KEY, queryFn: fetchProfiles, staleTime: 60_000 })

  const save = useMutation({
    mutationFn: (patch: Record<string, unknown>) => saveOrchestration(patch),
    onError: err => host.notify({ kind: 'error', message: errText(err) }),
    onSuccess: () => void qc.invalidateQueries({ queryKey: ORCHESTRATION_KEY })
  })

  if (!settings || !roster) {
    return null
  }

  return (
    <div className="flex flex-col gap-4 border-t border-(--ui-stroke-tertiary) px-4 py-3">
      <div className="flex flex-wrap items-end gap-4">
        <ProfilePicker
          label={k.orchestratorProfile}
          onSave={name => save.mutate({ orchestrator_profile: name })}
          profiles={roster.profiles}
          value={settings.orchestrator_profile}
        />
        <ProfilePicker
          label={k.defaultAssignee}
          onSave={name => save.mutate({ default_assignee: name })}
          profiles={roster.profiles}
          value={settings.default_assignee}
        />
        <label className="flex cursor-pointer items-center gap-2 pb-1.5 text-[0.75rem] text-(--ui-text-secondary)">
          <Switch
            aria-label={k.autoDecompose}
            checked={settings.auto_decompose}
            onCheckedChange={checked => save.mutate({ auto_decompose: checked })}
            size="xs"
          />
          {k.autoDecompose}
        </label>
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
