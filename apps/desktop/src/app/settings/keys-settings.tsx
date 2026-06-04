import { useCallback, useEffect, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Input } from '@/components/ui/input'
import { deleteEnvVar, getEnvVars, revealEnvVar, setEnvVar } from '@/hermes'
import { Check, Eye, EyeOff, Save, Settings2, Trash2, Zap } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { notify, notifyError } from '@/store/notifications'
import type { EnvVarInfo } from '@/types/hermes'

import { CONTROL_TEXT } from './constants'
import {
  asText,
  includesQuery,
  prettyName,
  providerGroup,
  providerPriority,
  redactedValue,
  withoutKey
} from './helpers'
import { LoadingState, Pill, SectionHeading, SettingsContent } from './primitives'
import type { EnvPatch, EnvRowProps, ProviderGroup, SearchProps } from './types'

interface EnvActionsProps {
  varKey: string
  info: EnvVarInfo
  saving: string | null
  onEdit: () => void
  onClear: (key: string) => void
  onReveal: (key: string) => void
  isRevealed: boolean
  showReveal?: boolean
}

function EnvActions({
  varKey,
  info,
  saving,
  onEdit,
  onClear,
  onReveal,
  isRevealed,
  showReveal = true
}: EnvActionsProps) {
  return (
    <div className="flex shrink-0 items-center gap-1.5">
      {info.url && (
        <Button asChild size="xs" title="Open provider docs" variant="ghost">
          <a href={info.url} rel="noreferrer" target="_blank">
            Docs
          </a>
        </Button>
      )}
      {info.is_set && showReveal && (
        <Button
          onClick={() => onReveal(varKey)}
          size="icon-xs"
          title={isRevealed ? 'Hide value' : 'Reveal value'}
          variant="ghost"
        >
          {isRevealed ? <EyeOff /> : <Eye />}
        </Button>
      )}
      <Button onClick={onEdit} size="xs" variant="outline">
        {info.is_set ? 'Replace' : 'Set'}
      </Button>
      {info.is_set && (
        <Button
          disabled={saving === varKey}
          onClick={() => onClear(varKey)}
          size="icon-xs"
          title="Clear value"
          variant="ghost"
        >
          <Trash2 />
        </Button>
      )}
    </div>
  )
}

function EnvVarRow({
  varKey,
  info,
  edits,
  revealed,
  saving,
  setEdits,
  onSave,
  onClear,
  onReveal,
  compact = false
}: EnvRowProps) {
  const isEditing = edits[varKey] !== undefined
  const isRevealed = revealed[varKey] !== undefined
  const value = isRevealed ? revealed[varKey] : info.redacted_value
  const startEdit = () => setEdits(c => ({ ...c, [varKey]: '' }))

  if (compact && !isEditing) {
    return (
      <div className="flex items-center justify-between gap-3 py-1.5">
        <div className="min-w-0">
          <div className="truncate font-mono text-[0.72rem] text-muted-foreground">{varKey}</div>
          <div className="truncate text-[0.68rem] text-muted-foreground/70">{info.description}</div>
        </div>
        <EnvActions
          info={info}
          isRevealed={isRevealed}
          onClear={onClear}
          onEdit={startEdit}
          onReveal={onReveal}
          saving={saving}
          showReveal={false}
          varKey={varKey}
        />
      </div>
    )
  }

  return (
    <div className="grid gap-2 rounded-xl bg-background/55 p-3">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-mono text-xs font-medium">{varKey}</span>
            <Pill tone={info.is_set ? 'primary' : 'muted'}>
              {info.is_set && <Check className="size-3" />}
              {info.is_set ? 'Set' : 'Not set'}
            </Pill>
          </div>
          <p className="mt-1 text-xs leading-5 text-muted-foreground">{info.description}</p>
        </div>
        <EnvActions
          info={info}
          isRevealed={isRevealed}
          onClear={onClear}
          onEdit={startEdit}
          onReveal={onReveal}
          saving={saving}
          varKey={varKey}
        />
      </div>

      {!isEditing && info.is_set && (
        <div
          className={cn(
            'rounded-md px-3 py-2 font-mono text-xs',
            isRevealed ? 'bg-background text-foreground' : 'bg-muted/30 text-muted-foreground'
          )}
        >
          {value || '---'}
        </div>
      )}

      {isEditing && (
        <div className="flex flex-wrap items-center gap-2">
          <Input
            autoFocus
            className={cn('min-w-56 flex-1 font-mono', CONTROL_TEXT)}
            onChange={e => setEdits(c => ({ ...c, [varKey]: e.target.value }))}
            placeholder={info.is_set ? 'Replace current value' : 'Enter value'}
            type={info.is_password ? 'password' : 'text'}
            value={edits[varKey]}
          />
          <Button disabled={saving === varKey || !edits[varKey]} onClick={() => onSave(varKey)} size="sm">
            <Save />
            {saving === varKey ? 'Saving' : 'Save'}
          </Button>
          <Button onClick={() => setEdits(c => withoutKey(c, varKey))} size="sm" variant="outline">
            <Codicon name="close" />
            Cancel
          </Button>
        </div>
      )}
    </div>
  )
}

function EnvProviderGroup({
  group,
  rowProps
}: {
  group: ProviderGroup
  rowProps: Omit<EnvRowProps, 'varKey' | 'info'>
}) {
  const setCount = group.entries.filter(([, info]) => info.is_set).length
  // Default-expand providers that already have at least one key set; the
  // user is much more likely to be coming back to edit those than to start
  // configuring a fresh provider from scratch.
  const [expanded, setExpanded] = useState(setCount > 0)

  return (
    <div className="overflow-hidden rounded-xl bg-background/60">
      <button
        className="flex w-full items-center justify-between gap-3 bg-transparent px-3 py-2.5 text-left hover:bg-accent/50"
        onClick={() => setExpanded(e => !e)}
        type="button"
      >
        <span className="flex min-w-0 items-center gap-2">
          <Zap className="size-4 shrink-0 text-muted-foreground" />
          <span className="truncate text-sm font-medium">
            {group.name === 'Other' ? 'Other providers' : group.name}
          </span>
          {setCount > 0 && <Pill tone="primary">{setCount} set</Pill>}
        </span>
        <span className="text-xs text-muted-foreground">{group.entries.length} keys</span>
      </button>
      {expanded && (
        <div className="grid gap-2 bg-muted/20 p-3">
          {group.entries.map(([key, info]) => (
            <EnvVarRow compact={!info.is_set} info={info} key={key} varKey={key} {...rowProps} />
          ))}
        </div>
      )}
    </div>
  )
}

export function KeysSettings({ query }: SearchProps) {
  const [vars, setVars] = useState<Record<string, EnvVarInfo> | null>(null)
  const [edits, setEdits] = useState<Record<string, string>>({})
  const [revealed, setRevealed] = useState<Record<string, string>>({})
  const [saving, setSaving] = useState<string | null>(null)

  // We used to hide ~80% of rows behind a global "Show advanced" toggle, but
  // everything in this view is configuration-level — "advanced" was a poor
  // distinction. The full list is rendered now and provider groups
  // default-collapsed-unless-set keep the surface manageable.
  useEffect(() => {
    try {
      window.localStorage.removeItem('desktop.settings.keys.show_advanced')
    } catch {
      // Ignore — old key cleanup is best-effort.
    }
  }, [])

  useEffect(() => {
    let cancelled = false

    void (async () => {
      try {
        const next = await getEnvVars()

        if (!cancelled) {
          setVars(next)
        }
      } catch (err) {
        notifyError(err, 'API keys failed to load')
      }
    })()

    return () => void (cancelled = true)
  }, [])

  const filterEnv = useCallback((info: EnvVarInfo, key: string, q: string, cat: string, extra?: string) => {
    if (asText(info.category) !== cat) {
      return false
    }

    if (!q) {
      return true
    }

    return (
      key.toLowerCase().includes(q) ||
      includesQuery(info.description, q) ||
      Boolean(extra && extra.toLowerCase().includes(q))
    )
  }, [])

  const providerGroups = useMemo<ProviderGroup[]>(() => {
    if (!vars) {
      return []
    }

    const q = query.trim().toLowerCase()

    const entries = Object.entries(vars).filter(([key, info]) =>
      filterEnv(info, key, q, 'provider', providerGroup(key))
    )

    const groups = new Map<string, [string, EnvVarInfo][]>()

    for (const entry of entries) {
      const name = providerGroup(entry[0])
      groups.set(name, [...(groups.get(name) ?? []), entry])
    }

    return Array.from(groups, ([name, entries]) => ({
      name,
      priority: providerPriority(name),
      entries: entries.sort(([a], [b]) => a.localeCompare(b)),
      hasAnySet: entries.some(([, info]) => info.is_set)
    })).sort((a, b) => a.priority - b.priority || a.name.localeCompare(b.name))
  }, [filterEnv, query, vars])

  const otherGroups = useMemo(() => {
    if (!vars) {
      return []
    }

    const q = query.trim().toLowerCase()

    const labels: Record<string, string> = {
      tool: 'Tools',
      messaging: 'Messaging',
      setting: 'Settings'
    }

    return ['tool', 'messaging', 'setting'].flatMap(cat => {
      const entries = Object.entries(vars)
        .filter(([key, info]) => filterEnv(info, key, q, cat))
        .sort(([a], [b]) => a.localeCompare(b))

      return entries.length === 0 ? [] : [{ category: cat, label: labels[cat] ?? prettyName(cat), entries }]
    })
  }, [filterEnv, query, vars])

  function patchVar(key: string, patch: EnvPatch) {
    setVars(c => (c ? { ...c, [key]: { ...c[key], ...patch } } : c))
  }

  function clearLocalState(key: string) {
    setEdits(c => withoutKey(c, key))
    setRevealed(c => withoutKey(c, key))
  }

  async function handleSave(key: string) {
    const value = edits[key]

    if (!value) {
      return
    }

    setSaving(key)

    try {
      await setEnvVar(key, value)
      patchVar(key, { is_set: true, redacted_value: redactedValue(value) })
      clearLocalState(key)
      notify({ kind: 'success', title: 'Credential saved', message: `${key} updated.` })
    } catch (err) {
      notifyError(err, `Failed to save ${key}`)
    } finally {
      setSaving(null)
    }
  }

  async function handleClear(key: string) {
    if (!window.confirm(`Remove ${key} from .env?`)) {
      return
    }

    setSaving(key)

    try {
      await deleteEnvVar(key)
      patchVar(key, { is_set: false, redacted_value: null })
      clearLocalState(key)
      notify({ kind: 'success', title: 'Credential removed', message: `${key} removed.` })
    } catch (err) {
      notifyError(err, `Failed to remove ${key}`)
    } finally {
      setSaving(null)
    }
  }

  async function handleReveal(key: string) {
    if (revealed[key]) {
      setRevealed(c => withoutKey(c, key))

      return
    }

    try {
      const result = await revealEnvVar(key)
      setRevealed(c => ({ ...c, [key]: result.value }))
    } catch (err) {
      notifyError(err, `Failed to reveal ${key}`)
    }
  }

  if (!vars) {
    return <LoadingState label="Loading API keys and credentials..." />
  }

  const rowProps = {
    edits,
    revealed,
    saving,
    setEdits,
    onSave: handleSave,
    onClear: handleClear,
    onReveal: handleReveal
  }

  const configuredCount = providerGroups.filter(g => g.hasAnySet).length

  return (
    <SettingsContent>
      <div className="mb-6">
        <SectionHeading
          icon={Zap}
          meta={`${configuredCount} of ${providerGroups.length} configured`}
          title="LLM providers"
        />
        <div className="grid gap-2">
          {providerGroups.map(group => (
            <EnvProviderGroup group={group} key={group.name} rowProps={rowProps} />
          ))}
        </div>
      </div>

      {otherGroups.map(group => (
        <div className="mb-6" key={group.category}>
          <SectionHeading
            icon={Settings2}
            meta={`${group.entries.filter(([, i]) => i.is_set).length} of ${group.entries.length} set`}
            title={group.label}
          />
          <div className="grid gap-2">
            {group.entries.map(([key, info]) => (
              <EnvVarRow info={info} key={key} varKey={key} {...rowProps} />
            ))}
          </div>
        </div>
      ))}
    </SettingsContent>
  )
}
