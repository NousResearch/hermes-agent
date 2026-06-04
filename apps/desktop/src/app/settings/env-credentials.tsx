import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Input } from '@/components/ui/input'
import { deleteEnvVar, getEnvVars, revealEnvVar, setEnvVar } from '@/hermes'
import { Check, Eye, EyeOff, type IconComponent, Save, Trash2 } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { notify, notifyError } from '@/store/notifications'
import type { EnvVarInfo } from '@/types/hermes'

import { CONTROL_TEXT } from './constants'
import { asText, includesQuery, redactedValue, withoutKey } from './helpers'
import { Pill } from './primitives'
import type { EnvRowProps } from './types'

// Shared filter used by every credential surface (Providers + Keys pages):
// category gate first, then a free-text match across key name + description.
export function filterEnv(info: EnvVarInfo, key: string, q: string, cat: string, extra?: string): boolean {
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

export function EnvVarRow({
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
    <div className="grid gap-2 rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-tertiary)/20 p-3">
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

export function SettingsCategoryHeading({ count, icon: Icon, title }: CategoryHeadingProps) {
  return (
    <div className="mb-3 flex items-center gap-2 text-[length:var(--conversation-text-font-size)] font-medium">
      <Icon className="size-4 text-muted-foreground" />
      <span>{title}</span>
      {count && <Pill>{count}</Pill>}
    </div>
  )
}

// Owns the env-var fetch + the edit/reveal/save/delete lifecycle so multiple
// credential pages (Providers, Keys) share one source of truth and one set of
// mutation handlers instead of duplicating the plumbing.
export function useEnvCredentials(): UseEnvCredentials {
  const [vars, setVars] = useState<Record<string, EnvVarInfo> | null>(null)
  const [edits, setEdits] = useState<Record<string, string>>({})
  const [revealed, setRevealed] = useState<Record<string, string>>({})
  const [saving, setSaving] = useState<string | null>(null)

  // Best-effort cleanup of a retired localStorage flag (global "Show
  // advanced" toggle) — everything in these views is configuration-level.
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

  function patchVar(key: string, patch: Partial<Pick<EnvVarInfo, 'is_set' | 'redacted_value'>>) {
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

  // Direct save for a known value (no edit-state round-trip) — used by the
  // onboarding-style key form, which owns its own input. Returns a result so
  // the form can surface inline errors instead of only toasting.
  async function saveValue(key: string, value: string): Promise<{ message?: string; ok: boolean }> {
    const trimmed = value.trim()

    if (!trimmed) {
      return { message: 'Enter a value first.', ok: false }
    }

    setSaving(key)

    try {
      await setEnvVar(key, trimmed)
      patchVar(key, { is_set: true, redacted_value: redactedValue(trimmed) })
      clearLocalState(key)
      notify({ kind: 'success', message: `${key} updated.`, title: 'Credential saved' })

      return { ok: true }
    } catch (err) {
      notifyError(err, `Failed to save ${key}`)

      return { message: err instanceof Error ? err.message : 'Could not save credential.', ok: false }
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

  return {
    saveValue,
    vars,
    rowProps: {
      edits,
      revealed,
      saving,
      setEdits,
      onSave: handleSave,
      onClear: handleClear,
      onReveal: handleReveal
    }
  }
}

interface CategoryHeadingProps {
  count?: string
  icon: IconComponent
  title: string
}

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

interface UseEnvCredentials {
  rowProps: Omit<EnvRowProps, 'varKey' | 'info'>
  saveValue: (key: string, value: string) => Promise<{ message?: string; ok: boolean }>
  vars: Record<string, EnvVarInfo> | null
}
