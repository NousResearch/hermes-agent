import { useEffect, useRef, useState } from 'react'

import { deleteEnvVar, getEnvVars, revealEnvVar, setEnvVar } from '@/hermes'
import type { ProfileScope } from '@/hermes'
import { useI18n } from '@/i18n'
import { type IconComponent } from '@/lib/icons'
import { confirm } from '@/store/confirm'
import { notify, notifyError } from '@/store/notifications'
import type { EnvVarInfo } from '@/types/hermes'

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
// mutation handlers instead of duplicating the plumbing. The immutable scope
// owns both gateway and profile; null fails closed while ownership is unknown.
export function useEnvCredentials(scope: null | ProfileScope): UseEnvCredentials {
  const { t } = useI18n()
  const credentials = t.settings.credentials
  const toolsets = t.settings.toolsets
  const [vars, setVars] = useState<Record<string, EnvVarInfo> | null>(null)
  const [edits, setEdits] = useState<Record<string, string>>({})
  const [revealed, setRevealed] = useState<Record<string, string>>({})
  const [saving, setSaving] = useState<string | null>(null)
  const scopeRef = useRef(scope)
  scopeRef.current = scope

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

    // Everything keyed by var name is dropped together with the reload: the
    // cached `vars`, plus any in-flight edit or revealed value. Those maps are
    // keyed by name alone, so leaving a draft behind after the target profile
    // changed left its Save button live — writing the value into the profile
    // now being targeted instead of the one it was typed for.
    setVars(null)
    setEdits({})
    setRevealed({})

    if (!scope) {
      return
    }
    void (async () => {
      try {
        const next = await getEnvVars(scope)

        if (!cancelled) {
          setVars(next)
        }
      } catch (err) {
        notifyError(err, t.settings.keys.failedLoad)
      }
    })()

    return () => void (cancelled = true)
    // eslint-disable-next-line react-hooks/exhaustive-deps -- reload per target profile; copy is stable
  }, [scope])

  function patchVar(key: string, patch: Partial<Pick<EnvVarInfo, 'is_set' | 'redacted_value'>>) {
    setVars(c => (c ? { ...c, [key]: { ...c[key], ...patch } } : c))
  }

  function clearLocalState(key: string) {
    setEdits(c => withoutKey(c, key))
    setRevealed(c => withoutKey(c, key))
  }

  async function handleSave(key: string) {
    const value = edits[key]
    const target = scope

    if (!value || !target) {
      return
    }

    setSaving(key)

    try {
      await setEnvVar(key, value, target)

      if (scopeRef.current !== target) {
        return
      }

      patchVar(key, { is_set: true, redacted_value: redactedValue(value) })
      clearLocalState(key)
      notify({ kind: 'success', title: toolsets.savedTitle, message: toolsets.savedMessage(key) })
    } catch (err) {
      notifyError(err, toolsets.failedSave(key))
    } finally {
      if (scopeRef.current === target) {
        setSaving(null)
      }
    }
  }

  // Direct save for a known value (no edit-state round-trip) — used by the
  // onboarding-style key form, which owns its own input. Returns a result so
  // the form can surface inline errors instead of only toasting.
  async function saveValue(key: string, value: string): Promise<{ message?: string; ok: boolean }> {
    const trimmed = value.trim()
    const target = scope

    if (!trimmed || !target) {
      return { message: credentials.enterValueFirst, ok: false }
    }

    setSaving(key)

    try {
      await setEnvVar(key, trimmed, target)

      if (scopeRef.current !== target) {
        return { message: credentials.couldNotSave, ok: false }
      }

      patchVar(key, { is_set: true, redacted_value: redactedValue(trimmed) })
      clearLocalState(key)
      notify({ kind: 'success', message: toolsets.savedMessage(key), title: toolsets.savedTitle })

      return { ok: true }
    } catch (err) {
      notifyError(err, toolsets.failedSave(key))

      return { message: err instanceof Error ? err.message : credentials.couldNotSave, ok: false }
    } finally {
      if (scopeRef.current === target) {
        setSaving(null)
      }
    }
  }

  async function handleClear(key: string) {
    const target = scope

    if (!target) {
      return
    }

    if (!(await confirm({ destructive: true, title: toolsets.removeConfirm(key) }))) {
      return
    }

    setSaving(key)

    try {
      await deleteEnvVar(key, target)

      if (scopeRef.current !== target) {
        return
      }

      patchVar(key, { is_set: false, redacted_value: null })
      clearLocalState(key)
      notify({ kind: 'success', title: toolsets.removedTitle, message: toolsets.removedMessage(key) })
    } catch (err) {
      notifyError(err, toolsets.failedRemove(key))
    } finally {
      if (scopeRef.current === target) {
        setSaving(null)
      }
    }
  }

  async function handleReveal(key: string) {
    const target = scope

    if (!target) {
      return
    }

    if (revealed[key]) {
      setRevealed(c => withoutKey(c, key))

      return
    }

    try {
      const result = await revealEnvVar(key, target)

      if (scopeRef.current !== target) {
        return
      }

      setRevealed(c => ({ ...c, [key]: result.value }))
    } catch (err) {
      notifyError(err, toolsets.failedReveal(key))
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

interface UseEnvCredentials {
  rowProps: Omit<EnvRowProps, 'varKey' | 'info'>
  saveValue: (key: string, value: string) => Promise<{ message?: string; ok: boolean }>
  vars: Record<string, EnvVarInfo> | null
}
