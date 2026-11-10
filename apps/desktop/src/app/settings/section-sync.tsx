import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { saveHermesConfig } from '@/hermes'
import { useI18n } from '@/i18n'
import { notify } from '@/store/notifications'
import { $profiles, normalizeProfileKey } from '@/store/profile'
import type { HermesConfigRecord } from '@/types/hermes'

import { getNested, setNested } from './helpers'

interface SectionSyncProps {
  fields: string[]
  profile: string
  source: HermesConfigRecord
}

function sectionPatch(source: HermesConfigRecord, fields: string[]): HermesConfigRecord {
  return fields.reduce<HermesConfigRecord>((patch, field) => {
    const value = getNested(source, field)

    return value === undefined ? patch : setNested(patch, field, value)
  }, {})
}

/** Copies only the fields rendered by the active config section to other profiles. */
export function SectionSync({ fields, profile, source }: SectionSyncProps) {
  const { t } = useI18n()
  const copy = t.settings.sectionSync
  const profiles = useStore($profiles)
  const sourceKey = normalizeProfileKey(profile)

  const targets = useMemo(
    () => profiles.filter(item => normalizeProfileKey(item.name) !== sourceKey),
    [profiles, sourceKey]
  )

  const [open, setOpen] = useState(false)
  const [selected, setSelected] = useState<Set<string>>(() => new Set(targets.map(target => target.name)))
  const [error, setError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    if (!open) {
      return
    }

    const currentTargets = new Set(targets.map(target => target.name))

    setSelected(previous => new Set([...previous].filter(target => currentTargets.has(target))))
  }, [open, targets])

  if (targets.length === 0 || fields.length === 0) {
    return null
  }

  const toggle = (name: string) => {
    setSelected(previous => {
      const next = new Set(previous)

      if (next.has(name)) {
        next.delete(name)
      } else {
        next.add(name)
      }

      return next
    })
  }

  const sync = async () => {
    const targetNames = new Set(targets.map(target => target.name))
    const activeSelected = [...selected].filter(target => targetNames.has(target))

    if (activeSelected.length === 0) {
      setError(copy.noTargets)

      return
    }

    setSaving(true)
    setError(null)
    const patch = sectionPatch(source, fields)

    const results = await Promise.allSettled(
      activeSelected.map(async target => {
        const result = await saveHermesConfig(patch, target)

        if (!result.ok) {
          throw new Error(copy.failed)
        }
      })
    )

    const succeeded = results.filter(result => result.status === 'fulfilled').length

    setSaving(false)

    if (succeeded !== results.length) {
      setError(copy.failed)

      return
    }

    notify({ kind: 'success', message: copy.saved(succeeded), title: copy.action })
    setOpen(false)
  }

  return (
    <div className="mb-5 rounded-lg border border-(--ui-stroke-tertiary) p-3">
      <div className="flex items-center justify-between gap-3">
        <p className="text-[length:var(--conversation-caption-font-size)] text-(--ui-text-secondary)">
          {copy.description}
        </p>
        <Button
          onClick={() => {
            setError(null)
            setSelected(new Set(targets.map(target => target.name)))
            setOpen(value => !value)
          }}
          size="sm"
          variant="outline"
        >
          {copy.action}
        </Button>
      </div>
      {open ? (
        <div className="mt-3 grid gap-2">
          <label className="flex items-center gap-2 text-sm">
            <input
              checked={selected.size === targets.length}
              onChange={() =>
                setSelected(selected.size === targets.length ? new Set() : new Set(targets.map(target => target.name)))
              }
              type="checkbox"
            />
            {copy.selectAll}
          </label>
          {targets.map(target => (
            <label className="flex items-center gap-2 text-sm" key={target.name}>
              <input checked={selected.has(target.name)} onChange={() => toggle(target.name)} type="checkbox" />
              {target.name}
            </label>
          ))}
          {error ? (
            <p className="text-sm text-(--ui-danger)" role="status">
              {error}
            </p>
          ) : null}
          <div className="flex gap-2">
            <Button disabled={saving} onClick={() => void sync()} size="sm">
              {copy.apply}
            </Button>
            <Button disabled={saving} onClick={() => setOpen(false)} size="sm" variant="ghost">
              {copy.cancel}
            </Button>
          </div>
        </div>
      ) : null}
    </div>
  )
}
