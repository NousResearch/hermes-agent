import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'

import type { SubagentPreset } from './presets'

export function PresetEditor({ preset, onCancel, onSave }: { preset: SubagentPreset | null; onCancel: () => void; onSave: (preset: SubagentPreset) => void }) {
  const [name, setName] = useState(preset?.name ?? '')
  const [character, setCharacter] = useState(preset?.character ?? '')
  const [skillNames, setSkillNames] = useState(preset?.skills.map(skill => skill.name).join(', ') ?? '')

  return (
    <form
      className="grid gap-3 rounded-lg border border-border/60 bg-muted/20 p-3"
      onSubmit={event => {
        event.preventDefault()
        const trimmedName = name.trim()
        const trimmedCharacter = character.trim()

        if (!trimmedName || !trimmedCharacter) {return}
        const now = new Date().toISOString()
        onSave({
          id: preset?.id ?? `custom-${Date.now()}`,
          name: trimmedName,
          character: trimmedCharacter,
          skills: [...new Set(skillNames.split(',').map(name => name.trim()).filter(Boolean))].map(name => ({ name })),
          created_at: preset?.created_at ?? now,
          updated_at: now
        })
      }}
    >
      <label className="grid gap-1 text-xs font-medium">Nazwa<Input aria-label="Preset name" onChange={event => setName(event.target.value)} value={name} /></label>
      <label className="grid gap-1 text-xs font-medium">Sposób pracy<textarea aria-label="Character instructions" className="min-h-24 rounded-md border border-input bg-background px-2 py-1.5 text-xs outline-none focus-visible:ring-2 focus-visible:ring-ring" onChange={event => setCharacter(event.target.value)} value={character} /></label>
      <label className="grid gap-1 text-sm font-medium">Skille do tej roli (nazwy oddziel przecinkami)<Input aria-label="Skille do roli" onChange={event => setSkillNames(event.target.value)} placeholder="raport-tygodniowy, research" value={skillNames} /></label>
      <div className="flex justify-end gap-2">
        <Button onClick={onCancel} size="sm" type="button" variant="ghost">Anuluj</Button>
        <Button size="sm" type="submit">Zapisz rolę</Button>
      </div>
    </form>
  )
}
