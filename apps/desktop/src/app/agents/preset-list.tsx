import { useState } from 'react'
import { useNavigate } from 'react-router'

import { NEW_CHAT_ROUTE } from '@/app/routes'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'

import { preparePresetTask } from './launch-preset'
import type { PresetOwner } from './preset-store'
import { type PresetSkillAvailability, skillAvailability, type SubagentPreset } from './presets'

export function PresetList({ owner, presets, skills, onEdit }: { owner: PresetOwner; presets: readonly SubagentPreset[]; skills: readonly { name: string; enabled: boolean }[]; onEdit: (preset: SubagentPreset) => void }) {
  const navigate = useNavigate()
  const [tasks, setTasks] = useState<Record<string, string>>({})
  const [availability, setAvailability] = useState<Record<string, PresetSkillAvailability[]>>({})

  const inspect = (preset: SubagentPreset) => {
    setAvailability(current => ({ ...current, [preset.id]: skillAvailability(preset.skills, skills) }))
  }

  const prepare = (preset: SubagentPreset) => {
    const task = (tasks[preset.id] ?? '').trim()

    if (!task) {return}
    const resolved = availability[preset.id] ?? skillAvailability(preset.skills, skills)
    preparePresetTask(owner, preset, task, resolved)
    navigate(NEW_CHAT_ROUTE)
  }

  return (
    <section className="grid gap-3">
      <div className="flex items-center justify-between gap-2"><h2 className="text-sm font-semibold">Gotowi do pomocy</h2><span className="text-xs text-muted-foreground">{presets.length} ról</span></div>
      {presets.map(preset => {
        const states = availability[preset.id]

        return (
          <article className="grid gap-2 rounded-lg border border-border/60 p-3" key={preset.id}>
            <div className="flex items-start justify-between gap-2"><div><h3 className="text-sm font-medium">{preset.name}</h3><p className="mt-1 text-xs leading-relaxed text-muted-foreground">{preset.character}</p></div><Button onClick={() => onEdit(preset)} size="xs" variant="ghost">Edytuj</Button></div>
            <div className="flex flex-wrap gap-1">{preset.skills.map(skill => <span className="rounded bg-muted px-1.5 py-0.5 text-[0.65rem] text-muted-foreground" key={skill.name}>{skill.name}{states?.find(item => item.name === skill.name)?.state === 'missing' ? ' · niedostępny' : ''}</span>)}</div>
            {states?.some(item => item.state !== 'enabled') ? <p className="text-xs text-amber-600 dark:text-amber-400">Niektóre skille są niedostępne. Włącz je w Narzędziach przed zleceniem zadania.</p> : null}
            <div className="flex gap-2"><Input aria-label={`${preset.name} task`} onChange={event => setTasks(current => ({ ...current, [preset.id]: event.target.value }))} placeholder="Co chcesz zlecić?" value={tasks[preset.id] ?? ''} /><Button disabled={!tasks[preset.id]?.trim()} onClick={() => prepare(preset)} size="sm">Przygotuj zadanie</Button></div>
            {!states ? <Button className="justify-self-start" onClick={() => inspect(preset)} size="xs" variant="text">Sprawdź skille</Button> : null}
          </article>
        )
      })}
    </section>
  )
}
