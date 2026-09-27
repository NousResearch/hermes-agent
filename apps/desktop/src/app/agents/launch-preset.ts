import type { PresetOwner } from './preset-store'
import type { PresetSkillAvailability, SubagentPreset } from './presets'

export interface PreparedPresetTask {
  id: string
  owner: PresetOwner
  preset_id: string
  text: string
}

const pending = new Map<string, PreparedPresetTask>()
let sequence = 0

const ownerKey = (owner: PresetOwner) => `${owner.connectionId || 'local'}\u0000${owner.profile.trim() || 'default'}`

export function preparePresetTask(
  owner: PresetOwner,
  preset: SubagentPreset,
  task: string,
  availability: readonly PresetSkillAvailability[]
): PreparedPresetTask {
  const capturedOwner = { connectionId: owner.connectionId || 'local', profile: owner.profile.trim() || 'default' }
  const resolvedSkills = availability.filter(skill => skill.state === 'enabled').map(skill => skill.name)

  const text = [
    `Jesteś Czesiek, koordynator rozmowy. Przekaż wykonanie zadania subagentowi przez dostępne narzędzie delegowania. Nie twierdź, że delegowano, dopóki narzędzie tego nie potwierdzi. Jeśli delegowanie jest niedostępne, poinformuj użytkownika.`,
    `Rozmawiaj naturalnie po polsku. Dopytaj o istotne niewiadome, a po raporcie subagenta podsumuj wynik i kolejny krok. Nie wymyślaj postępu ani zakończenia pracy.`,
    `Character: ${preset.character.trim()}`,
    resolvedSkills.length ? `Resolved skills: ${resolvedSkills.join(', ')}` : 'Resolved skills: none',
    `Task: ${task.trim()}`
  ].join('\n')

  const prepared = {
    id: `preset-task-${++sequence}`,
    owner: capturedOwner,
    preset_id: preset.id,
    text
  }

  pending.set(ownerKey(capturedOwner), prepared)

  return prepared
}

export function takePreparedPresetTask(owner: PresetOwner): PreparedPresetTask | null {
  const key = ownerKey(owner)
  const prepared = pending.get(key) ?? null

  if (prepared) {pending.delete(key)}

  return prepared
}

export function cancelPreparedPresetTask(id: string): void {
  for (const [key, prepared] of pending) {
    if (prepared.id === id) {pending.delete(key)}
  }
}
