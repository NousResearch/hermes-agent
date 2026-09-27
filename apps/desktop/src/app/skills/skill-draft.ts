export interface SkillDraft {
  name: string
  description: string
  category: string
  instructions: string
}

export function buildSkillDraft(draft: SkillDraft) {
  const name = draft.name.trim()
  const category = draft.category.trim()
  const instructions = draft.instructions.replace(/^\uFEFF/, '').trim()

  if (!/^[a-z0-9][a-z0-9_-]{0,63}$/.test(name)) {
    throw new Error('Nazwa: użyj 1–64 małych liter bez polskich znaków, cyfr i myślników.')
  }

  if (category && !/^[a-z0-9][a-z0-9_-]{0,63}$/.test(category)) {
    throw new Error('Kategoria może zawierać małe litery, cyfry i myślniki.')
  }

  if (!instructions) {
    throw new Error('Wklej instrukcję skilla albo opisz kolejne kroki.')
  }

  if (/^https?:\/\/\S+$/.test(instructions)) {
    throw new Error('Wklej skopiowaną instrukcję, nie sam adres strony.')
  }

  if (instructions.length > 99000) {
    throw new Error('Skill jest za długi. Maksymalnie 99 000 znaków.')
  }

  // Full documents are preserved; the backend parses YAML and scans the skill.
  if (instructions.startsWith('---')) {
    if (!/^---\r?\n[\s\S]+?\r?\n---\r?\n\s*\S/.test(instructions)) {
      throw new Error('Niekompletny SKILL.md: sprawdź nagłówek i treść instrukcji.')
    }

    return { name, category, content: instructions }
  }

  const description = draft.description.trim()

  if (!description || description.length > 60) {
    throw new Error('Opisz, kiedy używać skilla, w maksymalnie 60 znakach.')
  }

  return {
    name,
    category,
    content: `---\nname: ${JSON.stringify(name)}\ndescription: ${JSON.stringify(description)}\n---\n\n${instructions}\n`
  }
}
