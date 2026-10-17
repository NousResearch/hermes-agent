import type { Translations } from '@/i18n/types'
import type { OfficialSkillInfo, SkillInfo } from '@/types/hermes'

/** Presentation only: same-named user/hub skills keep their own description. */
export function skillDescription(skill: SkillInfo, copy?: Translations['skills']): string {
  return skill.provenance === 'bundled'
    ? (copy?.skillDescriptions?.[skill.name] ?? skill.description)
    : skill.description
}

export function officialSkillDescription(skill: OfficialSkillInfo, copy?: Translations['skills']): string {
  return copy?.skillDescriptions?.[skill.name] ?? skill.description
}
