/** Skill install targets shared by the public Skills Hub and Desktop's skill deep link. */
export interface SkillCatalogTarget {
  name: string
  source: string
  identifier?: string
  installIdentifier?: string
}

export function skillCatalogInstallIdentifier(skill: SkillCatalogTarget): string | null {
  if (skill.installIdentifier) {
    return skill.installIdentifier
  }

  const identifier = skill.identifier

  // Optional skills have a source-qualified name lookup. Other snapshots
  // without an identifier cannot name one exact community package.
  if (!identifier) {
    return skill.source === 'optional' ? `official/${skill.name}` : null
  }

  return skill.source.toLowerCase() === 'clawhub' && !identifier.startsWith('clawhub/')
    ? `clawhub/${identifier}`
    : identifier
}

export function skillCatalogInstallUrl(skill: SkillCatalogTarget): string | null {
  const identifier = skillCatalogInstallIdentifier(skill)

  return identifier ? `hermes://skill/install?${new URLSearchParams({ identifier })}` : null
}
