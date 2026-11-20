/** Install targets shared by the public docs galleries and native catalog. */
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

  // Older snapshots omit identifiers for local skills. Optional skills have a
  // source-qualified name lookup; bundled skills are not in that source.
  if (!identifier) {
    return skill.source === 'built-in' ? null : skill.source === 'optional' ? `official/${skill.name}` : skill.name
  }

  return skill.source.toLowerCase() === 'clawhub' && !identifier.startsWith('clawhub/')
    ? `clawhub/${identifier}`
    : identifier
}
