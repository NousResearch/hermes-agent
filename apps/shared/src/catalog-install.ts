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

export function skillCatalogInstallUrl(skill: SkillCatalogTarget): string | null {
  const identifier = skillCatalogInstallIdentifier(skill)

  return identifier ? `hermes://skill/install?${new URLSearchParams({ identifier })}` : null
}

export function pluginCatalogInstallUrl(plugin: { name: string }): string {
  return `hermes://plugin/install?${new URLSearchParams({ catalog: plugin.name })}`
}

/** The link that hands this plugin to another Hermes user: a curated catalog
 *  name wins (the recipient's backend resolves the reviewed pin), a git
 *  repo otherwise. A hand-copied folder has neither — nothing to link. */
export function pluginInstallShareUrl(plugin: { catalogName?: string | null; repo?: string | null }): string | null {
  const catalogName = plugin.catalogName?.trim()

  if (catalogName) {
    return pluginCatalogInstallUrl({ name: catalogName })
  }

  const repo = plugin.repo?.trim()

  return repo ? `hermes://plugin/install?${new URLSearchParams({ repo })}` : null
}
