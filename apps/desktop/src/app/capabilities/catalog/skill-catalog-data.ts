import type { OfficialSkillInfo, SkillHubSourcesResponse, SkillInfo } from '@/types/hermes'
import type { CatalogManagedInstallation, CatalogSource } from '@/types/skill-catalog'

import { usageOf } from '../skills/skills-data'

import { type CatalogEntry, parseCatalog } from './catalog-data'

export function buildSkillCatalog(skills: SkillInfo[], hubData: SkillHubSourcesResponse | undefined, officialData: { skills: OfficialSkillInfo[] } | undefined, updatesData: { updates: { name?: string }[] } | undefined, sources: CatalogSource[]) {
  const registeredSources = new Set(sources.map(source => source.id))

    const updatable = new Set((updatesData?.updates ?? []).map(row => row.name))
    const skillsById = new Map<string, SkillInfo>()
    const skillsByName = new Map(skills.filter(skill => !skill.provider).map(skill => [skill.name, skill]))
    const installedByName = new Map<string, CatalogEntry>()
    const installedByIdentifier = new Map<string, CatalogEntry>()
    const installedIdentifiers = new Set(Object.keys(hubData?.installed ?? {}))

    const localEntries = skills.flatMap(skill => parseCatalog(
      'skills',
      [{
        name: skill.name,
        description: skill.description,
        category: skill.category,
        source: skill.provider && skill.identifier && registeredSources.has(skill.identifier.split(':')[0]) ? skill.identifier.split(':')[0] : skill.provenance === 'bundled' ? 'built-in' : skill.provenance === 'hub' ? 'hub' : 'local',
        ...(skill.provider && skill.identifier && registeredSources.has(skill.identifier.split(':')[0]) ? { provider: skill.provider, identifier: skill.identifier, author: skill.provider.author, version: skill.installed_version } : {})
      }]
    ).map(entry => {

      const installed = {
        ...entry,
        id: entry.provider ? entry.identifier : `installed:${skill.name}`,
        ...(entry.provider ? { management: skill as CatalogManagedInstallation } : {}),
        installIdentifier: entry.provider ? entry.identifier : null,
        locked: !!skill.locked,
        usage: usageOf(skill),
        updateAvailable: entry.provider ? skill.installed_version !== skill.latest_version && !!skill.latest_version : skill.provenance === 'hub' && updatable.has(skill.name)
      }

      skillsById.set(installed.id, skill)
      installedByName.set(skill.name, installed)

      if (entry.provider) { installedByIdentifier.set(entry.identifier, installed) }

      return installed
    }))

    for (const [identifier, record] of Object.entries(hubData?.installed ?? {})) {
      const installed = record.name ? installedByName.get(record.name) : undefined

      if (installed && skillsById.get(installed.id)?.provenance === 'hub') {
        installedByIdentifier.set(identifier, installed)
      }
    }

    const official = officialData?.skills ?? []

    const officialEntries = parseCatalog(
      'skills',
      official.map(skill => ({
        ...skill,
        source: 'optional',
        installIdentifier: skill.identifier
      }))
    )

    const officialByIdentifier = new Map(officialEntries.map(entry => [entry.identifier, entry]))

    for (const skill of official) {
      if (!skill.installed) {
        continue
      }

      installedIdentifiers.add(skill.identifier)
      const installed = installedByName.get(skill.name)

      if (installed && skillsById.get(installed.id)?.provenance === 'hub') {
        installedByIdentifier.set(skill.identifier, installed)
      }
    }

    const officialFor = (entry: CatalogEntry) => {
      if (entry.source !== 'optional') {
        return undefined
      }

      const identifier = entry.installIdentifier ?? entry.identifier
      const exact = officialByIdentifier.get(identifier)

      if (exact) {
        return exact
      }

      if (identifier !== entry.name && identifier !== `official/${entry.name}`) {
        return undefined
      }

      // Old public snapshots omit optional identifiers. Only the official
      // optional namespace can use this fallback, never community lookalikes.
      return officialEntries.find(skill => skill.name === entry.name && skill.category === entry.category)
    }

    const matchInstalled = (entry: CatalogEntry): CatalogEntry | undefined => {
      const exact = installedByIdentifier.get(entry.installIdentifier ?? entry.identifier)

      if (exact) {
        return exact
      }

      if (entry.source === 'built-in') {
        const bundled = installedByName.get(entry.name)

        if (bundled && skillsById.get(bundled.id)?.provenance === 'bundled') {
          return bundled
        }
      }

      const optional = officialFor(entry)

      return optional ? (installedByIdentifier.get(optional.identifier) ?? optional) : undefined
    }

    // CatalogBrowser merges these supplemental rows with the public feed.
    // Optional rows remain available even offline / ahead of the snapshot;
    // membership here is NOT installation — isInstalled owns that decision.
    const entries = [
      ...localEntries,
      ...officialEntries.filter(entry => !installedByIdentifier.has(entry.identifier) && !skillsByName.has(entry.name))
    ]

    return { entries, skillsById, skillsByName, installedIdentifiers, matchInstalled, officialFor }

}
