import { persistentAtom } from '@/lib/persisted'

// Project ids are scoped by the active profile's projects.db. Keep the view
// atom in a dependency-light module so profile switching can clear a project
// id before the asynchronous gateway/catalog swap starts.
export const ALL_PROJECTS = '__all_projects__'

const PROJECT_SCOPE_KEY = 'hermes.desktop.projectScope'

export const $projectScope = persistentAtom<string>(PROJECT_SCOPE_KEY, ALL_PROJECTS, {
  decode: raw => raw || ALL_PROJECTS,
  encode: value => value || ALL_PROJECTS
})

export function resetProjectScope(): void {
  $projectScope.set(ALL_PROJECTS)
}
