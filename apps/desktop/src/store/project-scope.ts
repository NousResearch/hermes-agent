import { persistentAtom } from '@/lib/persisted'

// ── Project scope (the "you're inside a project" view, mirroring profile scope)─
// The sidebar's grouped view is a project switcher: ALL_PROJECTS shows the
// project overview (a list you drill into), and a concrete id means you've
// "entered" that project so only its worktrees/branches/sessions show. This is
// pure view state (localStorage), distinct from the durable active-project
// pointer in projects.db — though entering a project also makes it active so new
// chats land there, exactly as selecting a profile does.
//
// Project ids belong to ONE backend's projects.db, so switching profile or
// connection must leave the scope (store/profile, store/gateway-switch). It lives
// in this dependency-light module so those stores can reset it synchronously,
// before the fresh draft resolves its cwd from it.
export const ALL_PROJECTS = '__all_projects__'

const PROJECT_SCOPE_KEY = 'hermes.desktop.projectScope'

export const $projectScope = persistentAtom<string>(PROJECT_SCOPE_KEY, ALL_PROJECTS, {
  decode: raw => raw || ALL_PROJECTS,
  encode: value => value || ALL_PROJECTS
})

export function exitProjectScope(): void {
  $projectScope.set(ALL_PROJECTS)
}
