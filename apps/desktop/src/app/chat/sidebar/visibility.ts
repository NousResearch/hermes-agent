export function shouldShowFlatSessionsWithProjects(worktreeGroupingActive: boolean, inProject: boolean): boolean {
  return worktreeGroupingActive && !inProject
}
