export interface SkillHubQueryState {
  pending: boolean
  showLanding: boolean
  showResults: boolean
}

export function getSkillHubQueryState(query: string, term: string): SkillHubQueryState {
  const normalizedQuery = query.trim()
  const showLanding = normalizedQuery.length === 0
  const showResults = !showLanding && normalizedQuery === term

  return {
    pending: !showLanding && !showResults,
    showLanding,
    showResults
  }
}
