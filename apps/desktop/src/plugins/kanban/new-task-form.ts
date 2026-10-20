export interface NewTaskMetadataInput {
  priority: string
  projectId: string
  selectedSkills: string[]
  skillsText: string
}

export interface NewTaskMetadata {
  priority: number
  project_id?: string
  skills?: string[]
}

export const DEFAULT_SKILL_SUGGESTIONS = [
  'sam-project-orchestration',
  'hermes-agent',
  'github-pr-workflow',
  'systematic-debugging',
  'test-driven-development',
  'deferred-reminders',
  'user-facing-agent-communication',
  'youtube-content',
  'recording-transcription-pipelines',
  'yootheme-development'
]

export function parseSkills(selectedSkills: string[], skillsText: string): string[] {
  const seen = new Set<string>()

  const values = [...selectedSkills, ...skillsText.split(',')]
    .map(value => value.trim())
    .filter(Boolean)

  return values.filter(value => {
    if (seen.has(value)) {
      return false
    }

    seen.add(value)

    return true
  })
}

export function toggleSkill(selectedSkills: string[], skill: string): string[] {
  return selectedSkills.includes(skill) ? selectedSkills.filter(value => value !== skill) : [...selectedSkills, skill]
}

export function buildCreateTaskMetadata(input: NewTaskMetadataInput): NewTaskMetadata {
  const priority = Number(input.priority) || 0
  const skills = parseSkills(input.selectedSkills, input.skillsText)

  return {
    priority,
    ...(input.projectId ? { project_id: input.projectId } : {}),
    ...(skills.length ? { skills } : {})
  }
}
