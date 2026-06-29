import type { StudyProject, StudySchedule } from '@/types/hermes'

export type StudySchemaResult<T> = { ok: true; data: T } | { ok: false; errors: string[] }

const PROJECT_ID_RE = /^[a-z0-9][a-z0-9-]{2,63}$/
const SCHEDULE_ID_RE = /^[a-z0-9][a-z0-9-]{2,79}$/
const DATE_RE = /^\d{4}-\d{2}-\d{2}$/
const DATETIME_WITH_OFFSET_RE = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:[+-]\d{2}:\d{2}|Z)$/

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function requireString(data: Record<string, unknown>, key: string, errors: string[]): string | undefined {
  const value = data[key]
  if (typeof value !== 'string') {
    errors.push(`${key} must be a string`)
    return undefined
  }
  if (!value.trim()) {
    errors.push(`${key} must not be empty`)
    return undefined
  }
  return value
}

function parseDate(value: unknown, path: string, errors: string[]): string | undefined {
  if (typeof value !== 'string' || !DATE_RE.test(value)) {
    errors.push(`${path} must be ISO date YYYY-MM-DD`)
    return undefined
  }
  const parsed = new Date(`${value}T00:00:00Z`)
  if (Number.isNaN(parsed.getTime()) || parsed.toISOString().slice(0, 10) !== value) {
    errors.push(`${path} must be a valid ISO date`)
    return undefined
  }
  return value
}

function parseDateTime(value: unknown, path: string, errors: string[]): Date | undefined {
  if (typeof value !== 'string') {
    errors.push(`${path} must be a string`)
    return undefined
  }
  if (!DATETIME_WITH_OFFSET_RE.test(value)) {
    errors.push(`${path} must include timezone offset`)
    return undefined
  }
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) {
    errors.push(`${path} must be a valid ISO datetime`)
    return undefined
  }
  return parsed
}

function validateSubjects(value: unknown, errors: string[]): Set<string> {
  const subjectIds = new Set<string>()
  if (!Array.isArray(value) || value.length === 0) {
    errors.push('subjects must be a non-empty array')
    return subjectIds
  }
  value.forEach((subject, index) => {
    const path = `subjects[${index}]`
    if (!isRecord(subject)) {
      errors.push(`${path} must be an object`)
      return
    }
    const id = subject.id
    if (typeof id !== 'string' || !id.trim()) {
      errors.push(`${path}.id must be a non-empty string`)
    } else if (subjectIds.has(id)) {
      errors.push(`${path}.id must be unique`)
    } else {
      subjectIds.add(id)
    }
    if (typeof subject.label !== 'string' || !subject.label.trim()) {
      errors.push(`${path}.label must be a non-empty string`)
    }
    if (subject.target_score !== undefined && typeof subject.target_score !== 'number') {
      errors.push(`${path}.target_score must be a number`)
    }
  })
  return subjectIds
}

function validatePromptPolicy(value: unknown, errors: string[]): void {
  if (!isRecord(value)) {
    errors.push('prompt_policy must be an object')
    return
  }
  for (const key of [
    'base_max_chars',
    'intent_max_chars',
    'domain_max_chars',
    'project_summary_max_chars',
    'total_max_chars',
  ]) {
    if (!Number.isInteger(value[key]) || Number(value[key]) <= 0) {
      errors.push(`prompt_policy.${key} must be a positive integer`)
    }
  }
  if (value.updates_apply !== 'next_session') {
    errors.push('prompt_policy.updates_apply must be next_session')
  }
}

export function validateStudyProject(input: unknown): StudySchemaResult<StudyProject> {
  const errors: string[] = []
  if (!isRecord(input)) {
    return { ok: false, errors: ['project must be an object'] }
  }
  if (input.schema_version !== 'study_project.v1') {
    errors.push('schema_version must be study_project.v1')
  }
  const projectId = requireString(input, 'project_id', errors)
  if (projectId && !PROJECT_ID_RE.test(projectId)) {
    errors.push('project_id must match ^[a-z0-9][a-z0-9-]{2,63}$')
  }
  for (const key of ['title', 'domain', 'exam_type', 'timezone', 'phase', 'domain_pack', 'created_at', 'updated_at']) {
    requireString(input, key, errors)
  }
  parseDate(input.exam_date, 'exam_date', errors)
  validateSubjects(input.subjects, errors)
  validatePromptPolicy(input.prompt_policy, errors)
  parseDateTime(input.created_at, 'created_at', errors)
  parseDateTime(input.updated_at, 'updated_at', errors)

  return errors.length > 0 ? { ok: false, errors } : { ok: true, data: input as unknown as StudyProject }
}

export function validateStudySchedule(
  input: unknown,
  project?: StudyProject,
): StudySchemaResult<StudySchedule> {
  const errors: string[] = []
  if (!isRecord(input)) {
    return { ok: false, errors: ['schedule must be an object'] }
  }
  if (input.schema_version !== 'study_schedule.v1') {
    errors.push('schema_version must be study_schedule.v1')
  }
  const scheduleId = requireString(input, 'schedule_id', errors)
  if (scheduleId && !SCHEDULE_ID_RE.test(scheduleId)) {
    errors.push('schedule_id must match ^[a-z0-9][a-z0-9-]{2,79}$')
  }
  const projectId = requireString(input, 'project_id', errors)
  if (project && projectId && project.project_id !== projectId) {
    errors.push('project_id must match project manifest')
  }
  for (const key of ['title', 'timezone']) {
    requireString(input, key, errors)
  }
  const range = isRecord(input.range) ? input.range : undefined
  if (!range) {
    errors.push('range must be an object')
  }
  const rangeStart = range ? parseDate(range.start, 'range.start', errors) : undefined
  const rangeEnd = range ? parseDate(range.end, 'range.end', errors) : undefined
  if (rangeStart && rangeEnd && rangeEnd < rangeStart) {
    errors.push('range.end must be on or after range.start')
  }
  validatePhases(input.phases, errors)
  validateEvents(input.events, errors, rangeStart, rangeEnd, project)

  return errors.length > 0 ? { ok: false, errors } : { ok: true, data: input as unknown as StudySchedule }
}

function validatePhases(value: unknown, errors: string[]): void {
  if (!Array.isArray(value)) {
    errors.push('phases must be an array')
    return
  }
  value.forEach((phase, index) => {
    const path = `phases[${index}]`
    if (!isRecord(phase)) {
      errors.push(`${path} must be an object`)
      return
    }
    for (const key of ['id', 'title', 'goal']) {
      if (typeof phase[key] !== 'string' || !String(phase[key]).trim()) {
        errors.push(`${path}.${key} must be a non-empty string`)
      }
    }
    const start = parseDate(phase.start, `${path}.start`, errors)
    const end = parseDate(phase.end, `${path}.end`, errors)
    if (start && end && end < start) {
      errors.push(`${path}.end must be on or after start`)
    }
  })
}

function validateEvents(
  value: unknown,
  errors: string[],
  rangeStart: string | undefined,
  rangeEnd: string | undefined,
  project: StudyProject | undefined,
): void {
  if (!Array.isArray(value)) {
    errors.push('events must be an array')
    return
  }
  const subjectIds = project ? new Set(project.subjects.map(subject => subject.id)) : undefined
  const eventIds = new Set<string>()
  value.forEach((event, index) => {
    const path = `events[${index}]`
    if (!isRecord(event)) {
      errors.push(`${path} must be an object`)
      return
    }
    const id = event.id
    if (typeof id !== 'string' || !id.trim()) {
      errors.push(`${path}.id must be a non-empty string`)
    } else if (eventIds.has(id)) {
      errors.push(`${path}.id must be unique`)
    } else {
      eventIds.add(id)
    }
    for (const key of ['title', 'subject_id', 'type', 'status']) {
      if (typeof event[key] !== 'string' || !String(event[key]).trim()) {
        errors.push(`${path}.${key} must be a non-empty string`)
      }
    }
    if (subjectIds && typeof event.subject_id === 'string' && !subjectIds.has(event.subject_id)) {
      errors.push(`${path}.subject_id must exist in project subjects`)
    }
    const start = parseDateTime(event.start, `${path}.start`, errors)
    const end = parseDateTime(event.end, `${path}.end`, errors)
    if (!Number.isInteger(event.duration_minutes) || Number(event.duration_minutes) < 1 || Number(event.duration_minutes) > 720) {
      errors.push(`${path}.duration_minutes must be an integer from 1 to 720`)
    }
    if (start && end) {
      if (end <= start) {
        errors.push(`${path}.end must be after start`)
      }
      if (Number.isInteger(event.duration_minutes)) {
        const actualMinutes = Math.floor((end.getTime() - start.getTime()) / 60000)
        if (actualMinutes !== event.duration_minutes) {
          errors.push(`${path}.duration_minutes does not match start/end`)
        }
      }
      const startDate = String(event.start).slice(0, 10)
      const endDate = String(event.end).slice(0, 10)
      if (rangeStart && rangeEnd && (startDate < rangeStart || startDate > rangeEnd || endDate < rangeStart || endDate > rangeEnd)) {
        errors.push(`${path} must fall inside range`)
      }
    }
    if (!Array.isArray(event.goals) || !event.goals.every(goal => typeof goal === 'string' && goal.trim())) {
      errors.push(`${path}.goals must be an array of non-empty strings`)
    }
  })
}
