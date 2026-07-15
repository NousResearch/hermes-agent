import { existsSync, readFileSync } from 'node:fs'
import { resolve } from 'node:path'

import { describe, expect, it } from 'vitest'

import { validateStudyProject, validateStudySchedule } from './study-schemas'

interface ContractFixture {
  data: unknown
  kind: 'project' | 'schedule'
  name: string
  valid: boolean
}

const rootFixturePath = resolve(process.cwd(), 'plugins/study_os/contracts/fixtures.json')
const fixturesPath = existsSync(rootFixturePath)
  ? rootFixturePath
  : resolve(process.cwd(), '../../plugins/study_os/contracts/fixtures.json')

const fixtures = JSON.parse(readFileSync(fixturesPath, 'utf8')) as {
  cases: ContractFixture[]
}

describe('generated StudyOS contract guards', () => {
  it.each(fixtures.cases)('$name', fixture => {
    const result = fixture.kind === 'project' ? validateStudyProject(fixture.data) : validateStudySchedule(fixture.data)

    expect(result.ok).toBe(fixture.valid)
  })

  it('reports generated single-object schedule invariants', () => {
    const fixture = fixtures.cases.find(item => item.name === 'schedule-invalid-duration')
    const result = validateStudySchedule(fixture?.data)

    expect(result.ok).toBe(false)

    if (result.ok) {
      throw new Error('Expected invalid schedule fixture')
    }

    expect(result.errors).toContain('events[0].duration_minutes does not match start/end')
  })
})
