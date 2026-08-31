import { describe, expect, it, vi } from 'vitest'

vi.mock('@/app/settings/field-copy', () => ({ defineFieldCopy: (value: unknown) => value }))
vi.mock('@/app/settings/constants', () => ({ FIELD_DESCRIPTIONS: {}, FIELD_LABELS: {} }))

const [{ ar }, { en }, { ja }, { zhHant }, { zh }] = await Promise.all([
  import('./ar'),
  import('./en'),
  import('./ja'),
  import('./zh-hant'),
  import('./zh')
])

const locales = [en, ja, zh, zhHant, ar]

describe('todo status copy', () => {
  it('is complete in every supported locale', () => {
    for (const locale of locales) {
      const copy = locale.statusStack
      expect(copy.markDone).toBeTruthy()
      expect(copy.markDoneAria('task')).toContain('task')
      expect(copy.reopen).toBeTruthy()
      expect(copy.reopenAria('task')).toContain('task')
      expect(copy.retryTaskSync).toBeTruthy()
      expect(copy.syncingTask).toBeTruthy()
      expect(copy.markDoneTitle).toBeTruthy()
      expect(copy.reopenTitle).toBeTruthy()
      expect(copy.taskChanged).toBeTruthy()
      expect(copy.taskMissing).toBeTruthy()
      expect(copy.taskSessionMissing).toBeTruthy()
      expect(copy.taskSyncFailed).toBeTruthy()
      expect(copy.taskUpdateFailed).toBeTruthy()
      expect(copy.taskUpdatesUnavailable).toBeTruthy()
      expect(copy.statusPending).toBeTruthy()
      expect(copy.statusInProgress).toBeTruthy()
      expect(copy.statusCompleted).toBeTruthy()
      expect(copy.statusCancelled).toBeTruthy()
    }
  })
})
