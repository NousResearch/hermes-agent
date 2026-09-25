import { describe, expect, it } from 'vitest'

import { ko } from './ko'
import type { Translations } from './types'

describe('desktop Korean locale resource', () => {
  it('exports the desktop translation resource shape', () => {
    const resource: Translations = ko

    expect(resource.language.label).toBe('언어')
    expect(resource.settings.appearance.title).toBe('외관')
    expect(resource.notifications.more(3)).toBe('알림 3개 더 보기')
    expect(resource.boot.desktopBootFailedWithMessage('IPC')).toBe('데스크톱 부팅 실패: IPC')
  })
})
