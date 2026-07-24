import { describe, expect, it } from 'vitest'

import { en } from './en'
import { ko } from './ko'
import type { Translations } from './types'

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function translationKeyPaths(value: unknown, prefix = ''): string[] {
  if (!isRecord(value)) {
    return prefix ? [prefix] : []
  }

  return Object.keys(value)
    .sort()
    .flatMap(key => translationKeyPaths(value[key], prefix ? `${prefix}.${key}` : key))
}

describe('desktop Korean locale resource', () => {
  it('exports the desktop translation resource shape', () => {
    const resource: Translations = ko

    expect(Object.keys(resource).sort()).toEqual(Object.keys(en).sort())
    expect(resource.language.label).toBe('언어')
    expect(resource.settings.appearance.title).toBe('외관')
    expect(resource.notifications.more(3)).toBe('알림 3개 더 보기')
    expect(resource.boot.desktopBootFailedWithMessage('IPC')).toBe('데스크톱 부팅 실패: IPC')
  })

  it('includes every default locale translation key', () => {
    const koreanPaths = new Set(translationKeyPaths(ko))
    const missingPaths = translationKeyPaths(en).filter(path => !koreanPaths.has(path))

    expect(missingPaths).toEqual([])
  })
})
