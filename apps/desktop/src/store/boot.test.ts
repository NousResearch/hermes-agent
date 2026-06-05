import { afterEach, describe, expect, it } from 'vitest'

import { setRuntimeI18nLanguage } from '@/i18n'

import { $desktopBoot, failDesktopBoot } from './boot'

afterEach(() => {
  setRuntimeI18nLanguage('en')
})

describe('desktop boot store', () => {
  it('localizes boot failure detail messages in English', () => {
    setRuntimeI18nLanguage('en')

    failDesktopBoot('gateway unavailable')

    expect($desktopBoot.get().message).toBe('Desktop boot failed: gateway unavailable')
  })

  it('localizes boot failure detail messages in Simplified Chinese', () => {
    setRuntimeI18nLanguage('zh')

    failDesktopBoot('网关不可用')

    expect($desktopBoot.get().message).toBe('桌面启动失败：网关不可用')
  })
})
