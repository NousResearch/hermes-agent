import { describe, expect, it } from 'vitest'

import { zh } from '@/i18n/zh'

import { formatStageName, friendlyBootError, stageLabel } from './boot-install-format'

describe('formatStageName', () => {
  it('title-cases hyphenated ids and leaves short ids alone', () => {
    expect(formatStageName('system-packages')).toBe('System packages')
    expect(formatStageName('uv')).toBe('uv')
    expect(formatStageName('repository')).toBe('Repository')
  })
})

describe('stageLabel', () => {
  const labels = zh.install.stageLabels

  it('maps known installer stage ids to localized labels', () => {
    expect(stageLabel('prerequisites', labels)).toBe('前置环境')
    expect(stageLabel('repository', labels)).toBe('拉取程序')
    expect(stageLabel('venv', labels)).toBe('Python 环境')
    expect(stageLabel('python-deps', labels)).toBe('Python 依赖')
    expect(stageLabel('node-deps', labels)).toBe('Node 依赖')
    expect(stageLabel('config', labels)).toBe('写入配置')
    expect(stageLabel('setup', labels)).toBe('初始化')
    expect(stageLabel('gateway', labels)).toBe('启动网关')
    expect(stageLabel('complete', labels)).toBe('完成')
  })

  it('maps the windows-specific prerequisite sub-stages too', () => {
    expect(stageLabel('uv', labels)).toBe('前置环境')
    expect(stageLabel('system-packages', labels)).toBe('前置环境')
    expect(stageLabel('dependencies', labels)).toBe('Python 依赖')
    expect(stageLabel('config-templates', labels)).toBe('写入配置')
    expect(stageLabel('bootstrap-marker', labels)).toBe('完成')
  })

  it('falls back to formatStageName for unknown ids', () => {
    expect(stageLabel('some-future-stage', labels)).toBe('Some future stage')
  })
})

describe('friendlyBootError', () => {
  const copy = zh.boot.failure.errorMap

  it('returns null for empty/blank input', () => {
    expect(friendlyBootError(null, copy)).toBeNull()
    expect(friendlyBootError(undefined, copy)).toBeNull()
    expect(friendlyBootError('   ', copy)).toBeNull()
  })

  it('maps a cancelled-by-user transcript to the friendly cancel message', () => {
    const raw =
      "Error invoking remote method 'hermes:connection': Error: Hermes bootstrap failed at stage " +
      "'prerequisites': cancelled by user. Check /Users/x/logs/desktop.log for the full transcript."

    expect(friendlyBootError(raw, copy)).toBe('安装已取消')
  })

  it('maps the plain cancel strings', () => {
    expect(friendlyBootError('Hermes install was cancelled.', copy)).toBe('安装已取消')
    expect(friendlyBootError('bootstrap cancelled by user', copy)).toBe('安装已取消')
  })

  it('maps network failures', () => {
    expect(friendlyBootError('getaddrinfo ENOTFOUND github.com', copy)).toBe(copy.network)
    expect(friendlyBootError('Failed to download dependencies (timed out)', copy)).toBe(copy.network)
  })

  it('maps a non-cancel prerequisites failure', () => {
    const raw = "Hermes bootstrap failed at stage 'prerequisites': git not found."

    expect(friendlyBootError(raw, copy)).toBe(copy.prerequisites)
  })

  it('falls back to the generic message for anything else', () => {
    expect(friendlyBootError('some unexpected boot error', copy)).toBe(copy.unknown)
  })

  it('does not leak "Hermes" into the friendly messages', () => {
    for (const value of Object.values(copy)) {
      expect(value).not.toMatch(/Hermes/i)
    }
  })
})
