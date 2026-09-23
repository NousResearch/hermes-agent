// @vitest-environment jsdom
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { getCustomEndpoints } from '@/hermes'
import { ko } from '@/i18n/ko'

import { CustomEndpointsSettings } from './custom-endpoints-settings'

vi.mock('@/i18n', () => ({ useI18n: () => ({ t: ko }) }))

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getCustomEndpoints: vi.fn()
}))

afterEach(cleanup)

const endpoint = {
  api_key_preview: null,
  base_url: 'http://127.0.0.1:8081/v1',
  context_length: null,
  has_api_key: true,
  id: 'axet-proxy',
  is_current: true,
  model: 'qwen3-32b',
  models: [],
  name: 'Axet Proxy',
  source: 'custom-endpoints'
}

describe('custom endpoints in Korean', () => {
  it('translates the UI copy and leaves protocol and identifier values alone', async () => {
    vi.mocked(getCustomEndpoints).mockResolvedValue({ endpoints: [endpoint] } as never)

    render(<CustomEndpointsSettings />)

    await waitFor(() => expect(screen.getByText('사용 중')).toBeTruthy())

    // Ordinary UI copy is translated.
    expect(screen.getByText('자동 감지')).toBeTruthy()
    expect(screen.getByText('연결 테스트')).toBeTruthy()
    expect(screen.getByText('엔드포인트 URL')).toBeTruthy()
    expect(screen.getByText('새 대화에 사용')).toBeTruthy()

    // Wire-protocol names are identifiers, not copy, so they read the same everywhere.
    expect(screen.getByText('Chat Completions')).toBeTruthy()
    expect(screen.getByText('Responses API')).toBeTruthy()
    expect(screen.getByText('Anthropic Messages')).toBeTruthy()

    // So are the values the user typed and the sample values we suggest.
    expect(screen.getByDisplayValue('http://127.0.0.1:8081/v1')).toBeTruthy()
    expect(screen.getByDisplayValue('qwen3-32b')).toBeTruthy()
    expect(screen.getByPlaceholderText('axet-proxy')).toBeTruthy()
  })

  it('keeps the reasoning effort levels as the tokens providers use', () => {
    expect(ko.shell.modelOptions.low).toBe('low')
    expect(ko.shell.modelOptions.high).toBe('high')
    expect(ko.shell.modelOptions.xhigh).toBe('xhigh')
    expect(ko.shell.modelOptions.ultra).toBe('ultra')
    // `Fast` is an ordinary toggle rather than an effort token.
    expect(ko.shell.modelOptions.fast).toBe('빠름')
  })

  it('translates the Browser settings section, which falls back to its English label without a catalog entry', () => {
    expect(ko.settings.sections.browser).toBe('브라우저')
  })
})
