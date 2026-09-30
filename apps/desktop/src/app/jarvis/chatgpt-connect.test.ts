import { describe, expect, it, vi } from 'vitest'

import { type ChatGptConnectDeps, chatGptConnected, chatGptWorkModel, connectChatGpt } from './chatgpt-connect'

const provider = (models: string[], authenticated = true) => ({ authenticated, models, name: 'ChatGPT', slug: 'openai-codex' })

describe('chatGptWorkModel', () => {
  it('takes the newest generation the subscription lists and skips small and dated variants', () => {
    expect(chatGptWorkModel([provider(['gpt-5.5', 'gpt-6-mini', 'gpt-6', 'gpt-5.6'])])).toBe('gpt-6')
    expect(chatGptWorkModel([provider(['gpt-5.4-mini', 'gpt-5.5-2026-08-01', 'gpt-5.5'])])).toBe('gpt-5.5')
  })

  it('falls back to any listed model, and to nothing when not signed in or empty', () => {
    expect(chatGptWorkModel([provider(['codex-max'])])).toBe('codex-max')
    expect(chatGptWorkModel([provider(['gpt-6'], false)])).toBeUndefined()
    expect(chatGptWorkModel([provider([])])).toBeUndefined()
    expect(chatGptConnected([provider(['gpt-6'])])).toBe(true)
    expect(chatGptConnected(undefined)).toBe(false)
  })
})

function deps(statuses: string[], overrides: Partial<ChatGptConnectDeps> = {}): ChatGptConnectDeps {
  const queue = [...statuses]

  return {
    cancel: vi.fn().mockResolvedValue(undefined),
    loadOptions: vi.fn().mockResolvedValue({ providers: [provider(['gpt-5.5', 'gpt-6'])] }),
    openUrl: vi.fn(),
    poll: vi.fn(async () => ({ session_id: 's', status: (queue.shift() ?? 'pending') as never })),
    setDefaultModel: vi.fn().mockResolvedValue(undefined),
    sleep: vi.fn().mockResolvedValue(undefined),
    start: vi.fn().mockResolvedValue({ expires_in: 600, flow: 'device_code', poll_interval: 3, session_id: 's', user_code: 'ABCD-1234', verification_url: 'https://auth.openai.com/codex/device' }),
    ...overrides
  }
}

describe('connectChatGpt', () => {
  it('shows the code, opens the page, waits through pending, then lands on the best model', async () => {
    const d = deps(['pending', 'pending', 'approved'])
    const onCode = vi.fn()

    const result = await connectChatGpt(d, undefined, { onCode })

    expect(onCode).toHaveBeenCalledWith({ url: 'https://auth.openai.com/codex/device', userCode: 'ABCD-1234' })
    expect(d.openUrl).toHaveBeenCalledWith('https://auth.openai.com/codex/device')
    expect(result).toMatchObject({ model: 'gpt-6', ok: true })
    expect(d.setDefaultModel).toHaveBeenCalledWith('openai-codex', 'gpt-6')
  })

  it('reports a refusal or expiry without picking a model', async () => {
    expect(await connectChatGpt(deps(['denied']), undefined, { onCode: vi.fn() })).toMatchObject({ ok: false, reason: 'denied' })

    const d = deps(['expired'])

    expect(await connectChatGpt(d, undefined, { onCode: vi.fn() })).toMatchObject({ ok: false, reason: 'expired' })
    expect(d.setDefaultModel).not.toHaveBeenCalled()
  })

  it('cancels the server-side session when the user gives up, and survives a failing start', async () => {
    const controller = new AbortController()
    const d = deps([], { sleep: vi.fn(async () => controller.abort()) })

    expect(await connectChatGpt(d, undefined, { onCode: vi.fn(), signal: controller.signal })).toMatchObject({ ok: false, reason: 'cancelled' })
    expect(d.cancel).toHaveBeenCalledWith('s')

    const broken = deps([], { start: vi.fn().mockRejectedValue(new Error('no network')) })

    expect(await connectChatGpt(broken, undefined, { onCode: vi.fn() })).toMatchObject({ message: 'no network', ok: false, reason: 'error' })
  })
})
