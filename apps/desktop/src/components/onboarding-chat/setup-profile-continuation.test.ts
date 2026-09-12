import { beforeEach, expect, it, vi } from 'vitest'

import type { GatewayRequest } from '@/app/session/hooks/use-prompt-actions/utils'
import { ensureSetupProfile } from './setup-profile'

beforeEach(() => vi.clearAllMocks())

it('marks an existing or new guide through the canonical backend onboarding entry', async () => {
  const request = vi.fn(async () => ({ ok: true }))
  await ensureSetupProfile(request as GatewayRequest)
  expect(request).toHaveBeenCalledOnce()
  expect(request).toHaveBeenCalledWith('profiles.ensure_onboarding', { soul: expect.any(String) })
})

it('falls back only for an older backend, not a failed authoritative onboarding write', async () => {
  const old = vi.fn().mockRejectedValueOnce(new Error('method not found -32601')).mockResolvedValueOnce({ ok: true })
  await ensureSetupProfile(old as GatewayRequest)
  expect(old).toHaveBeenLastCalledWith('profiles.create', expect.objectContaining({ name: 'hermes-setup', share_auth: true }))
  const offline = vi.fn().mockRejectedValue(new Error('Connection closed'))
  await expect(ensureSetupProfile(offline as GatewayRequest)).rejects.toThrow('Connection closed')
  expect(offline).toHaveBeenCalledOnce()
})
