import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from '@/hermes'

import { $confirmRequest, runConfirm, settleConfirm } from './confirm'
import { $connectionsRegistry } from './connection-registry-state'
import type * as HubActions from './hub-actions'
import { installHubSkill } from './hub-actions'
import { notify } from './notifications'
import { requestSkillInstallFromDeepLink } from './skill-deeplink-install'

vi.mock('./hub-actions', async importOriginal => ({
  ...(await importOriginal<typeof HubActions>()),
  installHubSkill: vi.fn()
}))
vi.mock('./notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))

beforeEach(() => {
  setApiRequestConnection('local')
  setApiRequestProfile('research')
  vi.mocked(installHubSkill).mockResolvedValue(undefined)
})

afterEach(() => {
  settleConfirm(false)
  $connectionsRegistry.set(null)
  setApiRequestConnection(null)
  setApiRequestProfile(null)
  vi.clearAllMocks()
})

describe('skill link installation', () => {
  it('shows the exact source and readable destination, and cancellation installs nothing', async () => {
    const pending = requestSkillInstallFromDeepLink('clawhub/apple-design')
    expect($confirmRequest.get()).toMatchObject({
      title: 'Install “apple-design”?',
      details: [
        { label: 'Source', value: 'clawhub/apple-design' },
        { label: 'Install to', value: 'This computer · research' }
      ]
    })
    settleConfirm(false)
    await pending
    expect(installHubSkill).not.toHaveBeenCalled()
    expect(notify).not.toHaveBeenCalled()
  })

  it('uses the registered remote name without changing the install destination', async () => {
    setApiRequestConnection('server-a')
    $connectionsRegistry.set({
      version: 2, primary: 'server-a', secureTokenStorage: false,
      connections: [{ id: 'server-a', kind: 'remote', label: 'Homelab', tokenSet: false, tokenPreview: null }]
    })
    const pending = requestSkillInstallFromDeepLink('official/research/example')
    const request = $confirmRequest.get()!
    expect(request.details?.[1].value).toBe('Homelab · research')
    await runConfirm(request)
    expect(installHubSkill).toHaveBeenCalledExactlyOnceWith('official/research/example', {
      connectionId: 'server-a', profile: 'research'
    })
    expect(notify).toHaveBeenCalledWith(expect.objectContaining({ title: '“example” installed' }))
    settleConfirm(true)
    await pending
  })

  it('keeps a failed action retryable without sending a success notification', async () => {
    vi.mocked(installHubSkill).mockRejectedValueOnce(new Error('Download failed'))
    const pending = requestSkillInstallFromDeepLink('clawhub/apple-design')
    const request = $confirmRequest.get()!
    await expect(runConfirm(request)).rejects.toThrow('Download failed')
    expect($confirmRequest.get()).toBe(request)
    expect(request.phase).toBeUndefined()
    expect(notify).not.toHaveBeenCalled()
    await runConfirm(request)
    expect(request.phase).toBe('done')
    expect(notify).toHaveBeenCalledExactlyOnceWith(expect.objectContaining({ title: '“apple-design” installed' }))
    settleConfirm(true)
    await pending
  })

  it('refuses a changed destination before installation or retry', async () => {
    const pending = requestSkillInstallFromDeepLink('clawhub/apple-design')
    setApiRequestConnection('server-b')
    await expect(runConfirm($confirmRequest.get()!)).rejects.toThrow('The destination changed')
    expect(installHubSkill).not.toHaveBeenCalled()
    settleConfirm(false)
    await pending
  })

  it('does not report completion when a profile switch abandons polling', async () => {
    vi.mocked(installHubSkill).mockImplementationOnce(async () => { setApiRequestProfile('other') })
    const pending = requestSkillInstallFromDeepLink('clawhub/apple-design')
    const request = $confirmRequest.get()!
    await expect(runConfirm(request)).rejects.toThrow('The destination changed')
    expect(request.phase).toBeUndefined()
    expect(notify).not.toHaveBeenCalled()
    settleConfirm(false)
    await pending
  })
})
