import { describe, expect, it, vi } from 'vitest'

import { configureChatGptSession } from './chatgpt-session'

describe('configureChatGptSession', () => {
  it('denies guest permissions and downloads while preserving the isolated cookie session', () => {
    let permissionRequest: ((permission: string, callback: (allowed: boolean) => void) => void) | undefined
    let permissionCheck: ((permission: string) => boolean) | undefined
    let downloadHandler: ((event: { preventDefault: () => void }) => void) | undefined

    configureChatGptSession({
      setPermissionCheckHandler(handler) {
        permissionCheck = (_permission: string) => handler(null, _permission)
      },
      setPermissionRequestHandler(handler) {
        permissionRequest = (_permission: string, callback: (allowed: boolean) => void) =>
          handler(null, _permission, callback, {})
      },
      on(event, handler) {
        expect(event).toBe('will-download')
        downloadHandler = handler
      }
    })

    const permissionCallback = vi.fn()
    permissionRequest?.('clipboard-read', permissionCallback)
    expect(permissionCallback).toHaveBeenCalledWith(false)
    expect(permissionCheck?.('media')).toBe(false)

    const preventDefault = vi.fn()
    downloadHandler?.({ preventDefault })
    expect(preventDefault).toHaveBeenCalledOnce()
  })
})
