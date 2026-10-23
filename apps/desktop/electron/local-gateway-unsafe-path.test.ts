import { expect, test, vi } from 'vitest'

import { redialLocalGateway } from './local-gateway'

test('an unsafe control path surfaces without forgetting and re-ensuring the same owner', async () => {
  const failure = new Error('Unsafe gateway control path')
  const ensure = vi.fn(async () => 'owner')
  const forget = vi.fn()
  const use = vi.fn(async () => { throw failure })

  await expect(redialLocalGateway({ ensure, forget, use })).rejects.toBe(failure)
  expect(ensure).toHaveBeenCalledOnce()
  expect(use).toHaveBeenCalledOnce()
  expect(forget).not.toHaveBeenCalled()
})

test.each(['Gateway ticket control socket missing', 'Invalid gateway control pointer', 'Noncanonical gateway profile'])(
  'a stale %s still refreshes its endpoint once', async message => {
    const ensure = vi.fn(async () => 'owner')
    const forget = vi.fn()
    const use = vi.fn().mockRejectedValueOnce(new Error(message)).mockResolvedValue('fresh ticket')

    await expect(redialLocalGateway({ ensure, forget, use })).resolves.toBe('fresh ticket')
    expect(ensure).toHaveBeenCalledTimes(2)
    expect(forget).toHaveBeenCalledOnce()
  }
)

// The unsafe refusal is not stale, so nothing re-ensures past it: it must name the path and the
// repair (like the group-writable home's `chmod g-w`), never the bare 'Unsafe gateway control path'.
test.skipIf(process.platform === 'win32')('each unsafe control-path refusal names the path and the command that repairs it', async () => {
  const fs = await import('node:fs/promises')
  const path = await import('node:path')
  const { isStaleLocalGatewayError, mintLocalGatewayTicket } = await import('./local-gateway')
  const { shortSocketTmpDir } = await import('./local-gateway.test-helpers')
  const home = await shortSocketTmpDir('desktop-unsafe-msg-')
  const endpoint = (profile_id: string) => ({ profile_id, instance_id: 'owner', authority_epoch: 1, runtime_protocol: 1, api_origin: 'http://127.0.0.1:1234', capabilities: ['session-authority-v1'], supervisor: 'none' })

  const failure = async (profile_id = home) => {
    const error = await mintLocalGatewayTicket(endpoint(profile_id)).catch(e => e)
    expect(isStaleLocalGatewayError(error)).toBe(false)

    return error.message
  }

  const sock = path.join(home, 'gateway.sock')
  const pointer = path.join(home, 'gateway.sock.path')

  try {
    await fs.chmod(home, 0o757)
    expect(await failure()).toBe(`Unsafe gateway control path: profile home '${home}' is writable by other users (mode 757): run chmod o-w '${home}'`)

    await fs.chmod(home, 0o775)
    expect(await failure()).toBe(`Unsafe gateway control path: profile home is group-writable and this Desktop cannot verify the group is private: run chmod g-w '${home}'`)

    await fs.chmod(home, 0o700)
    await fs.writeFile(sock, '', { mode: 0o600 })
    expect(await failure()).toBe(`Unsafe gateway control path: '${sock}' should be a socket but is a regular file: remove it, then run hermes gateway restart`)

    await fs.rm(sock)
    await fs.writeFile(pointer, '/nowhere/control.sock\n')
    await fs.chmod(pointer, 0o644)
    expect(await failure()).toBe(`Unsafe gateway control path: '${pointer}' is accessible to other users (mode 644): run chmod go-rwx '${pointer}'`)

    if (process.getuid?.() !== 0) {
      expect(await failure('/')).toBe(`Unsafe gateway control path: '/' is owned by uid 0, not this user (uid ${process.getuid?.()}): run sudo chown ${process.getuid?.()} '/'`)
    }
  } finally {
    await fs.chmod(home, 0o700)
    await fs.rm(home, { recursive: true, force: true })
  }
})
