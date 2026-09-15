import assert from 'node:assert/strict'

import { test } from 'vitest'

import { registerDeepLinkProtocolOutsideTests } from './deep-link-protocol-registration'

type RegistrationArgs = [protocol: string, executable?: string, args?: string[]]
type Registrar = (...args: RegistrationArgs) => boolean

const registrations: { mode: string; register: (registrar: Registrar) => void; expected: RegistrationArgs }[] = [
  {
    mode: 'dev',
    register: (registrar) => {
      registrar('hermes', '/synthetic/electron', ['/synthetic/desktop'])
    },
    expected: ['hermes', '/synthetic/electron', ['/synthetic/desktop']]
  },
  {
    mode: 'packaged',
    register: (registrar) => {
      registrar('hermes')
    },
    expected: ['hermes']
  }
]

for (const { mode, register, expected } of registrations) {
  for (const workerIndex of ['0', '', '3']) {
    test(`does not call the ${mode} protocol registrar when TEST_WORKER_INDEX is ${JSON.stringify(workerIndex)}`, () => {
      const calls: RegistrationArgs[] = []

      const registrar: Registrar = (...args) => {
        calls.push(args)

        return true
      }

      registerDeepLinkProtocolOutsideTests(workerIndex, () => {
        register(registrar)
      })

      assert.deepEqual(calls, [])
    })
  }

  test(`preserves the ${mode} protocol registration when TEST_WORKER_INDEX is absent`, () => {
    const calls: RegistrationArgs[] = []

    const registrar: Registrar = (...args) => {
      calls.push(args)

      return true
    }

    registerDeepLinkProtocolOutsideTests(undefined, () => {
      register(registrar)
    })

    assert.deepEqual(calls, [expected])
  })
}

test('leaves production registration errors to the existing caller', () => {
  const error = new Error('synthetic registration failure')

  assert.throws(() => {
    registerDeepLinkProtocolOutsideTests(undefined, () => {
      throw error
    })
  }, (thrown) => thrown === error)
})
