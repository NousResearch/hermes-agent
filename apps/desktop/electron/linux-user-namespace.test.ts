import assert from 'node:assert/strict'

import { test } from 'vitest'

import { detectUserNamespaceSandbox, isUserNamespaceRestriction } from './linux-user-namespace'

const CLONE_GATE = '/proc/sys/kernel/unprivileged_userns_clone'
const MAX_NS_GATE = '/proc/sys/user/max_user_namespaces'

test('isUserNamespaceRestriction follows the Debian clone gate first', () => {
  assert.equal(
    isUserNamespaceRestriction(() => '1'),
    false
  )
  assert.equal(
    isUserNamespaceRestriction(() => '0'),
    true
  )
  assert.equal(
    isUserNamespaceRestriction(() => '0\n'),
    true
  )
  assert.equal(
    isUserNamespaceRestriction(() => 'N'),
    true
  )

  // Not a Debian-derived kernel: fall through to the namespace count.
  const values: Record<string, string> = { [MAX_NS_GATE]: '15000' }
  assert.equal(
    isUserNamespaceRestriction(path => values[path] ?? null),
    false
  )
})

test('isUserNamespaceRestriction treats a zero namespace budget as restricted', () => {
  const values: Record<string, string> = { [MAX_NS_GATE]: '0' }
  assert.equal(
    isUserNamespaceRestriction(path => values[path] ?? null),
    true
  )
})

test('isUserNamespaceRestriction is permissive when no gate answers', () => {
  // An unreadable gate is not a restriction — the kernel is not blocking
  // unprivileged namespaces, we just could not ask.
  assert.equal(
    isUserNamespaceRestriction(() => null),
    false
  )
  assert.equal(
    isUserNamespaceRestriction(() => ''),
    false
  )
  assert.equal(
    isUserNamespaceRestriction(() => {
      throw new Error('EACCES')
    }),
    false
  )
})

test('detectUserNamespaceSandbox only answers for Linux', () => {
  assert.equal(detectUserNamespaceSandbox({ platform: 'darwin', readFileSync: () => '1' }), false)
  assert.equal(detectUserNamespaceSandbox({ platform: 'win32', readFileSync: () => '1' }), false)

  const values: Record<string, string> = { [CLONE_GATE]: '1' }
  assert.equal(detectUserNamespaceSandbox({ platform: 'linux', readFileSync: path => values[path] ?? '' }), true)
})

test('detectUserNamespaceSandbox reports a restricted kernel', () => {
  const values: Record<string, string> = { [CLONE_GATE]: '0' }
  assert.equal(detectUserNamespaceSandbox({ platform: 'linux', readFileSync: path => values[path] ?? '' }), false)
})

test('detectUserNamespaceSandbox survives an unreadable gate', () => {
  assert.equal(
    detectUserNamespaceSandbox({
      platform: 'linux',
      readFileSync: () => {
        throw new Error('ENOENT')
      }
    }),
    true
  )
})
