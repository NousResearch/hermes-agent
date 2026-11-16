import { afterEach, describe, expect, it } from 'vitest'

import { $butterbarDismissed, $butterbarItems, dismissButterbar, registerButterbar } from './butterbar'

const offs: (() => void)[] = []

const register = (...args: Parameters<typeof registerButterbar>) => {
  const off = registerButterbar(...args)
  offs.push(off)

  return off
}

const ids = () => $butterbarItems.get().map(item => item.id)

afterEach(() => {
  offs.splice(0).forEach(off => off())
  $butterbarDismissed.set([])
})

describe('butterbar store', () => {
  it('is empty until something registers', () => {
    expect(ids()).toEqual([])
  })

  it('orders by priority, then registration order, and re-registering replaces in place', () => {
    register({ id: 'a', node: 'a' })
    register({ id: 'b', node: 'b', priority: 2 })
    register({ id: 'c', node: 'c' })
    register({ id: 'a', node: 'a2' })

    expect(ids()).toEqual(['b', 'a', 'c'])
    expect($butterbarItems.get()[1].node).toBe('a2')
  })

  it('a stale unregister does not remove the item that replaced it', () => {
    const off = register({ id: 'a', node: 'first' })
    register({ id: 'a', node: 'second' })
    off()

    expect($butterbarItems.get().map(item => item.node)).toEqual(['second'])
  })

  it('a keyed close persists across re-registration; a keyless close lasts this run', () => {
    const keyed = { id: 'keyed', node: 'k', persistKey: 'notice-v1' }
    const off = register(keyed)
    register({ id: 'keyless', node: 'x', closeable: true })

    dismissButterbar(keyed)
    dismissButterbar($butterbarItems.get().find(item => item.id === 'keyless')!)
    off()
    register(keyed)

    expect(ids()).toEqual([])
    expect($butterbarDismissed.get()).toEqual(['notice-v1'])

    register({ ...keyed, persistKey: 'notice-v2' })
    expect(ids()).toEqual(['keyed'])
  })
})
