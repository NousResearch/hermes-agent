---
name: expect-api
description: Assertions with matchers, asymmetric matchers, and custom matchers
---

# Expect API

Vitest uses Chai assertions with Jest-compatible API.

## Basic Assertions

```ts
import { expect, test } from 'vitest'

test('assertions', () => {
  // Equality
  expect(1 + 1).toBe(2)              // Strict equality (===)
  expect({ a: 1 }).toEqual({ a: 1 }) // Deep equality

  // Truthiness
  expect(true).toBeTruthy()
  expect(false).toBeFalsy()
  expect(null).toBeNull()
  expect(undefined).toBeUndefined()
  expect('value').toBeDefined()

  // Numbers
  expect(10).toBeGreaterThan(5)
  expect(10).toBeGreaterThanOrEqual(10)
  expect(5).toBeLessThan(10)
  expect(0.1 + 0.2).toBeCloseTo(0.3, 5)

  // Strings
  expect('hello world').toMatch(/world/)
  expect('hello').toContain('ell')

  // Arrays
  expect([1, 2, 3]).toContain(2)
  expect([{ a: 1 }]).toContainEqual({ a: 1 })
  expect([1, 2, 3]).toHaveLength(3)

  // Objects
  expect({ a: 1, b: 2 }).toHaveProperty('a')
  expect({ a: 1, b: 2 }).toHaveProperty('a', 1)
  expect({ a: { b: 1 } }).toHaveProperty('a.b', 1)
  expect({ a: 1 }).toMatchObject({ a: 1 })

  // Types
  expect('string').toBeTypeOf('string')
  expect(new Date()).toBeInstanceOf(Date)
})
```

## Negation

```ts
expect(1).not.toBe(2)
expect({ a: 1 }).not.toEqual({ a: 2 })
```

## Error Assertions

```ts
// Sync errors - wrap in function
expect(() => throwError()).toThrow()
expect(() => throwError()).toThrow('message')
expect(() => throwError()).toThrow(/pattern/)
expect(() => throwError()).toThrow(CustomError)

// Async errors - use rejects
await expect(asyncThrow()).rejects.toThrow('error')
```

## Promise Assertions

```ts
// Resolves
await expect(Promise.resolve(1)).resolves.toBe(1)
await expect(fetchData()).resolves.toEqual({ data: true })

// Rejects
await expect(Promise.reject('error')).rejects.toBe('error')
await expect(failingFetch()).rejects.toThrow()
```

## Spy/Mock Assertions

```ts
const fn = vi.fn()
fn('arg1', 'arg2')
fn('arg3')

expect(fn).toHaveBeenCalled()
expect(fn).toHaveBeenCalledTimes(2)
expect(fn).toHaveBeenCalledWith('arg1', 'arg2')
expect(fn).toHaveBeenLastCalledWith('arg3')
expect(fn).toHaveBeenNthCalledWith(1, 'arg1', 'arg2')

expect(fn).toHaveReturned()
expect(fn).toHaveReturnedWith(value)

// v4 additions
expect(fn).toHaveBeenCalledExactlyOnceWith('arg1', 'arg2')
expect(fnA).toHaveBeenCalledBefore(fnB)
expect(fnA).toHaveBeenCalledAfter(fnB)
```

### Chai-Style Spy Assertions (4.1+)

Sinon-chai-compatible aliases, useful when migrating from Sinon:

```ts
expect(spy).to.have.been.called
expect(spy).to.have.been.calledOnce
expect(spy).to.have.been.calledWith('arg1', 'arg2')
expect(spy).to.have.been.calledOnceWith('arg')
```

### Conditional Mock Exhaustion (v5)

Assert every `vi.when` behavior was consumed:

```ts
const w = vi.when(spy).calledWith(1).thenReturnOnce('a')
spy(1)
expect(w).toHaveBeenExhausted()
```

## Asymmetric Matchers

Use inside `toEqual`, `toHaveBeenCalledWith`, etc:

```ts
expect({ id: 1, name: 'test' }).toEqual({
  id: expect.any(Number),
  name: expect.any(String),
})

expect({ a: 1, b: 2, c: 3 }).toEqual(
  expect.objectContaining({ a: 1 })
)

expect([1, 2, 3, 4]).toEqual(
  expect.arrayContaining([1, 3])
)

expect('hello world').toEqual(
  expect.stringContaining('world')
)

expect('hello world').toEqual(
  expect.stringMatching(/world$/)
)

expect({ value: null }).toEqual({
  value: expect.anything() // Matches anything except null/undefined
})

// Negate with expect.not
expect([1, 2]).toEqual(
  expect.not.arrayContaining([3])
)

// toBeOneOf - value matches any option (great for optional props)
expect(user).toEqual({
  name: expect.any(String),
  middleName: expect.toBeOneOf([expect.any(String), undefined]),
})

// schemaMatching (4.0+) - matches any Standard Schema (Zod, Valibot, ArkType)
import { z } from 'zod'
expect(payload).toEqual({
  email: expect.schemaMatching(z.string().email()),
})
expect(repo.save).toHaveBeenCalledWith(expect.schemaMatching(UserSchema))
```

## Soft Assertions

Prefer `expect.soft` for **non-critical assertions** — it marks the test failed but continues so all failures are reported together:

```ts
expect.soft(response.status).toBe(200) // non-critical, keeps going
expect.soft(response.headers.get('x-id')).toBeTruthy()
expect(response.body).toBeDefined() // critical: hard expect stops on failure
```

## Type-Narrowing Assertions (4.0+)

`expect.assert` throws at runtime **and** narrows the TypeScript type (unlike `toBeTruthy`/`toBeDefined`, which return `void`):

```ts
const user = cache.get('alice') // { id, name } | undefined
expect.assert(user)             // throws if undefined, narrows below
expect(user.name).toBe('Alice') // no `!`, no `as`

// Narrows typeof / instanceof too
expect.assert(typeof input === 'string')
input.toUpperCase()

// Chai assert helpers via the same namespace
expect.assert.isDefined(maybeUser)
expect.assert.instanceOf(error, MyError)
```

## Poll Assertions

Retry until passes:

```ts
await expect.poll(() => fetchStatus()).toBe('ready')

await expect.poll(
  () => document.querySelector('.element'),
  { interval: 100, timeout: 5000 }
).toBeTruthy()
```

## Assertion Count

```ts
test('async assertions', async () => {
  expect.assertions(2) // Exactly 2 assertions must run
  
  await doAsync((data) => {
    expect(data).toBeDefined()
    expect(data.id).toBe(1)
  })
})

test('at least one', () => {
  expect.hasAssertions() // At least 1 assertion must run
})
```

## Extending Matchers

```ts
expect.extend({
  toBeWithinRange(received, floor, ceiling) {
    const pass = received >= floor && received <= ceiling
    return {
      pass,
      message: () => 
        `expected ${received} to be within range ${floor} - ${ceiling}`,
    }
  },
})

test('custom matcher', () => {
  expect(100).toBeWithinRange(90, 110)
})
```

## Snapshot Assertions

```ts
expect(data).toMatchSnapshot()
expect(data).toMatchInlineSnapshot(`{ "id": 1 }`)
await expect(result).toMatchFileSnapshot('./expected.json')

expect(() => throw new Error('fail')).toThrowErrorMatchingSnapshot()
```

## Key Points

- Use `toBe` for primitives, `toEqual` for objects/arrays
- `toStrictEqual` checks undefined properties and array sparseness
- Always `await` async assertions (`resolves`, `rejects`, `poll`)
- Use context's `expect` in concurrent tests for correct tracking
- `toThrow` requires wrapping sync code in a function
- Use `expect.soft` for non-critical assertions; reserve hard `expect` for must-pass conditions
- Use `expect.assert` (not `toBeTruthy`) when you also need TypeScript narrowing

<!-- 
Source references:
- https://vitest.dev/api/expect.html
- https://vitest.dev/guide/recipes/type-narrowing
- https://vitest.dev/guide/recipes/schema-matching
-->
