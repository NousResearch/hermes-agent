---
name: type-testing
description: Test TypeScript types with expectTypeOf and assertType
---

# Type Testing

Test TypeScript types without runtime execution.

## Setup

Type tests use `.test-d.ts` extension:

```ts
// math.test-d.ts
import { expectTypeOf } from 'vitest'
import { add } from './math'

test('add returns number', () => {
  expectTypeOf(add).returns.toBeNumber()
})
```

## Configuration

```ts
defineConfig({
  test: {
    typecheck: {
      enabled: true,
      
      // Only type check
      only: false,
      
      // Checker: 'tsc' or 'vue-tsc'
      checker: 'tsc',
      
      // Include patterns
      include: ['**/*.test-d.ts'],
      
      // tsconfig to use
      tsconfig: './tsconfig.json',
    },
  },
})
```

## expectTypeOf API

```ts
import { expectTypeOf } from 'vitest'

// Basic type checks
expectTypeOf<string>().toBeString()
expectTypeOf<number>().toBeNumber()
expectTypeOf<boolean>().toBeBoolean()
expectTypeOf<null>().toBeNull()
expectTypeOf<undefined>().toBeUndefined()
expectTypeOf<void>().toBeVoid()
expectTypeOf<never>().toBeNever()
expectTypeOf<any>().toBeAny()
expectTypeOf<unknown>().toBeUnknown()
expectTypeOf<object>().toBeObject()
expectTypeOf<Function>().toBeFunction()
expectTypeOf<[]>().toBeArray()
expectTypeOf<symbol>().toBeSymbol()
```

## Value Type Checking

```ts
const value = 'hello'
expectTypeOf(value).toBeString()

const obj = { name: 'test', count: 42 }
expectTypeOf(obj).toExtend<{ name: string }>()
expectTypeOf(obj).toHaveProperty('name')
```

## Function Types

```ts
function greet(name: string): string {
  return `Hello, ${name}`
}

expectTypeOf(greet).toBeFunction()
expectTypeOf(greet).parameters.toEqualTypeOf<[string]>()
expectTypeOf(greet).returns.toBeString()

// Parameter checking
expectTypeOf(greet).parameter(0).toBeString()
```

## Object Types

```ts
interface User {
  id: number
  name: string
  email?: string
}

expectTypeOf<User>().toHaveProperty('id')
expectTypeOf<User>().toHaveProperty('name').toBeString()

// Check shape
expectTypeOf({ id: 1, name: 'test' }).toExtend<User>()
```

## Equality vs Matching

`toMatchTypeOf` is **deprecated** (expect-type v1.2+) — use `toExtend` for subset matching:

```ts
interface A { x: number }
interface B { x: number; y: string }

// toExtend - subset matching (replaces toMatchTypeOf)
expectTypeOf<B>().toExtend<A>()  // B extends A

// toEqualTypeOf - exact match
expectTypeOf<A>().not.toEqualTypeOf<B>()  // Not exact match
expectTypeOf<A>().toEqualTypeOf<{ x: number }>()  // Exact match
```

## Branded Types

```ts
type UserId = number & { __brand: 'UserId' }
type PostId = number & { __brand: 'PostId' }

expectTypeOf<UserId>().not.toEqualTypeOf<PostId>()
expectTypeOf<UserId>().not.toEqualTypeOf<number>()
```

## Generic Types

```ts
function identity<T>(value: T): T {
  return value
}

expectTypeOf(identity<string>).returns.toBeString()
expectTypeOf(identity<number>).returns.toBeNumber()
```

## Nullable Types

```ts
type MaybeString = string | null | undefined

expectTypeOf<MaybeString>().toBeNullable()
expectTypeOf<string>().not.toBeNullable()
```

## assertType

Assert a value matches a type (no assertion at runtime):

```ts
import { assertType } from 'vitest'

function getUser(): User | null {
  return { id: 1, name: 'test' }
}

test('returns user', () => {
  const result = getUser()
  
  // @ts-expect-error - should fail type check
  assertType<string>(result)
  
  // Correct type
  assertType<User | null>(result)
})
```

## Using @ts-expect-error

Test that code produces type error:

```ts
test('rejects wrong types', () => {
  function requireString(s: string) {}
  
  // @ts-expect-error - number not assignable to string
  requireString(123)
})
```

## Running Type Tests

```bash
# Run type tests
vitest typecheck

# Run alongside unit tests
vitest --typecheck

# Type tests only
vitest --typecheck.only
```

## Mixed Test Files

Combine runtime and type tests:

```ts
// user.test.ts
import { describe, expect, expectTypeOf, test } from 'vitest'
import { createUser } from './user'

describe('createUser', () => {
  test('runtime: creates user', () => {
    const user = createUser('John')
    expect(user.name).toBe('John')
  })

  test('types: returns User type', () => {
    expectTypeOf(createUser).returns.toExtend<{ name: string }>()
  })
})
```

## Key Points

- Use `.test-d.ts` for type-only tests
- `expectTypeOf` for type assertions
- `toExtend` for subset matching (`toMatchTypeOf` is deprecated)
- `toEqualTypeOf` for exact type matching
- Use `@ts-expect-error` to test type errors
- Run with `vitest typecheck` or `--typecheck`

<!-- 
Source references:
- https://vitest.dev/guide/testing-types.html
- https://vitest.dev/api/expect-typeof.html
-->
