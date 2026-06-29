# Pact.js Utils Zod to Pact

## Principle

Use `zodToPactMatchers` from `@seontechnologies/pactjs-utils` to derive Pact V3 matchers directly from a Zod schema so you never maintain two representations of the same response shape. The schema is the source of truth for types; plain example values (or `.openapi({ example })` metadata) supply the concrete example data.

## Rationale

### Problems with hand-written matcher helpers

- **Duplication**: Teams that already define response shapes in Zod (or generate OpenAPI from Zod) then redefine the same shape again as hand-written `{ id: integer(...), name: string(...) }` matcher objects.
- **Silent drift**: Every schema change must be applied in both places; miss one and the contract drifts silently from the real response shape.
- **Boilerplate helpers per test file**: Consumer tests end up with local `propMatcherFoo(x) => ({ ... })` helpers that mirror the type exactly.
- **Over-specification**: Importing the provider's full 20-field schema produces a contract that forces the provider to return every field — breaking consumer-driven testing's core benefit (consumer only asserts what it reads).

### Solutions

- **`zodToPactMatchers(schema, example)`** — walks a Zod schema and emits the right `MatchersV3.*` call per field (`string()`, `integer()`, `decimal()`, `boolean()`, `nullValue()`, `eachLike(...)` for arrays, recursive objects, first option for unions, first value for enums, literal-typed matchers for literals).
- **Three-step example resolution**: (1) the `example` arg wins, (2) `.openapi({ example })` metadata (if `@asteasolutions/zod-to-openapi` is installed), (3) a type-appropriate default (`'string'`, `1.0`, `true`, no-arg `integer()`).
- **Consumer-curated schemas**: You choose which schema to pass, so you can include only the fields the consumer actually reads — keeping contracts lean and consumer-driven.

## Pattern Examples

### Example 1: Consumer-curated schema (mandatory pattern)

```typescript
// pact/http/helpers/consumer-schemas.ts
import { z } from 'zod';

// Only the fields this consumer actually reads — NOT the shared full-response schema
export const ConsumerMovieSchema = z.object({
  id: z.number().int(),
  name: z.string(),
  year: z.number().int(),
  rating: z.number(),
  director: z.string(),
});
```

### Example 2: Replacing hand-written matcher helpers

```typescript
// ❌ Before — hand-written helper duplicates the shape defined in Movie type
const propMatcherNoId = (movie: Omit<Movie, 'id'>) => ({
  name: string(movie.name),
  year: integer(movie.year),
  rating: decimal(movie.rating),
  director: string(movie.director),
});

await pact
  .addInteraction()
  .given('No movies exist')
  .uponReceiving('a request to add a new movie')
  .withRequest('POST', '/movies', setJsonContent({ body: movieWithoutId }))
  .willRespondWith(
    200,
    setJsonContent({
      body: {
        status: 200,
        data: { id: integer(), ...propMatcherNoId(movieWithoutId) },
      },
    }),
  );
```

```typescript
// ✅ After — schema defines types, plain object provides examples
import { zodToPactMatchers, setJsonContent } from '@seontechnologies/pactjs-utils';
import { ConsumerMovieSchema } from '../helpers/consumer-schemas';

await pact
  .addInteraction()
  .given('No movies exist')
  .uponReceiving('a request to add a new movie')
  .withRequest('POST', '/movies', setJsonContent({ body: movieWithoutId }))
  .willRespondWith(
    200,
    setJsonContent({
      body: {
        status: 200,
        data: zodToPactMatchers(ConsumerMovieSchema, { id: 1, ...movieWithoutId }),
      },
    }),
  );
```

### Example 3: Array responses with `eachLike`

```typescript
import { PactV4, MatchersV3 } from '@pact-foundation/pact';
import { zodToPactMatchers, setJsonContent } from '@seontechnologies/pactjs-utils';
import { ConsumerMovieSchema } from '../helpers/consumer-schemas';

const { eachLike } = MatchersV3;
const pact = new PactV4({ consumer: 'Movies Web', provider: 'Movies API' });
const movie = { id: 1, name: 'My movie', year: 1999, rating: 8.5, director: 'John Doe' };

await pact
  .addInteraction()
  .given('Movies exist')
  .uponReceiving('a request for all movies')
  .withRequest('GET', '/movies')
  .willRespondWith(
    200,
    setJsonContent({
      body: {
        status: 200,
        data: eachLike(zodToPactMatchers(ConsumerMovieSchema, movie) as Parameters<typeof eachLike>[0]),
      },
    }),
  );
// data expands to: eachLike({ id: integer(1), name: string('My movie'), year: integer(1999), rating: decimal(8.5), director: string('John Doe') })
```

### Example 4: Message Pact tests (Kafka / async)

```typescript
import { PactV4, MatchersV3 } from '@pact-foundation/pact';
import { zodToPactMatchers } from '@seontechnologies/pactjs-utils';
import { ConsumerMovieSchema } from '../../http/helpers/consumer-schemas';

const { string } = MatchersV3;

// Schema-derived matchers — no manual matcher construction, no outer like() wrapper
const movieValue = zodToPactMatchers(ConsumerMovieSchema, {
  id: 1,
  name: 'Inception',
  year: 2010,
  rating: 8.8,
  director: 'Christopher Nolan',
});

await messagePact
  .addAsynchronousInteraction()
  .given('An existing movie exists')
  .expectsToReceive('a movie-created event', (builder) => {
    builder.withJSONContent({
      topic: string('movie-created'),
      messages: [{ key: string('1'), value: movieValue }],
    });
  });
```

Note: `zodToPactMatchers` on an object schema already wraps each field in the right matcher, so the extra `like()` wrapper from hand-written versions is not needed — each field carries its own type constraint.

### Example 5: OpenAPI example metadata (optional peer)

```typescript
import { z } from 'zod';
import { extendZodWithOpenApi } from '@asteasolutions/zod-to-openapi';

extendZodWithOpenApi(z);

const MovieSchema = z.object({
  name: z.string().openapi({ example: 'Inception' }),
  year: z.number().int().openapi({ example: 2010 }),
});

// No second argument needed — examples come from the schema itself
zodToPactMatchers(MovieSchema);
// → { name: string('Inception'), year: integer(2010) }
```

## Zod to Pact V3 Mapping

| Zod type                                      | Pact V3 matcher                           |
| --------------------------------------------- | ----------------------------------------- |
| `z.string()`                                  | `string(example ?? 'string')`             |
| `z.number().int()`                            | `integer(example)` (no-arg if no example) |
| `z.number()`                                  | `decimal(example ?? 1.0)`                 |
| `z.boolean()`                                 | `boolean(example ?? true)`                |
| `z.null()`                                    | `nullValue()`                             |
| `z.object({...})`                             | recursive object of field matchers        |
| `z.array(...)`                                | `eachLike(itemMatchers)`                  |
| `z.union([...])`                              | first option's matcher                    |
| `z.literal('x')` / number / bool              | typed matcher with literal value          |
| `z.enum([...])`                               | `string(firstValue)`                      |
| `z.optional()` / `.nullable()` / `.default()` | unwraps to the inner schema               |
| anything else                                 | `like(example ?? null)` fallback          |

## Key Points

- **Consumer-curated schema is mandatory**: Define schemas that describe only what the consumer actually reads. Do **not** pass the shared full-response schema, and do **not** `import` the provider-side schema — that turns contract tests into schema tests and blocks the provider from deprecating unused fields.
- **Example precedence**: `example` argument > `.openapi({ example })` metadata > type default. The example only sets the placeholder value; Pact matchers check type/shape, not exact values.
- **Optional peer**: `@asteasolutions/zod-to-openapi` is an optional peer dependency. If it's not installed, openapi-example extraction silently becomes a no-op and only the `example` argument / defaults are used.
- **Optional peer (zod)**: `zod` itself is declared as an optional peer of `@seontechnologies/pactjs-utils` so consumers who don't use `zodToPactMatchers` don't need it; consumers who do use it must have zod installed.
- **Object wrapping**: When passing an object result into `eachLike(...)`, cast to `Parameters<typeof eachLike>[0]` — `zodToPactMatchers` returns `unknown` by design to stay compatible with both primitive and composite matcher shapes.
- **Arrays without examples**: If the example array is empty, the first item's field matchers are derived from the schema (and `.openapi({ example })` metadata, if present).
- **No extra `like()` wrapper**: For objects returned from `zodToPactMatchers`, do not wrap the whole object in `like()`; each field is already a matcher.
- **Works for HTTP and message pacts**: The same function produces matchers for request/response bodies and for Kafka / async message payloads.
- **TypeScript**: Import `z` as a runtime value when defining schemas (`import { z } from 'zod'`). If you need a schema type in helper signatures, import it separately (for example, `import type { ZodTypeAny } from 'zod'`).

## Related Fragments

- `pactjs-utils-overview.md` — installation, utility table, decision tree
- `pactjs-utils-consumer-helpers.md` — `createProviderState`, `setJsonContent`, `setJsonBody`
- `pactjs-utils-provider-verifier.md` — `buildVerifierOptions` integration
- `contract-testing.md` — foundational patterns with raw Pact.js, Provider Scrutiny Protocol (required fields / enums / data types / nested structures)

## Anti-Patterns

### Wrong: Passing the provider's full response schema

```typescript
// ❌ Importing the shared server-side schema forces the provider to return every field
import { FullMovieSchema } from '@shared/schemas/movie'; // 20 fields

data: zodToPactMatchers(FullMovieSchema, movie);
```

This creates a contract that requires the provider to return all 20 fields, even the ones this consumer never reads — breaking consumer-driven testing and blocking future field deprecation.

### Right: Consumer-curated schema beside the pact tests

```typescript
// ✅ pact/http/helpers/consumer-schemas.ts — only the fields this consumer reads
export const ConsumerMovieSchema = z.object({
  id: z.number().int(),
  name: z.string(),
  year: z.number().int(),
  rating: z.number(),
  director: z.string(),
});

data: zodToPactMatchers(ConsumerMovieSchema, movie);
```

### Wrong: Hand-written matcher helper duplicating the schema

```typescript
// ❌ Local helper that mirrors the TS type — drifts silently on every schema change
const propMatcherNoId = (movie: Omit<Movie, 'id'>) => ({
  name: string(movie.name),
  year: integer(movie.year),
  rating: decimal(movie.rating),
  director: string(movie.director),
});
```

### Right: `zodToPactMatchers` with a consumer-curated schema

```typescript
// ✅ Schema is the single source of truth; plain object supplies examples
data: zodToPactMatchers(ConsumerMovieSchema, { id: 1, ...movieWithoutId });
```

### Wrong: Wrapping the whole object result in `like()`

```typescript
// ❌ Redundant — each field is already a matcher
value: like(zodToPactMatchers(ConsumerMovieSchema, movie));
```

### Right: Use the object directly

```typescript
// ✅ Each field carries its own type constraint
value: zodToPactMatchers(ConsumerMovieSchema, movie);
```

_Source: @seontechnologies/pactjs-utils library, pactjs-utils docs (`docs/zod-to-pact/`), pact-js consumer sample repos, Pact docs on consumer-driven contracts_
