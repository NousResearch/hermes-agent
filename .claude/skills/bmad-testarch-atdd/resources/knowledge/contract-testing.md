# Contract Testing Essentials (Pact)

## Principle

Contract testing validates API contracts between consumer and provider services without requiring integrated end-to-end tests. Store consumer contracts alongside integration specs, version contracts semantically, and publish on every CI run. Provider verification before merge surfaces breaking changes immediately, while explicit fallback behavior (timeouts, retries, error payloads) captures resilience guarantees in contracts.

> **Pact.js Utils Note**: When `tea_use_pactjs_utils` is enabled, prefer the patterns in the `pactjs-utils-*.md` fragments over the raw Pact.js patterns shown below. The pactjs-utils library eliminates boilerplate for provider states, verifier configuration, and request filters. See `pactjs-utils-overview.md` for the decision tree.

## Rationale

Traditional integration testing requires running both consumer and provider simultaneously, creating slow, flaky tests with complex setup. Contract testing decouples services: consumers define expectations (pact files), providers verify against those expectations independently. This enables parallel development, catches breaking changes early, and documents API behavior as executable specifications. Pair contract tests with API smoke tests to validate data mapping and UI rendering in tandem.

> **Recommended**: When `tea_use_pactjs_utils` is enabled, use `@seontechnologies/pactjs-utils` utilities instead of the manual patterns below. The library handles JsonMap conversion, verifier configuration, and request filter assembly automatically. See the `pactjs-utils-overview.md`, `pactjs-utils-consumer-helpers.md`, `pactjs-utils-provider-verifier.md`, and `pactjs-utils-request-filter.md` fragments for the simplified approach.

## Pattern Examples

### Example 1: Pact Consumer Test (Frontend → Backend API)

**Context**: React application consuming a user management API, defining expected interactions.

**Implementation**:

```typescript
// tests/contract/user-api.pact.spec.ts
import { PactV3, MatchersV3 } from '@pact-foundation/pact';
import { getUserById, createUser, User } from '@/api/user-service';

const { like, eachLike, string, integer } = MatchersV3;

/**
 * Consumer-Driven Contract Test
 * - Consumer (React app) defines expected API behavior
 * - Generates pact file for provider to verify
 * - Runs in isolation (no real backend required)
 */

const provider = new PactV3({
  consumer: 'user-management-web',
  provider: 'user-api-service',
  dir: './pacts', // Output directory for pact files
  logLevel: 'warn',
});

describe('User API Contract', () => {
  describe('GET /users/:id', () => {
    it('should return user when user exists', async () => {
      // Arrange: Define expected interaction
      await provider
        .given('user with id 1 exists') // Provider state
        .uponReceiving('a request for user 1')
        .withRequest({
          method: 'GET',
          path: '/users/1',
          headers: {
            Accept: 'application/json',
            Authorization: like('Bearer token123'), // Matcher: any string
          },
        })
        .willRespondWith({
          status: 200,
          headers: {
            'Content-Type': 'application/json',
          },
          body: like({
            id: integer(1),
            name: string('John Doe'),
            email: string('john@example.com'),
            role: string('user'),
            createdAt: string('2025-01-15T10:00:00Z'),
          }),
        })
        .executeTest(async (mockServer) => {
          // Act: Call consumer code against mock server
          const user = await getUserById(1, {
            baseURL: mockServer.url,
            headers: { Authorization: 'Bearer token123' },
          });

          // Assert: Validate consumer behavior
          expect(user).toEqual(
            expect.objectContaining({
              id: 1,
              name: 'John Doe',
              email: 'john@example.com',
              role: 'user',
            }),
          );
        });
    });

    it('should handle 404 when user does not exist', async () => {
      await provider
        .given('user with id 999 does not exist')
        .uponReceiving('a request for non-existent user')
        .withRequest({
          method: 'GET',
          path: '/users/999',
          headers: { Accept: 'application/json' },
        })
        .willRespondWith({
          status: 404,
          headers: { 'Content-Type': 'application/json' },
          body: {
            error: 'User not found',
            code: 'USER_NOT_FOUND',
          },
        })
        .executeTest(async (mockServer) => {
          // Act & Assert: Consumer handles 404 gracefully
          await expect(getUserById(999, { baseURL: mockServer.url })).rejects.toThrow('User not found');
        });
    });
  });

  describe('POST /users', () => {
    it('should create user and return 201', async () => {
      const newUser: Omit<User, 'id' | 'createdAt'> = {
        name: 'Jane Smith',
        email: 'jane@example.com',
        role: 'admin',
      };

      await provider
        .given('no users exist')
        .uponReceiving('a request to create a user')
        .withRequest({
          method: 'POST',
          path: '/users',
          headers: {
            'Content-Type': 'application/json',
            Accept: 'application/json',
          },
          body: newUser,
        })
        .willRespondWith({
          status: 201,
          headers: { 'Content-Type': 'application/json' },
          body: like({
            id: integer(2),
            name: string('Jane Smith'),
            email: string('jane@example.com'),
            role: string('admin'),
            createdAt: string('2025-01-15T11:00:00Z'),
          }),
        })
        .executeTest(async (mockServer) => {
          const createdUser = await createUser(newUser, {
            baseURL: mockServer.url,
          });

          expect(createdUser).toEqual(
            expect.objectContaining({
              id: expect.any(Number),
              name: 'Jane Smith',
              email: 'jane@example.com',
              role: 'admin',
            }),
          );
        });
    });
  });
});
```

**package.json scripts** (when using pactjs-utils conventions, prefer `test:pact:consumer` naming — see `pact-consumer-framework-setup.md`):

```json
{
  "scripts": {
    "test:pact:consumer": "./scripts/check-pact-determinism.sh 'npm run test:pact:consumer:run' 3 ./pacts",
    "test:pact:consumer:run": "vitest run --config vitest.config.pact.ts",
    "publish:pact": ". ./scripts/env-setup.sh && ./scripts/publish-pact.sh"
  }
}
```

**Key Points**:

- **Consumer-driven**: Frontend defines expectations, not backend
- **Matchers (Postel's Law)**: Use `like`, `string`, `integer` matchers in `willRespondWith` (responses) for flexible matching. Do NOT use `like()` on request bodies in `withRequest` — the consumer controls what it sends, so request bodies should use exact values. This follows Postel's Law: be strict in what you send (requests), be lenient in what you accept (responses).
- **Provider states**: given() sets up test preconditions
- **Isolation**: No real backend needed, runs fast
- **Pact generation**: Automatically creates JSON pact files

---

### Example 2: Pact Provider Verification (Backend validates contracts)

**Context**: Node.js/Express API verifying pacts published by consumers.

**Implementation**:

```typescript
// tests/contract/user-api.provider.spec.ts
import { Verifier, VerifierOptions } from '@pact-foundation/pact';
import { server } from '../../src/server'; // Your Express/Fastify app
import { seedDatabase, resetDatabase } from '../support/db-helpers';

/**
 * Provider Verification Test
 * - Provider (backend API) verifies against published pacts
 * - State handlers setup test data for each interaction
 * - Runs before merge to catch breaking changes
 */

describe('Pact Provider Verification', () => {
  let serverInstance;
  const PORT = 3001;

  beforeAll(async () => {
    // Start provider server
    serverInstance = server.listen(PORT);
    console.log(`Provider server running on port ${PORT}`);
  });

  afterAll(async () => {
    // Cleanup
    await serverInstance.close();
  });

  it('should verify pacts from all consumers', async () => {
    const opts: VerifierOptions = {
      // Provider details
      provider: 'user-api-service',
      providerBaseUrl: `http://localhost:${PORT}`,

      // Pact Broker configuration
      pactBrokerUrl: process.env.PACT_BROKER_BASE_URL,
      pactBrokerToken: process.env.PACT_BROKER_TOKEN,
      publishVerificationResult: process.env.CI === 'true',
      providerVersion: process.env.GITHUB_SHA || 'dev',

      // State handlers: Setup provider state for each interaction
      stateHandlers: {
        'user with id 1 exists': async () => {
          await seedDatabase({
            users: [
              {
                id: 1,
                name: 'John Doe',
                email: 'john@example.com',
                role: 'user',
                createdAt: '2025-01-15T10:00:00Z',
              },
            ],
          });
          return 'User seeded successfully';
        },

        'user with id 999 does not exist': async () => {
          // Ensure user doesn't exist
          await resetDatabase();
          return 'Database reset';
        },

        'no users exist': async () => {
          await resetDatabase();
          return 'Database empty';
        },
      },

      // Request filters: Add auth headers to all requests
      requestFilter: (req, res, next) => {
        // Mock authentication for verification
        req.headers['x-user-id'] = 'test-user';
        req.headers['authorization'] = 'Bearer valid-test-token';
        next();
      },

      // Timeout for verification
      timeout: 30000,
    };

    // Run verification
    await new Verifier(opts).verifyProvider();
  });
});
```

**CI integration**:

```yaml
# .github/workflows/contract-test-provider.yml
# NOTE: Canonical naming is contract-test-provider.yml per pactjs-utils conventions
name: Pact Provider Verification
on:
  pull_request:
  push:
    branches: [main]

jobs:
  verify-contracts:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version-file: '.nvmrc'

      - name: Install dependencies
        run: npm ci

      - name: Start database
        run: docker-compose up -d postgres

      - name: Run migrations
        run: npm run db:migrate

      - name: Verify pacts
        run: npm run test:pact:provider:remote:contract
        env:
          PACT_BROKER_BASE_URL: ${{ secrets.PACT_BROKER_BASE_URL }}
          PACT_BROKER_TOKEN: ${{ secrets.PACT_BROKER_TOKEN }}
          GITHUB_SHA: ${{ github.sha }}
          GITHUB_BRANCH: ${{ github.head_ref || github.ref_name }}

      - name: Can I Deploy?
        if: github.ref == 'refs/heads/main'
        run: npm run can:i:deploy:provider
```

**Key Points**:

- **State handlers**: Setup provider data for each given() state
- **Request filters**: Add auth/headers for verification requests
- **CI publishing**: Verification results sent to broker
- **can-i-deploy**: Safety check before production deployment
- **Database isolation**: Reset between state handlers

---

### Example 3: Contract CI Integration (Consumer & Provider Workflow)

**Context**: Simplified overview of consumer and provider CI coordination. For the complete consumer CI workflow with env blocks, concurrency, and breaking-change detection, see `pact-consumer-framework-setup.md` Example 5.

**Implementation**:

```yaml
# .github/workflows/contract-test-consumer.yml (Consumer side)
# NOTE: Canonical naming is contract-test-consumer.yml per pactjs-utils conventions
name: Pact Consumer Tests
on:
  pull_request:
  push:
    branches: [main]

jobs:
  consumer-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version-file: '.nvmrc'

      - name: Install dependencies
        run: npm ci

      - name: Run consumer contract tests
        run: npm run test:pact:consumer

      - name: Publish pacts to broker
        run: npm run publish:pact

      - name: Can I deploy consumer? (main only)
        if: github.ref == 'refs/heads/main' && env.PACT_BREAKING_CHANGE != 'true'
        run: npm run can:i:deploy:consumer

      - name: Record consumer deployment (main only)
        if: github.ref == 'refs/heads/main'
        run: npm run record:consumer:deployment --env=dev
```

```yaml
# .github/workflows/contract-test-provider.yml (Provider side)
# NOTE: Canonical naming is contract-test-provider.yml per pactjs-utils conventions
name: Pact Provider Verification
on:
  pull_request:
  push:
    branches: [main]
  repository_dispatch:
    types: [pact_changed] # Webhook from Pact Broker

jobs:
  verify-contracts:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version-file: '.nvmrc'

      - name: Install dependencies
        run: npm ci

      - name: Start dependencies
        run: docker-compose up -d

      - name: Run provider verification
        run: npm run test:pact:provider:remote:contract
        env:
          PACT_BROKER_BASE_URL: ${{ secrets.PACT_BROKER_BASE_URL }}
          PACT_BROKER_TOKEN: ${{ secrets.PACT_BROKER_TOKEN }}
          GITHUB_SHA: ${{ github.sha }}
          GITHUB_BRANCH: ${{ github.head_ref || github.ref_name }}

      - name: Can I deploy provider? (main only)
        if: github.ref == 'refs/heads/main' && env.PACT_BREAKING_CHANGE != 'true'
        run: npm run can:i:deploy:provider

      - name: Record provider deployment (main only)
        if: github.ref == 'refs/heads/main'
        run: npm run record:provider:deployment --env=dev
```

**Pact Broker Webhook Configuration**:

```json
{
  "events": [
    {
      "name": "contract_content_changed"
    }
  ],
  "request": {
    "method": "POST",
    "url": "https://api.github.com/repos/your-org/user-api/dispatches",
    "headers": {
      "Authorization": "Bearer ${user.githubToken}",
      "Content-Type": "application/json",
      "Accept": "application/vnd.github.v3+json"
    },
    "body": {
      "event_type": "pact_changed",
      "client_payload": {
        "pact_url": "${pactbroker.pactUrl}",
        "consumer": "${pactbroker.consumerName}",
        "provider": "${pactbroker.providerName}"
      }
    }
  }
}
```

**Key Points**:

- **Automatic trigger**: Consumer pact changes trigger provider verification via webhook
- **Branch tracking**: Pacts published per branch for feature testing
- **can-i-deploy**: Safety gate before production deployment
- **Record deployment**: Track which version is in each environment
- **Parallel dev**: Consumer and provider teams work independently

---

### Example 4: Resilience Coverage (Testing Fallback Behavior)

**Context**: Capture timeout, retry, and error handling behavior explicitly in contracts.

**Implementation**:

```typescript
// tests/contract/user-api-resilience.pact.spec.ts
import { PactV3, MatchersV3 } from '@pact-foundation/pact';
import { getUserById, ApiError } from '@/api/user-service';

const { like, string } = MatchersV3;

const provider = new PactV3({
  consumer: 'user-management-web',
  provider: 'user-api-service',
  dir: './pacts',
});

describe('User API Resilience Contract', () => {
  /**
   * Test 500 error handling
   * Verifies consumer handles server errors gracefully
   */
  it('should handle 500 errors with retry logic', async () => {
    await provider
      .given('server is experiencing errors')
      .uponReceiving('a request that returns 500')
      .withRequest({
        method: 'GET',
        path: '/users/1',
        headers: { Accept: 'application/json' },
      })
      .willRespondWith({
        status: 500,
        headers: { 'Content-Type': 'application/json' },
        body: {
          error: 'Internal server error',
          code: 'INTERNAL_ERROR',
          retryable: true,
        },
      })
      .executeTest(async (mockServer) => {
        // Consumer should retry on 500
        try {
          await getUserById(1, {
            baseURL: mockServer.url,
            retries: 3,
            retryDelay: 100,
          });
          fail('Should have thrown error after retries');
        } catch (error) {
          expect(error).toBeInstanceOf(ApiError);
          expect((error as ApiError).code).toBe('INTERNAL_ERROR');
          expect((error as ApiError).retryable).toBe(true);
        }
      });
  });

  /**
   * Test 429 rate limiting
   * Verifies consumer respects rate limits
   */
  it('should handle 429 rate limit with backoff', async () => {
    await provider
      .given('rate limit exceeded for user')
      .uponReceiving('a request that is rate limited')
      .withRequest({
        method: 'GET',
        path: '/users/1',
      })
      .willRespondWith({
        status: 429,
        headers: {
          'Content-Type': 'application/json',
          'Retry-After': '60', // Retry after 60 seconds
        },
        body: {
          error: 'Too many requests',
          code: 'RATE_LIMIT_EXCEEDED',
        },
      })
      .executeTest(async (mockServer) => {
        try {
          await getUserById(1, {
            baseURL: mockServer.url,
            respectRateLimit: true,
          });
          fail('Should have thrown rate limit error');
        } catch (error) {
          expect(error).toBeInstanceOf(ApiError);
          expect((error as ApiError).code).toBe('RATE_LIMIT_EXCEEDED');
          expect((error as ApiError).retryAfter).toBe(60);
        }
      });
  });

  /**
   * Test timeout handling
   * Verifies consumer has appropriate timeout configuration
   */
  it('should timeout after 10 seconds', async () => {
    await provider
      .given('server is slow to respond')
      .uponReceiving('a request that times out')
      .withRequest({
        method: 'GET',
        path: '/users/1',
      })
      .willRespondWith({
        status: 200,
        headers: { 'Content-Type': 'application/json' },
        body: like({ id: 1, name: 'John' }),
      })
      .withDelay(15000) // Simulate 15 second delay
      .executeTest(async (mockServer) => {
        try {
          await getUserById(1, {
            baseURL: mockServer.url,
            timeout: 10000, // 10 second timeout
          });
          fail('Should have timed out');
        } catch (error) {
          expect(error).toBeInstanceOf(ApiError);
          expect((error as ApiError).code).toBe('TIMEOUT');
        }
      });
  });

  /**
   * Test partial response (optional fields)
   * Verifies consumer handles missing optional data
   */
  it('should handle response with missing optional fields', async () => {
    await provider
      .given('user exists with minimal data')
      .uponReceiving('a request for user with partial data')
      .withRequest({
        method: 'GET',
        path: '/users/1',
      })
      .willRespondWith({
        status: 200,
        headers: { 'Content-Type': 'application/json' },
        body: {
          id: integer(1),
          name: string('John Doe'),
          email: string('john@example.com'),
          // role, createdAt, etc. omitted (optional fields)
        },
      })
      .executeTest(async (mockServer) => {
        const user = await getUserById(1, { baseURL: mockServer.url });

        // Consumer handles missing optional fields gracefully
        expect(user.id).toBe(1);
        expect(user.name).toBe('John Doe');
        expect(user.role).toBeUndefined(); // Optional field
        expect(user.createdAt).toBeUndefined(); // Optional field
      });
  });
});
```

**API client with retry logic**:

```typescript
// src/api/user-service.ts
import axios, { AxiosInstance, AxiosRequestConfig } from 'axios';

export class ApiError extends Error {
  constructor(
    message: string,
    public code: string,
    public retryable: boolean = false,
    public retryAfter?: number,
  ) {
    super(message);
  }
}

/**
 * User API client with retry and error handling
 */
export async function getUserById(
  id: number,
  config?: AxiosRequestConfig & { retries?: number; retryDelay?: number; respectRateLimit?: boolean },
): Promise<User> {
  const { retries = 3, retryDelay = 1000, respectRateLimit = true, ...axiosConfig } = config || {};

  let lastError: Error;

  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      const response = await axios.get(`/users/${id}`, axiosConfig);
      return response.data;
    } catch (error: any) {
      lastError = error;

      // Handle rate limiting
      if (error.response?.status === 429) {
        const retryAfter = parseInt(error.response.headers['retry-after'] || '60');
        throw new ApiError('Too many requests', 'RATE_LIMIT_EXCEEDED', false, retryAfter);
      }

      // Retry on 500 errors
      if (error.response?.status === 500 && attempt < retries) {
        await new Promise((resolve) => setTimeout(resolve, retryDelay * attempt));
        continue;
      }

      // Handle 404
      if (error.response?.status === 404) {
        throw new ApiError('User not found', 'USER_NOT_FOUND', false);
      }

      // Handle timeout
      if (error.code === 'ECONNABORTED') {
        throw new ApiError('Request timeout', 'TIMEOUT', true);
      }

      break;
    }
  }

  throw new ApiError('Request failed after retries', 'INTERNAL_ERROR', true);
}
```

**Key Points**:

- **Resilience contracts**: Timeouts, retries, errors explicitly tested
- **State handlers**: Provider sets up each test scenario
- **Error handling**: Consumer validates graceful degradation
- **Retry logic**: Exponential backoff tested
- **Optional fields**: Consumer handles partial responses

---

### Example 5: Pact Broker Housekeeping & Lifecycle Management

**Context**: Automated broker maintenance to prevent contract sprawl and noise.

**Implementation**:

```typescript
// scripts/pact-broker-housekeeping.ts
/**
 * Pact Broker Housekeeping Script
 * - Archive superseded contracts
 * - Expire unused pacts
 * - Tag releases for environment tracking
 */

import { execFileSync } from 'node:child_process';

const PACT_BROKER_BASE_URL = process.env.PACT_BROKER_BASE_URL!;
const PACT_BROKER_TOKEN = process.env.PACT_BROKER_TOKEN!;
const PACTICIPANT = 'user-api-service';

/**
 * Tag release with environment
 */
function tagRelease(version: string, environment: 'staging' | 'production') {
  console.log(`🏷️  Tagging ${PACTICIPANT} v${version} as ${environment}`);

  execFileSync(
    'pact-broker',
    [
      'create-version-tag',
      '--pacticipant',
      PACTICIPANT,
      '--version',
      version,
      '--tag',
      environment,
      '--broker-base-url',
      PACT_BROKER_BASE_URL,
      '--broker-token',
      PACT_BROKER_TOKEN,
    ],
    { stdio: 'inherit' },
  );
}

/**
 * Record deployment to environment
 */
function recordDeployment(version: string, environment: 'staging' | 'production') {
  console.log(`📝 Recording deployment of ${PACTICIPANT} v${version} to ${environment}`);

  execFileSync(
    'pact-broker',
    [
      'record-deployment',
      '--pacticipant',
      PACTICIPANT,
      '--version',
      version,
      '--environment',
      environment,
      '--broker-base-url',
      PACT_BROKER_BASE_URL,
      '--broker-token',
      PACT_BROKER_TOKEN,
    ],
    { stdio: 'inherit' },
  );
}

/**
 * Clean up old pact versions (retention policy)
 * Keep: last 30 days, all production tags, latest from each branch
 */
function cleanupOldPacts() {
  console.log(`🧹 Cleaning up old pacts for ${PACTICIPANT}`);

  execFileSync(
    'pact-broker',
    [
      'clean',
      '--pacticipant',
      PACTICIPANT,
      '--broker-base-url',
      PACT_BROKER_BASE_URL,
      '--broker-token',
      PACT_BROKER_TOKEN,
      '--keep-latest-for-branch',
      '1',
      '--keep-min-age',
      '30',
    ],
    { stdio: 'inherit' },
  );
}

/**
 * Check deployment compatibility
 */
function canIDeploy(version: string, toEnvironment: string): boolean {
  console.log(`🔍 Checking if ${PACTICIPANT} v${version} can deploy to ${toEnvironment}`);

  try {
    execFileSync(
      'pact-broker',
      [
        'can-i-deploy',
        '--pacticipant',
        PACTICIPANT,
        '--version',
        version,
        '--to-environment',
        toEnvironment,
        '--broker-base-url',
        PACT_BROKER_BASE_URL,
        '--broker-token',
        PACT_BROKER_TOKEN,
        '--retry-while-unknown',
        '10',
        '--retry-interval',
        '30',
      ],
      { stdio: 'inherit' },
    );
    return true;
  } catch (error) {
    console.error(`❌ Cannot deploy to ${toEnvironment}`);
    return false;
  }
}

/**
 * Main housekeeping workflow
 */
async function main() {
  const command = process.argv[2];
  const version = process.argv[3];
  const environment = process.argv[4] as 'staging' | 'production';

  switch (command) {
    case 'tag-release':
      tagRelease(version, environment);
      break;

    case 'record-deployment':
      recordDeployment(version, environment);
      break;

    case 'can-i-deploy':
      const canDeploy = canIDeploy(version, environment);
      process.exit(canDeploy ? 0 : 1);

    case 'cleanup':
      cleanupOldPacts();
      break;

    default:
      console.error('Unknown command. Use: tag-release | record-deployment | can-i-deploy | cleanup');
      process.exit(1);
  }
}

main();
```

**package.json scripts**:

```json
{
  "scripts": {
    "pact:tag": "ts-node scripts/pact-broker-housekeeping.ts tag-release",
    "pact:record": "ts-node scripts/pact-broker-housekeeping.ts record-deployment",
    "pact:can-deploy": "ts-node scripts/pact-broker-housekeeping.ts can-i-deploy",
    "pact:cleanup": "ts-node scripts/pact-broker-housekeeping.ts cleanup"
  }
}
```

**Deployment workflow integration**:

```yaml
# .github/workflows/deploy-production.yml
name: Deploy to Production
on:
  push:
    tags:
      - 'v*'

jobs:
  verify-contracts:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Check pact compatibility
        run: npm run pact:can-deploy ${{ github.ref_name }} production
        env:
          PACT_BROKER_BASE_URL: ${{ secrets.PACT_BROKER_BASE_URL }}
          PACT_BROKER_TOKEN: ${{ secrets.PACT_BROKER_TOKEN }}

  deploy:
    needs: verify-contracts
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to production
        run: ./scripts/deploy.sh production

      - name: Record deployment in Pact Broker
        run: npm run pact:record ${{ github.ref_name }} production
        env:
          PACT_BROKER_BASE_URL: ${{ secrets.PACT_BROKER_BASE_URL }}
          PACT_BROKER_TOKEN: ${{ secrets.PACT_BROKER_TOKEN }}
```

**Scheduled cleanup**:

```yaml
# .github/workflows/pact-housekeeping.yml
name: Pact Broker Housekeeping
on:
  schedule:
    - cron: '0 2 * * 0' # Weekly on Sunday at 2 AM

jobs:
  cleanup:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Cleanup old pacts
        run: npm run pact:cleanup
        env:
          PACT_BROKER_BASE_URL: ${{ secrets.PACT_BROKER_BASE_URL }}
          PACT_BROKER_TOKEN: ${{ secrets.PACT_BROKER_TOKEN }}
```

**Key Points**:

- **Automated tagging**: Releases tagged with environment
- **Deployment tracking**: Broker knows which version is where
- **Safety gate**: can-i-deploy blocks incompatible deployments
- **Retention policy**: Keep recent, production, and branch-latest pacts
- **Webhook triggers**: Provider verification runs on consumer changes

---

## Provider Scrutiny Protocol

When generating consumer contract tests, the agent **MUST** analyze provider source code — or the provider's OpenAPI/Swagger spec — before writing any Pact interaction. Generating contracts from consumer-side assumptions alone leads to mismatches that only surface during provider verification — wrong response shapes, wrong status codes, wrong field names, wrong types, missing required fields, and wrong enum values.

**Source priority**: Provider source code is the most authoritative reference. When an OpenAPI/Swagger spec exists (`openapi.yaml`, `openapi.json`, `swagger.json`), use it as a complementary or alternative source — it documents the provider's contract explicitly and can be faster to parse than tracing through handler code. When both exist, cross-reference them; if they disagree, the source code wins.

### Provider Endpoint Comment

Every Pact interaction MUST include a provider endpoint comment immediately above the `.given()` call:

```typescript
// Provider endpoint: server/src/routes/userRouteHandlers.ts -> GET /api/v2/users/:userId
await provider.given('user with id 1 exists').uponReceiving('a request for user 1');
```

**Format**: `// Provider endpoint: <relative-path-to-handler> -> <METHOD> <route-pattern>`

If the provider source is not accessible, use: `// Provider endpoint: TODO — provider source not accessible, verify manually`

### Seven-Point Scrutiny Checklist

Before generating each Pact interaction, read the provider route handler and/or OpenAPI spec and verify:

| #   | Check                 | What to Read (source code / OpenAPI spec)                         | Common Mismatch                                               |
| --- | --------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------- |
| 1   | **Response shape**    | Handler's `res.json()` calls / OpenAPI `responses.content.schema` | Nested object vs flat; array wrapper vs direct                |
| 2   | **Status codes**      | Handler's `res.status()` calls / OpenAPI `responses` keys         | 200 vs 201 for creation; 204 vs 200 for delete                |
| 3   | **Field names**       | Response type/DTO definitions / OpenAPI `schema.properties`       | `transaction_id` vs `transactionId`; `fraud_score` vs `score` |
| 4   | **Enum values**       | Validation schemas, constants / OpenAPI `schema.enum`             | `"active"` vs `"ACTIVE"`; `"pending"` vs `"in_progress"`      |
| 5   | **Required fields**   | Request validation (Joi, Zod) / OpenAPI `schema.required`         | Missing required header; optional field assumed required      |
| 6   | **Data types**        | TypeScript types, DB models / OpenAPI `schema.type` + `format`    | `string` ID vs `number` ID; ISO date vs Unix timestamp        |
| 7   | **Nested structures** | Response builder, serializer / OpenAPI `$ref` + `allOf`/`oneOf`   | `{ data: { items: [] } }` vs `{ items: [] }`                  |

### Scrutiny Evidence Block

Document what was found from provider source and/or OpenAPI spec as a block comment in the test file:

```typescript
/*
 * Provider Scrutiny Evidence:
 * - Handler: server/src/routes/userRouteHandlers.ts:45
 * - OpenAPI: server/openapi.yaml paths./api/v2/users/{userId}.get (if available)
 * - Response type: UserResponseDto (server/src/types/user.ts:12)
 * - Status: 200 (line 52), 404 (line 48)
 * - Fields: { id: number, name: string, email: string, role: "user" | "admin", createdAt: string }
 * - Required request headers: Authorization (Bearer token)
 * - Validation: Zod schema at server/src/validation/user.ts:8
 */
```

### Graceful Degradation

When provider source code is not accessible (different repo, no access, closed source):

1. **OpenAPI/Swagger spec available**: Use the spec as the source of truth for response shapes, status codes, and field names
2. **Pact Broker has existing contracts**: Use `pact_mcp` tools to fetch existing provider states and verified interactions as reference
3. **Neither available**: Generate contracts from consumer-side types but use the TODO form of the mandatory comment: `// Provider endpoint: TODO — provider source not accessible, verify manually` and add a `provider_scrutiny: "pending"` field to the output JSON
4. **Never silently guess**: If you cannot verify, document what you assumed and why

---

## Contract Testing Checklist

Before implementing contract testing, verify:

- [ ] **Pact Broker setup**: Hosted (Pactflow) or self-hosted broker configured
- [ ] **Consumer tests**: Generate pacts in CI, publish to broker on merge
- [ ] **Provider verification**: Runs on PR, verifies all consumer pacts
- [ ] **State handlers**: Provider implements all given() states
- [ ] **can-i-deploy**: Blocks deployment if contracts incompatible
- [ ] **Webhooks configured**: Consumer changes trigger provider verification
- [ ] **Retention policy**: Old pacts archived (keep 30 days, all production tags)
- [ ] **Resilience tested**: Timeouts, retries, error codes in contracts
- [ ] **Provider endpoint comments**: Every Pact interaction has `// Provider endpoint:` comment
- [ ] **Provider scrutiny completed**: Seven-point checklist verified for each interaction
- [ ] **Scrutiny evidence documented**: Block comment with handler, types, status codes, and fields

## Integration Points

- Used in workflows: `*automate` (integration test generation), `*ci` (contract CI setup)
- Related fragments: `test-levels-framework.md`, `ci-burn-in.md`, `pact-consumer-framework-setup.md` (consumer vitest `fileParallelism: false` + `pool: 'forks'` + `singleFork: true`), `pactjs-utils-consumer-helpers.md` (PactV4 one-interaction-per-`it()` rule), `pactjs-utils-provider-verifier.md` (provider vitest `pool: 'forks'` + `singleFork: true` — same rule as consumer), `pact-broker-webhooks.md` (PactFlow → GitHub webhook auth, PAT rotation, staleness monitoring)
- Tools: Pact.js, Pact Broker (Pactflow or self-hosted), Pact CLI

---

## Pact.js Utils Accelerator

When `tea_use_pactjs_utils` is enabled, the following utilities replace manual boilerplate:

| Manual Pattern (raw Pact.js)                             | Pact.js Utils Equivalent                                                          | Benefit                                                                                                    |
| -------------------------------------------------------- | --------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| Manual `JsonMap` casting for `.given()` params           | `createProviderState({ name, params })`                                           | Type-safe, auto-conversion of Date/null/nested objects                                                     |
| Repeated builder callbacks for query/header/body         | `setJsonContent({ query, headers, body })`                                        | Reusable callback for `.withRequest(...)` and `.willRespondWith(...)`                                      |
| Inline body lambda `(builder) => builder.jsonBody(body)` | `setJsonBody(body)`                                                               | Body-only shorthand for cleaner response builders                                                          |
| 30+ lines of `VerifierOptions` assembly                  | `buildVerifierOptions({ provider, port, includeMainAndDeployed, stateHandlers })` | One-call setup, env-aware, flow auto-detection                                                             |
| Manual broker URL + selector logic from env vars         | `handlePactBrokerUrlAndSelectors({ ..., options })`                               | Mutates options in-place with broker URL and selectors                                                     |
| DIY Express middleware for auth injection                | `createRequestFilter({ tokenGenerator })`                                         | Bearer prefix contract prevents double-prefix bugs                                                         |
| Manual CI branch/tag extraction                          | `getProviderVersionTags()`                                                        | CI-aware (GitHub Actions, GitLab CI, etc.)                                                                 |
| Message verifier config assembly                         | `buildMessageVerifierOptions({ provider, messageProviders })`                     | Same one-call pattern for Kafka/async contracts                                                            |
| Inline no-op filter `(req, res, next) => next()`         | `noOpRequestFilter`                                                               | Pre-built pass-through for no-auth providers                                                               |
| Hand-written matcher helper duplicating a Zod/TS type    | `zodToPactMatchers(ConsumerMovieSchema, example)`                                 | Single source of truth for response shape; consumer-curated scope keeps contracts lean and consumer-driven |

See the `pactjs-utils-*.md` knowledge fragments for complete examples and anti-patterns (`pactjs-utils-zod-to-pact.md` covers the consumer-curated schema pattern).

### PactV4 Determinism & FFI Safety (Mandatory)

Four rules that together prevent both (a) non-deterministic pact generation failures that cause `Cannot change pact content for already published pact` errors at PactFlow publish, and (b) "request was expected but not received" flakes observed on Linux CI once a consumer+provider pair has more than one `.pacttest.ts` file:

1. **Consumer Vitest `fileParallelism: false`** in `vitest.config.pact.ts` — prevents parallel workers from racing on the shared pact JSON. See `pact-consumer-framework-setup.md` Example 2.
2. **Consumer Vitest `pool: 'forks'` + `poolOptions.forks.singleFork: true`** in `vitest.config.pact.ts` — same config as the provider side (`pactjs-utils-provider-verifier.md` Example 7). Best current understanding: the `@pact-foundation/pact` napi-rs binding is not robust across Vitest worker threads sharing a process; serialization alone (via `fileParallelism: false`) is insufficient on the default threads pool in Vitest v1. Forks + `singleFork: true` runs every pact file in one subprocess with a coherent FFI handle and eliminated a reproducible Linux-CI flake on two repos (`pactjs-utils`, `seon-mcp-server`). Single-file consumer suites have not been observed to flake; this rule is still recommended as a future-proof. See `pact-consumer-framework-setup.md` Example 2.
3. **One `addInteraction()` per `it()` block** — see `pactjs-utils-consumer-helpers.md` Example 6.
4. **Determinism gate** runs the consumer suite N times and fails on byte-different pact JSON before publish — see `pact-consumer-framework-setup.md` Example 10 (`scripts/check-pact-determinism.sh`).

Provider suites require the same `pool: 'forks'` + `singleFork: true` combination — see `pactjs-utils-provider-verifier.md` Example 7.

### Webhook Auth & Staleness

When `can-i-deploy` in a consumer repo times out with `There is no verified pact between <consumer> and the version of <provider> currently in <env>` — check the provider's PactFlow webhook. Silent failures from an expired/revoked GitHub PAT are the most common non-code cause of this symptom. See `pact-broker-webhooks.md` for the dedicated-machine-user pattern, classic-PAT-with-`repo`-scope rationale, rotation runbook, and staleness monitoring options.

_Source: Pact consumer/provider sample repos, Murat contract testing blog, Pact official documentation, @seontechnologies/pactjs-utils library_
