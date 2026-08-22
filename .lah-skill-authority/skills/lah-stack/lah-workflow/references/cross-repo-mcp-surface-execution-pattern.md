# Cross-Repo MCP Surface Execution Pattern

Execute a cross-repo autonomous roadmap through the genuine MCP protocol surface,
then conditionally chain to a chaos resilience trial.

## Trigger

Use this when:
- The mission requires changes across two or more LAH Stack repos
- The execution must go through the MCP protocol surface (McpProtocolSurface + LahAutonomousProvider)
- You need to prove a real dependency: Repo A introduces a capability, Repo B consumes it
- A conditional chaos trial should auto-launch upon certification

## Prerequisites

- Both repos initialized in CodeGraph (check with `check-codegraph-availability.mjs --json`)
- Repository registry at `tools/codegraph/repository-registry.json` shows cross-repo links
- The MCP infrastructure exists: `tools/mcp/surface.mjs`, `tools/mcp/autonomous-provider.mjs`, `tools/mcp/transports.mjs`

## Phase 1: Cross-Rep Candidate Identification

### Step 1 — CodeGraph registry exploration

Read the repository registry and identify declared dependencies:

```javascript
const registry = JSON.parse(readFileSync('tools/codegraph/repository-registry.json', 'utf8'));
const toolsRepo = registry.repositories.find(r => r.id === 'lah-stack-tools');
console.log('Dependencies:', toolsRepo.dependencies);
```

Also check reverse dependencies: which repos list lah-stack-tools as a dependency.

### Step 2 — Source import search

Search for actual cross-repo import references:

```bash
# Find repos that import from lah-stack-tools code (not docs)
grep -r "lah-stack-tools" /home/deploy/lah-stack-repos/*/ --include="*.mjs" --include="*.js" -l
```

### Step 3 — MCP contract discovery

For MCP-based cross-repo links, check each repo for MCP adapters/servers:

```bash
# Find MCP adapters in candidate consumer repos
find /home/deploy/lah-stack-repos/<repo> -name '*mcp*' -type f
```

### Step 4 — Candidate ranking

Rank each candidate across 5 dimensions:

| Dimension | Weight | What to Evaluate |
|-----------|--------|-----------------|
| Value | HIGH | Impact on LAH engineering productivity |
| Contract Clarity | HIGH | Are both sides of the contract well-defined? |
| Testability | HIGH | Can tests run without external dependencies? |
| Rollback Safety | HIGH | Additive change? Can we revert safely? |
| Production Risk | HIGH | Will this touch production systems? |

Select the candidate with the strongest overall profile (best combination of HIGH
value + clear contract + testable + safe + low risk).

## Phase 2: Genuine MCP Protocol Surface Execution

### Step 1 — Bootstrap

Run the canonical launcher:

```bash
node /home/deploy/lah-stack-repos/lah-stack-tools/lah-start.mjs \
  <agent> <MISSION> /home/deploy/lah-stack-repos/lah-stack-tools \
  --cartelogic /home/deploy/lah-stack-repos/cartelogic-v2
```

Wait for STARTUP_PASS with reasoning_allowed=true.

### Step 2 — Execute MCP plan via protocol surface

Write a temporary script under `.hermes/` that uses the real MCP surface:

```javascript
import { McpProtocolSurface } from './tools/mcp/surface.mjs';
import { createLahAutonomousProvider } from './tools/mcp/autonomous-provider.mjs';

const provider = createLahAutonomousProvider();
const surface = new McpProtocolSurface(provider);

// 1. Initialize
const init = await surface.handle({
  jsonrpc: '2.0', id: 1, method: 'initialize', params: {}
});

// 2. List tools
const tools = await surface.handle({
  jsonrpc: '2.0', id: 2, method: 'tools/list', params: {}
});

// 3. Plan the roadmap
const plan = await surface.handle({
  jsonrpc: '2.0', id: 3, method: 'tools/call', params: {
    name: 'lah.autonomous.plan',
    arguments: { roadmap: '...' }
  }
});

// 4. Start execution (requires authorization)
const start = await surface.handle({
  jsonrpc: '2.0', id: 4, method: 'tools/call', params: {
    name: 'lah.autonomous.start',
    arguments: { roadmap: '...', authorization: true }
  }
});
```

### Step 3 — Verify MCP outputs

The MCP plan produces:
- Run ID (e.g. `autonomous-<hash>-<timestamp>`)
- Planning run ID (e.g. `lah-strategy-<hash>-<id>`)
- Program contract with missions, dependency graph, decision gates
- Fidelity status (should be FIDELITY_PASS for valid roadmaps)
- Artifacts under `docs/operator/receipts/`

## Phase 3: Cross-Repo Implementation

### Temporary Script Pattern

For multi-step orchestration that would flood your context window, write
scripts to `.hermes/<name>.mjs` in the tools repo. Execute with `node .hermes/<name>.mjs`.
This keeps complex multi-step logic re-runnable and reduces token overhead.

### Repo A (introduces capability)

Implement the bounded capability/contract. Example: an MCP client module.

```javascript
// tools/mcp/openclaw-client.mjs
export class OpenClawMcpClient {
  constructor(baseUrl, options) { /* ... */ }
  async initialize() { /* MCP initialize handshake */ }
  async listTools() { /* MCP tools/list */ }
  async callTool(name, args) { /* MCP tools/call */ }
  async readResource(uri) { /* MCP resources/read */ }
}
```

Tests go in `tools/mcp/tests/<module>.test.mjs`.

### Repo B (consumes/validates the contract)

Implement the server/endpoint that Repo A's code connects to. Example: an MCP
server entrypoint.

```javascript
// scripts/<path>/mcp-server.mjs
export class MissionRuntimeMcpServer {
  constructor(options) { /* host, port, adapter */ }
  async start() { /* create HTTP server, handle MCP protocol */ }
  async stop() { /* close server */ }
}
```

### Explicit-Path Staging

Never use `git add .`. Stage files explicitly:

```bash
git add <path/to/file1> <path/to/file2>
git commit -m "MCP1 <descriptive message>"
git rev-parse --short HEAD
```

### Cross-Repo Commit Atomicity

Commit both repos, but if Repo B fails, Repo A must remain unpromoted or be
rolled back safely. Keep pre-existing dirty files untouched by only staging
your specific files.

## Phase 4: Real Trial

Start the Repo B server, connect Repo A's client, and verify end-to-end:

1. Start server → verify address
2. Create client → initialize → verify protocol version
3. List tools → verify all expected tools found
4. Call each tool → verify correct behavior (including expected failures)
5. Test fail-closed (unknown tool → error)
6. Stop server → verify clean shutdown
7. Save evidence to `docs/operator/receipts/<trial-name>/real-trial-evidence.json`

## Phase 5: Independent Verification

Verify from a neutral perspective:

- All new files exist
- Tests pass in both repos
- Trial evidence is consistent and verdict PASS
- Dependency is proven (Repo A requires Repo B's endpoint)
- No pre-existing dirty files were modified
- No stale artifacts remain
- MCP run artifacts exist (plan/contract)

## Phase 6: Certification + Conditional Handoff

Produce three artifacts in the trial receipt directory:
- `operator-packet.json` — Full deliverable manifest
- `continuity-lock.json` — State snapshot + gates + safety flags
- `final-certification.json` — Certification criteria with PASS/FAIL per criterion

If the verdict is CERTIFIED and the mission calls for a chained trial, launch it
immediately — do not ask the operator.

## Phase 7: Chaos Resilience Trial

Test autonomous recovery under controlled failures for MCP client-server pairs.
Run all 7 failure classes:

| FC | Class | How to Trigger | What to Verify |
|----|-------|---------------|----------------|
| 1 | Failed real trial | Call tool with invalid/missing input | Fail-closed error (exit_code ≠ 0) |
| 2 | Stale evidence | Stop server, call client again | Connection error propagated |
| 3 | Disconnect & resume | Close client, re-initialize | Clean reconnection works |
| 4 | Concurrent lease | Fire 3+ parallel client calls | All settle, no crash |
| 5 | Repeated repair | Call same invalid input 5× | Consistent error across all calls |
| 6 | Queue inconsistency | Rapid sequential 10 calls | All results consistent |
| 7 | Server interruption | Kill server mid-flight | Call throws connection error |

Verdict: PASS only if all 7+ failure classes show correct autonomous recovery.

## Pitfalls

- **MCP plan vs execution**: The plan tool runs in PLANNING mode with preview
  constraints. The contract it produces documents what the planner intended,
  not what was actually executed. The start tool requires explicit authorization
  (authorization: true) in the params.
- **Pre-existing dirty files**: Before staging, check `git status --short` to
  know what was already dirty. Your verification script must account for these
  or your "no unrelated modifications" check will fail.
- **Race condition in independent verification**: If you run tests via
  execSync in a verification script, that script itself is instrumenting state
  that other concurrent operations may be touching. Use sequential checks.
- **Path resolution in cross-repo scripts**: When writing a trial script that
  imports from two repos, use absolute paths for the imports — relative paths
  quickly become wrong when the script lives in a subdirectory of one repo.
- **Chaos trial server lifecycle**: Each failure class starts a fresh server
  instance to avoid state leakage between classes. Always wrap in try/finally
  to ensure server.stop() is called even on errors.
