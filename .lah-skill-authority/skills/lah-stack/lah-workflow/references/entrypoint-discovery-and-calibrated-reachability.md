# Entrypoint Discovery and Calibrated Reachability

Systematically discover real execution surfaces in a monorepo, then compute reachability from multiple categories of entrypoints (not just a single "canonical" set). This prevents false orphan/unreachable classifications when tools are independent CLIs, workers, or test infrastructure rather than dead code.

Use this pattern during any repository-hygiene or dead-code-detection mission where false unreachable findings are expected due to monorepo architecture.

## When to Use

This pattern is appropriate when:
- The repo is a **tools monorepo** (many independent CLIs, scripts, workers)
- Entrypoints are shell wrappers, shebang scripts, or npm scripts (not a single app entry)
- The dependency-graph collector returns 90%+ "unreachable" modules
- You need to distinguish "truly orphaned code" from "independent tool that happens to not share imports with the main app"
- You need a before/after metrics comparison to prove improvement

Do NOT use this pattern for:
- Single-application repos with one clear entrypoint
- Libraries (where all modules are exported, not executed)
- Repos where the dependency graph already shows >50% reachable

## Phase 1: Authority Model

Before discovery, establish the entrypoint authority categories:

| Category | Examples | graph_root | Role |
|----------|----------|------------|------|
| `CANONICAL_PRODUCTION_ENTRYPOINT` | Main app server, primary library index | true | production |
| `CANONICAL_CLI_ENTRYPOINT` | Root shebang files, bin/ wrappers | true | cli |
| `INDEPENDENT_TOOL_ENTRYPOINT` | tools/*/cli.mjs, tool index.mjs | true | tooling |
| `WORKER_ENTRYPOINT` | worker-loop.mjs, background-worker.mjs | true | worker |
| `MAINTENANCE_ENTRYPOINT` | scripts/*.mjs (audit, certify) | true | tooling |
| `CERTIFICATION_ENTRYPOINT` | Acceptance trial scripts | false | certification |
| `TEST_ENTRYPOINT` | *.test.mjs files | false | test |
| `IMPLICIT_FRAMEWORK_ENTRYPOINT` | Convention-based loaders | false | dynamic |
| `DYNAMIC_REGISTRY_ROOT` | Plugin discovery directories | true | dynamic |
| `GENERATED_ENTRYPOINT` | Build output | false | generated |
| `EXTERNAL_CONSUMER_BOUNDARY` | package.json exports | false | boundary |
| `UNRESOLVED_ENTRYPOINT` | Declared but missing | false | error |

## Phase 2: Systematic Discovery

Run discovery from 6 orthogonal evidence sources. Deduplicate by path, merging evidence and keeping max confidence.

### Source 1: Package Metadata

Scan `package.json` for:
- **bin** — declared executables (highest confidence, 0.95)
- **main** — primary module entry (0.95 if exists)
- **module** — ESM entry (0.9)
- **exports** — public API surface (0.7, graph_root=false)
- **scripts** — npm scripts that run `node <file>` (0.85)

Pitfall: If `package.json` is empty (`{}`), all package-based discovery yields nothing. Fall through to filesystem conventions.

### Source 2: Filesystem Conventions

Scan these patterns (skip node_modules, .codegraph, .git):

```
bin/*                          → CLI wrappers (check shebang for bash vs script)
<root>/*.mjs with #!           → Root-level executables
tools/*/cli.*                  → Tool CLIs (each tool may have one)
tools/*/index.mjs              → Tool module entry points
scripts/*                      → Maintenance scripts
```

For each file found:
1. Read shebang (first line) to detect `#!/usr/bin/env node` vs bash
2. Capture file type (mjs, js, bash/shell)
3. Confidence: shebang=0.9, no shebang=0.75, shell=0.9 (wrappers)

Pitfall: Shell wrappers in `bin/` (bash scripts) are valid graph roots for *documentation* but they cannot be followed by static import analysis (depcruise). Their targets (node CLI entries) should be discovered separately.

### Source 3: Test Infrastructure

Every `*.test.mjs` file is a test entrypoint. Walk `test/` and `tools/` recursively, excluding `node_modules`, `fixtures`, and `.codegraph`.

- Authority: `TEST_ENTRYPOINT`
- graph_root: **false** — test reachability must NOT be counted as production reachability
- Confidence: 0.95

Pitfall: Tests often import modules that are not imported by production code. A module "reachable from tests only" is NOT "reachable" for production hygiene purposes. Keep this category separate.

### Source 4: Worker Detection

Known worker file patterns in a tools monorepo:
```
tools/*/worker*.mjs
tools/*/background-worker*.mjs
bin/*-worker
```

- Authority: `WORKER_ENTRYPOINT`
- graph_root: true (workers are independent execution roots)
- Confidence: 0.85

### Source 5: Dynamic Registry Roots

Directories where modules are loaded by convention (not explicit import):
```
tools/*/src/cli/          → Subcommand registries (batch-runner adapters, etc.)
tools/*/src/commands/     → Command-pattern directories
```

- Authority: `DYNAMIC_REGISTRY_ROOT`
- graph_root: true (modules under these dirs may be loaded dynamically)
- Confidence: 0.7–0.8
- Path is a directory, not a file

### Source 6: Child-Process Targets (Optional)

Scan for `spawn(`, `spawnSync(`, `exec(`, `execFile(` calls to detect subprocess targets. Extract the command argument if it resolves to a local file.

This is optional because:
- Targets may be external binaries
- Targets may be dynamically constructed
- High false-positive rate from `.exec(` regex matches

## Phase 3: Deduplication and Merging

The same file may be discovered by multiple sources (e.g., `tools/control-plane/cli.mjs` found by Source 2 AND Source 6). When merging:

```js
seen.set(c.path, createEntrypointCandidate({
  ...existing,  // Keep higher-confidence existing values
  evidence: [...new Set([...existing.evidence, ...c.evidence])],
  discovery_sources: [...new Set([...existing.discovery_sources, ...c.discovery_sources])],
  confidence: Math.max(existing.confidence, c.confidence),
}));
```

## Phase 4: Entrypoint Validation

For every candidate, validate:

1. **File exists** — if not, mark `MISSING_FILE`
2. **Syntax check** — run `node --check <file>` for .mjs files (5s timeout)
3. **Shebang check** — detect `#!/usr/bin/env node` vs bash
4. **Generated check** — look for `@generated`, `Generated by`, or path contains `generated/` or `/dist/`

Validation statuses: `VALID`, `MISSING_FILE`, `PARSE_ERROR`, `SHELL_WRAPPER`, `GENERATED`

## Phase 5: Calibrated Reachability

### Group Construction

Categorize validated graph roots:

```js
const groups = {
  primary:   roots.filter(c => c.authority === 'CANONICAL_CLI_ENTRYPOINT' || c.authority === 'CANONICAL_PRODUCTION_ENTRYPOINT'),
  independent: roots.filter(c => c.authority === 'INDEPENDENT_TOOL_ENTRYPOINT'),
  workers:   roots.filter(c => c.authority === 'WORKER_ENTRYPOINT'),
  test:      roots.filter(c => c.authority === 'TEST_ENTRYPOINT'),
  dynamic:   roots.filter(c => c.authority === 'DYNAMIC_REGISTRY_ROOT'),
};
```

### Multi-Root BFS

For each group, run BFS from all roots in that group:

```js
function bfsFromRoots(roots, moduleMap) {
  const reachable = new Set();
  const queue = [...roots];
  while (queue.length > 0) {
    const node = queue.shift();
    if (reachable.has(node)) continue;
    reachable.add(node);
    const mod = moduleMap.get(node);
    if (mod) {
      for (const dep of (mod.dependencies || [])) {
        if (dep.resolved && !dep.coreModule && !reachable.has(dep.resolved))
          queue.push(dep.resolved);
      }
    }
  }
  return reachable;
}
```

### Per-Module Classification

Each module gets a reachability record with ALL applicable classes:

```js
{
  path: 'tools/example/module.mjs',
  reachable: true,  // true if ANY root group reaches it
  reachability_classes: [
    'REACHABLE_FROM_PRIMARY_CANONICAL_ENTRYPOINT',
    'REACHABLE_FROM_INDEPENDENT_TOOL_ENTRYPOINT'
  ],
  reached_from: ['root-cloe', 'tool-cli-control-plane'],
  confidence: 0.95
}
```

### 10 Reachability Classes

| Class | reachable | Meaning |
|-------|-----------|---------|
| `REACHABLE_FROM_PRIMARY_CANONICAL_ENTRYPOINT` | true | Normal production/app code |
| `REACHABLE_FROM_INDEPENDENT_TOOL_ENTRYPOINT` | true | Tool module, not dead code |
| `REACHABLE_FROM_TEST_ENTRYPOINT_ONLY` | true | Test-only import, not production-reachable |
| `REACHABLE_FROM_DYNAMIC_OR_IMPLICIT_ROOT` | true | Loaded by convention or worker |
| `TRULY_UNREACHABLE_FROM_ALL_VALIDATED_ROOTS` | false | Confirmed orphan — highest confidence |
| `UNRESOLVED_DUE_TO_MISSING_ENTRYPOINT` | false | May be reachable if we had better entrypoints |
| `UNRESOLVED_DUE_TO_DYNAMIC_LOADING` | false | Has dynamic imports — static analysis incomplete |
| `UNRESOLVED_DUE_TO_GRAPH_PARSE_FAILURE` | false | Could not parse |
| `EXCLUDED_FROM_ANALYSIS` | false | Generated, historical, or vendor |
| `HISTORICAL_OR_CERTIFICATION_ARTIFACT` | false | Certification evidence |

### Candidate Mapping Rule

Only `TRULY_UNREACHABLE_FROM_ALL_VALIDATED_ROOTS` may map to:
```
classification: UNREACHABLE_FROM_CANONICAL_ENTRYPOINT
recommended_action: REVIEW
approval_required: true
```

All other unresolved categories must use `KEEP` or `REVIEW`. Never `DELETE`.

## Phase 6: Completeness Score

Evaluate entrypoint coverage quality:

```js
const requiredScore = requiredEntrypoints.length > 0
  ? resolvedRequired.length / requiredEntrypoints.length
  : 0.5;  // No required entrypoints = neutral

const discoveredScore = discoveredGraphRoots.length > 0
  ? Math.min(discoveredGraphRoots.length / 20, 1)  // 20+ = full score
  : 0;

const validationScore = validatedGraphRoots.length > 0
  ? 1 - ((missingFiles + parseErrors) / Math.max(validatedGraphRoots.length, 1))
  : 0;

const graphCoverage = graph?.summary
  ? Math.min(graph.summary.reachableCount / Math.max(graph.summary.totalModules, 1) * 10, 1)
  : 0;

const completenessScore = requiredScore * 0.4 + discoveredScore * 0.3
  + validationScore * 0.2 + graphCoverage * 0.1;
```

Confidence thresholds:
- **HIGH**: score >= 0.8
- **MEDIUM**: score >= 0.5
- **LOW**: score < 0.5

## Phase 7: Before/After Metrics

Required comparison table:

| Metric | Before | After |
|--------|--------|-------|
| Configured entrypoints | X | X |
| Resolved entrypoints | X | X |
| Validated graph roots | X | X |
| Primary canonical roots | X | X |
| Independent tool roots | X | X |
| Worker roots | X | X |
| Test roots | X | X |
| Dynamic roots | X | X |
| Total modules | X | X |
| Reachable (any root) | X | X |
| Primary reachable | X | X |
| Tool-reachable | X | X |
| Test-only reachable | X | X |
| Dynamic/implicit reachable | X | X |
| Truly unreachable | X | X |
| Unresolved (incomplete evidence) | X | X |
| Completeness score | X | X |
| Confidence | X | X |

## Real-World Example: lah-stack-tools

A tools monorepo with 945 source files, 8 bin/ wrappers, 30+ tool directories, 2 workers, 2 dynamic registry roots.

**Phase 2 discovery**: 361 candidates from 6 sources
**Phase 4 validation**: 54 high-confidence graph roots (>= 0.7)
**Phase 5 calibration**:
- 3 primary-reachable (root `cloe.mjs`, `lah-start.mjs`, `lah-guard.mjs`)
- 37 tool-reachable (control-plane, autoresearch-wrapper, strategic-planner, etc.)
- 4 worker-reachable
- 8 truly unreachable (confirmed orphans, mostly test fixtures)
- 845 unresolved (incomplete evidence — tool entries are shell wrappers spawn targets)

**Phase 6 score**: 0.70 (MEDIUM) — many independent tools are shell wrappers that spawn subprocesses, not import-graph reachable.

## Pitfalls

| Trap | Symptom | Fix |
|------|---------|-----|
| **Bash wrappers as graph roots** | Shell wrappers in bin/ are detected as CLIs but depcruise can't follow bash | Discover the actual node entry the wrapper invokes |
| **Duplicate counts** | Same file found by 2+ sources | Deduplicate by path with evidence merging |
| **Test reachability leakage** | A test-only import makes a module "reachable" in the unified graph | Keep test reachability in a separate category — never count as production |
| **Over-counting tool roots** | Every tool's index.mjs is a graph root — but many just re-export | Only use graph_root=true tools that have their own CLI or are standalone |
| **Dynamic roots inflate reachable** | A directory root makes all children "reachable" | Use dynamic roots as evidence only, not as reachability BFS seeds |
| **False confidence from low entrypoint count** | Only 2 entrypoints found = high reachable = deceptive | The completeness score penalizes low entrypoint coverage |
| **Ignoring bin/ entirely** | Shell wrappers are not import-graph-reachable but they ARE real execution surfaces | Document them separately in the report |
