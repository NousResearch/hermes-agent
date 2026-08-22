# External Tool Evidence Collector Pattern

Integrate a third-party analysis tool as a **read-only evidence producer** into a LAH Stack authority. The tool produces structural facts; the LAH authority interprets them and makes classification decisions. The tool never becomes the deletion authority.

Used for: dependency-cruiser, Knip, jscpd, Semgrep, Vulture, or any CLI tool that outputs structured analysis.

## Collector Contract

```
{
  id: string,             // Stable identifier (e.g. 'dependency-graph')
  version: number,        // Contract version (1)
  mode: 'read-only',      // Never mutates the repository
  isAvailable(context),   // Check tool availability + capture version
  collect(context),       // Run tool, normalize output, classify candidates
  describe()              // Metadata for operator documentation
}
```

## Integration Steps

### 1. Tool Detection (`isAvailable`)

```js
isAvailable(context = {}) {
  try {
    const result = spawnSync('npx', ['--yes', '<tool>', '--version'], {
      encoding: 'utf8', timeout: 30000, cwd: context.repoPath,
    });
    if (result.status === 0) return { available: true, version: versionStr, error: null };
    return { available: false, version: null, error: `Exit: ${result.status}` };
  } catch (err) {
    return { available: false, version: null, error: err.message };
  }
}
```

Pitfall: Use `spawnSync` (not `execSync`) to avoid shell injection. Use `npx --yes` to auto-download without prompt. Set `maxBuffer` for large outputs (depcruise can produce 10MB+).

### 2. Deterministic Invocation (`collect`)

```js
// Build CLI args dynamically
const args = ['--yes', '<tool>'];
if (config.configPath) args.push('--config', config.configPath);
else args.push('--no-config'); // Prevent auto-discovery failures
args.push('--output-type', 'json');
if (config.includeOnly) args.push('--include-only', ...patterns);
args.push('--do-not-follow', 'node_modules'); // Always exclude externals
if (config.ignoredPaths) for (const p of config.ignoredPaths) args.push('--exclude', p);
args.push(repoPath); // Final arg = target directory

const result = spawnSync('npx', args, { ... });
```

Key invariants captured:
- Working directory
- Configuration path (or `--no-config`)
- Tool version (from `isAvailable`)
- Exit code
- Duration (wall clock)
- Output hash (SHA-256 of stdout, first 16 hex chars)

### 3. Stable Failure Semantics

Every failure returns a structured error response with NO fabricated findings:

| Status | When | Behavior |
|--------|------|----------|
| `COLLECTOR_UNAVAILABLE` | Tool not found/not installed | No graph, empty candidates |
| `COLLECTOR_FAILED` | Non-zero exit | No graph, bounded raw output for debugging |
| `MALFORMED_OUTPUT` | JSON parse failure | No graph, bounded raw output |
| `PARTIAL_GRAPH` | Warnings in output | Graph produced, warnings recorded |
| `COLLECTOR_COMPLETE` | Success | Full graph + candidates |

### 4. Entrypoint Normalization

Tools return paths relative to the repo root; users may provide absolute entrypoint paths. Normalize before comparison:

```js
const normalizedEntrypoints = (config.entrypoints || []).map(ep => {
  if (isAbsolute(ep)) return relative(repoPath, ep);
  return ep;
});
```

### 5. Protective Classification Ordering

Classification priority determines which label applies when a module has multiple characteristics. Order matters — the first match wins:

```
Priority 1: CANONICAL_AUTHORITY (configured entrypoints — must preserve)
Priority 2: DYNAMICALLY_REFERENCED (dynamic incoming edges — override static reachability)
Priority 3: AMBIGUOUS_DORMANT_CAPABILITY (has import() calls — dynamic loader)
Priority 4: RUNTIME_REFERENCED (reachable via static analysis)
Priority 5: MANUAL_REVIEW_REQUIRED (unresolved imports, cycles)
Priority 6: UNREACHABLE_FROM_CANONICAL_ENTRYPOINT (not reachable, no counter-evidence)
```

Rule: **No classification may ever recommend DELETE based solely on tool output.** Always use REVIEW or KEEP.

### 6. Source-Level Scanning Fallback

Static analysis tools (depcruise, etc.) may miss imports that are:
- Unresolvable local imports (module file doesn't exist)
- Dynamic imports with template literals (`import(`template`)`)

Add a source scanner as secondary evidence:

```js
_scanSourceFile(filePath) {
  const content = readFileSync(filePath, 'utf8');
  // Detect static imports: import ... from '...'
  /^import\s/.test(line) && /from\s+['"]/.test(line)
  // Detect dynamic imports: import(...)
  /\bimport\s*\(/.test(line)
  // Detect CJS require
  /\brequire\s*\(/.test(line)
}
```

Use the scan to create `MANUAL_REVIEW_REQUIRED` candidates for modules that have import statements but zero resolved dependencies in the graph.

### 7. Core Module and External Filtering

Filter out modules that are not part of the source codebase:

```js
const coreModules = new Set(['fs', 'path', 'readline', 'url', ...]);
graph.modules = graph.modules.filter(m => {
  if (coreModules.has(m.source)) return false;         // Node built-ins
  if (m.source.startsWith('node:')) return false;       // node: protocol
  if (m.source.includes('node_modules')) return false;  // External packages
  return true;
});
```

Recalculate reachability/unreachable counts after filtering.

### 8. Zero-Mutation Guarantee

The collector MUST NOT:
- Delete files
- Rewrite imports
- Alter analysis tool configuration files
- Make final removal decisions
- Bypass the LAH candidate classifier

The collector IS allowed to:
- Read source files for scanning
- Write reports to `docs/repo-hygiene/`
- Create temporary analysis configs

## Verification Checklist

- [ ] Tool availability detection works with and without tool installed
- [ ] Version is captured correctly
- [ ] Non-zero exit returns `COLLECTOR_FAILED` with structured output
- [ ] Malformed JSON returns `MALFORMED_OUTPUT`
- [ ] Empty entrypoints array doesn't crash
- [ ] Missing entrypoint files are reported as `missingEntrypoints`
- [ ] Core Node modules are filtered from candidates
- [ ] No candidate ever has `recommended_action: 'DELETE'`
- [ ] Dynamic/ambiguous modules never get automatic KEEP without review
- [ ] Source files are not modified during collection
- [ ] CLI works with `--format json` and `--format markdown`
- [ ] Collector can be disabled — central authority still works
- [ ] Output hash is deterministic (same input → same hash)

## Known Limitations

- Only analyzes statically resolvable imports
- Dynamic/string-based imports are flagged but not followed
- TypeScript requires transpiler plugins (not bundled by default)
- Cannot detect runtime-only code paths
- Entrypoint completeness directly affects reachability accuracy
- Tool must be available via npx or locally installed
