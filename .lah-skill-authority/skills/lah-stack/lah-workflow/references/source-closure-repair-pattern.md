# Source-Closure Repair Pattern

Use when a commit has been pushed with untracked source files or missing exports,
and you need to produce a self-contained commit whose clean checkout can install,
load, and test without depending on external files.

## 5-Class Missing-Import Taxonomy

When analyzing scanner output or diagnostic reports, classify each finding:

| Class | Meaning | Action |
|-------|---------|--------|
| **REQUIRED_PRODUCTION_SOURCE** | A source file needed by production code that is not tracked. The file exists in the dirty workspace. | Add the file to Git. Verify its imports also resolve. |
| **MODIFIED_PRODUCTION_SOURCE** | A tracked file that is modified in the dirty workspace to add an export or function that a committed file already imports. The committed version is incomplete. | Stage the modified version. The export was missing from the commit. |
| **DEAD_OR_STALE_IMPORT** | A production import target that does not exist anywhere — not in the commit, not in the dirty workspace. | Determine if the import is from a removed feature or a pre-existing test infrastructure defect. If out of scope, document and skip. |
| **FALSE_POSITIVE_SCANNER_MATCH** | A path pattern that the scanner's regex matched but is not an actual import. Common causes: string fixtures in tests, `resolve()` calls, dynamic `import()` to tracked files, JSDoc comments, negative test assertions. | Trace back to the source line. If it's a string literal, `resolve()`, or dynamic import to a tracked file → false positive. |
| **REQUIRED_TEST_SOURCE** | A file needed by test code that is not tracked. Only classify as this if the test is part of the commit's test suite and the missing file is actually required for test execution. | Add if the test is in scope. Otherwise, document as pre-existing defect. |

## Two-Level Check Protocol

File-level import resolution is NOT sufficient. You must also verify named exports:

### Level 1 — Path resolution
```javascript
// Does the import path resolve to an existing file?
const resolved = resolve(dirname(sourceFile), importPath);
// Try exact, .js, /index.js, /index.mjs
```
This catches missing files but NOT missing exports.

### Level 2 — Module loading
```javascript
// Does the module actually load without errors?
await import('./src/server.js');  // catches missing exports
```
A file may exist and the import path may resolve, but the required export may not exist in the committed version of the dependency.

## Clean Worktree Proof Protocol

After committing the repair, prove closure in a FRESH detached worktree:

```bash
git worktree add --detach /tmp/<unique-name> <SHA>
cd /tmp/<unique-name>/<app-dir>
git status --short          # must be clean
npm ci --ignore-scripts     # install without lifecycle scripts
git status --short          # must still be clean
```

Then run these checks in order:
1. **Source-closure checker** (if one exists) — `node bin/check-source-closure.mjs`
2. **Closure import tests** — focused test verifying each production module loads
3. **Server import** — `await import('./src/server.js')` to catch missing exports
4. **Existing test suite** — bridge tests, gateway tests, related unit tests
5. **Docker Compose config validation** — `docker compose config` with temp placeholder `.env`
6. **Docker build** — `docker build --build-arg GIT_COMMIT=<SHA>` from the clean worktree
7. **Verify running containers untouched** — `docker ps` confirms no restart

## Common Pitfalls

| Trap | Symptom | Fix |
|------|---------|-----|
| Scanner matches string fixture content | Test assertions like `source.includes("from './executor.js'")` or `resolve(__dirname, '../src/routes/...')` are flagged as missing imports | Trace each finding back to its source line; classify as FALSE_POSITIVE_SCANNER_MATCH |
| Server imports but auth doesn't export | Import path resolves, module file exists, but the specific named export wasn't committed | Add Level 2 (module loading) check; always `await import()` the entry point in the clean checkout |
| Previous worktree reused | The old worktree may have cached state, node_modules, or partial files | Always create a `git worktree add --detach` for final proof; never reuse the old worktree |
| `.env` required by Compose | `docker compose config` needs `.env` but it must not be committed | Create a temporary placeholder `.env` with non-secret values, validate, then remove it |
