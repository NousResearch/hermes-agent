---
name: systematic-debugging
description: Use when encountering any bug, test failure, or unexpected behavior. 4-phase root cause investigation — NO fixes without understanding the problem first.
version: 1.1.0
author: Hermes Agent (adapted from obra/superpowers)
license: MIT
metadata:
  hermes:
    tags: [debugging, troubleshooting, problem-solving, root-cause, investigation]
    related_skills: [test-driven-development, writing-plans, subagent-driven-development]
---

# Systematic Debugging

## Overview

Random fixes waste time and create new bugs. Quick patches mask underlying issues.

**Core principle:** ALWAYS find root cause before attempting fixes. Symptom fixes are failure.

**Violating the letter of this process is violating the spirit of debugging.**

## The Iron Law

```
NO FIXES WITHOUT ROOT CAUSE INVESTIGATION FIRST
```

If you haven't completed Phase 1, you cannot propose fixes.

## When to Use

Use for ANY technical issue:
- Test failures
- Bugs in production
- Unexpected behavior
- Performance problems
- Build failures
- Integration issues

**Use this ESPECIALLY when:**
- Under time pressure (emergencies make guessing tempting)
- "Just one quick fix" seems obvious
- You've already tried multiple fixes
- Previous fix didn't work
- You don't fully understand the issue

**Don't skip when:**
- Issue seems simple (simple bugs have root causes too)
- You're in a hurry (rushing guarantees rework)
- Someone wants it fixed NOW (systematic is faster than thrashing)

## The Four Phases

You MUST complete each phase before proceeding to the next.

---

## Phase 1: Root Cause Investigation

**BEFORE attempting ANY fix:**

### 1. Read Error Messages Carefully

- Don't skip past errors or warnings
- They often contain the exact solution
- Read stack traces completely
- Note line numbers, file paths, error codes

**Action:** Use `read_file` on the relevant source files. Use `search_files` to find the error string in the codebase.

**Python Import Errors — Additional Checks:**

`ModuleNotFoundError: No module named 'X'` can mean:
1. Package `X` is not installed → `pip install X`
2. Package is installed but import name differs from pip name (e.g., `import ultralytics` for `ultralytics` package)
3. Directory/package rename: code imports `X` but actual package is `x_subpackage`
4. `sys.path` issue: package is installed but not on the Python path

When the package DOES exist locally (e.g., `from collab.server import` but package is named `collaboration`):
- The import name in the code tells you what the author EXPECTED the package to be named
- The actual directory name tells you what it IS named
- **The mismatch is the bug** — fix the import, not the directory

### 2. Reproduce Consistently

- Can you trigger it reliably?
- What are the exact steps?
- Does it happen every time?
- If not reproducible → gather more data, don't guess

**Action:** Use the `terminal` tool to run the failing test or trigger the bug:

```bash
# Run specific failing test
pytest tests/test_module.py::test_name -v

# Run with verbose output
pytest tests/test_module.py -v --tb=long
```

### 3. Check Recent Changes

**Before investigating: pull latest code from remote**

```bash
git fetch origin && git pull origin <branch>
```

(Replace `<branch>` with the relevant branch — typically `main` for most repos.)

- What changed that could cause this?
- Git diff, recent commits
- New dependencies, config changes

**Action:**

```bash
# Recent commits
git log --oneline -10

# Uncommitted changes
git diff

# Changes in specific file
git log -p --follow src/problematic_file.py | head -100
```

### 4. Gather Evidence in Multi-Component Systems

**WHEN system has multiple components (API → service → database, CI → build → deploy):**

**BEFORE proposing fixes, add diagnostic instrumentation:**

For EACH component boundary:
- Log what data enters the component
- Log what data exits the component
- Verify environment/config propagation
- Check state at each layer

Run once to gather evidence showing WHERE it breaks.
THEN analyze evidence to identify the failing component.
THEN investigate that specific component.

### 5. Trace Data Flow

**WHEN error is deep in the call stack:**

- Where does the bad value originate?
- What called this function with the bad value?
- Keep tracing upstream until you find the source
- Fix at the source, not at the symptom

**Action:** Use `search_files` to trace references:

```python
# Find where the function is called
search_files("function_name(", path="src/", file_glob="*.py")

# Find where the variable is set
search_files("variable_name\\s*=", path="src/", file_glob="*.py")
```

### Phase 1 Completion Checklist

- [ ] Error messages fully read and understood
- [ ] Issue reproduced consistently
- [ ] Recent changes identified and reviewed
- [ ] Evidence gathered (logs, state, data flow)
- [ ] Problem isolated to specific component/code
- [ ] Root cause hypothesis formed

**STOP:** Do not proceed to Phase 2 until you understand WHY it's happening.

---

## Phase 2: Pattern Analysis

**Find the pattern before fixing:**

### 1. Find Working Examples

- Locate similar working code in the same codebase
- What works that's similar to what's broken?

**Action:** Use `search_files` to find comparable patterns:

```python
search_files("similar_pattern", path="src/", file_glob="*.py")
```

### 2. Compare Against References

- If implementing a pattern, read the reference implementation COMPLETELY
- Don't skim — read every line
- Understand the pattern fully before applying

### 3. Identify Differences

- What's different between working and broken?
- List every difference, however small
- Don't assume "that can't matter"

### 4. Understand Dependencies

- What other components does this need?
- What settings, config, environment?
- What assumptions does it make?

---

## Phase 3: Hypothesis and Testing

**Scientific method:**

### 1. Form a Single Hypothesis

- State clearly: "I think X is the root cause because Y"
- Write it down
- Be specific, not vague

### 2. Test Minimally

- Make the SMALLEST possible change to test the hypothesis
- One variable at a time
- Don't fix multiple things at once

### 3. Verify Before Continuing

- Did it work? → Phase 4
- Didn't work? → Form NEW hypothesis
- DON'T add more fixes on top

### 4. When You Don't Know

- Say "I don't understand X"
- Don't pretend to know
- Ask the user for help
- Research more

---

## Phase 4: Implementation

**Fix the root cause, not the symptom:**

### 1. Create Failing Test Case

- Simplest possible reproduction
- Automated test if possible
- MUST have before fixing
- Use the `test-driven-development` skill

### 2. Implement Single Fix

- Address the root cause identified
- ONE change at a time
- No "while I'm here" improvements
- No bundled refactoring

### 3. Verify Fix

```bash
# Run the specific regression test
pytest tests/test_module.py::test_regression -v

# Run full suite — no regressions
pytest tests/ -q
```

### 4. If Fix Doesn't Work — The Rule of Three

- **STOP.**
- Count: How many fixes have you tried?
- If < 3: Return to Phase 1, re-analyze with new information
- **If ≥ 3: STOP and question the architecture (step 5 below)**
- DON'T attempt Fix #4 without architectural discussion

**Cascading Latent Bugs — Special Case:**

When the codebase has pre-existing bugs (not introduced by your fixes), fixing one bug can surface another. This is distinct from the "wrong architecture" pattern. Each individual bug has a real root cause — they were simply all present but only the first was visible.

**When cascading latent bugs occur:**
- Error #1 blocks import of module X
- Fix #1 unblocks X, revealing that module Y references a non-existent name
- Fix #2 unblocks Y, revealing that class Z is missing a required field
- This continues until all are fixed

**When to stop and report vs. continue:**
- After fixing 3+ bugs in the same file/component, PAUSE and give the user a status report
- "I've fixed 3 import/type errors in collab_api.py. The codebase has systematic import inconsistencies. There are likely more latent bugs. Do you want me to continue fixing or would you prefer to take a different approach?"
- This respects the user's time and lets them decide if the codebase needs a more thorough audit

### 4b. Editing Minified/Bundled JS Files

When the build system (Vite, Webpack, etc.) has a frozen/persistent cache and you need to hotfix the built output:

**sed vs patch tool for single-line JS:**
- `patch` tool can produce syntax errors on minified single-line JavaScript because it may misalign comma operators and expressions
- `sed` is more reliable for targeted string replacements in minified JS

```bash
# Example: fix a coordinate in bundled minified JS
sed -i 's/old_coord/new_coord/g' dist/assets/bundle.js
```

**Root cause vs symptom in built files:**
- When error originates in built JS, FIRST check if source files are wrong
- If source is correct but build is stale → fix the build system, not the output
- Only patch built output when build system itself is broken (e.g., Vite frozen cache)

### 4c. When Build System Has Frozen Cache

Symptoms: `npm run build` produces identical output despite source changes. MD5 of build artifacts doesn't change.

```bash
# Verify build is stale - compare MD5 before/after
md5sum dist/assets/bundle.js
# Modify source file
touch src/file.js && npm run build
md5sum dist/assets/bundle.js  # Same MD5 = frozen cache
```

**Workarounds (in order of preference):**
1. Clear node_modules/.vite/ cache directory
2. Delete dist/ completely before rebuild
3. Patch Vite's module resolution (advanced - see vite-build-cache-debug skill)
4. Patch the built output directly with sed

### 4b. Canvas 2D Rendering — Shadow vs Body Coordinate Space

**Common bug pattern:** A game entity renders its shadow correctly (world-space) but body parts at wrong screen position. Shadow appears at player's world position, body always at top-left corner.

**Root cause:** Body parts use canvas-absolute coordinates while shadow uses world-space coordinates. The canvas is already translated by camera offset before render(), so body parts need `ctx.translate(playerX, playerY)` before drawing.

**Diagnosis:** Compare shadow render code vs body render code in the same `render()` method. Shadow uses `this.x + offset` (world coords after camera transform). Body parts use fixed offsets from canvas center like `e.fillRect(-8, 12, 7, 16)` without the translate.

**Fix pattern:**
```javascript
// WRONG: body parts drawn at canvas coordinates (top-left of screen)
const i = this.x + this.width/2;
e.fillRect(-8, 12, 7, 16);  // relative to canvas origin!

// RIGHT: translate canvas to player position first, THEN draw body parts
e.save();
e.translate(i, this.y + this.height/2);  // move origin to player anchor
e.fillRect(-8, 12, 7, 16);  // now these are relative to player
e.restore();
```

**Key insight:** In Canvas 2D with camera transform (`e.translate(-camera.x, -camera.y)`), ALL world objects render correctly because they use world coords. But player-local decorations (body parts relative to player center) need their own translate FROM player world position.

**Y-offset math:** If player's feet are at `this.y + this.height` and anchor is `this.y + this.height/2`:
- Legs should start at `anchor + 12` (below anchor, toward ground)
- Body, head above anchor (anchor - N)
- NOT the reverse

### 4b. When Build System Has Frozen Cache

Symptoms: `npm run build` produces identical output despite source changes. MD5 of build artifacts doesn't change.

```bash
# Verify build is stale - compare MD5 before/after
md5sum dist/assets/bundle.js
# Modify source file
touch src/file.js && npm run build
md5sum dist/assets/bundle.js  # Same MD5 = frozen cache
```

**Workarounds (in order of preference):**
1. Clear node_modules/.vite/ cache directory: `rm -rf node_modules/.vite/`
2. Delete dist/ completely before rebuild: `rm -rf dist/ && npm run build`
3. Patch the built output directly with sed (when above don't work)

**Direct patch workflow for frozen Vite cache:**
```bash
# sed is more reliable than patch tool for single-line minified JS
sed -i 's/old_pattern/new_pattern/g' dist/assets/bundle.js
node --check dist/assets/bundle.js  # verify syntax before committing
git add dist/assets/bundle.js && git commit -m "fix: [description]" && git push
```

**Why sed over patch tool:** The `patch` tool can misalign comma operators and expressions in minified single-line JavaScript, producing syntax errors. sed operates on raw string replacement at the byte level.

**Important — determine what's actually served:**
- For GitHub Pages with gh-pages branch: check BOTH `dist/` AND root-level `assets/` directory
- GitHub Pages serves from root, not from `dist/`
- The served HTML may reference `./assets/bundle.js` (root-level) rather than `dist/assets/bundle.js`
- Always verify: `curl -s "https://domain.com/assets/bundle.js" | grep -o 'pattern_to_find'`

### 4m. Debugging React Production Bundles — Function Signature Analysis

**Symptom:** `ReferenceError: X is not defined` at `index-XXXXX.js:LINE:COL` in a React production build. TypeScript compiles fine. The error is inside a `.map()` callback within a React component.

**Root Cause Pattern:** A React component (often an inline function inside another component) uses a variable in its JSX/render body, but that variable is NOT in the function's destructured parameters — it "accidentally" works because the outer scope has a variable of the same name, but in the minified bundle the function gets its own scope and the variable is undefined.

**Diagnosis — analyze the minified bundle function signature:**

1. Find the minified function name: React components get minified to short names like `yK`, `mK`, `vb`. Search by the error line or by the source component name pattern.

2. Extract the function signature from the bundle:
```bash
# Find function declaration with its parameters in the bundle
grep -o 'yK=function(.{0,500})' dist/assets/bundle.js | head -1
# or find the full signature
grep -n "function yK" dist/assets/bundle.js
```

3. Compare what the function RECEIVES vs what it USES:

In the bundle, a React component looks like:
```javascript
function yK({column, proposals, droppableId, onCardClick, isDropTarget, isCollapsed, onToggleCollapse}) {
    // ... function body uses selectedProposalIds and onToggleSelectProposal
}
```

If `selectedProposalIds` and `onToggleSelectProposal` are used in the body but NOT in the parameters → that's the bug.

4. Verify by checking the JSX call site in the bundle — it DOES pass these props:
```javascript
p.jsx(yK, {column, proposals, droppableId, onCardClick, isDropTarget, isCollapsed, onToggleCollapse, selectedProposalIds, onToggleSelectProposal})
```
Note: even if the call site passes them, if the function signature doesn't destructure them, they won't be available in the function scope.

**Fix:** Add the missing parameters to the function signature in the SOURCE file — NOT in the bundle:
```javascript
// BEFORE (bug):
function DroppableColumn({ column, proposals, droppableId, onCardClick, isDropTarget, isCollapsed, onToggleCollapse }) {

// AFTER (fix):
function DroppableColumn({ column, proposals, droppableId, onCardClick, isDropTarget, isCollapsed, onToggleCollapse, selectedProposalIds, onToggleSelectProposal }) {
```

**Why TypeScript doesn't catch this:** TypeScript sees `selectedProposalIds` referenced in the body and finds it in the outer scope (from props passed by the parent), so no error. But when the component is called, the outer scope's variables aren't automatically available as local parameters.

**Key insight:** In React, when a component function destructures its props, ALL props used in the body must be in the destructuring list — no accidental outer-scope fallbacks. The TWO places that must match: (1) the JSX call site passes the props, AND (2) the function signature destructures them.

### 4j. Browser Line Numbers Are Unreliable

When hitting `ModuleNotFoundError` in Python, BEFORE attempting fixes, scan ALL imports in the affected file:

```bash
# Find every import in a Python file — catch latent issues early
grep -rn "^from \|^import \|from \. " path/to/file.py

# Find all imports across a package
grep -rn "^from \|^import " path/to/package/ --include="*.py"
```

This surfaces ALL import issues at once instead of discovering them one by one.

### 4f. FastAPI Catch-All Route Intercepting API Calls

**Symptom:** Requests to API endpoints return "Method Not Allowed" or HTML (when JSON expected). The route exists in your router but requests never reach it.

**Root Cause:** When using `app.include_router(router)` with a catch-all `/{full_path:path}` route, FastAPI routes are matched in registration order. If the catch-all is registered AFTER the included router, it intercepts ALL paths including `/api/...` before the router can handle them.

**Pattern that breaks:**
```python
app.include_router(collab_router)  # has routes like /api/collab/agents

@app.get("/{full_path:path}")       # registered AFTER — catches EVERYTHING
async def serve_spa(full_path: str):
    return FileResponse("dist/index.html")  # serves HTML for /api/agents too!
```

**Fix — check for API prefix first:**
```python
@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    # Return 404 for API paths so FastAPI can try next matching route
    if full_path.startswith("api/"):
        from fastapi.responses import JSONResponse
        return JSONResponse({"detail": "Not Found"}, status_code=404)
    file_path = Path("dist") / full_path
    if full_path and file_path.is_file():
        return FileResponse(file_path)
    return FileResponse("dist/index.html")
```

**Better pattern — mount API at prefix, separate from SPA:**
```python
# API router mounted first at known prefix
app.include_router(api_router, prefix="/api")
# SPA fallback only catches non-API paths (registered last as catch-all)
```

**Diagnosis:** `curl -v http://host:port/api/route` — if HTML comes back instead of JSON, the SPA fallback is intercepting. The `status_code=404` approach lets FastAPI continue route matching rather than short-circuiting.

### 4g. JavaScript — Shallow Copy Object + indexOf Returns -1 Bug

**Symptom:** `TypeError: Cannot set properties of undefined (setting 'X')` — error occurs on a line that sets a property on an object that appears to exist.

**Root cause pattern:**

```javascript
// BAD: shallow copy creates a NEW object with different reference
this.animatingCandy = { ...candy };  // creates new object
const idx = this.candies.indexOf(this.animatingCandy);  // returns -1!
this.candies[idx].matched = true;   // TypeError: cannot set properties of undefined
```

### 4h. Browser Console — Catching Silent Errors with window.onerror

**Symptom:** Button click has no visible effect, no console errors appear, but function isn't executing.

**Problem:** Some errors are caught by the browser but don't appear in console output.

**Diagnosis pattern:**
```javascript
// Set global error handler to catch uncaught errors
window.onerror = function(msg, url, line, col, error) {
    console.error('GLOBAL ERROR: ' + msg + ' at line ' + line);
    return false;
};

// Then try calling the function directly
try { startLevel(0, 0); } catch(e) { console.error(e.message); }
```

**Key insight:** Browser console might show no errors even when errors occur. Direct function calls via `browser_console` can reveal errors that are swallowed by the page's error handling.

**For ES Module scope issues:** Variables declared with `let`/`const` are NOT accessible on `window` by default. Check with:
```javascript
window.hasOwnProperty('variableName')  // returns false for undeclared
typeof variableName  // returns 'undefined' if not accessible in scope
```

When `Cannot access 'X' before initialization` occurs in a function but `X` appears to be declared — the function may be defined in a scope where the variable isn't yet accessible (ES module temporal dead zone). Verify the declaration order in source code.

### 4i. Latent Bug Cascade — One Fix Reveals Another

**Pattern:** Fix one bug and a NEW error immediately appears. Each fix unblocks the next latent bug.

**Example from debugging a game:**
1. Error #1: `Cannot read properties of null (reading 'addEventListener')` on achievement button
   - Fix: Add optional chaining (`?.`) null check
2. Push fix, test — game still doesn't start
3. New error: `Cannot access 'currentWorld' before initialization`  
   - Root cause: ES module variable scoping — `let currentWorld` declared at line 1269, function at line 2807 tries to use it before it's accessible in that scope's execution order

**When this happens:**
- Don't panic — the first fix was correct
- The second error was already present, just hidden
- Continue systematically: fix one, test, fix next
- Report to user: "Found N latent bugs — here's the full status"

**Rule:** When fixing cascading latent bugs, after each fix verify the page still loads and test the specific feature. Don't assume multiple fixes are needed simultaneously.

### 4j. Browser Line Numbers Are Unreliable in HTML Files with CDN Scripts

**Symptom:** Console error reports "at line 3465:51" but the HTML file only has 2000 lines.

**Root Cause:** When an HTML file contains `<script src="https://CDN/...">` tags, the browser counts line numbers from the start of the HTML document INCLUDING all the CDN script content. A CDN script like three.min.js might be 10000+ lines on its own, so a browser-reported line 3465 in your inline `<script>` block could actually be at line 10 of your source file.

**Diagnosis:**
1. NEVER trust browser line numbers at face value for HTML files with CDN `<script src>` tags
2. Instead, search by the error message content or function name:
   ```bash
   # Find the error string in your source
   search_files("Cannot read properties of null", path="/path/to/project", file_glob="*.html")
   # Or find the specific function
   search_files("achievementBtn", path="/path/to/project", file_glob="*.html")
   ```
3. The browser might also report a line in a CDN script (e.g., "at three.min.js:1253") even though the real problem is in your code calling that CDN function with wrong arguments

**Fix verification:** After finding the actual line, confirm the fix was applied by downloading the deployed file and checking the exact content — don't rely on browser line numbers to verify the fix location.

**Pattern to add to Phase 1:** When error line number exceeds total lines in the source file, suspect CDN script line counting.

### 4j. Async/Await Chain Silent Failure — Click Handler Does Nothing

**Symptom:** User clicks a button, nothing happens. No console errors. No visible reaction. The page appears fine.

**Root Cause Mechanism:** `async/await` chain failure is **silent**. When an `await` rejects, it throws synchronously into the caller's await expression — but if nobody `await`s that call, or if the rejection happens inside a Promise that isn't observed, the error is **swallowed**. The chain simply stops. No console error. No uncaught exception. Just... nothing.

**Classic Pattern — Click handler starts async chain, nothing happens:**
```javascript
// User clicks start button
startBtn.addEventListener('click', startGame);  // registered but nobody awaits

async function startGame() {
  await initScene();      // runs OK
  await loadModels();     // REJECTS — model URL unreachable
  await startLevel();     // NEVER REACHED — chain stops here
  // No error logged. No crash. Just stops.
}
```

**Why the console might show no errors:**
- If `startGame` is called without `await` (which is the case inside addEventListener)
- If the rejection happens in a Promise that nothing observes
- If there's a try-catch somewhere swallowing it

**Debug Flow:**
1. Identify the click handler and trace its execution path
2. Call the handler function **directly** via `browser_console`:
   ```javascript
   startGame()  // observe: does it throw? Does it log? Does it stop silently?
   ```
3. If direct call works but clicking doesn't → the event registration is broken
4. If direct call also fails → trace inside the async chain, one `await` at a time:
   ```javascript
   async function trace() {
     try { console.log('1'); await initScene(); console.log('2'); } catch(e) { console.error('failed at 2:', e); }
     try { console.log('3'); await loadModels(); console.log('4'); } catch(e) { console.error('failed at 4:', e); }
     try { console.log('5'); await startLevel(); console.log('6'); } catch(e) { console.error('failed at 6:', e); }
   }
   trace()
   ```
5. Verify network resources with `curl` — don't trust URLs in code:
   ```bash
   curl -I --max-time 5 "https://threejs.org/examples/models/gltf/RobotExpressive/RobotExpressive.glb"
   ```

**Verification Checklist for "Click Does Nothing":**
- [ ] Button exists and `btn.click()` from console triggers handler
- [ ] Handler function exists and is callable
- [ ] Each `await` in the chain actually resolves (add per-step logging)
- [ ] External resources (CDN models, images) are actually reachable
- [ ] DOM elements exist when event listeners are registered

**Symptom:** `TypeError: Cannot set properties of undefined (setting 'X')` — error occurs on a line that sets a property on an object that appears to exist.

**Root cause pattern:**

```javascript
// BAD: shallow copy creates a NEW object with different reference
this.animatingCandy = { ...candy };  // creates new object
const idx = this.candies.indexOf(this.animatingCandy);  // returns -1!
this.candies[idx].matched = true;   // TypeError: cannot set properties of undefined
```

**Why it happens:** `indexOf` uses reference equality. `{ ...candy }` creates a new object. `this.candies.indexOf(newObject)` looks for `newObject` in `this.candies` array and returns `-1` because no element in the array is that exact object reference.

**Diagnosis steps:**
1. Search for `indexOf` calls that might receive shallow copies
2. Check for spread operator `{ ...obj }` creating copies that are then used in `indexOf`
3. Verify the variable isn't reassigned somewhere between creation and `indexOf` call
4. Add `console.log(idx, obj)` before the failing line to confirm idx is -1

**Fix pattern — use indices instead of object references:**

```javascript
// GOOD: store the index, not the object
this.animatingCandyIndex = this.candies.indexOf(candy);
this.animatingTargetCup = targetIndex;
// In update():
if (this.animatingCandyIndex >= 0) {
    this.candies[this.animatingCandyIndex].matched = true;
}
```

**When shallow copy IS appropriate (and indexOf still works):**
- When you need to mutate the copy without affecting the original
- When you never need to find the copy in the original array

**When shallow copy FAILS with indexOf:**
- When the copy needs to be found in the original array
- When `indexOf` or `findIndex` is used for lookups

**Verification:** `node --check` catches syntax errors but NOT logic errors. Write minimal test:
```javascript
const arr = [{ id: 1 }, { id: 2 }];
const copy = { ...arr[0] };
console.log(arr.indexOf(copy));  // -1 — proves the bug pattern
```

### 4k. Graph-Based Node Unlock Debugging — "Click Does Nothing" But No Error

**Symptom:** Clicking nodes on a game map (canvas-based) does nothing. No preview popup. No error. Button appears to be dead.

**Root Cause:** Not a click handler problem — a game state problem. The `unlockedNodes` array only contained `["f1r1"]` (rest node). Combat/elite/boss nodes were locked. The click handler correctly detected the click and checked `isNodeUnlocked()`, which returned false.

**Diagnosis:**
```javascript
// First step: check game state
gameState.unlockedNodes       // What nodes are unlocked?
gameState.completedNodes     // What nodes are completed?
gameState.currentFloor      // Which floor?

// Verify the click is registering
const fakeEvent = { clientX: rect.left + X, clientY: rect.top + Y };
handleMapClick(fakeEvent);   // Direct call to handler
```

**Pattern:** Many Roguelike/map games use a graph structure for node unlocking:
- `FLOORS[n].edges = [['nodeA', 'nodeB'], ...]` defines which nodes connect
- `completeNode(nodeId)` marks node done and calls `unlockNode()` for adjacent nodes via edges
- Only adjacent (connected via edge) nodes get unlocked
- First node is typically pre-unlocked to start the progression

**Fix:** Click the pre-unlocked starting node first to begin the unlock chain.

**Verification:** After clicking starting node, `unlockedNodes` should grow to include adjacent nodes. Then click the next adjacent node.

### 5. If 3+ Fixes Failed: Question Architecture

**Pattern indicating an architectural problem:**
- Each fix reveals new shared state/coupling in a different place
- Fixes require "massive refactoring" to implement
- Each fix creates new symptoms elsewhere

**STOP and question fundamentals:**
- Is this pattern fundamentally sound?
- Are we "sticking with it through sheer inertia"?
- Should we refactor the architecture vs. continue fixing symptoms?

**Discuss with the user before attempting more fixes.**

This is NOT a failed hypothesis — this is a wrong architecture.

---

## Red Flags — STOP and Follow Process

If you catch yourself thinking:
- "Quick fix for now, investigate later"
- "Just try changing X and see if it works"
- "Add multiple changes, run tests"
- "Skip the test, I'll manually verify"
- "It's probably X, let me fix that"
- "I don't fully understand but this might work"
- "Pattern says X but I'll adapt it differently"
- "Here are the main problems: [lists fixes without investigation]"
- Proposing solutions before tracing data flow
- **"One more fix attempt" (when already tried 2+)**
- **Each fix reveals a new problem in a different place**

**ALL of these mean: STOP. Return to Phase 1.**

**If 3+ fixes failed:** Question the architecture (Phase 4 step 5).

## Common Rationalizations

| Excuse | Reality |
|--------|---------|
| "Issue is simple, don't need process" | Simple issues have root causes too. Process is fast for simple bugs. |
| "Emergency, no time for process" | Systematic debugging is FASTER than guess-and-check thrashing. |
| "Just try this first, then investigate" | First fix sets the pattern. Do it right from the start. |
| "I'll write test after confirming fix works" | Untested fixes don't stick. Test first proves it. |
| "Multiple fixes at once saves time" | Can't isolate what worked. Causes new bugs. |
| "Reference too long, I'll adapt the pattern" | Partial understanding guarantees bugs. Read it completely. |
| "I see the problem, let me fix it" | Seeing symptoms ≠ understanding root cause. |
| "One more fix attempt" (after 2+ failures) | 3+ failures = architectural problem. Question the pattern, don't fix again. |

## Quick Reference

| Phase | Key Activities | Success Criteria |
|-------|---------------|------------------|
| **1. Root Cause** | Read errors, reproduce, check changes, gather evidence, trace data flow | Understand WHAT and WHY |
| **2. Pattern** | Find working examples, compare, identify differences | Know what's different |
| **3. Hypothesis** | Form theory, test minimally, one variable at a time | Confirmed or new hypothesis |
| **4. Implementation** | Create regression test, fix root cause, verify | Bug resolved, all tests pass |

## Hermes Agent Integration

### Investigation Tools

Use these Hermes tools during Phase 1:

- **`search_files`** — Find error strings, trace function calls, locate patterns
- **`read_file`** — Read source code with line numbers for precise analysis
- **`terminal`** — Run tests, check git history, reproduce bugs
- **`web_search`/`web_extract`** — Research error messages, library docs

### With delegate_task

For complex multi-component debugging, dispatch investigation subagents:

```python
delegate_task(
    goal="Investigate why [specific test/behavior] fails",
    context="""
    Follow systematic-debugging skill:
    1. Read the error message carefully
    2. Reproduce the issue
    3. Trace the data flow to find root cause
    4. Report findings — do NOT fix yet

    Error: [paste full error]
    File: [path to failing code]
    Test command: [exact command]
    """,
    toolsets=['terminal', 'file']
)
```

### With test-driven-development

When fixing bugs:
1. Write a test that reproduces the bug (RED)
2. Debug systematically to find root cause
3. Fix the root cause (GREEN)
4. The test proves the fix and prevents regression

## Real-World Impact

From debugging sessions:
- Systematic approach: 15-30 minutes to fix
- Random fixes approach: 2-3 hours of thrashing
- First-time fix rate: 95% vs 40%
- New bugs introduced: Near zero vs common

**No shortcuts. No guessing. Systematic always wins.**
