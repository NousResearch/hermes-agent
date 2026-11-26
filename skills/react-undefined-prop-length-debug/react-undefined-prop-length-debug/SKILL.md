---
name: react-undefined-prop-length-debug
category: software-development
description: Debug white-screen React crashes where a prop .length access fails on undefined, focusing on execution-order traps with ??= defaults.
---

# React undefined prop .length crash — debugging method

## Symptoms
White screen with:
```
TypeError: Cannot read properties of undefined (reading 'length')
    at aF (index-XXXX.js:248:1193)
```
Where `aF` is a minified component/function name.

## Root cause pattern
Component receives an `undefined` prop and immediately accesses `.length` on it, sometimes INSIDE a conditional that evaluates BEFORE `??=` default assignment executes:

```jsx
// BUG: toasts is undefined, ??= runs AFTER the if() check
function ToastContainer({ toasts }) {
  if (toasts.length === 0) return null;  // CRASH here
  toasts ??= [];
  // ...
}
```

Also common with optional chaining that wasn't applied:
```jsx
// BUG: proposals may be null
project.proposals.length > RECENT_PROPOSALS_PER_PROJECT
```

## Debugging method (minified bundle)

1. Find the crash location in the minified bundle: `aF` at line 248, col 1193
2. Search the SOURCE for the crash site using the minified name:
   ```bash
   grep -n "function aF\|const aF" src/some-file.js
   ```
3. The crash site usually has `.length` on a prop that may be undefined/null
4. Check if `??=` or `||=` default assignment exists on the same line or nearby
5. If `??=` appears AFTER a `.length` access, it's an execution-order issue

## Systematic search approach (don't assume initial location)

When debugging `.length` crashes, don't assume the crash is in the file you're looking at:
- Initial suspicion was App.jsx `project.proposals.length` — NOT the crash site
- Actual crash was in Toast.jsx (a completely different component)
- The error message only says "aF" which mapped to ToastContainer, not App.jsx
- Approach: grep ALL components for `.length` access on props that could be undefined:
  ```bash
  grep -rn "\.length" src/components/ src/pages/
  ```
- For each match, check if it's behind a guard OR if `??=` runs before the access
- Toast.jsx pattern: `if (toasts.length === 0)` crash, `toasts ??= []` after = execution order bug

## Fix patterns

**Always guard before .length:**
```jsx
// For arrays
if (!toasts || toasts.length === 0) return null;
if (!Array.isArray(toasts)) return null;

// For optional chaining on object chains
project.proposals?.length || 0
hasMore = (project.proposals?.length || 0) > RECENT_PROPOSALS_PER_PROJECT
```

**Key rule:** The guard must come BEFORE any `.length` access, not after.

## Verification
- Browser console should show 0 errors after fix
- Page should render correctly (even if showing token prompt, no white screen)
