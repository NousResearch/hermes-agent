# Freeze-While-Building Pattern

Alternative to recursive `deepFreeze` for constructing immutable governance objects.
Used successfully in CLOE_X402_LOT_D (commit 0ad78d9) after deepFreeze crashed on pre-frozen sub-objects.

## Problem

Lot A's `deepFreeze()` does:

```javascript
value[key] = deepFreeze(value[key]);  // TypeError: Cannot assign to read only property
```

This crashes when a property is already frozen (e.g. a binding object passed into a packet constructor, or a `SUBMISSION_STATE` enum value). The `seen` WeakSet is per-call, so re-freezing an already-frozen object at a different level in the tree still crashes.

## Solution: Freeze During Construction

Instead of building a mutable tree and recursively freezing it at the end, freeze each nested structure as you build it:

```javascript
export function createImmutableObject(params) {
  // Freeze nested arrays immediately
  const items = params.items ? Object.freeze([...params.items]) : Object.freeze([]);

  // Freeze nested objects immediately
  const metadata = params.metadata ? Object.freeze({
    key: params.metadata.key || null,
    value: params.metadata.value || null
  }) : null;

  // Freeze the top-level object
  const result = {
    id: params.id,
    items,
    metadata
  };
  return Object.freeze(result);
}
```

## Key Rules

1. **Freeze arrays at the point of creation:** `Object.freeze([...source])` not just `[...source]`
2. **Freeze nested objects inline:** wrap `{...}` in `Object.freeze({...})`
3. **Top-level `Object.freeze` is sufficient** if all children are already frozen
4. **Never call `deepFreeze` on a tree containing already-frozen sub-objects**

## When This Pattern Fails

If a deeply nested tree has branches you don't control during construction (e.g. a passed-in binding that's already frozen), use `safeDeepFreeze` V2 (reconstruct frozen parents into new mutable copies) instead. See `references/x402-readonly-transport-parser-pattern.md` pitfall 1.

## Evidence Refs Sorting

When encoding arrays into a binding ID, always sort a **copy** — never the original:

```javascript
// RIGHT — does not mutate caller's array
fields.push(encodeString((arr ? [...arr].sort() : []).join(',')));

// WRONG — mutates caller's array, crashes if frozen
fields.push(encodeString((arr || []).sort().join(',')));
```

## Caller Reference Isolation

When extracting fields into a snapshot, create new arrays and objects rather than storing references:

```javascript
// RIGHT — isolated copy
const snapshot = Object.freeze({
  items: Object.freeze([...source.items]),
  nested: source.nested ? Object.freeze({
    field: source.nested.field || null
  }) : null
});

// WRONG — shares reference with caller
const snapshot = {
  items: source.items,  // caller can mutate after construction
  nested: source.nested
};
```
