# Strict Object Validation — `__proto__` Prototype Getter Trap

## When to use

When implementing strict plain-object validation that rejects non-plain objects (class instances, accessors) by inspecting the prototype. Any security-critical schema parser or receipt model that validates input as a plain object risks this trap.

## The bug

`Object.getOwnPropertyDescriptors(Object.prototype)` returns `__proto__` as a getter/setter pair:

```js
Object.getOwnPropertyDescriptor(Object.prototype, '__proto__')
// → { get: [Getter], set: [Setter], enumerable: false, configurable: true }
```

A naive accessor-detection loop that rejects **any** descriptor with a getter falsely rejects **every** plain object:

```js
// BROKEN — rejects ALL objects, not just class instances
const descriptors = Object.getOwnPropertyDescriptors(proto);
for (const key of Object.keys(descriptors)) {
  if (key !== 'constructor' && descriptors[key].get) { return false; }
}
```

All objects created via `{}` or `new Object()` inherit `Object.prototype`, whose `__proto__` getter triggers the rejection. Broke `createX402LahbApprovalReceipt` in CLOE_X402_LOT_E5_V1.

## The fix

Exclude `Object.prototype` from the accessor check:

```js
if (proto && proto !== Object.prototype) {
  const descriptors = Object.getOwnPropertyDescriptors(proto);
  for (const key of Object.keys(descriptors)) {
    if (key !== 'constructor' && descriptors[key].get) { return false; }
  }
}
```

Only prototypes beyond `Object.prototype` are security-relevant.

## Checklist

- [ ] Accessor detection explicitly skips `Object.prototype`
- [ ] `__proto__` getter is not treated as a security signal
- [ ] Tests include `{ key: 'value' }` to prove it passes validation
- [ ] Tests include `new SomeClass()` to prove it is still rejected
