# SSR and Frameworks

Web 3D often fails at framework boundaries, not inside the scene itself.

## Rule zero

If a framework renders on the server, never assume WebGL/browser APIs are safe at import time.

## What to check

- Is this Vite client-only?
- Is this Next.js with SSR?
- Is this Astro with islands?
- Is the component mounted only on the client?

## Framework posture

### Vite / client-only React

Usually straightforward.

Main concerns:

- component boundaries
- asset paths
- cleanup/remount behavior

### Next.js

Be careful with:

- browser-only imports
- client component boundaries
- dynamic imports for heavy 3D surfaces
- hydration mismatch risks

### Astro

Decide whether the 3D surface is:

- a plain client-side script
- a React island
- a deferred enhancement

Do not overcomplicate a simple page with a full app boundary if a small client-only surface will do.

## Safe mental model

The 3D surface is usually a client-only island. Treat it that way unless proven otherwise.

## Common failures

- touching `window` during SSR
- importing browser-only code too high in the tree
- assuming assets resolve the same way in dev and build
- mounting/unmounting canvases without cleanup
