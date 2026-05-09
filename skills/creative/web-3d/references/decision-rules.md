# Decision Rules

This skill only makes sense if it chooses the right path instead of flattening every request into "install Three.js."

## Primary split

### `p5js`

Choose `p5js` when the user wants:

- generative visuals
- browser art
- particles / noise / flow fields
- sketch-like experimentation
- shaders as visual art rather than product UI infrastructure

Do **not** force `three` just because something is "3D." If the scene is artistic, procedural, and mostly self-contained, `p5js` is usually the better Hermes-native fit.

### `three`

Choose vanilla `three` when:

- the result is a standalone scene or viewer
- the repo is framework-light or plain web
- render-loop ownership should stay explicit
- you want minimal abstraction between scene code and WebGL output

Good fits:

- small model viewer
- isolated hero section
- one-canvas microsite surface
- low-abstraction prototype

### `@react-three/fiber`

Choose `R3F` when:

- the repo already uses React
- the scene must coexist with app state
- the surface should be a real component
- interactivity is tied to normal UI/state flows
- composition with existing layout/components matters

Good fits:

- React landing page hero
- componentized product visualization
- interactive feature section
- model viewer integrated with filters or UI controls

### `blender-mcp`

Choose `blender-mcp` first when:

- authored assets are the real missing piece
- modeling, animation, material tuning, or camera work still needs to happen
- the web layer is mostly a presentation shell around the asset

Typical split:

- Blender creates the asset
- `web-3d` integrates and presents it on the web

### `touchdesigner-mcp`

Choose `touchdesigner-mcp` when:

- the work is about operator networks
- performance art / installation / live VJ logic is central
- the output is not naturally a product-UI surface

## Standalone vs repo implementation

### Standalone artifact

Prefer a standalone artifact when:

- the user asks for a demo, spike, or proof of concept
- there is no existing repo
- the result is meant to be shared visually before integration

### Repo implementation

Prefer repo implementation when:

- the request clearly targets an existing app
- the scene must live in production code
- design system, layout, or framework constraints are already real

Do not generate a detached HTML proof when the real problem is "put this in our app."

## Surface taxonomy

Classify the request before choosing code shape:

- **Hero surface**: needs composition, fallback, restrained motion
- **Viewer**: needs asset normalization, controls, camera discipline
- **Background surface**: needs low distraction and strong performance discipline
- **Feature section**: needs harmony with surrounding UI
- **Product scene**: needs state coordination and component reuse
- **Shader surface**: needs careful performance constraints and progressive enhancement

## Anti-patterns

- Choosing `three` for every 3D mention
- Treating `R3F` as mandatory inside every React repo
- Using a heavy model viewer where a 2D motion layer would suffice
- Building asset-authoring logic inside a web-surface skill
- Re-implementing `p5js` workflows under a new name
