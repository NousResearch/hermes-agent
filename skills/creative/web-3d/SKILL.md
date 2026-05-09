---
name: web-3d
description: "Interactive 3D surfaces for the web: choose between vanilla Three.js, React Three Fiber, or adjacent Hermes skills, then implement and verify the result inside a real web stack."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [webgl, 3d, threejs, react-three-fiber, gltf, shaders, frontend, interactive, ui]
    related_skills: [p5js, blender-mcp, touchdesigner-mcp, claude-design]
    requires_toolsets: [terminal]
---

# Web 3D

Build and debug interactive 3D surfaces for the web using the repo's real stack.

This skill is for one specific problem:

- **shipping or prototyping interactive 3D inside a web product**

That includes:

- hero sections
- product scenes
- model viewers
- shader-backed app surfaces
- React-owned 3D components
- standalone demos that are meant to become web UI later

This skill is **not** a general 3D skill and **not** a replacement for other creative skills.

It is a workflow layer:

- it classifies the request
- chooses the right implementation path
- composes with adjacent skills when needed
- implements and verifies the result using Hermes' existing file, terminal, and browser surfaces

## Relationship to other creative skills

This skill complements existing creative skills instead of replacing them:

- `p5js` -> browser-native generative art, sketch-style WebGL, shaders, and creative coding
- `blender-mcp` -> upstream 3D asset and scene authoring before web integration
- `touchdesigner-mcp` -> alternative real-time media workflow for operator-network and installation-style visuals
- `claude-design` / `sketch` -> design exploration and non-3D artifact work
- `popular-web-designs` -> visual vocabulary and design reference material

Use `web-3d` when the central problem is **3D implementation inside a web stack**, not asset authoring, generative art, or general design exploration.

## When to use

Use when the user wants:

- interactive 3D inside a website or web app
- a 3D landing-page hero
- a product visualization surface
- a model viewer for `.glb` / `.gltf`
- pointer-reactive, scroll-reactive, or shader-based 3D UI
- a React component that owns a 3D canvas
- a standalone browser demo that will likely become app UI later

## When NOT to use

- **Pure generative art, browser sketch, or canvas-first experimentation** -> use `p5js`
- **3D scene authoring, modeling, rigging, animation, or material work in a DCC** -> use `blender-mcp`
- **Real-time installation / VJ / operator-graph work** -> use `touchdesigner-mcp`
- **Static mockups, non-3D prototypes, or general HTML design exploration** -> use `claude-design` or `sketch`
- **A diagram, not a scene** -> use `architecture-diagram`, `concept-diagrams`, or `excalidraw`

## Core idea

This skill does four things:

1. Classify the request
2. Choose the right implementation path
3. Implement using the repo's actual stack
4. Verify visual correctness, console health, cleanup, and performance

Do not reduce this to "install three and start coding."

The highest-value behavior is choosing the right implementation path:

- `p5js` vs `three`
- `three` vs `@react-three/fiber`
- web implementation vs asset-authoring handoff
- standalone demo vs real repo integration

## Decision rules

See `references/decision-rules.md` for full detail.

### Choose `react-three-fiber` when

- the repo already uses React
- the 3D scene must live inside component state
- the surface needs to interact with UI controls, route state, or application data
- the result should be reusable as a component instead of a page-local script

### Choose vanilla `three` when

- the project is standalone or framework-light
- a single isolated canvas is enough
- low-level control matters more than React composition
- the result should be a minimal viewer / hero / demo with explicit render-loop ownership

### Choose `p5js` instead when

- the request is generative-art-first, scene-graph-second
- the main value is motion language, particles, noise, or visual experimentation
- exact scene hierarchy, loaders, or reusable 3D UI components are not important

### Choose `blender-mcp` before web integration when

- the scene itself still needs to be authored
- the user needs meshes, cameras, materials, or animation built first
- the web task is mainly "display an authored 3D asset" instead of "design the 3D system"

Treat `blender-mcp` as an upstream authoring handoff, not as an alternative web implementation path.

### Choose `touchdesigner-mcp` instead when

- the work is operator-network-based
- the output is installation/performance-oriented
- the environment is closer to real-time media systems than product UI

Treat `touchdesigner-mcp` as an adjacent medium, not as the normal owner for product-web 3D.

## Delivery modes

This skill should support four concrete output modes:

1. **Standalone viewer**
   - one local HTML file or a tiny Vite page
   - good for demos, viewers, and spikes
2. **Repo-integrated React component**
   - `SceneHero.tsx`, `ModelViewer.tsx`, `InteractiveCanvas.tsx`
   - good for real product work
3. **Framework page/section integration**
   - Vite / Next / Astro / similar app surface
   - a section, route, or hero inside an existing product
4. **Asset-backed model surface**
   - `.glb` / `.gltf` viewer with lights, camera, controls, fallback, and performance constraints

Choose the smallest mode that fits the request.
Do not widen scope just because a neighboring creative skill exists.

## Workflow

```
INSPECT -> CLASSIFY -> CHOOSE PATH -> BUILD MINIMUM -> ADD INTERACTION -> VERIFY -> TIGHTEN
```

### Step 1 - Inspect the repo and runtime

Before writing code, determine:

- framework: React / Vite / Next / Astro / plain HTML
- package manager: npm / pnpm / yarn / bun
- whether SSR is involved
- whether 3D must coexist with an existing design system
- where the surface lives: page, section, component, modal, hero, viewer
- whether assets already exist (`.glb`, `.gltf`, textures, HDRIs)

If the task is in an existing repo, build in that repo's real stack. Do not force a standalone HTML artifact if the real target is a component or page.

### Step 2 - Classify the surface

Decide what kind of thing this is:

- hero section
- model viewer
- product scene
- background surface
- shader-first ambient layer
- data-driven interactive scene
- marketing/demo artifact

The classification should change the implementation:

- a hero section needs strong composition and graceful fallback
- a viewer needs camera, controls, and asset hygiene
- a product scene needs responsiveness and UI coexistence
- a shader surface needs performance discipline and progressive enhancement

### Step 3 - Choose the implementation path

Use:

- `references/react-three-fiber.md`
- `references/vanilla-three.md`
- `references/ssr-and-frameworks.md`

Do not mix paradigms casually. Pick one render ownership model and commit:

- R3F component tree
- vanilla Three.js loop
- adjacent skill handoff when this is not the right owner

### Step 4 - Build the minimum render path first

Before polish:

- get a canvas rendering
- establish camera + scene + renderer
- confirm resize behavior
- confirm render loop ownership
- confirm unmount/cleanup path

No postprocessing, fancy materials, or elaborate controls until the minimum scene is stable.

### Step 5 - Add the real product layer

Then add what the brief actually needs:

- lights
- controls
- assets
- interactions
- shader passes
- UI coordination
- scroll or pointer response

Do not add everything just because the library allows it.

### Step 6 - Verify aggressively

Use the browser toolset when available.

Always check:

- visual output
- browser console
- resize behavior
- asset loading
- SSR/hydration safety
- cleanup/dispose path
- performance on the intended surface

See `references/verification-checklist.md`.

### Step 7 - Tighten and simplify

Before declaring success:

- remove unnecessary abstractions
- cap pixel density when needed
- dispose textures/materials/geometries correctly
- reduce asset weight if the scene is too heavy
- ensure there is a non-WebGL or low-power fallback when the surface is user-facing

## Default implementation standards

- Prefer **existing repo stack** over standalone HTML when the task lives in an app
- Prefer **component boundaries** over global script blobs in React repos
- Prefer **one clear surface** over multiple competing canvases
- Prefer **small scene graphs** over ornamental complexity
- Prefer **real verification** over source-only confidence

## Verification checklist

Before returning work:

- [ ] Scene renders without console errors
- [ ] Canvas resizes correctly
- [ ] No obvious SSR or hydration crash path
- [ ] Controls and interactions work as intended
- [ ] Assets load from correct paths
- [ ] Cleanup/dispose path exists
- [ ] Performance is acceptable on the target surface
- [ ] A fallback exists if the surface is public-facing

Load `references/verification-checklist.md` for the detailed list.

## Common failure modes

- importing browser-only code into SSR path
- creating multiple render loops accidentally
- leaking geometry, material, texture, or controls on remount
- rendering at unbounded DPR on laptops/phones
- shipping a 3D scene that visually fights the page around it
- using 3D where 2D motion or CSS would have solved the problem better
- forcing `three` into a repo where `R3F` is the better integration layer
- forcing `R3F` into a tiny standalone demo where plain `three` is simpler
- absorbing adjacent skills instead of routing to them

## File map

```
SKILL.md
references/
  decision-rules.md
  vanilla-three.md
  react-three-fiber.md
  gltf-assets.md
  ssr-and-frameworks.md
  performance-and-debugging.md
  verification-checklist.md
templates/
  standalone-viewer.html
  react-scene-component.tsx
scripts/
  setup.sh
  serve.sh
```

## Usage pattern

1. Read the repo and classify the request
2. Load the most relevant reference file(s), not all of them
3. Build the smallest correct path
4. Verify with browser + console
5. Only then add finish and polish
