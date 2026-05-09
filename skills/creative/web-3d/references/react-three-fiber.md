# React Three Fiber

Use `@react-three/fiber` when the 3D surface belongs inside a React app as a real component.

## Choose R3F when

- the repo already uses React
- the scene should compose with normal components
- state or props should drive the scene
- the canvas belongs to a route, section, or reusable component

## Do not choose R3F just because React exists

If the task is a tiny isolated spike, vanilla `three` may still be simpler.

## Minimum component structure

```tsx
import { Canvas } from "@react-three/fiber";

function SceneContents() {
  return (
    <mesh>
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial color="#ffffff" />
    </mesh>
  );
}

export function SceneHero() {
  return (
    <Canvas dpr={[1, 2]} camera={{ position: [0, 0, 4], fov: 45 }}>
      <ambientLight intensity={0.5} />
      <directionalLight position={[2, 2, 2]} intensity={1.2} />
      <SceneContents />
    </Canvas>
  );
}
```

## Integration rules

- Keep 3D concerns in a dedicated component boundary
- Do not spread scene setup across random UI files
- Keep canvas sizing explicit
- Treat `Canvas` as infrastructure, not a casual decoration

## State rules

- Use props/state for real UI coupling
- Do not push every animation detail into React state
- Avoid causing React rerenders at frame rate

Per-frame animation belongs inside the render loop, not normal app state.

## Suspense and loaders

If using asset loaders:

- isolate loading states
- do not block unrelated page UI unnecessarily
- provide a visible fallback if the surface is important

## Controls

Only add controls if:

- the user explicitly needs scene manipulation
- the viewer metaphor requires it

Do not put `OrbitControls` on a marketing hero unless it improves the brief.

## Cleanup posture

R3F handles some lifecycle cleanup, but you still need discipline:

- unmount scene-local listeners
- clean custom resources
- verify remount behavior
- ensure controls/helpers are not leaked

## Best fits

- product hero in React app
- model viewer with UI filters
- section-level interactive 3D
- reusable 3D UI component
