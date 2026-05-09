# Vanilla Three.js

Use vanilla `three` when the result is a standalone web surface or when explicit render-loop ownership is more important than framework composition.

## Minimum structure

Build this first:

1. scene
2. camera
3. renderer
4. one visible object
5. resize handling
6. deterministic cleanup

Do not start with loaders, postprocessing, or controls.

## Minimum file shape

```js
import * as THREE from "three";

const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
camera.position.set(0, 0, 4);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setSize(width, height);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

const geometry = new THREE.BoxGeometry(1, 1, 1);
const material = new THREE.MeshStandardMaterial({ color: 0xffffff });
const mesh = new THREE.Mesh(geometry, material);
scene.add(mesh);

function frame() {
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}

frame();
```

## Required concerns

### Resize

Always update:

- renderer size
- camera aspect
- camera projection matrix

### DPR discipline

Never leave DPR uncapped on user-facing surfaces.

Default:

```js
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
```

Drop to `1.5` or `1` if perf is marginal.

### Cleanup

When the scene is destroyed:

- cancel animation loop
- remove listeners
- dispose geometry
- dispose material
- dispose textures
- dispose controls if present
- dispose renderer if the surface unmounts permanently

### Controls

Add controls only if the task actually needs them.

Orbiting everything by default is lazy and often harms composition.

## Good use cases

- model viewer
- isolated hero
- microsite scene
- small shader-backed section

## Bad use cases

- React component systems needing deep state integration
- complex UI/state choreography where `R3F` is cleaner
