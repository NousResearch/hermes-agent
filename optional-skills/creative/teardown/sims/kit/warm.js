// Shader warm-up before a live page's first frame. three.js links a program the first time something draws with it,
// and the frame waits on every one in turn: 1-2 s on a phone or an integrated GPU, and a frozen page meanwhile. Here
// all of them start at once and the page awaits them without blocking (renderer.compileAsync, KHR_parallel_shader_compile):
// the scene's materials and the ones held by post passes (GTAO, bloom, ink, output), which compileAsync never meets
// on its own because they only reach the renderer inside composer.render().
//
// A warmed program only helps when its cache key matches the one the frame asks for, so everything here mirrors the
// frame: each set compiles against the target it draws into (tone mapping, colour space), pass quads have no normals
// like three's fullscreen triangle, override materials (GTAO's normals) see the scene's lights and shadows, and the
// two settings three only fixes on its first render (PCFSoftShadowMap, OutputPass defines) are fixed first.
import * as THREE from 'three';

// three's FullScreenQuad geometry: position and uv, no normal
const TRI = new THREE.BufferGeometry();
TRI.setAttribute('position', new THREE.Float32BufferAttribute([-1, 3, 0, -1, -1, 0, 3, -1, 0], 3));
TRI.setAttribute('uv', new THREE.Float32BufferAttribute([0, 2, 0, 0, 2, 0], 2));
// a stand-in for the scene's meshes under an override material
const MESH = new THREE.PlaneGeometry(1, 1);

// every material a pass holds directly or in an array (GTAOPass, UnrealBloomPass, ShaderPass, OutputPass)
const materialsOf = (pass) => Object.values(pass).flatMap((v) => (v?.isMaterial ? [v] : Array.isArray(v) ? v.filter((m) => m?.isMaterial) : []));
// drawn over the scene's own meshes instead of on a quad
const isOverride = (m) => m.isMeshNormalMaterial || m.isMeshDepthMaterial || m.isMeshDistanceMaterial;

/**
 * jobs: [{scene, target}] or [{materials, target, lights, override}]; target null is the canvas, lights a scene whose
 * lights apply, override true for materials drawn as scene.overrideMaterial whatever their type.
 */
export function warmUp(renderer, camera, jobs) {
  // three swaps the removed PCFSoftShadowMap for PCFShadowMap inside its first shadow render, after any warm-up
  if (renderer.shadowMap.type === THREE.PCFSoftShadowMap) renderer.shadowMap.type = THREE.PCFShadowMap;
  const prev = renderer.getRenderTarget();
  const waits = jobs.map(({scene, materials = [], target = null, lights = null, override = false}) => {
    let root = scene;
    if (!root) {
      root = new THREE.Scene();
      for (const m of materials) root.add(new THREE.Mesh(override || isOverride(m) ? MESH : TRI, m));
    }
    renderer.setRenderTarget(target);
    return renderer.compileAsync(root, camera, lights);
  });
  renderer.setRenderTarget(prev);
  return Promise.all(waits);
}

/** The scene and every enabled pass of an EffectComposer; the last pass draws to the canvas when the composer does. */
export function warmComposer(renderer, composer, scene, camera, jobs = []) {
  const target = composer.renderTarget1;
  const passes = composer.passes.filter((p) => p.enabled);
  const last = composer.renderToScreen ? passes[passes.length - 1] : null;
  // OutputPass picks its defines from the renderer on its first render: run that part now, with the draw left out
  for (const p of passes.filter((x) => x.isOutputPass)) {
    const quad = p._fsQuad;
    p._fsQuad = {render() {}};
    p.render(renderer, composer.renderTarget2, composer.renderTarget1);
    p._fsQuad = quad;
  }
  const mats = passes.filter((p) => p !== last).flatMap(materialsOf);
  return warmUp(renderer, camera, [
    {scene, target},
    {materials: mats.filter((m) => !isOverride(m)), target},
    {materials: mats.filter(isOverride), target, lights: scene},
    {materials: last ? materialsOf(last) : [], target: null},
    ...jobs,
  ]);
}
