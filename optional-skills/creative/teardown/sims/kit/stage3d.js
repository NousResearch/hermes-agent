import * as THREE from 'three';
import {EffectComposer} from 'three/addons/postprocessing/EffectComposer.js';
import {OutputPass} from 'three/addons/postprocessing/OutputPass.js';
import {RenderPass} from 'three/addons/postprocessing/RenderPass.js';
import {UnrealBloomPass} from 'three/addons/postprocessing/UnrealBloomPass.js';

/**
 * Renderer + scene + camera + bloom. The camera's projection centre is shifted
 * to the middle of the HUD viewport (between header and bottom panel), so the
 * subject is centred in the visible area, not behind the terminal.
 */
export function createStage(canvas, hud, opts = {}) {
  const {
    bg = '#09090a',
    fov = 38,
    bloom = {strength: 0.85, radius: 0.55, threshold: 0.62},
    capture = false,
    exposure = 1,
  } = opts;
  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: false,
    preserveDrawingBuffer: true,
    powerPreference: 'high-performance',
  });
  renderer.setClearColor(new THREE.Color(bg), 1);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = exposure;
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(bg);
  const camera = new THREE.PerspectiveCamera(fov, 1, 0.5, 6000);

  const target = new THREE.WebGLRenderTarget(4, 4, {type: THREE.HalfFloatType, samples: 4});
  const composer = new EffectComposer(renderer, target);
  composer.addPass(new RenderPass(scene, camera));
  const bloomPass = new UnrealBloomPass(new THREE.Vector2(4, 4), bloom.strength, bloom.radius, bloom.threshold);
  composer.addPass(bloomPass);
  composer.addPass(new OutputPass());

  const size = {w: 1, h: 1, centre: 0.5};
  // Director crop: render a sub-rectangle of the base view (crisp punch-ins).
  const crop = {z: 1, fx: 0, fy: 0, x0: 0, y0: 0};
  const applyView = () => {
    const {w, h, centre} = size;
    const offY = h / 2 - centre;
    if (crop.z <= 1.0001) {
      crop.x0 = 0;
      crop.y0 = 0;
      camera.setViewOffset(w, h, 0, offY, w, h);
    } else {
      const cw = w / crop.z;
      const ch = h / crop.z;
      crop.x0 = Math.min(Math.max(crop.fx - cw / 2, 0), w - cw);
      crop.y0 = Math.min(Math.max(crop.fy - ch / 2, 0), h - ch);
      camera.setViewOffset(w, h, crop.x0, offY + crop.y0, cw, ch);
    }
    camera.updateProjectionMatrix();
  };
  // opts.pr fixes the pixel ratio (a capture that wants the 3D at the output resolution, a page that adapts it)
  let prFixed = opts.pr;
  const resize = () => {
    const w = canvas.clientWidth || innerWidth;
    const h = canvas.clientHeight || innerHeight;
    const pr = prFixed ?? (capture ? 1 : Math.min(devicePixelRatio || 1, 2));
    size.w = w;
    size.h = h;
    renderer.setPixelRatio(pr);
    renderer.setSize(w, h, false);
    composer.setPixelRatio(pr);
    composer.setSize(w, h);
    const vp = hud.viewport();
    size.centre = (vp.top + (h - vp.bottom)) / 2;
    camera.aspect = w / h;
    applyView();
  };
  /** Zoom z around base-screen point (fx, fy); returns {z, x0, y0} for mapping back. */
  const setCrop = (z, fx, fy) => {
    crop.z = Math.max(1, z);
    crop.fx = fx;
    crop.fy = fy;
    applyView();
    return {z: crop.z, x0: crop.x0, y0: crop.y0};
  };

  const v = new THREE.Vector3();
  /** World point → CSS px on screen, plus whether it's in front of the camera. */
  const project = (p) => {
    v.set(p[0] ?? p.x, p[1] ?? p.y, p[2] ?? p.z).project(camera);
    return {x: (v.x * 0.5 + 0.5) * size.w, y: (-v.y * 0.5 + 0.5) * size.h, visible: v.z < 1};
  };

  const st = {
    THREE,
    renderer,
    scene,
    camera,
    composer,
    bloomPass,
    size,
    resize,
    setPixelRatio: (v) => {
      prFixed = v;
      resize();
    },
    project,
    render: () => composer.render(),
    setCrop,
    hudViewport: () => hud.viewport(),
  };
  window.__STAGE = st;
  return st;
}
