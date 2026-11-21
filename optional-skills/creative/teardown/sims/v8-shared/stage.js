// The V8's stage: a pale grey studio, a soft shadow on an invisible floor, bright soft boxes for the gloss-black parts
// to catch, GTAO kept off the floor. frame(th, cam, ops) poses the engine for crank angle th and draws one frame.
import * as THREE from 'three';
import {EffectComposer} from 'three/addons/postprocessing/EffectComposer.js';
import {GTAOPass} from 'three/addons/postprocessing/GTAOPass.js';
import {OutputPass} from 'three/addons/postprocessing/OutputPass.js';
import {Pass} from 'three/addons/postprocessing/Pass.js';
import {RenderPass} from 'three/addons/postprocessing/RenderPass.js';
import {buildEngine} from './engine.js';

const V3 = THREE.Vector3;
export const BG = {top: '#f3f3f4', mid: '#e9eaec', low: '#e3e4e6'};

// look (every field optional, the film's stage when absent): ao, the GTAO blend (0.75); shadow, 'soft' (VSM, a wide
// blur) or 'hard' (PCF, a crisp edge); studio, 'full' (four soft boxes) or 'plain' (the top box alone, a flatter read
// on the gloss); engine, the options for buildEngine. The A/B film's version B is the one caller that sets them.
// lo: the teardown A/B's version B, cheaper all at once: no environment, no AO, no MSAA, basic hard-edged shadows and
// a brighter sky light so the flat-shaded parts still read. engine.caps asks for a stencil buffer for the section caps.
export function createV8Stage({canvas, look = {}}) {
  const lo = !!look.lo;
  const hard = look.shadow === 'hard';
  const renderer = new THREE.WebGLRenderer({canvas, antialias: false, preserveDrawingBuffer: true, powerPreference: 'high-performance'});
  renderer.toneMapping = THREE.NeutralToneMapping;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = lo ? THREE.BasicShadowMap : hard ? THREE.PCFShadowMap : THREE.VSMShadowMap;
  renderer.localClippingEnabled = true;
  const scene = new THREE.Scene();
  {
    // a screen-space backdrop: pale at the top, a touch darker toward the floor
    const c = document.createElement('canvas');
    c.width = 4;
    c.height = 512;
    const g = c.getContext('2d');
    const gr = g.createLinearGradient(0, 0, 0, 512);
    gr.addColorStop(0, BG.top);
    gr.addColorStop(0.55, BG.mid);
    gr.addColorStop(1, BG.low);
    g.fillStyle = gr;
    g.fillRect(0, 0, 4, 512);
    const tex = new THREE.CanvasTexture(c);
    tex.colorSpace = THREE.SRGBColorSpace;
    scene.background = tex;
  }
  const camera = new THREE.PerspectiveCamera(26, 16 / 9, 0.02, 60);
  const composer = new EffectComposer(renderer, new THREE.WebGLRenderTarget(4, 4, {type: THREE.HalfFloatType, samples: lo ? 0 : 8, stencilBuffer: !!look.engine?.caps}));
  composer.addPass(new RenderPass(scene, camera));
  const gtao = new GTAOPass(scene, camera, 4, 4);
  gtao.output = GTAOPass.OUTPUT.Default;
  gtao.blendIntensity = look.ao ?? 0.75;
  gtao.updateGtaoMaterial({radius: 0.05, distanceExponent: 1.4, thickness: 0.6, scale: 1.0, samples: 16, distanceFallOff: 1.0, screenSpaceRadius: false});
  gtao.updatePdMaterial({lumaPhi: 10, depthPhi: 2, normalPhi: 3, radius: 6, rings: 2, samples: 16});
  class Toggle extends Pass {
    constructor(fn) {
      super();
      this.needsSwap = false;
      this.fn = fn;
    }
    render() {
      this.fn();
    }
  }
  // the floor's shadow catcher and the section cap stay out of the AO pass
  const aoSkip = (on) => {
    shade.visible = on;
    if (eng.cap) eng.cap.visible = on;
  };
  composer.addPass(new Toggle(() => aoSkip(false)));
  composer.addPass(gtao);
  composer.addPass(new Toggle(() => aoSkip(true)));
  if (lo) gtao.enabled = false;
  composer.addPass(new OutputPass());

  // a bright studio for the gloss to read: a big top softbox, two side strips, a warm kicker
  if (!lo) {
    const env = new THREE.Scene();
    env.background = new THREE.Color('#d9dadd');
    const box = (w, h, pos, k) => {
      const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({color: new THREE.Color(1, 1, 1).multiplyScalar(k), side: THREE.DoubleSide}));
      m.position.copy(pos);
      m.lookAt(0, 0, 0);
      env.add(m);
    };
    box(12, 12, new V3(0, 14, 2), 2.6);
    if (look.studio !== 'plain') {
      box(3, 14, new V3(-12, 4, 6), 3.4);
      box(3, 14, new V3(12, 4, -4), 2.2);
      box(18, 2, new V3(0, 2, 14), 1.6);
    }
    const g = new THREE.Mesh(new THREE.CircleGeometry(40, 32), new THREE.MeshBasicMaterial({color: '#c9cacd', side: THREE.DoubleSide}));
    g.rotation.x = -Math.PI / 2;
    g.position.y = -2;
    env.add(g);
    const pm = new THREE.PMREMGenerator(renderer);
    scene.environment = pm.fromScene(env, 0.02).texture;
    scene.environmentIntensity = 1.0;
    pm.dispose();
  }
  const sun = new THREE.DirectionalLight('#ffffff', 2.2);
  sun.position.set(-1.2, 3.2, 1.6);
  sun.castShadow = true;
  Object.assign(sun.shadow.camera, {left: -0.8, right: 0.8, top: 0.8, bottom: -0.8, near: 0.5, far: 8});
  sun.shadow.mapSize.set(hard || lo ? 2048 : 4096, hard || lo ? 2048 : 4096);
  sun.shadow.radius = hard ? 1 : 14;
  sun.shadow.blurSamples = 24;
  sun.shadow.bias = -0.0004;
  sun.shadow.normalBias = 0.002;
  scene.add(sun, sun.target);
  scene.add(new THREE.HemisphereLight('#ffffff', '#cfd0d3', lo ? 2.2 : 0.7));

  // the floor: shadow only, on a plane just under the pan
  const FLOOR_Y = -0.14;
  const shade = new THREE.Mesh(
    new THREE.PlaneGeometry(20, 20),
    new THREE.ShadowMaterial({color: '#5d636c', opacity: 0.28, polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -8}),
  );
  shade.rotation.x = -Math.PI / 2;
  shade.position.y = FLOOR_Y;
  shade.receiveShadow = true;
  scene.add(shade);

  const eng = buildEngine(look.engine);
  scene.add(eng.root);
  if (eng.cap) scene.add(eng.cap);

  const size = {w: 1600, h: 900};
  const resize = (w, h, pr) => {
    size.w = w;
    size.h = h;
    renderer.setPixelRatio(pr);
    renderer.setSize(w, h, false);
    composer.setPixelRatio(pr);
    composer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  // cam: target (tx, ty, tz), azimuth from +z toward +x, elevation, distance (radians, metres), sx/sy screen shift
  const frame = (th, cam, ops = {}) => {
    eng.update(th, ops);
    const {tx, ty, tz, az, el, d} = cam;
    camera.position.set(tx + d * Math.cos(el) * Math.sin(az), ty + d * Math.sin(el), tz + d * Math.cos(el) * Math.cos(az));
    camera.lookAt(tx, ty, tz);
    camera.setViewOffset(size.w, size.h, -(cam.sx ?? 0), -(cam.sy ?? 0), size.w, size.h);
    camera.updateMatrixWorld();
    composer.render();
  };
  const pv = new V3();
  const project = (x, y, z) => {
    pv.set(x, y, z).project(camera);
    if (pv.z > 1) return null;
    return [(pv.x * 0.5 + 0.5) * size.w, (-pv.y * 0.5 + 0.5) * size.h];
  };
  const setQuality = ({samples, ao, shadow}) => {
    if (samples !== undefined) {
      for (const t of [composer.renderTarget1, composer.renderTarget2]) {
        t.samples = samples;
        t.dispose();
      }
    }
    if (ao !== undefined) gtao.enabled = ao;
    if (shadow !== undefined) {
      sun.shadow.mapSize.set(shadow, shadow);
      sun.shadow.map?.dispose();
      sun.shadow.map = null;
    }
  };
  const setCut = (mode, zWorld, normal) => {
    eng.setCut(mode, zWorld, normal);
    gtao.normalMaterial.clippingPlanes = mode === 'section' ? [eng.CLIP] : null;
    gtao.normalMaterial.needsUpdate = true;
  };
  return {renderer, scene, camera, eng, composer, gtao, sun, resize, frame, project, size, FLOOR_Y, setQuality, setCut};
}
