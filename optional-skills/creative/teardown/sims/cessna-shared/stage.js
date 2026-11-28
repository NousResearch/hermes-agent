// The landing gear's stage: a warm peach studio over a runway (centreline dashes, edge bars, threshold letters drawn
// in a shader and a canvas), soft shadows, GTAO kept off the ground. frame(state, cam) poses the gear, slides the
// runway under it by the distance rolled and draws one frame.
import * as THREE from 'three';
import {EffectComposer} from 'three/addons/postprocessing/EffectComposer.js';
import {GTAOPass} from 'three/addons/postprocessing/GTAOPass.js';
import {OutputPass} from 'three/addons/postprocessing/OutputPass.js';
import {Pass} from 'three/addons/postprocessing/Pass.js';
import {RenderPass} from 'three/addons/postprocessing/RenderPass.js';
import {buildGear, GROUND} from './gear.js';

const V3 = THREE.Vector3;
export const COL = {bg: '#f2e6d2', ground: '#efe0c6', mark: '#d6bf9a', shadow: '#7a5a3a'};

const GROUND_VERT = /* glsl */ `varying vec3 vW; void main(){ vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`;
const GROUND_FRAG = /* glsl */ `uniform vec3 uG; uniform vec3 uM; uniform vec3 uBg; uniform float uRoll; uniform vec3 uC; varying vec3 vW;
  float box(vec2 p, vec2 b){ vec2 d = abs(p) - b; return max(d.x, d.y); }
  void main(){
    // runway frame: x across, z along; the runway slides under the gear by uRoll
    float z = vW.z + uRoll;
    float x = vW.x - 1.4;
    float m = 0.0;
    // centreline dashes, 0.4 wide, 3 m on 2 m off
    float cz = mod(z, 5.0);
    m = max(m, step(abs(x), 0.2) * step(cz, 3.0));
    // edge bars, short blocks every 2.4 m
    float ez = mod(z, 2.4);
    m = max(m, step(abs(abs(x) - 6.0), 0.35) * step(ez, 1.2));
    // touchdown zone stripes
    float tz = mod(z + 1.0, 18.0);
    m = max(m, step(abs(abs(x) - 2.6), 0.9) * step(abs(mod(x + 0.4, 0.8) - 0.4), 0.22) * step(tz, 4.0));
    // a speckle so the ground reads as a surface
    float n = fract(sin(dot(floor(vW.xz * 40.0), vec2(12.9898, 78.233))) * 43758.5453);
    vec3 col = mix(uG, uM, m * 0.85) * (0.985 + 0.03 * n);
    float d = length(vW.xz - uC.xz);
    gl_FragColor = vec4(mix(col, uBg, smoothstep(5.0, 16.0, d)), 1.0);
  }`;

// the airframe as a ghost: a fresnel shell, bright at the silhouette, nearly clear face on
const GHOST_VERT = /* glsl */ `varying vec3 vN; varying vec3 vV; void main(){ vec4 mv = modelViewMatrix * vec4(position, 1.0); vN = normalize(normalMatrix * normal); vV = normalize(-mv.xyz); gl_Position = projectionMatrix * mv; }`;
const GHOST_FRAG = /* glsl */ `uniform vec3 uCol; uniform float uA; varying vec3 vN; varying vec3 vV;
  void main(){ float f = pow(1.0 - abs(dot(normalize(vN), normalize(vV))), 2.4); gl_FragColor = vec4(uCol, uA * (0.05 + 0.6 * f)); }`;
// the Skymaster's shape around the gear plate (the plate top is the belly, y = 0; the nose toward +z): a pod with the
// tractor engine forward and the pusher aft, a high wing, twin booms and the tail between them
function buildGhost(mat) {
  const g = new THREE.Group();
  const lathe = (prof, seg = 48) => new THREE.LatheGeometry(prof.map(([r, z]) => new THREE.Vector2(r, z)), seg);
  const add = (geo, f) => {
    const m = new THREE.Mesh(geo, mat);
    m.renderOrder = 6;
    f(m);
    g.add(m);
    return m;
  };
  // the pod along z (lathe about y, then turned onto z), a little flattened
  const pod = lathe([[0.001, -2.6], [0.18, -2.5], [0.42, -2.0], [0.56, -1.1], [0.6, 0.0], [0.58, 1.0], [0.48, 1.8], [0.3, 2.35], [0.12, 2.6], [0.001, 2.65]]);
  add(pod, (m) => {
    m.rotation.x = Math.PI / 2;
    m.scale.set(1, 1, 0.92);
    m.position.set(0, 0.56, 0.2);
  });
  // the high wing
  add(new THREE.BoxGeometry(9.2, 0.08, 0.95), (m) => m.position.set(0, 1.18, 0.15));
  // twin booms and the tail
  for (const x of [-1.25, 1.25]) {
    add(new THREE.CylinderGeometry(0.1, 0.07, 3.6, 20), (m) => {
      m.rotation.x = Math.PI / 2;
      m.position.set(x, 1.05, -1.95);
    });
    add(new THREE.BoxGeometry(0.06, 0.75, 0.55), (m) => m.position.set(x, 1.38, -3.6));
  }
  add(new THREE.BoxGeometry(2.6, 0.05, 0.42), (m) => m.position.set(0, 1.62, -3.65));
  // the two propellers' discs
  add(new THREE.CylinderGeometry(0.85, 0.85, 0.01, 48), (m) => {
    m.rotation.x = Math.PI / 2;
    m.position.set(0, 0.62, 2.9);
  });
  add(new THREE.CylinderGeometry(0.85, 0.85, 0.01, 48), (m) => {
    m.rotation.x = Math.PI / 2;
    m.position.set(0, 0.62, -2.5);
  });
  return g;
}

export function createCessnaStage({canvas}) {
  const renderer = new THREE.WebGLRenderer({canvas, antialias: false, preserveDrawingBuffer: true, powerPreference: 'high-performance'});
  renderer.toneMapping = THREE.NeutralToneMapping;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.VSMShadowMap;
  renderer.localClippingEnabled = true;
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(COL.bg);
  scene.fog = new THREE.Fog(COL.bg, 8, 22);
  const camera = new THREE.PerspectiveCamera(28, 16 / 9, 0.02, 80);
  const composer = new EffectComposer(renderer, new THREE.WebGLRenderTarget(4, 4, {type: THREE.HalfFloatType, samples: 8}));
  composer.addPass(new RenderPass(scene, camera));
  const gtao = new GTAOPass(scene, camera, 4, 4);
  gtao.output = GTAOPass.OUTPUT.Default;
  gtao.blendIntensity = 0.8;
  gtao.updateGtaoMaterial({radius: 0.06, distanceExponent: 1.4, thickness: 0.6, scale: 1.0, samples: 16, distanceFallOff: 1.0, screenSpaceRadius: false});
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
  const groundVis = (v) => () => {
    ground.visible = v;
    shade.visible = v;
    for (const o of ghosts) o.visible = v && o.userData.on;
  };
  const ghosts = [];
  composer.addPass(new Toggle(groundVis(false)));
  composer.addPass(gtao);
  composer.addPass(new Toggle(groundVis(true)));
  composer.addPass(new OutputPass());

  {
    const env = new THREE.Scene();
    env.background = new THREE.Color('#e9dcc6');
    const box = (w, h, pos, k) => {
      const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({color: new THREE.Color(1, 0.98, 0.95).multiplyScalar(k), side: THREE.DoubleSide}));
      m.position.copy(pos);
      m.lookAt(0, 0, 0);
      env.add(m);
    };
    box(12, 12, new V3(0, 14, 2), 2.4);
    box(3, 14, new V3(-12, 4, 6), 3.0);
    box(3, 14, new V3(12, 4, -4), 2.0);
    box(18, 2, new V3(0, 2, 14), 1.5);
    const g = new THREE.Mesh(new THREE.CircleGeometry(40, 32), new THREE.MeshBasicMaterial({color: '#d8c6a8', side: THREE.DoubleSide}));
    g.rotation.x = -Math.PI / 2;
    g.position.y = -2;
    env.add(g);
    const pm = new THREE.PMREMGenerator(renderer);
    scene.environment = pm.fromScene(env, 0.02).texture;
    scene.environmentIntensity = 0.95;
    pm.dispose();
  }
  const SUN_DIR = new V3(-0.35, 0.82, 0.45).normalize();
  const sun = new THREE.DirectionalLight('#fff6ea', 2.3);
  sun.castShadow = true;
  const EXT = 1.6;
  Object.assign(sun.shadow.camera, {left: -EXT, right: EXT, top: EXT, bottom: -EXT, near: 0.5, far: 20});
  sun.shadow.mapSize.set(4096, 4096);
  sun.shadow.radius = 10;
  sun.shadow.blurSamples = 20;
  sun.shadow.bias = -0.0003;
  sun.shadow.normalBias = 0.002;
  scene.add(sun, sun.target);
  scene.add(new THREE.HemisphereLight('#fffaf2', '#d9c6a6', 0.65));

  const groundMat = new THREE.ShaderMaterial({
    vertexShader: GROUND_VERT,
    fragmentShader: GROUND_FRAG,
    uniforms: {uG: {value: new THREE.Color(COL.ground)}, uM: {value: new THREE.Color(COL.mark)}, uBg: {value: new THREE.Color(COL.bg)}, uRoll: {value: 0}, uC: {value: new V3()}},
    toneMapped: false,
  });
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(200, 200), groundMat);
  ground.rotation.x = -Math.PI / 2;
  scene.add(ground);
  const shade = new THREE.Mesh(
    new THREE.PlaneGeometry(200, 200),
    new THREE.ShadowMaterial({color: COL.shadow, opacity: 0.3, polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -8}),
  );
  shade.rotation.x = -Math.PI / 2;
  shade.receiveShadow = true;
  scene.add(shade);

  const gear = buildGear();
  scene.add(gear.root);
  const ghostMat = new THREE.ShaderMaterial({vertexShader: GHOST_VERT, fragmentShader: GHOST_FRAG, uniforms: {uCol: {value: new THREE.Color('#ffffff')}, uA: {value: 0}}, transparent: true, depthWrite: false, side: THREE.DoubleSide});
  const ghost = buildGhost(ghostMat);
  ghost.userData.on = false;
  ghost.visible = false;
  gear.root.add(ghost);
  ghosts.push(ghost);

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
  const tmp = new V3();
  const right = new V3().crossVectors(new V3(0, 1, 0), SUN_DIR).normalize();
  const upL = new V3().crossVectors(SUN_DIR, right);
  let texel = (2 * EXT) / sun.shadow.mapSize.x;

  // st: {stroke, roll (wheel spin), height (m above the down-locked ride), dist (m rolled)}
  // cam: {tx, ty, tz, az, el, d (radians, metres), sx, sy}; the target is relative to the gear
  const frame = (st, cam) => {
    gear.update(st);
    ghostMat.uniforms.uA.value = st.fuselage ?? 0;
    ghost.userData.on = (st.fuselage ?? 0) > 0.01;
    ghost.visible = ghost.userData.on;
    const y = GROUND + (st.height ?? 0);
    gear.root.position.set(0, y, 0);
    groundMat.uniforms.uRoll.value = st.dist ?? 0;
    const tx = cam.tx;
    const ty = cam.ty + y;
    const tz = cam.tz;
    camera.position.set(tx + cam.d * Math.cos(cam.el) * Math.sin(cam.az), ty + cam.d * Math.sin(cam.el), tz + cam.d * Math.cos(cam.el) * Math.cos(cam.az));
    camera.lookAt(tx, ty, tz);
    camera.setViewOffset(size.w, size.h, -(cam.sx ?? 0), -(cam.sy ?? 0), size.w, size.h);
    camera.updateMatrixWorld();
    groundMat.uniforms.uC.value.set(tx, 0, tz);
    tmp.set(0, y * 0.5, 0);
    const rr = Math.round(tmp.dot(right) / texel) * texel;
    const uu = Math.round(tmp.dot(upL) / texel) * texel;
    sun.target.position.copy(right).multiplyScalar(rr).addScaledVector(upL, uu).addScaledVector(SUN_DIR, tmp.dot(SUN_DIR));
    sun.position.copy(sun.target.position).addScaledVector(SUN_DIR, 8);
    sun.target.updateMatrixWorld();
    composer.render();
  };
  const pv = new V3();
  const project = (x, y, z) => {
    pv.set(x, y, z).project(camera);
    if (pv.z > 1) return null;
    return [(pv.x * 0.5 + 0.5) * size.w, (-pv.y * 0.5 + 0.5) * size.h];
  };
  const setQuality = ({samples, ao, shadow}) => {
    if (samples !== undefined)
      for (const t of [composer.renderTarget1, composer.renderTarget2]) {
        t.samples = samples;
        t.dispose();
      }
    if (ao !== undefined) gtao.enabled = ao;
    if (shadow !== undefined) {
      sun.shadow.mapSize.set(shadow, shadow);
      sun.shadow.map?.dispose();
      sun.shadow.map = null;
      texel = (2 * EXT) / shadow;
    }
  };
  return {renderer, scene, camera, gear, composer, gtao, sun, resize, frame, project, size, ghosts, setQuality};
}
