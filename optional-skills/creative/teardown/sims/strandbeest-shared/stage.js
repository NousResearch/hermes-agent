// The beest's stage: a warm studio over pale sand. PBR with VSM shadows and GTAO (the sand kept out of the AO pass),
// an unlit sand floor whose ripples fade into the backdrop, shadows laid over it. frame(theta, t, cam) poses the
// beast from the crank angle and draws one frame; the view size is whatever the caller's canvas box is.
import * as THREE from 'three';
import {EffectComposer} from 'three/addons/postprocessing/EffectComposer.js';
import {GTAOPass} from 'three/addons/postprocessing/GTAOPass.js';
import {OutputPass} from 'three/addons/postprocessing/OutputPass.js';
import {Pass} from 'three/addons/postprocessing/Pass.js';
import {RenderPass} from 'three/addons/postprocessing/RenderPass.js';
import {ShaderPass} from 'three/addons/postprocessing/ShaderPass.js';
import {bodyAt, buildBeast, PAD, S} from './beast.js';
import {FOOT_MIN} from './jansen.js';

const V3 = THREE.Vector3;
export const PALETTES = {
  sand: {
    bg: '#eee7dc',
    sand: '#ebe2d3',
    tube: '#f4f1eb',
    joint: '#d1322a',
    steel: '#646870',
    foot: '#8c8780',
    sail: '#fbf9f5',
    shadow: ['#5b4936', 0.34],
    env: {bg: '#e4ddd2', ground: '#d6ccbb', k: 0.8},
    hemi: ['#fbf7f1', '#dacdb8', 0.62],
  },
  // tan dune under a pale sky: the white frame stands off a darker ground
  dune: {
    bg: '#e9e2d6',
    sand: '#d2bc98',
    tube: '#f7f4ef',
    joint: '#d1322a',
    steel: '#4e525a',
    foot: '#6f6a64',
    sail: '#fbf9f5',
    shadow: ['#4a3220', 0.42],
    env: {bg: '#dde2e4', ground: '#c9b391', k: 0.85},
    hemi: ['#f4f7fa', '#c7ae86', 0.6],
    hero: '#2a2622',
    trace: '#e2672a',
    print: '#6a4f30',
    grain: '#86683f',
  },
  // a cool slate studio: the most contrast the light console can take
  slate: {
    bg: '#cfd5da',
    sand: '#bcc1c3',
    tube: '#f8f7f4',
    joint: '#d8352b',
    steel: '#3f444b',
    foot: '#5d6166',
    sail: '#fbfbf9',
    shadow: ['#1d2731', 0.4],
    env: {bg: '#cfd5da', ground: '#aeb4b8', k: 0.85},
    hemi: ['#f2f5f8', '#a9afb3', 0.62],
  },
};
const INK_FRAG = /* glsl */ `
  uniform sampler2D tDiffuse; uniform sampler2D tDepth; uniform sampler2D tNormal;
  uniform float uNear; uniform float uFar; uniform vec2 uPx; uniform vec3 uInk; uniform float uStrength;
  varying vec2 vUv;
  float lz(vec2 uv){ float d = texture2D(tDepth, uv).x; return uNear * uFar / (uFar - d * (uFar - uNear)); }
  vec3 nn(vec2 uv){ return normalize(texture2D(tNormal, uv).xyz * 2.0 - 1.0); }
  void main(){
    vec4 c = texture2D(tDiffuse, vUv);
    float z0 = lz(vUv);
    float zl = lz(vUv - vec2(uPx.x, 0.0)), zr = lz(vUv + vec2(uPx.x, 0.0));
    float zd = lz(vUv - vec2(0.0, uPx.y)), zu = lz(vUv + vec2(0.0, uPx.y));
    float lap = (abs(zl + zr - 2.0 * z0) + abs(zu + zd - 2.0 * z0)) / max(z0, 1e-3);
    float eD = smoothstep(0.004, 0.016, lap);
    vec3 n0 = nn(vUv);
    float eN = max(max(1.0 - dot(n0, nn(vUv - vec2(uPx.x, 0.0))), 1.0 - dot(n0, nn(vUv + vec2(uPx.x, 0.0)))),
                   max(1.0 - dot(n0, nn(vUv - vec2(0.0, uPx.y))), 1.0 - dot(n0, nn(vUv + vec2(0.0, uPx.y)))));
    float e = max(eD, smoothstep(0.3, 0.6, eN));
    c.rgb = mix(c.rgb, uInk, e * uStrength);
    gl_FragColor = c;
  }`;

const SAND_VERT = /* glsl */ `varying vec3 vW; void main(){ vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`;
const SAND_FRAG = /* glsl */ `uniform vec3 uSand; uniform vec3 uBg; uniform vec3 uC; varying vec3 vW;
  float h(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
  float n(vec2 p){ vec2 i = floor(p); vec2 f = fract(p); f = f * f * (3.0 - 2.0 * f);
    return mix(mix(h(i), h(i + vec2(1, 0)), f.x), mix(h(i + vec2(0, 1)), h(i + vec2(1, 1)), f.x), f.y); }
  void main(){
    vec2 p = vW.xz;
    // wind ripples: long crests bent by a slow noise, light on the windward face
    float bend = n(p * 0.35) * 2.6 + n(p * 1.1) * 0.6;
    float ph = (p.y * 2.9 + p.x * 0.55 + bend) * 6.2831;
    float rip = sin(ph) * 0.5 + 0.5;
    rip = pow(rip, 3.0) - 0.25 * sin(ph + 1.2);
    float grain = n(p * 70.0) * 0.6 + n(p * 190.0) * 0.4;
    float pat = n(p * 0.22);
    vec3 col = uSand * (0.985 + 0.055 * rip - 0.05 * grain + 0.04 * (pat - 0.5));
    // fine fade near the camera's look point keeps the texture under the beast, the horizon melts into the backdrop
    float d = length(p - uC.xz);
    float fade = smoothstep(5.0, 17.0, d);
    gl_FragColor = vec4(mix(col, uBg, fade), 1.0);
  }`;

// footprints: pressed ovals that appear at touchdown and fade over ~20 s; one instance per planted step
const PRINT_VERT = /* glsl */ `attribute vec4 aP; uniform float uT; varying vec2 vUv; varying float vA;
  void main(){
    vec2 c = position.xy;
    float ca = cos(aP.w), sa = sin(aP.w);
    vec2 q = vec2(c.x * 0.085, c.y * 0.105);
    vec2 r = vec2(q.x * ca - q.y * sa, q.x * sa + q.y * ca);
    vUv = c;
    float age = uT - aP.z;
    vA = step(0.0, age) * smoothstep(0.0, 0.15, age) * (1.0 - smoothstep(14.0, 24.0, age));
    gl_Position = projectionMatrix * viewMatrix * vec4(aP.x + r.x, 0.0012, aP.y + r.y, 1.0);
  }`;
const PRINT_FRAG = /* glsl */ `uniform vec3 uCol; varying vec2 vUv; varying float vA;
  void main(){
    float d = length(vUv * 2.0);
    // a pressed hollow with a raised, lit rim pushed out around it
    float hollow = 1.0 - smoothstep(0.6, 1.0, d);
    float rim = smoothstep(0.75, 0.95, d) * (1.0 - smoothstep(1.0, 1.25, d));
    float a = (hollow * 0.5 + rim * 0.16) * vA;
    if (a < 0.003) discard;
    gl_FragColor = vec4(mix(uCol, vec3(1.0), rim * 0.7), a);
  }`;
// blown sand: grains streaming low over the dune with the wind, wrapped in a box around the camera's look point
const GRAIN_VERT = /* glsl */ `attribute vec3 aSeed; uniform float uT; uniform float uFlow; uniform vec3 uC; uniform vec2 uDir; uniform float uPr; uniform float uAmt;
  varying float vA;
  void main(){
    vec3 box = vec3(12.0, 1.0, 9.0);
    vec2 p = aSeed.xz * box.xz + uDir * uFlow * (0.65 + 0.7 * aSeed.y);
    vec2 rel = mod(p - uC.xz + box.xz * 0.5, box.xz);
    vec2 edge = min(rel, box.xz - rel);
    vec3 w = vec3(rel.x + uC.x - box.x * 0.5, 0.012 + aSeed.y * aSeed.y * aSeed.y * 0.55 + 0.03 * sin(uT * 2.3 + aSeed.x * 40.0), rel.y + uC.z - box.z * 0.5);
    vec4 mv = viewMatrix * vec4(w, 1.0);
    gl_Position = projectionMatrix * mv;
    gl_PointSize = uPr * (2.2 + 3.2 * aSeed.z) * clamp(7.0 / -mv.z, 0.5, 2.5);
    vA = uAmt * (0.25 + 0.6 * aSeed.z) * smoothstep(0.0, 1.5, min(edge.x, edge.y));
  }`;
const GRAIN_FRAG = /* glsl */ `uniform vec3 uCol; varying float vA;
  void main(){ vec2 c = gl_PointCoord - 0.5; float a = (1.0 - smoothstep(0.15, 0.5, length(c))) * vA; if (a < 0.01) discard; gl_FragColor = vec4(uCol, a); }`;

export function createBeestStage({canvas, palette = 'sand', ink = 0}) {
  const P = PALETTES[palette] || PALETTES.sand;
  const renderer = new THREE.WebGLRenderer({canvas, antialias: false, preserveDrawingBuffer: true, powerPreference: 'high-performance'});
  renderer.toneMapping = THREE.NeutralToneMapping;
  renderer.toneMappingExposure = 1.0;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.VSMShadowMap;
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(P.bg);
  if (P.sky) {
    // a screen-space sky: the top colour down to the horizon, the horizon colour below it
    const c = document.createElement('canvas');
    c.width = 4;
    c.height = 512;
    const g = c.getContext('2d');
    const gr = g.createLinearGradient(0, 0, 0, 512);
    gr.addColorStop(0, P.sky[0]);
    gr.addColorStop(0.55, P.sky[1]);
    gr.addColorStop(1, P.sky[1]);
    g.fillStyle = gr;
    g.fillRect(0, 0, 4, 512);
    const tex = new THREE.CanvasTexture(c);
    tex.colorSpace = THREE.SRGBColorSpace;
    scene.background = tex;
  }
  scene.fog = new THREE.Fog(P.bg, 9, 24);
  const camera = new THREE.PerspectiveCamera(22, 16 / 9, 0.05, 200);
  const rt = new THREE.WebGLRenderTarget(4, 4, {type: THREE.HalfFloatType, samples: 8});
  const composer = new EffectComposer(renderer, rt);
  composer.addPass(new RenderPass(scene, camera));
  const gtao = new GTAOPass(scene, camera, 4, 4);
  gtao.output = GTAOPass.OUTPUT.Default;
  gtao.blendIntensity = 0.85;
  gtao.updateGtaoMaterial({radius: 0.12, distanceExponent: 1.4, thickness: 1.0, scale: 1.0, samples: 16, distanceFallOff: 1.0, screenSpaceRadius: false});
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
  // what the AO pass (and the ink drawn from it) must not see: the ground, the sheets, the overlays, and the ghosted
  // frame while a leg is isolated
  let isoNow = 0;
  const groundAO = (v) => () => {
    floor.visible = v;
    shade.visible = v;
    prints.visible = v;
    grains.visible = v;
    for (const s of beast.sails) s.visible = v;
    for (const m of beast.traces) m.visible = v && m.userData.on;
    if (isoNow > 0.5) for (const m of beast.main) m.visible = v;
  };
  composer.addPass(new Toggle(groundAO(false)));
  // ink only (a weak GPU): GTAO renders just its normal and depth buffer for the outline, skips the AO and the
  // denoise, and leaves the picture where it is
  let aoOn = true;
  const gtaoRender = gtao.render.bind(gtao);
  gtao.render = (renderer, writeBuffer, readBuffer, ...rest) => {
    if (aoOn) return gtaoRender(renderer, writeBuffer, readBuffer, ...rest);
    gtao._overrideVisibility();
    gtao._renderOverride(renderer, gtao.normalMaterial, gtao.normalRenderTarget, 0x7777ff, 1.0);
    gtao._restoreVisibility();
  };
  composer.addPass(gtao);
  composer.addPass(new Toggle(groundAO(true)));
  // ink: depth creases and normal breaks from the AO pass's g-buffer, which holds the beast alone
  const inkPass = new ShaderPass(
    new THREE.ShaderMaterial({
      uniforms: {
        tDiffuse: {value: null},
        tDepth: {value: gtao.depthTexture},
        tNormal: {value: gtao.normalTexture},
        uNear: {value: camera.near},
        uFar: {value: camera.far},
        uPx: {value: new THREE.Vector2(1 / 1920, 1 / 1080)},
        uInk: {value: new THREE.Color().setRGB(0.07, 0.06, 0.055, THREE.LinearSRGBColorSpace)},
        uStrength: {value: ink},
      },
      vertexShader: /* glsl */ `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
      fragmentShader: INK_FRAG,
    }),
  );
  inkPass.enabled = ink > 0;
  composer.addPass(inkPass);
  composer.addPass(new OutputPass());

  // studio environment: warm soft boxes over a sand-coloured ground, prefiltered
  {
    const env = new THREE.Scene();
    env.background = new THREE.Color(P.env.bg);
    const box = (w, h, pos, k, tint = [1, 0.985, 0.95]) => {
      const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({color: new THREE.Color(...tint).multiplyScalar(k), side: THREE.DoubleSide}));
      m.position.copy(pos);
      m.lookAt(0, 0, 0);
      env.add(m);
    };
    box(16, 7, new V3(-12, 11, 9), 3.0);
    box(10, 10, new V3(11, 14, -7), 2.2);
    box(20, 3, new V3(2, 5, 17), 1.7);
    const g = new THREE.Mesh(new THREE.CircleGeometry(40, 32), new THREE.MeshBasicMaterial({color: P.env.ground, side: THREE.DoubleSide}));
    g.rotation.x = -Math.PI / 2;
    g.position.y = -1.5;
    env.add(g);
    const pm = new THREE.PMREMGenerator(renderer);
    scene.environment = pm.fromScene(env, 0.03).texture;
    scene.environmentIntensity = P.env.k;
    pm.dispose();
  }
  const SUN_DIR = new V3(-0.55, 0.66, 0.52).normalize();
  const sun = new THREE.DirectionalLight('#fff7ec', 2.4);
  sun.castShadow = true;
  const EXT = 3.0;
  Object.assign(sun.shadow.camera, {left: -EXT, right: EXT, top: EXT, bottom: -EXT, near: 0.5, far: 24});
  sun.shadow.mapSize.set(4096, 4096);
  sun.shadow.radius = 7;
  sun.shadow.blurSamples = 20;
  sun.shadow.bias = -0.0003;
  sun.shadow.normalBias = 0.003;
  scene.add(sun, sun.target);
  scene.add(new THREE.HemisphereLight(...P.hemi));

  // sand
  const sandMat = new THREE.ShaderMaterial({
    vertexShader: SAND_VERT,
    fragmentShader: SAND_FRAG,
    uniforms: {uSand: {value: new THREE.Color(P.sand)}, uBg: {value: new THREE.Color(P.bg)}, uC: {value: new V3()}},
    toneMapped: false,
  });
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(400, 400), sandMat);
  floor.rotation.x = -Math.PI / 2;
  scene.add(floor);
  const shade = new THREE.Mesh(new THREE.PlaneGeometry(400, 400), new THREE.ShadowMaterial({color: P.shadow[0], opacity: P.shadow[1], fog: false, polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -8}));
  shade.rotation.x = -Math.PI / 2;
  shade.position.y = 0.0004;
  shade.receiveShadow = true;
  scene.add(shade);

  const beast = buildBeast(P);
  scene.add(beast.group);

  // footprints, set by the caller as [x, z, born, angle] per step
  const printGeo = new THREE.InstancedBufferGeometry();
  {
    const base = new THREE.PlaneGeometry(1, 1);
    printGeo.index = base.index;
    printGeo.setAttribute('position', base.attributes.position);
    printGeo.setAttribute('aP', new THREE.InstancedBufferAttribute(new Float32Array(4), 4));
    printGeo.instanceCount = 0;
  }
  const printMat = new THREE.ShaderMaterial({
    vertexShader: PRINT_VERT,
    fragmentShader: PRINT_FRAG,
    uniforms: {uT: {value: 0}, uCol: {value: new THREE.Color(P.print || '#7a5c3a')}},
    transparent: true,
    depthWrite: false,
    // the quad is laid on the ground by a mirrored mapping, so its winding flips
    side: THREE.DoubleSide,
    polygonOffset: true,
    polygonOffsetFactor: -4,
    polygonOffsetUnits: -8,
  });
  const prints = new THREE.Mesh(printGeo, printMat);
  prints.frustumCulled = false;
  prints.renderOrder = 1;
  scene.add(prints);
  const setPrints = (list) => {
    const a = new Float32Array(Math.max(4, list.length * 4));
    list.forEach((p, i) => a.set(p, i * 4));
    printGeo.setAttribute('aP', new THREE.InstancedBufferAttribute(a, 4));
    printGeo.instanceCount = list.length;
  };
  // the live page lays prints as the feet land: a ring of cap slots, unborn slots far in the future
  let ring = null;
  const usePrintRing = (cap) => {
    const a = new Float32Array(cap * 4);
    for (let i = 0; i < cap; i++) a[i * 4 + 2] = 1e9;
    const attr = new THREE.InstancedBufferAttribute(a, 4);
    attr.setUsage(THREE.DynamicDrawUsage);
    printGeo.setAttribute('aP', attr);
    printGeo.instanceCount = cap;
    ring = {attr, cap, i: 0};
  };
  const addPrint = (x, z, born, angle) => {
    ring.attr.array.set([x, z, born, angle], ring.i * 4);
    ring.i = (ring.i + 1) % ring.cap;
    ring.attr.needsUpdate = true;
  };

  // blown sand
  const NG = 9000;
  const grainGeo = new THREE.BufferGeometry();
  {
    const seed = new Float32Array(NG * 3);
    let r = 91;
    const rnd = () => (r = (r * 16807) % 2147483647) / 2147483647;
    for (let i = 0; i < NG * 3; i++) seed[i] = rnd();
    grainGeo.setAttribute('aSeed', new THREE.BufferAttribute(seed, 3));
    grainGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(NG * 3), 3));
  }
  const grainMat = new THREE.ShaderMaterial({
    vertexShader: GRAIN_VERT,
    fragmentShader: GRAIN_FRAG,
    uniforms: {uT: {value: 0}, uFlow: {value: 0}, uC: {value: new V3()}, uDir: {value: new THREE.Vector2(-0.86, 0.5)}, uPr: {value: 1}, uAmt: {value: 0}, uCol: {value: new THREE.Color(P.grain || '#8f7350')}},
    transparent: true,
    depthWrite: false,
  });
  const grains = new THREE.Points(grainGeo, grainMat);
  grains.frustumCulled = false;
  grains.renderOrder = 4;
  scene.add(grains);

  const size = {w: 1600, h: 900};
  const resize = (w, h, pr) => {
    size.w = w;
    size.h = h;
    grainMat.uniforms.uPr.value = pr;
    renderer.setPixelRatio(pr);
    renderer.setSize(w, h, false);
    composer.setPixelRatio(pr);
    composer.setSize(w, h);
    inkPass.material.uniforms.uPx.value.set(1 / (w * pr), 1 / (h * pr));
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };

  const tmp = new V3();
  const right = new V3().crossVectors(new V3(0, 1, 0), SUN_DIR).normalize();
  const upL = new V3().crossVectors(SUN_DIR, right);
  let texel = (2 * EXT) / sun.shadow.mapSize.x;

  /** The body's world position for crank angle theta: x along the walk, y of the axle. */
  const bodyPos = (theta) => {
    const b = bodyAt(theta);
    return {x: b.x * S, y: -(FOOT_MIN + b.low - PAD * 1.25) * S};
  };

  // ops: wind (m/s), flow (metres of wind run so far), iso, hero, trace, heroTrace; cam.sx / cam.sy shift the picture
  // on screen (CSS px) so the subject sits in the space the console leaves free
  const frame = (theta, t, cam, ops = {}) => {
    const b = bodyPos(theta);
    beast.group.position.set(b.x, b.y, 0);
    const wind = ops.wind ?? 6;
    isoNow = ops.iso ?? 0;
    beast.update(theta, t, 0.45 + 0.6 * Math.min(1.2, wind / 10), ops);
    for (const m of beast.traces) m.userData.on = m.visible;
    printMat.uniforms.uT.value = t;
    grainMat.uniforms.uT.value = t;
    grainMat.uniforms.uFlow.value = ops.flow ?? 0;
    grainMat.uniforms.uAmt.value = 0.25 + 0.75 * Math.min(1, Math.max(0, (wind - 5) / 7));

    const {tx, ty, tz, az, el, d} = cam;
    camera.position.set(tx + d * Math.cos(el) * Math.cos(az), ty + d * Math.sin(el), tz - d * Math.cos(el) * Math.sin(az));
    camera.lookAt(tx, ty, tz);
    camera.setViewOffset(size.w, size.h, -(cam.sx ?? 0), -(cam.sy ?? 0), size.w, size.h);
    camera.updateMatrixWorld();
    grainMat.uniforms.uC.value.set(tx, 0, tz);
    sandMat.uniforms.uC.value.set(tx, 0, tz);
    // the sand and its shadow catcher follow the camera in whole metres; the ripples are drawn in world space
    floor.position.x = shade.position.x = Math.round(tx);
    floor.position.z = shade.position.z = Math.round(tz);
    // the shadow box follows the beast, snapped to texels so it never shimmers
    tmp.set(b.x, 0.6, 0);
    const rr = Math.round(tmp.dot(right) / texel) * texel;
    const uu = Math.round(tmp.dot(upL) / texel) * texel;
    sun.target.position.copy(right).multiplyScalar(rr).addScaledVector(upL, uu).addScaledVector(SUN_DIR, tmp.dot(SUN_DIR));
    sun.position.copy(sun.target.position).addScaledVector(SUN_DIR, 10);
    sun.target.updateMatrixWorld();
    composer.render();
    return b;
  };
  // a world point to CSS px on the canvas, null behind the camera
  const pv = new V3();
  const project = (x, y, z) => {
    pv.set(x, y, z).project(camera);
    if (pv.z > 1) return null;
    return [(pv.x * 0.5 + 0.5) * size.w, (-pv.y * 0.5 + 0.5) * size.h];
  };
  // a point of the beast given in Jansen units of its own frame (u, v, z) to world
  const beastWorld = (u, v, z) => [beast.group.position.x + u * S, beast.group.position.y + v * S, z * S];
  // the live page's quality knobs: MSAA samples, the shadow map and its blur, the AO samples, how many grains blow
  const setQuality = ({samples, shadow, blur, radius, ao, pcf, grains: ng}) => {
    if (samples !== undefined) {
      for (const t of [composer.renderTarget1, composer.renderTarget2]) {
        t.samples = samples;
        t.dispose();
      }
    }
    if (shadow !== undefined) {
      sun.shadow.mapSize.set(shadow, shadow);
      sun.shadow.map?.dispose();
      sun.shadow.map = null;
      texel = (2 * EXT) / shadow;
    }
    if (blur !== undefined) sun.shadow.blurSamples = blur;
    if (radius !== undefined) sun.shadow.radius = radius;
    if (ao !== undefined) {
      aoOn = ao > 0;
      gtao.needsSwap = aoOn;
      if (aoOn) {
        gtao.updateGtaoMaterial({samples: ao});
        gtao.updatePdMaterial({samples: Math.max(4, ao)});
      }
    }
    if (pcf !== undefined) {
      const type = pcf ? THREE.PCFSoftShadowMap : THREE.VSMShadowMap;
      if (renderer.shadowMap.type !== type) {
        renderer.shadowMap.type = type;
        sun.shadow.map?.dispose();
        sun.shadow.map = null;
        scene.traverse((o) => {
          if (o.material) for (const m of [].concat(o.material)) m.needsUpdate = true;
        });
      }
    }
    if (ng !== undefined) grainGeo.setDrawRange(0, ng);
  };
  return {renderer, scene, camera, beast, composer, gtao, inkPass, sun, resize, frame, bodyPos, size, project, beastWorld, setPrints, usePrintRing, addPrint, setQuality};
}
