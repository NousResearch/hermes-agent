// The beest itself, built from nothing but tubes, red joint caps, crank throws, rubber feet and sails. One crankshaft
// runs the long axis (z); each station on it carries a crank throw and two mirrored Jansen legs (front and back),
// so the beast walks broadside along -x like the real ones on the beach. update(theta, t) poses every link from the
// solved linkage: instanced meshes, matrices recomposed per frame, nothing animated by hand.
import * as THREE from 'three';
import {FOOT_MIN, JOINTS, LINKS, PATH, solveLeg, solveMirror} from './jansen.js';

const V3 = THREE.Vector3;
export const S = 0.012; // metres per Jansen unit
export const PAD = 2.2; // foot pad half-height, units
export const CFG = {stations: 10, spacing: 26, half: 4.2};
// phase of each station's crank throw: a wave that runs down the spine
export const phaseOf = (s) => -s * ((Math.PI * 2) / CFG.stations);
// link layers inside a leg plane, toward (+1) or away from (-1) the station centre
const LAYER = {j: 1, k: 1, b: -1, c: -1, d: -1, e: 0, f: 0, g: -1, h: 0, i: 1};

// ------------------------------------------------------------ body motion without foot slip
// The body moves at minus the mean stance velocity of the feet on the ground, integrated over one crank turn.
const NT = 1440;
const legsAt = (th) => {
  const out = [];
  for (let s = 0; s < CFG.stations; s++) {
    const ps = th + phaseOf(s);
    out.push(solveLeg(ps).F, solveMirror(ps).F);
  }
  return out;
};
const MOTION = (() => {
  const X = new Float32Array(NT + 1);
  const Y = new Float32Array(NT + 1);
  const dth = (Math.PI * 2) / NT;
  let x = 0;
  let vPrev = 20;
  for (let n = 0; n <= NT; n++) {
    const th = n * dth;
    const a = legsAt(th - dth * 0.5);
    const b = legsAt(th + dth * 0.5);
    let ws = 0;
    let vs = 0;
    let low = Infinity;
    for (let q = 0; q < a.length; q++) {
      const lift = (a[q][1] + b[q][1]) / 2 - FOOT_MIN;
      low = Math.min(low, lift);
      const w = Math.max(0, 1 - lift / 1.2);
      ws += w;
      vs += (w * (b[q][0] - a[q][0])) / dth;
    }
    const v = ws > 0 ? vs / ws : vPrev;
    vPrev = v;
    X[n] = x;
    Y[n] = low;
    x -= v * dth;
  }
  return {X, Y, stride: X[NT]};
})();
/** Body offset along x (units) and the lowest foot's lift for crank angle theta (any number of turns). */
export function bodyAt(theta) {
  const turns = Math.floor(theta / (Math.PI * 2));
  const u = (theta / (Math.PI * 2) - turns) * NT;
  const n = Math.floor(u);
  const k = u - n;
  return {x: turns * MOTION.stride + MOTION.X[n] + (MOTION.X[n + 1] - MOTION.X[n]) * k, low: MOTION.Y[n] + (MOTION.Y[n + 1] - MOTION.Y[n]) * k};
}
export const STRIDE = MOTION.stride;
/** One leg's joints for station s, side 0 (front, foot toward -x) or 1 (the mirror), at crank angle theta. */
export const legAt = (s, side, theta) => (side ? solveMirror : solveLeg)(theta + phaseOf(s));
/** The leg plane's z, in units, for station s and side. */
export const legZ = (s, side) => -((CFG.stations - 1) * CFG.spacing) / 2 + s * CFG.spacing + (side ? CFG.half : -CFG.half);
/** Where a foot sits on its own path (0..1, the path's crank parameter) at crank angle theta. */
export const footParam = (s, side, theta) => {
  const ps = theta + phaseOf(s);
  const a = side ? Math.PI - ps : ps;
  return (((a / (Math.PI * 2)) % 1) + 1) % 1;
};

// the foot path as a lit trace: visible over the arc the foot swept in the last uLen of a turn
const TRACE_VERT = /* glsl */ `attribute float aS; varying float vS; void main(){ vS = aS; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;
const TRACE_FRAG = /* glsl */ `uniform vec3 uCol; uniform float uHead; uniform float uLen; uniform float uSign; uniform float uAlpha; varying float vS;
  void main(){
    float d = fract(uSign * (uHead - vS));
    float on = 1.0 - smoothstep(uLen - 0.004, uLen + 0.004, d);
    // the freshest part of the arc burns brighter
    float hot = 0.75 + 0.25 * (1.0 - d / max(uLen, 1e-3));
    if (on * uAlpha < 0.01) discard;
    gl_FragColor = vec4(uCol * hot, on * uAlpha);
  }`;

// ------------------------------------------------------------ the build
export function buildBeast(P) {
  const group = new THREE.Group();
  const N = CFG.stations;
  const z0 = -((N - 1) * CFG.spacing) / 2;
  const zs = (s) => z0 + s * CFG.spacing;

  const mats = {
    tube: new THREE.MeshPhysicalMaterial({color: P.tube, roughness: 0.34, metalness: 0, clearcoat: 0.35, clearcoatRoughness: 0.4}),
    joint: new THREE.MeshPhysicalMaterial({color: P.joint, roughness: 0.32, metalness: 0, clearcoat: 0.5, clearcoatRoughness: 0.3}),
    steel: new THREE.MeshStandardMaterial({color: P.steel, roughness: 0.32, metalness: 0.75}),
    foot: new THREE.MeshStandardMaterial({color: P.foot, roughness: 0.85, metalness: 0}),
    sail: new THREE.MeshStandardMaterial({color: P.sail, roughness: 0.7, metalness: 0, transparent: true, opacity: 0.86, side: THREE.DoubleSide, depthWrite: false}),
  };
  // the isolated leg keeps its own solid copies while everything else turns to glass
  mats.heroTube = mats.tube.clone();
  mats.heroJoint = mats.joint.clone();
  mats.heroFoot = mats.foot.clone();
  for (const k of ['tube', 'joint', 'steel', 'foot']) mats[k].transparent = true;
  const tubeCol = new THREE.Color(P.tube);
  const heroCol = new THREE.Color(P.hero || '#2a2622');
  const tubeGeo = new THREE.CylinderGeometry(1, 1, 1, 16, 1, true);
  const capGeo = new THREE.CylinderGeometry(1, 1, 1, 24, 1);
  // a joint cap: a short cylinder with a bevel, axis along z
  capGeo.rotateX(Math.PI / 2);
  const footGeo = new THREE.SphereGeometry(1, 24, 14);

  // counts
  const legs = N * 2;
  const nTubes = legs * LINKS.length + N * 6 + (N - 1) * 3 + N * 3; // legs, frame triangles, longerons, mast + boom + stay
  const nCaps = legs * (JOINTS.length + 1) + N * 3 + N * 2; // leg joints + the crank pin per leg, bearings, mast caps
  const nSteel = N * 3 + 1; // crank pins, two webs per station, the axle
  const tubes = new THREE.InstancedMesh(tubeGeo, mats.tube, nTubes);
  const caps = new THREE.InstancedMesh(capGeo, mats.joint, nCaps);
  const steel = new THREE.InstancedMesh(tubeGeo, mats.steel, nSteel);
  const feet = new THREE.InstancedMesh(footGeo, mats.foot, legs);
  const heroTubes = new THREE.InstancedMesh(tubeGeo, mats.heroTube, LINKS.length + 1);
  const heroCaps = new THREE.InstancedMesh(capGeo, mats.heroJoint, JOINTS.length + 2);
  const heroFeet = new THREE.InstancedMesh(footGeo, mats.heroFoot, 1);
  const main = [tubes, caps, steel, feet];
  const hero = [heroTubes, heroCaps, heroFeet];
  for (const m of [...main, ...hero]) {
    m.castShadow = true;
    m.receiveShadow = true;
    m.frustumCulled = false;
    m.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    group.add(m);
  }

  // foot traces, one closed loop per leg in its own plane, the crank parameter carried per vertex
  const traces = [];
  {
    const NP = PATH.length;
    const cum = new Float32Array(NP + 1);
    for (let n = 1; n <= NP; n++) {
      const a = PATH[n - 1];
      const b = PATH[n % NP];
      cum[n] = cum[n - 1] + Math.hypot(b[0] - a[0], b[1] - a[1]);
    }
    const SEG = 240;
    const RAD = 8;
    for (let st = 0; st < N; st++)
      for (let side = 0; side < 2; side++) {
        const zl = legZ(st, side);
        const pts = [];
        const par = [];
        for (let i = 0; i < SEG; i++) {
          // even steps along the loop, each tagged with the crank parameter it was reached at
          const target = (i / SEG) * cum[NP];
          let lo = 0;
          let hi = NP;
          while (hi - lo > 1) {
            const m = (lo + hi) >> 1;
            if (cum[m] <= target) lo = m;
            else hi = m;
          }
          const a = PATH[lo % NP];
          const b = PATH[(lo + 1) % NP];
          const k = (target - cum[lo]) / Math.max(1e-6, cum[lo + 1] - cum[lo]);
          const u = a[0] + (b[0] - a[0]) * k;
          const v = a[1] + (b[1] - a[1]) * k;
          pts.push(new V3((side ? -u : u) * S, (v - PAD * 0.25) * S, zl * S));
          par.push((lo + k) / NP);
        }
        const geo = new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts, true), SEG, 0.55 * S, RAD, true);
        const aS = new Float32Array(geo.attributes.position.count);
        for (let i = 0; i <= SEG; i++) for (let r = 0; r <= RAD; r++) aS[i * (RAD + 1) + r] = par[i % SEG];
        geo.setAttribute('aS', new THREE.BufferAttribute(aS, 1));
        const mat = new THREE.ShaderMaterial({
          vertexShader: TRACE_VERT,
          fragmentShader: TRACE_FRAG,
          uniforms: {uCol: {value: new THREE.Color(P.trace || '#e0662b')}, uHead: {value: 0}, uLen: {value: 1}, uSign: {value: side ? -1 : 1}, uAlpha: {value: 0}},
          transparent: true,
          depthWrite: false,
        });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.renderOrder = 3;
        mesh.frustumCulled = false;
        mesh.visible = false;
        mesh.userData = {st, side};
        group.add(mesh);
        traces.push(mesh);
      }
  }

  // sails: one cambered sheet per station, re-shaped every frame (wind fill + flutter)
  const SU = 14;
  const SV = 18;
  const sailTex = (() => {
    const c = document.createElement('canvas');
    c.width = c.height = 512;
    const g = c.getContext('2d');
    g.fillStyle = '#ffffff';
    g.fillRect(0, 0, 512, 512);
    g.strokeStyle = 'rgba(120,112,100,0.5)';
    g.lineWidth = 2;
    for (let k = 1; k < 6; k++) {
      g.beginPath();
      g.moveTo(0, (k * 512) / 6);
      g.lineTo(512, (k * 512) / 6);
      g.stroke();
    }
    g.strokeStyle = 'rgba(120,112,100,0.26)';
    for (let k = 1; k < 4; k++) {
      g.beginPath();
      g.moveTo((k * 512) / 4, 0);
      g.lineTo((k * 512) / 4, 512);
      g.stroke();
    }
    const t = new THREE.CanvasTexture(c);
    t.colorSpace = THREE.SRGBColorSpace;
    t.anisotropy = 8;
    return t;
  })();
  mats.sail.map = sailTex;
  const SAIL = {mast: 30, base: 30, height: 64, chord: 52, yaw: -0.5};
  const cy = Math.cos(SAIL.yaw);
  const sy = Math.sin(SAIL.yaw);
  // a point of the sail plane (u along the chord, zz off the plane) turned about the mast to meet the wind
  const yawU = (u, zz) => SAIL.mast + (u - SAIL.mast) * cy - zz * sy;
  const yawZ = (u, zz) => (u - SAIL.mast) * sy + zz * cy;
  const sails = [...Array(N)].map((_, s) => {
    const geo = new THREE.PlaneGeometry(1, 1, SU, SV);
    const mesh = new THREE.Mesh(geo, mats.sail);
    mesh.position.z = zs(s) * S;
    mesh.renderOrder = 2;
    mesh.frustumCulled = false;
    group.add(mesh);
    return mesh;
  });
  // the sail outline in (u, v) for chord fraction r (0 at the mast) and height fraction q
  const sailPt = (r, q) => {
    const chord = SAIL.chord * (1 - Math.pow(q, 1.35)) * (0.92 + 0.08 * Math.sin(q * Math.PI));
    return [SAIL.mast - r * chord, SAIL.base + q * SAIL.height - r * r * 10 * (1 - q)];
  };

  // ------------------------------------------------------------ per-frame pose
  const M4 = new THREE.Matrix4();
  const q = new THREE.Quaternion();
  const qZ = new THREE.Quaternion(); // caps are built along z already
  const Y = new V3(0, 1, 0);
  const dir = new V3();
  const mid = new V3();
  const sc = new V3();
  const pa = new V3();
  const pb = new V3();
  let it = 0;
  let ic = 0;
  let is = 0;
  let ht = 0;
  let hc = 0;
  const tube = (mesh, i, A, B, r) => {
    dir.subVectors(B, A);
    const L = dir.length();
    mid.addVectors(A, B).multiplyScalar(0.5);
    q.setFromUnitVectors(Y, dir.multiplyScalar(1 / L));
    sc.set(r * S, L, r * S);
    M4.compose(mid, q, sc);
    mesh.setMatrixAt(i, M4);
  };
  let toHero = false;
  const T = (u0, v0, z0_, u1, v1, z1, r = 1.15) => {
    pa.set(u0 * S, v0 * S, z0_ * S);
    pb.set(u1 * S, v1 * S, z1 * S);
    if (toHero) tube(heroTubes, ht++, pa, pb, r);
    else tube(tubes, it++, pa, pb, r);
  };
  const ST = (u0, v0, z0_, u1, v1, z1, r) => {
    pa.set(u0 * S, v0 * S, z0_ * S);
    pb.set(u1 * S, v1 * S, z1 * S);
    tube(steel, is++, pa, pb, r);
  };
  const CAP = (u, v, z, r = 2.1, len = 5.2) => {
    mid.set(u * S, v * S, z * S);
    sc.set(r * S, r * S, len * S);
    M4.compose(mid, qZ, sc);
    if (toHero) heroCaps.setMatrixAt(hc++, M4);
    else caps.setMatrixAt(ic++, M4);
  };
  const TOP = 20; // frame apex above the axle

  // ops: iso (0..1) isolates the hero leg {st, side}; trace (0..1) shows every foot loop, heroTrace (0..1) how much
  // of the hero's loop its foot has drawn so far
  const out = {heroJ: null, heroZ: 0};
  const update = (theta, t = 0, wind = 1, ops = {}) => {
    const iso = ops.iso ?? 0;
    const H = ops.hero ?? {st: 0, side: 0};
    it = 0;
    ic = 0;
    is = 0;
    ht = 0;
    hc = 0;
    // glass: the ghosted frame fades, the hero leg darkens to ink
    const ghost = 1 - 0.84 * iso;
    mats.tube.opacity = ghost;
    mats.joint.opacity = ghost;
    mats.foot.opacity = ghost;
    mats.steel.opacity = 1 - 0.6 * iso;
    mats.heroTube.color.copy(tubeCol).lerp(heroCol, iso);
    for (const m of hero) m.visible = iso > 0.001;
    const hw = CFG.half;
    // the crankshaft
    ST(0, 0, zs(0) - 14, 0, 0, zs(N - 1) + 14, 1.5);
    const feetM = [];
    for (let s = 0; s < N; s++) {
      const z = zs(s);
      const ps = theta + phaseOf(s);
      const legsS = [
        {J: solveLeg(ps), z: z - hw, w: 1},
        {J: solveMirror(ps), z: z + hw, w: -1},
      ];
      // crank throw: two webs and the pin both legs ride on
      const A = legsS[0].J.A;
      ST(0, 0, z - hw - 3.2, A[0], A[1], z - hw - 3.2, 1.6);
      ST(0, 0, z + hw + 3.2, A[0], A[1], z + hw + 3.2, 1.6);
      ST(A[0], A[1], z - hw - 3.2, A[0], A[1], z + hw + 3.2, 1.25);
      legsS.forEach(({J, z: zl, w}, side) => {
        toHero = iso > 0.001 && s === H.st && side === H.side;
        for (const [p, r, key] of LINKS) {
          const dz = LAYER[key] * w * 1.25;
          T(J[p][0], J[p][1], zl + dz, J[r][0], J[r][1], zl + dz);
        }
        for (const key of JOINTS) CAP(J[key][0], J[key][1], zl, key === 'P' ? 2.5 : 2.05, key === 'P' ? 6.2 : 5.0);
        CAP(J.A[0], J.A[1], zl, 1.9, 4.4);
        if (toHero) {
          // the crank m drawn as part of the leg, and the axle bearing it turns on
          T(0, 0, zl + w * 2.6, J.A[0], J.A[1], zl + w * 2.6, 1.3);
          CAP(0, 0, zl, 2.9, 4.6);
        }
        if (s === H.st && side === H.side) {
          out.heroJ = J;
          out.heroZ = zl;
        }
        feetM.push([J.F, zl, J.E, toHero]);
        toHero = false;
        // frame: axle bearing to the fixed pivot, pivot to the apex
        T(0, 0, zl, J.P[0], J.P[1], zl, 1.3);
        T(J.P[0], J.P[1], zl, 0, TOP, z, 1.3);
      });
      CAP(0, 0, z - hw, 2.9, 4.6);
      CAP(0, 0, z + hw, 2.9, 4.6);
      CAP(0, TOP, z, 2.4, 6);
      // mast, boom and a stay to the frame
      const top = sailPt(0, 1);
      const foot = sailPt(1, 0);
      T(SAIL.mast, TOP, z, top[0], top[1] + 4, z, 1.05);
      T(SAIL.mast, SAIL.base, z, yawU(foot[0], 0), foot[1], z + yawZ(foot[0], 0), 0.9);
      T(0, TOP, z, SAIL.mast, SAIL.base - 2, z, 1.0);
      CAP(SAIL.mast, SAIL.base, z, 1.8, 3.6);
      CAP(top[0], top[1] + 4, z, 1.6, 3.2);
    }
    // longerons: pivot rails front and back, the apex spine
    for (let s = 0; s < N - 1; s++) {
      const a = zs(s);
      const b = zs(s + 1);
      T(-38, -7.8, a - hw, -38, -7.8, b - hw, 1.2);
      T(38, -7.8, a + hw, 38, -7.8, b + hw, 1.2);
      T(0, TOP, a, 0, TOP, b, 1.3);
    }
    // feet: rubber pads under each lower link, flattened and tipped with the shin
    let fk = 0;
    for (let k = 0; k < feetM.length; k++) {
      const [F, zl, E, isHero] = feetM[k];
      dir.set(F[0] - E[0], F[1] - E[1], 0).normalize();
      q.setFromUnitVectors(Y, dir.multiplyScalar(-1));
      mid.set(F[0] * S, (F[1] - PAD * 0.25) * S, zl * S);
      sc.set(3.4 * S, PAD * S, 4.2 * S);
      M4.compose(mid, q, sc);
      if (isHero) heroFeet.setMatrixAt(0, M4);
      else feet.setMatrixAt(fk++, M4);
    }
    tubes.count = it;
    caps.count = ic;
    steel.count = is;
    feet.count = fk;
    heroTubes.count = ht;
    heroCaps.count = hc;
    for (const m of [...main, ...hero]) m.instanceMatrix.needsUpdate = true;

    // foot traces: the hero's loop draws itself, the rest fade in together
    const trace = ops.trace ?? 0;
    const heroTrace = ops.heroTrace ?? 0;
    for (const m of traces) {
      const {st, side} = m.userData;
      const isH = st === H.st && side === H.side;
      const u = m.material.uniforms;
      u.uHead.value = footParam(st, side, theta);
      u.uLen.value = isH && heroTrace > 0 ? Math.min(1, heroTrace) : 1;
      u.uAlpha.value = (isH ? Math.max(heroTrace > 0 ? 1 : 0, trace) : trace * (1 - 0.9 * iso)) * (ops.traceFade ?? 1);
      m.visible = u.uAlpha.value > 0.01 && u.uLen.value > 0.002;
    }

    // sails: filled by the wind, a flutter running aft along the leech
    for (let s = 0; s < N; s++) {
      const pos = sails[s].geometry.attributes.position;
      let n = 0;
      for (let iv = 0; iv <= SV; iv++) {
        const qv = 1 - iv / SV;
        for (let iu = 0; iu <= SU; iu++) {
          const r = iu / SU;
          const [u, v] = sailPt(r, qv);
          const fill = wind * 7.5 * Math.sin(Math.PI * Math.min(1, r * 1.15)) * (0.55 + 0.45 * qv);
          const flut = 0.9 * r * r * Math.sin(t * 9.0 - r * 7.0 + qv * 3.0 + s * 1.7) * (0.4 + 0.6 * wind);
          pos.setXYZ(n++, yawU(u, fill + flut) * S, v * S, yawZ(u, fill + flut) * S);
        }
      }
      pos.needsUpdate = true;
      sails[s].geometry.computeVertexNormals();
    }
  };
  return {group, update, mats, sails, traces, main, hero, out, length: ((N - 1) * CFG.spacing + 28) * S, zs};
}
