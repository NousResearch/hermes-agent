// The Cessna 337 main landing gear, built from code in metres: the white main plate with its channel rails, the
// hydraulic actuator on top, the rod-end connector and the brass gear rack, the pinion on a transverse rotary output
// shaft, a clevis (universal) joint at each end, the side tubes that spin about their own axis, the orange spring
// legs and the wheels. One degree of freedom: stroke 0..1 moves the rack 201 mm, turns the pinion 180°, and the side
// tubes carry the legs through the same 180° about their own axes. The right gear is built once; the left one is its
// mirror (scale.x = -1).
import * as THREE from 'three';
import {RoundedBoxGeometry} from 'three/addons/geometries/RoundedBoxGeometry.js';

const V3 = THREE.Vector3;
export const G = {
  pinionR: 0.064, // 32 teeth, 4 mm module
  stroke: 0.201, // rack travel for a full 180°
  wheelR: 0.205,
  wheelW: 0.135,
  shaftY: -0.1365,
  jointX: 0.22,
  tubeLen: 0.3,
  legLen: 0.5,
};
const norm = (v) => v.clone().normalize();
// the side tube's axis (outboard, down, aft) and the spring leg's direction when down and locked: half a turn about
// that axis carries the wheel from a wide stance on the runway to a pocket beside the plate, level with it, aft
export const AXIS = norm(new V3(1, -0.6, -1));
export const LEG0 = norm(new V3(1, -1, 0.8));
export const J = new V3(G.jointX, G.shaftY, 0);
export const E0 = J.clone().addScaledVector(AXIS, G.tubeLen);
export const TIP0 = E0.clone().addScaledVector(LEG0, G.legLen);
/** where the wheel centre is (right gear, assembly frame) at gear angle th (radians) */
export const tipAt = (th) => TIP0.clone().sub(J).applyAxisAngle(AXIS, th).add(J);
/** the assembly's height above the ground when down and locked: the tyre touches the runway */
export const GROUND = -(TIP0.y - G.wheelR);

// ------------------------------------------------------------ geometry
function gearGeo(teeth, rp, width, holeR = 0) {
  const ra = rp + 0.004;
  const rr = rp - 0.005;
  const sh = new THREE.Shape();
  for (let i = 0; i < teeth; i++) {
    const a0 = (i / teeth) * Math.PI * 2;
    const da = (Math.PI * 2) / teeth;
    const pts = [[rr, a0], [rr, a0 + da * 0.18], [ra, a0 + da * 0.34], [ra, a0 + da * 0.6], [rr, a0 + da * 0.76]];
    pts.forEach(([r, a], k) => (i === 0 && k === 0 ? sh.moveTo(r * Math.cos(a), r * Math.sin(a)) : sh.lineTo(r * Math.cos(a), r * Math.sin(a))));
  }
  sh.closePath();
  if (holeR) {
    const h = new THREE.Path();
    h.absarc(0, 0, holeR, 0, Math.PI * 2, true);
    sh.holes.push(h);
  }
  const g = new THREE.ExtrudeGeometry(sh, {depth: width, bevelEnabled: true, bevelThickness: 0.002, bevelSize: 0.0015, bevelSegments: 1, curveSegments: 4});
  g.translate(0, 0, -width / 2);
  return g;
}
function rackGeo(len, w, h, pitch) {
  // a bar along z with teeth on its underside
  const sh = new THREE.Shape();
  sh.moveTo(-len / 2, h / 2);
  sh.lineTo(len / 2, h / 2);
  sh.lineTo(len / 2, -h / 2);
  const n = Math.floor(len / pitch);
  for (let i = n - 1; i >= 0; i--) {
    const z0 = -len / 2 + i * pitch;
    sh.lineTo(z0 + pitch * 0.78, -h / 2);
    sh.lineTo(z0 + pitch * 0.62, -h / 2 - 0.008);
    sh.lineTo(z0 + pitch * 0.38, -h / 2 - 0.008);
    sh.lineTo(z0 + pitch * 0.22, -h / 2);
  }
  sh.lineTo(-len / 2, -h / 2);
  sh.closePath();
  const g = new THREE.ExtrudeGeometry(sh, {depth: w, bevelEnabled: true, bevelThickness: 0.002, bevelSize: 0.002, bevelSegments: 1});
  g.translate(0, 0, -w / 2);
  // shape x -> world z, shape y -> world y, depth -> world x
  g.rotateY(-Math.PI / 2);
  return g;
}
// the plate's plan form (x lateral, z fore-aft), extruded down by its thickness
function plateGeo() {
  const sh = new THREE.Shape();
  const P = [
    [-0.12, 0.62], [0.12, 0.62], [0.14, 0.32], [0.42, 0.22], [0.46, 0.12], [0.44, -0.06], [0.16, -0.16],
    [0.14, -0.78], [-0.14, -0.78], [-0.16, -0.16], [-0.44, -0.06], [-0.46, 0.12], [-0.42, 0.22], [-0.14, 0.32],
  ];
  P.forEach(([x, z], i) => (i ? sh.lineTo(x, z) : sh.moveTo(x, z)));
  sh.closePath();
  const g = new THREE.ExtrudeGeometry(sh, {depth: 0.026, bevelEnabled: true, bevelThickness: 0.004, bevelSize: 0.006, bevelSegments: 2});
  g.rotateX(Math.PI / 2);
  return g;
}
// a channel rail along z with lightening holes, its profile an upright plate
function railGeo(len, h) {
  const sh = new THREE.Shape();
  sh.moveTo(-len / 2, 0);
  sh.lineTo(len / 2, 0);
  sh.lineTo(len / 2, h * 0.7);
  sh.lineTo(len / 2 - 0.04, h);
  sh.lineTo(-len / 2 + 0.04, h);
  sh.lineTo(-len / 2, h * 0.7);
  sh.closePath();
  for (let i = 0; i < 6; i++) {
    const hole = new THREE.Path();
    hole.absarc(-len / 2 + 0.14 + i * ((len - 0.28) / 5), h * 0.48, h * 0.22, 0, Math.PI * 2, true);
    sh.holes.push(hole);
  }
  const g = new THREE.ExtrudeGeometry(sh, {depth: 0.012, bevelEnabled: true, bevelThickness: 0.002, bevelSize: 0.002, bevelSegments: 1, curveSegments: 18});
  g.translate(0, 0, -0.006);
  g.rotateY(-Math.PI / 2);
  return g;
}
// the spring leg: a flat orange bar swept from the tube end to the axle, outboard first and then down
function legGeo() {
  const tip = TIP0.clone().sub(E0);
  const out = new V3(1, 0, 0);
  const pts = [new V3(0, 0, 0), out.clone().multiplyScalar(0.12).add(new V3(0, -0.02, 0.02)), tip.clone().multiplyScalar(0.55).add(out.clone().multiplyScalar(0.1)), tip.clone().add(new V3(-0.02, 0.0, 0))];
  const curve = new THREE.CatmullRomCurve3(pts);
  const sh = new THREE.Shape();
  sh.moveTo(-0.048, -0.014);
  sh.lineTo(0.048, -0.014);
  sh.lineTo(0.048, 0.014);
  sh.lineTo(-0.048, 0.014);
  sh.closePath();
  return new THREE.ExtrudeGeometry(sh, {steps: 60, extrudePath: curve, bevelEnabled: false});
}
// a tyre: a rounded lathe profile; the sidewall lettering is a ring with a canvas texture
function tyreGeo() {
  const R = G.wheelR;
  const W = G.wheelW;
  const pts = [];
  const r0 = R * 0.55;
  for (let i = 0; i <= 24; i++) {
    const a = -Math.PI / 2 + (i / 24) * Math.PI;
    pts.push(new THREE.Vector2(R - (W / 2) * 0.55 + (W / 2) * 0.55 * Math.cos(a), (W / 2) * Math.sin(a)));
  }
  pts.unshift(new THREE.Vector2(r0, -W / 2 * 0.9));
  pts.push(new THREE.Vector2(r0, W / 2 * 0.9));
  const g = new THREE.LatheGeometry(pts, 64);
  g.rotateZ(Math.PI / 2);
  return g;
}
function letterTex() {
  const c = document.createElement('canvas');
  c.width = c.height = 1024;
  const g = c.getContext('2d');
  g.clearRect(0, 0, 1024, 1024);
  g.fillStyle = '#f4f1ea';
  g.font = '700 74px Inter, Arial, sans-serif';
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  const word = 'AEROGRIP';
  for (const [a0, w] of [[-Math.PI / 2, word], [Math.PI / 2, '6.00-6 6PLY']]) {
    const span = 0.11;
    [...w].forEach((ch, i) => {
      const a = a0 + (i - (w.length - 1) / 2) * span;
      g.save();
      g.translate(512 + Math.cos(a) * 395, 512 + Math.sin(a) * 395);
      g.rotate(a + Math.PI / 2);
      g.fillText(ch, 0, 0);
      g.restore();
    });
  }
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  return t;
}

export function buildGear() {
  const M = {
    white: new THREE.MeshPhysicalMaterial({color: '#f1efea', roughness: 0.35, metalness: 0, clearcoat: 0.35, clearcoatRoughness: 0.35}),
    orange: new THREE.MeshPhysicalMaterial({color: '#e8572a', roughness: 0.32, metalness: 0.05, clearcoat: 0.5, clearcoatRoughness: 0.25}),
    black: new THREE.MeshPhysicalMaterial({color: '#17181b', roughness: 0.35, metalness: 0.2, clearcoat: 0.5}),
    rubber: new THREE.MeshStandardMaterial({color: '#1b1c1e', roughness: 0.82, metalness: 0}),
    steel: new THREE.MeshStandardMaterial({color: '#c9ccd1', roughness: 0.22, metalness: 1}),
    brass: new THREE.MeshStandardMaterial({color: '#b28a3e', roughness: 0.3, metalness: 0.9}),
    hub: new THREE.MeshStandardMaterial({color: '#b8bcc2', roughness: 0.3, metalness: 0.8}),
    plate: null,
    body: null,
    letters: new THREE.MeshStandardMaterial({map: letterTex(), transparent: true, roughness: 0.6, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2}),
  };
  // the plate (and its rails and blocks) and the actuator body get their own white so x-ray and section reach only them
  M.plate = M.white.clone();
  M.body = M.white.clone();
  const all = [];
  const add = (parent, geo, mat) => {
    const m = new THREE.Mesh(geo, mat);
    m.castShadow = true;
    m.receiveShadow = true;
    parent.add(m);
    all.push(m);
    return m;
  };
  const root = new THREE.Group();

  // ------------------------------------------------------------ plate and rails
  const plate = new THREE.Group();
  root.add(plate);
  add(plate, plateGeo(), M.plate);
  const rail = railGeo(1.1, 0.07);
  for (const x of [-0.08, 0.08]) add(plate, rail, M.plate).position.set(x, 0.004, -0.12);
  // pillow blocks under the plate carrying the output shaft
  for (const x of [-0.11, 0.11]) {
    const b = add(plate, new RoundedBoxGeometry(0.05, 0.12, 0.07, 3, 0.012), M.plate);
    b.position.set(x, -0.026 - 0.06 + 0.006, 0);
  }

  // ------------------------------------------------------------ actuator, rod end, rack
  const act = new THREE.Group();
  root.add(act);
  const AY = 0.06;
  const body = add(act, new THREE.CylinderGeometry(0.042, 0.042, 0.46, 32), M.body);
  body.rotation.x = Math.PI / 2;
  body.position.set(0, AY, -0.39);
  for (const z of [-0.6, -0.17]) {
    const band = add(act, new THREE.CylinderGeometry(0.046, 0.046, 0.03, 32), M.black);
    band.rotation.x = Math.PI / 2;
    band.position.set(0, AY, z);
  }
  const clevis = add(act, new RoundedBoxGeometry(0.06, 0.07, 0.07, 3, 0.012), M.brass);
  clevis.position.set(0, AY - 0.005, -0.66);
  const pin = add(act, new THREE.CylinderGeometry(0.012, 0.012, 0.1, 12), M.black);
  pin.rotation.z = Math.PI / 2;
  pin.position.set(0, AY - 0.005, -0.66);
  const nipple = add(act, new THREE.CylinderGeometry(0.012, 0.012, 0.03, 12), M.orange);
  nipple.position.set(0, AY + 0.05, -0.55);
  const rod = add(act, new THREE.CylinderGeometry(0.016, 0.016, 1, 20), M.steel);
  rod.rotation.x = Math.PI / 2;
  const rodEnd = new THREE.Group();
  root.add(rodEnd);
  add(rodEnd, new RoundedBoxGeometry(0.07, 0.06, 0.08, 3, 0.012), M.orange).position.y = AY;
  const web = add(rodEnd, new RoundedBoxGeometry(0.03, 0.1, 0.05, 2, 0.008), M.orange);
  web.position.y = AY - 0.06;
  for (const s of [-1, 1]) {
    const b = add(rodEnd, new THREE.CylinderGeometry(0.008, 0.008, 0.01, 10), M.black);
    b.rotation.z = Math.PI / 2;
    b.position.set(s * 0.036, AY, 0.015);
  }
  const RACK_LEN = 0.34;
  const rack = add(root, rackGeo(RACK_LEN, 0.04, 0.034, 2 * Math.PI * G.pinionR / 32), M.brass);
  const RACK_Y = G.shaftY + G.pinionR + 0.017 + 0.004;

  // ------------------------------------------------------------ pinion and output shaft
  const shaft = new THREE.Group();
  shaft.position.set(0, G.shaftY, 0);
  root.add(shaft);
  const sh = add(shaft, new THREE.CylinderGeometry(0.024, 0.024, 2 * (G.jointX - 0.03), 24), M.black);
  sh.rotation.z = Math.PI / 2;
  const pinion = add(shaft, gearGeo(32, G.pinionR, 0.036, 0.02), M.brass);
  pinion.rotation.y = Math.PI / 2;
  for (const x of [-0.11, 0.11]) {
    const brg = add(shaft, new THREE.CylinderGeometry(0.03, 0.03, 0.05, 24), M.steel);
    brg.rotation.z = Math.PI / 2;
    brg.position.x = x;
  }

  // ------------------------------------------------------------ one side: clevis joint, side tube, leg, wheel
  const sides = [1, -1].map((sx) => {
    const side = new THREE.Group();
    side.scale.x = sx;
    root.add(side);
    // the shaft-side yoke turns with the shaft about x
    const yokeA = new THREE.Group();
    yokeA.position.copy(J);
    side.add(yokeA);
    add(yokeA, new RoundedBoxGeometry(0.05, 0.075, 0.075, 3, 0.014), M.orange).position.x = -0.03;
    for (const s of [-1, 1]) {
      const bolt = add(yokeA, new THREE.CylinderGeometry(0.009, 0.009, 0.012, 10), M.black);
      bolt.rotation.x = Math.PI / 2;
      bolt.position.set(-0.03, s * 0.02, 0.04);
    }
    // the tube group turns about the side tube's own axis; the tube, its yoke, the leg and the wheel ride on it
    const spin = new THREE.Group();
    spin.position.copy(J);
    side.add(spin);
    const yokeB = add(spin, new RoundedBoxGeometry(0.075, 0.06, 0.06, 3, 0.012), M.orange);
    yokeB.position.copy(AXIS).multiplyScalar(0.035);
    yokeB.quaternion.setFromUnitVectors(new V3(1, 0, 0), AXIS);
    const tube = add(spin, new THREE.CylinderGeometry(0.024, 0.024, G.tubeLen - 0.03, 20), M.steel);
    tube.position.copy(AXIS).multiplyScalar((G.tubeLen + 0.03) / 2);
    tube.quaternion.setFromUnitVectors(new V3(0, 1, 0), AXIS);
    // a stripe on the tube so its spin reads
    const stripe = add(spin, new THREE.BoxGeometry(0.006, G.tubeLen * 0.6, 0.044), M.black);
    stripe.position.copy(tube.position);
    stripe.quaternion.copy(tube.quaternion);
    const root2 = add(spin, new RoundedBoxGeometry(0.08, 0.07, 0.07, 3, 0.014), M.orange);
    root2.position.copy(E0).sub(J);
    const leg = add(spin, legGeo(), M.orange);
    leg.position.copy(E0).sub(J);
    // the wheel: axle along x when down, spinning on its axle
    const wheelMount = new THREE.Group();
    wheelMount.position.copy(TIP0).sub(J);
    spin.add(wheelMount);
    const fork = add(wheelMount, new RoundedBoxGeometry(0.05, 0.09, 0.06, 3, 0.012), M.black);
    fork.position.x = -0.07;
    const wheel = new THREE.Group();
    wheelMount.add(wheel);
    add(wheel, tyreGeo(), M.rubber);
    const hub = add(wheel, new THREE.CylinderGeometry(G.wheelR * 0.5, G.wheelR * 0.5, G.wheelW * 0.8, 40), M.hub);
    hub.rotation.z = Math.PI / 2;
    const cap = add(wheel, new THREE.CylinderGeometry(0.035, 0.035, G.wheelW * 0.95, 24), M.black);
    cap.rotation.z = Math.PI / 2;
    for (const s of [-1, 1]) {
      const ring = add(wheel, new THREE.RingGeometry(G.wheelR * 0.62, G.wheelR * 0.98, 64), M.letters);
      ring.rotation.y = (s * Math.PI) / 2;
      ring.position.x = s * (G.wheelW / 2 - 0.002);
      ring.castShadow = false;
    }
    return {side, yokeA, spin, wheel};
  });

  // display modes: x-ray (the plate and the actuator body go to glass), section (both cut at the centreline), exploded
  const CLIP = new THREE.Plane(new V3(-1, 0, 0), 0);
  let xrayNow = -1;
  let sectionNow = false;
  const setXray = (k) => {
    if (Math.abs(k - xrayNow) < 1e-3) return;
    xrayNow = k;
    for (const m of [M.plate, M.body]) {
      m.transparent = k > 0.001;
      m.opacity = 1 - 0.78 * k;
      m.depthWrite = k < 0.5;
      m.needsUpdate = true;
    }
  };
  const setSection = (on) => {
    if (on === sectionNow) return;
    sectionNow = on;
    for (const m of [M.plate, M.body]) {
      m.clippingPlanes = on ? [CLIP] : null;
      m.side = on ? THREE.DoubleSide : THREE.FrontSide;
      m.needsUpdate = true;
    }
  };
  const out = {theta: 0, d: 0, CLIP};
  /** stroke 0..1 (0 down and locked, 1 up), roll (wheel spin, radians), xray 0..1, section, explode 0..1 */
  const update = ({stroke = 0, roll = 0, xray = 0, section = false, explode = 0} = {}) => {
    setXray(xray);
    setSection(section);
    const e = explode;
    plate.position.y = 0.16 * e;
    act.position.y = 0.36 * e;
    const th = stroke * Math.PI;
    const d = stroke * G.stroke;
    out.theta = th;
    out.d = d;
    // the actuator strokes aft: the rod end, the rack with it
    const zRe = 0.32 - d;
    rodEnd.position.set(0, 0.36 * e, zRe);
    const rodFrom = -0.16;
    rod.scale.y = zRe - rodFrom;
    rod.position.set(0, AY, (zRe + rodFrom) / 2);
    rack.position.set(0, RACK_Y + 0.06 * e, zRe - 0.02 - RACK_LEN / 2);
    shaft.position.y = G.shaftY - 0.12 * e;
    // the rack turns the pinion and the shaft (rack aft = shaft turning about +x)
    shaft.rotation.x = th;
    for (const s of sides) {
      s.side.position.set(0.16 * e * s.side.scale.x, -0.12 * e, 0);
      s.yokeA.rotation.x = th;
      s.spin.quaternion.setFromAxisAngle(AXIS, th);
      s.wheel.rotation.x = roll;
    }
  };
  update();
  return {root, update, M, all, plate, act, rack, shaft, sides, out};
}
