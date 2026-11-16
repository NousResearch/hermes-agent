const DATA = __DATA__;
const follow = null;
const stage = document.querySelector(".td-stage");
const readout = document.querySelector("[data-readout]");
const still = matchMedia("(prefers-reduced-motion: reduce)");
const INDEX = (DATA.index * 180) / Math.PI;
let angle = 0;
let target = 0;
let frame = 0;
let last = 0;
let visible = false;
let touring = !still.matches;
let idle = 0;
let drawn = null;
let clock = 0;
let lift = 0;
let broken = 0;
let breaking = null;
let podHook;
const SPEED = 18;
const readingOf = (turn) => ((((INDEX + 90 - turn) / 3.6) % 100) + 100) % 100;
const describe = () => {
  const reading = Math.round(readingOf(target)) % 100;
  stage.setAttribute("aria-valuenow", String(reading));
  stage.setAttribute("aria-valuetext", `dial at ${reading}${broken ? ", the pod has broken off" : ""}`);
};
const hold = () => {
  touring = false;
  clearTimeout(idle);
  idle = setTimeout(() => {
    touring = !still.matches;
    run();
  }, 3600);
};
let own = false;
const show = (value) => {
  const text = `dial ${Math.round(readingOf(value)) % 100} · ${(((value % 360) + 360) % 360).toFixed(1)}°${broken ? " · pod off" : ""}`;
  if (readout.textContent !== text) readout.textContent = text;
};
const podAt = () => {
  if (!breaking && podHook !== undefined) return podHook;
  if (!breaking) return broken ? breakAt(DATA.breakEnd) : null;
  const t = Math.min(DATA.breakEnd, breaking.time);
  return breakAt(breaking.reverse ? DATA.breakEnd - t : t);
};
let paint = () => {};
const controller = TURN.mount(stage, DATA.turn, {
  follow,
  onHold: () => {
    touring = false;
    clearTimeout(idle);
  },
  onFrame: (turned) => {
    const value = turned.values.dial;
    if (!own) {
      angle = target = drawn = value;
      lift = turned.values.lift ?? lift;
      if ("pod" in turned.values) podHook = turned.values.pod;
      describe();
    }
    show(value);
    paint();
  },
});
const GLD = DATA.gl;
const LIGHT = (stage.closest("[data-theme]")?.getAttribute("data-theme") ?? "dark") !== "dark" ? 1 : 0;
const INK = `
uniform float uLight;
vec4 over(vec4 top,vec4 under){return top+under*(1.-top.a);}
vec4 inkFor(vec3 seen,float cover){vec3 absorbed=1.-seen;float a=max(max(absorbed.r,max(absorbed.g,absorbed.b)),cover);return vec4(vec3(a)-absorbed,a);}
vec4 finish(vec4 ink,vec2 frag){ink.rgb=min(ink.rgb,vec3(ink.a));float n=(hash12(frag)-.5)/255.*smoothstep(0.,1.5/255.,ink.a);return clamp(ink+vec4(n),0.,1.);}
`;
const BACK = `${glslPose("Dial")}${INK}
uniform vec3 uLed;
uniform vec3 uLedDir;
uniform float uOn;
const float FACE_Z=${GLD.faceZ.toFixed(3)};
const float FACE_R=${GLD.faceR.toFixed(3)};
const float HUB_R=${GLD.hubR.toFixed(3)};
void main(){
  vec2 vb=viewOf(gl_FragCoord.xy);
  vec3 p=onFloor(vb,FACE_Z);
  vec3 q=toDial(p);
  float r=length(q.xy);
  float face=(1.-smoothstep(FACE_R-3.,FACE_R-1.,r))*smoothstep(HUB_R+.2,HUB_R+1.6,r);
  vec3 v=p-uLed;
  float d2=dot(v,v);
  float d=sqrt(d2);
  float lens=.3+.7*pow(max(0.,dot(v/d,uLedDir)),1.5);
  float lambert=max(0.,(uLed.z-FACE_Z+.6)/d);
  float grain=.84+.16*noise2(vec2(atan(q.y,q.x)*180.,r*.6));
  float e=420.*lens*lambert/(d2+10.)*grain*face;
  float lit=1.-exp(-e);
  vec3 tint=vec3(.40,.84,.60);
  vec4 light=inkFor(mix(vec3(1.),tint,.7*lit),0.);
  vec3 glow=vec3(.25,1.,.58)*lit*.55;
  vec4 dark=vec4(glow,max(glow.r,max(glow.g,glow.b)));
  gl_FragColor=finish(mix(dark,light,uLight)*uOn,gl_FragCoord.xy);
}`;
const FRONT = `${GLSL_TURN}${INK}
uniform vec2 uLedVb;
uniform float uLedR;
uniform float uLedFace;
uniform float uOn;
uniform vec4 uSpark[24];
uniform float uSparkHeat[24];
vec4 sparkInk(float heat,float along,float young){
  vec3 tail=mix(vec3(.78,.2,.04),vec3(1.,.55,.08),smoothstep(.35,.8,heat));
  vec3 head=mix(tail,vec3(1.,.74,.2),smoothstep(.55,.9,heat)*smoothstep(.5,1.,along));
  vec3 seen=mix(head,vec3(1.,.95,.82),young*smoothstep(.8,1.,along));
  float a=.88*smoothstep(0.,.3,along)*smoothstep(.12,.45,heat);
  vec4 light=vec4(seen*a,a);
  vec3 hot=mix(vec3(1.,.35,.05),vec3(1.,.9,.6),young*along)*(.4+.6*heat);
  vec4 dark=vec4(hot*a,a);
  return mix(dark,light,uLight);
}
void main(){
  vec2 vb=viewOf(gl_FragCoord.xy);
  float px=uView.x/uCanvas.x;
  float soft=.7*px;
  float d=length(vb-uLedVb);
  float body=1.-smoothstep(uLedR-.6*px,uLedR+.6*px,d);
  float core=1.-smoothstep(uLedR*.32,uLedR*.55+.5*px,d);
  float rim=smoothstep(uLedR*.55,uLedR,d);
  vec3 seen=mix(mix(vec3(.18,.70,.43),vec3(.11,.52,.31),rim),vec3(.90,1.,.94),core);
  vec4 lightLed=inkFor(seen,.92)*body;
  vec3 hot=mix(vec3(.2,1.,.55),vec3(.92,1.,.95),core)*body;
  vec4 darkLed=vec4(hot,body);
  vec4 led=mix(darkLed,lightLed,uLight)*smoothstep(.02,.3,uLedFace)*uOn;
  vec4 sparks=vec4(0.);
  float w=max(.35+.6*px,1.25*px);
  for(int i=0;i<24;i++){
    float heat=uSparkHeat[i];
    if(heat<=0.)continue;
    vec4 s=uSpark[i];
    vec2 a=s.zw;
    vec2 b=s.xy;
    vec2 ab=b-a;
    float l2=max(dot(ab,ab),1e-6);
    float t=clamp(dot(vb-a,ab)/l2,0.,1.);
    float dist=length(vb-a-ab*t);
    float cover=1.-smoothstep(w*.5-.5*px,w*.5+.5*px,dist);
    if(cover<=0.)continue;
    sparks=over(sparkInk(heat,t,step(.65,heat))*cover,sparks);
  }
  vec4 ink=over(sparks,led)*(1.-coverOf(vb,soft));
  gl_FragColor=finish(ink,gl_FragCoord.xy);
}`;
const canvasOf = (name) => stage.querySelector(`[data-canvas="${name}"]`);
const view = [GLD.W, GLD.H];
const backLayer = glLayer({ canvas: canvasOf("back"), fragment: BACK, view, camera: GLD.camera, uniforms: ["uLed", "uLedDir", "uOn", "uDialX", "uDialY", "uDialZ", "uDialT"], fixed: { uLight: LIGHT } });
const frontLayer = backLayer ? glLayer({ canvas: canvasOf("front"), fragment: FRONT, view, camera: GLD.camera, uniforms: ["uLedVb", "uLedR", "uLedFace", "uOn", "uSpark", "uSparkHeat", "uCoverEdge", "uCoverCount"], fixed: { uLight: LIGHT } }) : null;
const roomy = frontLayer ? frontLayer.gl.getParameter(frontLayer.gl.MAX_FRAGMENT_UNIFORM_VECTORS) >= 128 : false;
if (backLayer && frontLayer && roomy) stage.setAttribute("data-gl", "");
const SPARKS = 24;
const sparkData = new Float32Array(SPARKS * 4);
const sparkHeat = new Float32Array(SPARKS);
const sparks = [];
let first = { x: 0, v: 0 };
let glClock = 0;
let seed = 7;
const random = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
const ignite = () => {
  const at = controller.anchor("dial", GLD.socket).world;
  const spin = ((angle - (drawn ?? angle)) * Math.PI) / 180;
  void spin;
  for (let i = 0; i < SPARKS; i++) {
    const a = random() * Math.PI * 2;
    const tilt = 0.35 + random() * 0.9;
    const speed = 16 + random() * 22;
    sparks[i] = { p: at.slice(), v: [Math.cos(a) * Math.sin(tilt) * speed, Math.sin(a) * Math.sin(tilt) * speed, Math.cos(tilt) * speed], age: -random() * 0.08, life: 0.3 + random() * 0.45 };
  }
};
const stepSparks = (dt) => {
  let live = false;
  for (let i = 0; i < SPARKS; i++) {
    const s = sparks[i];
    sparkHeat[i] = 0;
    if (!s) continue;
    s.age += dt;
    if (s.age < 0) {
      live = true;
      continue;
    }
    if (s.age > s.life) {
      sparks[i] = null;
      continue;
    }
    s.v[2] -= 98 * dt;
    for (let c = 0; c < 3; c++) s.p[c] += s.v[c] * dt;
    if (s.p[2] < GLD.faceZ) {
      sparks[i] = null;
      continue;
    }
    live = true;
  }
  return live;
};
const writeSparks = () => {
  const w = controller.cams().world;
  const project = (p) => [w.ox + w.m[0] * p[0] + w.m[1] * p[1] + w.m[2] * p[2], w.oy + w.m[3] * p[0] + w.m[4] * p[1] + w.m[5] * p[2]];
  for (let i = 0; i < SPARKS; i++) {
    const s = sparks[i];
    if (!s || s.age < 0) {
      sparkHeat[i] = 0;
      continue;
    }
    const head = project(s.p);
    const tail = project([s.p[0] - s.v[0] * 0.05, s.p[1] - s.v[1] * 0.05, s.p[2] - s.v[2] * 0.05]);
    sparkData.set([head[0], head[1], tail[0], tail[1]], 4 * i);
    sparkHeat[i] = 1 - s.age / s.life;
  }
};
paint = () => {
  if (!stage.hasAttribute("data-gl")) return;
  const led = controller.anchor(GLD.led.group, GLD.led.at, GLD.led.normal);
  const world = controller.cams().world;
  const facing = led.direction[0] * world.V[0] + led.direction[1] * world.V[1] + led.direction[2] * world.V[2];
  const on = first.x;
  backLayer.draw({ uLed: led.world, uLedDir: led.direction, uOn: on, ...TURN.poseUniforms(controller, "dial", "uDial") }, glClock);
  writeSparks();
  const live = sparks.some(Boolean);
  const cover = TURN.cover(controller, live ? [[GLD.led.part], ["hand.pod.socket"]] : [GLD.led.part]);
  frontLayer.draw({ uLedVb: led.screen, uLedR: 0.52 * world.k, uLedFace: facing, uOn: on, uSpark: sparkData, uSparkHeat: sparkHeat, uCoverEdge: cover.uCoverEdge, uCoverCount: cover.uCoverCount }, glClock);
  const u = on;
  const alpha = 0.92;
  stage.style.setProperty("--td-fallback", String(Math.max(0, (1 - u) / (1 - u * alpha + 1e-6))));
};
const draw = () => {
  own = true;
  controller.set({ dial: angle, lift, pod: podAt() });
  own = false;
  drawn = angle;
};
function tick(now) {
  const dt = last ? Math.min((now - last) / 1000, 1 / 30) : 1 / 60;
  last = now;
  const calm = still.matches;
  if (touring) {
    target += SPEED * dt;
    clock += dt;
  }
  angle = calm ? target : angle + (target - angle) * (1 - Math.exp(-dt / 0.12));
  if (Math.abs(target - angle) < 1e-3) angle = target;
  const nextLift = calm ? 0 : touring ? lift + (tapAt(clock) - lift) * (1 - Math.exp(-dt / 0.15)) : lift;
  let moved = Math.abs(nextLift - lift) > 1e-5;
  lift = nextLift;
  glClock += dt;
  const before = first.x;
  const lighting = stage.hasAttribute("data-gl") && settle(first, 1, dt, 0.35, calm);
  if (first.x !== before) moved = true;
  const sparking = stage.hasAttribute("data-gl") && stepSparks(calm ? 1 : dt);
  if (lighting || sparking) moved = true;
  if (breaking) {
    if (!breaking.reverse && breaking.time === 0 && !calm) ignite();
    breaking.time += calm ? DATA.breakEnd : dt;
    moved = true;
    if (breaking.time >= DATA.breakEnd) {
      broken = breaking.reverse ? 0 : 1;
      breaking = null;
      describe();
    }
  }
  if (moved || drawn === null || Math.abs(angle - drawn) > 1e-4) draw();
  const settled = !touring && angle === target && !breaking && !lighting && !sparking;
  frame = visible && !settled ? requestAnimationFrame(tick) : 0;
}
const run = () => {
  if (!frame && visible) {
    last = 0;
    frame = requestAnimationFrame(tick);
  }
};
new IntersectionObserver(
  (entries) => {
    visible = entries[entries.length - 1].isIntersecting;
    if (visible) run();
    else if (frame) {
      cancelAnimationFrame(frame);
      frame = 0;
    }
  },
  { rootMargin: "120px 0px" },
).observe(stage);
let drag = null;
stage.addEventListener("pointerdown", (event) => {
  hold();
  drag = { from: controller.angleAt(event.clientX, event.clientY, 17.2, "dial"), base: target, id: event.pointerId, captured: false };
});
stage.addEventListener("pointermove", (event) => {
  if (!drag || event.pointerId !== drag.id) return;
  hold();
  const now = controller.angleAt(event.clientX, event.clientY, 17.2, "dial");
  const delta = now - drag.from;
  if (!drag.captured && Math.abs(delta) > 1.5) {
    drag.captured = true;
    stage.setPointerCapture(event.pointerId);
    stage.setAttribute("data-drag", "");
  }
  if (!drag.captured) return;
  target = drag.base + delta;
  describe();
  run();
});
const release = (event) => {
  if (!drag || event.pointerId !== drag.id) return;
  drag = null;
  stage.removeAttribute("data-drag");
  try {
    stage.releasePointerCapture(event.pointerId);
  } catch {}
};
stage.addEventListener("pointerup", release);
stage.addEventListener("pointercancel", release);
stage.addEventListener("keydown", (event) => {
  if (event.key === "b" || event.key === "B") {
    event.preventDefault();
    podHook = undefined;
    if (breaking) breaking = { reverse: !breaking.reverse, time: DATA.breakEnd - breaking.time };
    else breaking = { reverse: Boolean(broken), time: 0 };
    run();
    return;
  }
  const step = event.shiftKey ? 36 : 3.6;
  let next = null;
  if (event.key === "ArrowRight" || event.key === "ArrowUp") next = Math.round(target / 3.6) * 3.6 - step;
  if (event.key === "ArrowLeft" || event.key === "ArrowDown") next = Math.round(target / 3.6) * 3.6 + step;
  if (event.key === "Home") next = INDEX + 90 - 360 * Math.round((INDEX + 90 - target) / 360);
  if (next === null) return;
  event.preventDefault();
  hold();
  target = next;
  describe();
  run();
});
if (still.matches) touring = false;
draw();
describe();
