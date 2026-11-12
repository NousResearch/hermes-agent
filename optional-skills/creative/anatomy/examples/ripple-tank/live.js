const D = __DATA__;
const root = document.querySelector(".rt");
const stage = root.querySelector(".rt-stage");
const readout = root.querySelector("[data-readout]");
const paperCanvas = stage.querySelector(".rt-paper-gl");
const waterCanvas = stage.querySelector(".rt-water-gl");
const front = stage.querySelector(".rt-front");
const mid = stage.querySelector(".rt-mid");
const farDipper = front.querySelector('[data-part="dipper-far"]');
const nearDipper = front.querySelector('[data-part="dipper-near"]');
const eccentric = front.querySelector('[data-part="eccentric"]');
const pointer = mid.querySelector('[data-part="pointer"]');
const needle = mid.querySelector('[data-part="needle"]');
const bar = front.querySelector('[data-part="bar"]');
const motor = front.querySelector('[data-part="motor"]');
const said = stage.querySelector(".rt-said");
const crestPath = stage.querySelector('[data-part="crests"]');
const nodalPath = stage.querySelector('[data-part="nodal"]');
const still = matchMedia("(prefers-reduced-motion: reduce)");
const pale = !!root.closest('[data-theme="light"]');
const TAU = Math.PI * 2;
const HIST = 32;
const clampTo = (value, low, high) => Math.min(high, Math.max(low, value));

const WAVE_GLSL = `
uniform vec2 uS1;uniform vec2 uS2;uniform float uPhase;uniform float uReach;
uniform sampler2D uHT;uniform vec4 uLo;uniform vec4 uSpan;uniform vec2 uHist;uniform float uIG;uniform float uKNow;uniform float uStep;
uniform float uSlope;uniform float uCaustic;uniform float uMag;
uniform vec4 uIn;uniform float uBeach;uniform vec4 uZ;uniform float uLight;uniform float uPale;uniform float uBed;uniform vec3 uLamp;uniform float uSpread;
const float HN=${HIST}.;
float gPx;vec2 gGrad;vec3 gHess;float gCurv;float gAmp;float gKA;float gMove;
float coneOf(vec3 p){float drop=uLamp.z-p.z;float spread=13.4+drop*uSpread;return 1.-smoothstep(spread*.72,spread,length(p.xy-uLamp.xy));}
float sdRect(vec2 p,vec4 r,float rad){vec2 c=.5*(r.xy+r.zw);vec2 h=.5*(r.zw-r.xy)-rad;vec2 q=abs(p-c)-h;return length(max(q,0.))+min(max(q.x,q.y),0.)-rad;}
float inRect(vec2 p,vec4 r,float rad,float soft){return 1.-smoothstep(-soft,soft,sdRect(p,r,rad));}
vec2 unpack(vec4 t){t=floor(t*255.+.5);return vec2(t.x*256.+t.y,t.z*256.+t.w)/65535.;}
float slotOf(float age){float s=age/uHist.y;return clamp(mix(s/max(uHist.x,1e-3),1.+s-uHist.x,step(uHist.x,s)),0.,HN-1.);}
vec2 rowAt(float u,float row){
  float n=floor(u);
  vec2 a=unpack(texture2D(uHT,vec2((n+.5)/HN,row)));
  vec2 b=unpack(texture2D(uHT,vec2((min(n+1.,HN-1.)+.5)/HN,row)));
  return mix(a,b,u-n);
}
void source(vec2 p,vec2 s,float scale){
  vec2 q=p-s;float r2=dot(q,q)+4.;float r=sqrt(r2);
  float ig=uLo.w+uSpan.w*rowAt(slotOf(r*uIG),.75).y;
  float u=slotOf(r*ig);
  vec2 kp=uLo.xy+uSpan.xy*rowAt(u,.25);
  float kE=kp.x;
  float dec=uLo.z+uSpan.z*rowAt(u,.75).x;
  float fine=smoothstep(2.5,5.,6.2832/kE*scale/gPx);
  float inv=1./(r+uReach);
  float A=uSlope/kE*fine*sqrt((uReach+6.2832/kE)*inv)*exp(-r*dec);
  float a1=-.5*inv-dec;
  float a2=a1*a1+.5*inv*inv;
  float th=kE*r-uPhase+kp.y;float cs=cos(th);float sn=sin(th);
  float g1=A*(a1*cs-kE*sn);
  float g2=A*(a2*cs-2.*a1*kE*sn-kE*kE*cs);
  vec2 n=q/r;float t=g1/r;
  gGrad+=g1*n;
  gHess+=vec3((g2-t)*n.x*n.x+t,(g2-t)*n.x*n.y,(g2-t)*n.y*n.y+t);
  gCurv+=A*kE*kE;
  gAmp+=A;
  gKA+=A*kE;
  gMove=max(gMove,abs(uKNow-kE)/max(ig,1e-6)*uStep/kE*fine);
}
void field(vec2 p,float scale){
  gGrad=vec2(0.);gHess=vec3(0.);gCurv=0.;gAmp=0.;gKA=0.;gMove=0.;
  source(p,uS1,scale);source(p,uS2,scale);
  float w=smoothstep(0.,uBeach,-sdRect(p,uIn,0.));
  gGrad*=w;gHess*=w;gCurv*=w;gAmp*=w;gKA*=w;gMove*=w;
}
vec3 dither(vec3 c){
  float m=max(c.r,max(c.g,c.b));
  float n=(hash12(gl_FragCoord.xy)+hash12(gl_FragCoord.yx*1.37+17.)-1.)/255.;
  return max(c+n*smoothstep(0.,1.5/255.,m),0.);
}
`;

const PACK = [
  [0.15, 0.85],
  [-256, 256],
  [0, 0.004],
  [0.0025, 0.005],
];

function textureOf(layer) {
  const gl = layer.gl;
  const texture = gl.createTexture();
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, HIST, 2, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
  return texture;
}

const PAPER_GLSL = `${WAVE_GLSL}
uniform vec4 uOut;uniform float uLampR;uniform vec4 uPaper;uniform vec4 uBase;uniform vec2 uRound;uniform vec4 uLegs;uniform vec4 uPost;
uniform vec4 uBar;uniform vec4 uBeam;uniform vec4 uMotor;uniform vec3 uHigh;uniform vec4 uBox;uniform float uBoxTop;uniform float uBlur;uniform vec3 uExpo;uniform vec3 uFoam;
float gPen;
vec2 hitAt(vec3 q,float z){return uLamp.xy+(q.xy-uLamp.xy)*(uLamp.z-z)/(uLamp.z-q.z);}
float penAt(vec3 q,float z){return max(uLampR*(z-q.z)/(uLamp.z-q.z),gPen);}
float rodLight(vec3 q,vec2 c,float rad,float top){
  vec2 ab=uLamp.xy-q.xy;float h=(top-q.z)/(uLamp.z-q.z);
  float t=clamp(dot(c-q.xy,ab)/max(dot(ab,ab),1e-4),0.,h);
  float pen=max(uLampR*t,gPen);
  return smoothstep(rad-pen,rad+pen,length(q.xy+ab*t-c));
}
float legClear(vec3 q,vec2 c){
  vec3 v=towardViewer();
  float top=(uLegs.w-q.z)/v.z;
  float t=clamp(dot(c-q.xy,v.xy)/max(dot(v.xy,v.xy),1e-4),0.,top);
  return smoothstep(uLegs.z+3.,uLegs.z+7.,length(q.xy+v.xy*t-c));
}
float causticAt(vec2 u){
  vec2 u0=u;
  field(u0,uMag);u0=u-uCaustic*gGrad;
  field(u0,uMag);u0=u-uCaustic*gGrad;
  field(u0,uMag);
  float c=uCaustic;
  float J=(1.+c*gHess.x)*(1.+c*gHess.z)-c*c*gHess.y*gHess.y;
  float b=sqrt(uBlur*uBlur+pow(.55*gPx/uMag,2.)+pow(.5*gMove,2.));
  float e=.6*pow(max(gKA/max(gAmp,1e-6)*b,1e-4),.6667);
  return sqrt(1.+e*e)/sqrt(J*J+e*e);
}
float lampAt(vec3 q){
  vec3 d=uLamp-q;float fall=pow(d.z/length(d),3.)*coneOf(q);
  float through=inRect(hitAt(q,uZ.y),uIn,0.,penAt(q,uZ.y));
  through*=mix(uFoam.z,1.,inRect(hitAt(q,uFoam.x),uIn+vec4(uFoam.y,uFoam.y,-uFoam.y,-uFoam.y),0.,max(penAt(q,uFoam.x),uFoam.y*.5)));
  float open=1.-inRect(hitAt(q,uZ.w),uOut,1.5,penAt(q,uZ.w));
  float caustic=1.;
  if(through>.001)caustic=causticAt(hitAt(q,uZ.z));
  float shade=rodLight(q,vec2(uLegs.x,uLegs.y),uLegs.z,uLegs.w)*rodLight(q,vec2(-uLegs.x,uLegs.y),uLegs.z,uLegs.w)*rodLight(q,vec2(uLegs.x,-uLegs.y),uLegs.z,uLegs.w)*rodLight(q,vec2(-uLegs.x,-uLegs.y),uLegs.z,uLegs.w);
  shade*=rodLight(q,uPost.xy,uPost.z,uPost.w);
  shade*=1.-.96*inRect(hitAt(q,uHigh.x),uBar,1.,penAt(q,uHigh.x));
  shade*=1.-.96*inRect(hitAt(q,uHigh.y),uBeam,1.,penAt(q,uHigh.y));
  shade*=1.-.96*inRect(hitAt(q,uHigh.z),uMotor,3.,penAt(q,uHigh.z));
  shade*=1.-.96*inRect(hitAt(q,uBoxTop),uBox,4.,penAt(q,uBoxTop));
  return fall*shade*(through*caustic*mix(.6,1.1,clamp(caustic-1.,0.,1.))*.9+open);
}
void main(){
  vec2 vb=viewOf(gl_FragCoord.xy);
  gPx=pixelOf()/uCam.z;
  float soft=.7*gPx;
  vec3 q=onFloor(vb,uZ.x);
  vec2 rest=q.xy;
  vec3 s=onFloor(vb,uZ.z);
  float wet=inRect(s.xy,uIn,0.,soft);
  vec2 shift=vec2(0.);
  if(wet>.001){
    field(s.xy,1.);
    vec3 n=normalize(vec3(-gGrad,1.));
    vec3 d=-towardViewer();
    vec3 inWater=refract(d,n,1./1.333);
    vec3 flatWater=refract(d,vec3(0.,0.,1.),1./1.333);
    vec3 below=refract(inWater,vec3(0.,0.,1.),1.333);
    shift=(inWater.xy/-inWater.z-flatWater.xy/-flatWater.z)*(uZ.z-uBed)+(below.xy/-below.z-d.xy/-d.z)*(uBed-uZ.x);
    shift*=wet*legClear(q,vec2(uLegs.x,uLegs.y))*legClear(q,vec2(-uLegs.x,uLegs.y))*legClear(q,vec2(uLegs.x,-uLegs.y))*legClear(q,vec2(-uLegs.x,-uLegs.y));
  }
  q.xy+=shift;
#ifdef GL_OES_standard_derivatives
  float stretch=length(fwidth(q.xy))/max(length(fwidth(rest)),1e-6);
#else
  float stretch=1.+8.*length(shift)/max(uZ.z-uZ.x,1.);
#endif
  gPen=1.5*soft*max(stretch,1.);
  float base=inRect(q.xy,uBase,uRound.y,soft);
  if(base<.001){gl_FragColor=vec4(0.);return;}
  float paper=inRect(q.xy,uPaper,uRound.x,soft);
  float e=lampAt(q)*mix(1.,.84,wet);
  if(uPale>.5){
    float lit=mix(1.-.05*exp(-e),1.-uExpo.x/(1.+pow(e/uExpo.y,uExpo.z)),paper);
    float off=(1.-mix(.95,1.-uExpo.x,paper))/.89*base;
    float ink=mix(off,(1.-lit)/.89*base,uLight);
    gl_FragColor=vec4(dither(vec3(.11,.11,.12)*ink),ink);
  }else{
    float glow=(1.-exp(-uExpo.x*e))*mix(.34,.92,paper)*base*uLight;
    vec3 c=dither(vec3(glow));
    gl_FragColor=vec4(c,c.r);
  }
}`;

const WATER_GLSL = `${WAVE_GLSL}
uniform vec2 uNear;uniform float uContact;uniform float uFloor;uniform float uRim;uniform float uTop;uniform vec4 uEnv;uniform vec3 uGlint;uniform float uTint;uniform vec4 uWall;uniform vec2 uGlass;
float beamAt(vec2 vb){
  float sum=0.;
  float jitter=hash12(gl_FragCoord.xy);
  for(int i=0;i<28;i++){
    float z=mix(uTop+1.,uLamp.z,(float(i)+jitter)/28.);
    vec3 p=onFloor(vb,z);
    float drop=uLamp.z-z;
    sum+=coneOf(p)*smoothstep(0.,16.,drop)/(1.+drop*drop/6400.);
  }
  return sum/28.;
}
float envAt(vec3 r,float blur){
  float el=asin(clamp(r.z,-1.,1.));
  float w=sqrt(uEnv.w*uEnv.w+blur*blur);
  float g=sqrt(uGlint.y*uGlint.y+blur*blur);
  float glint=(uGlint.y/g)*exp(-pow((el-asin(uCam.z)-uGlint.x)/g,2.));
  return mix(uEnv.x,uEnv.y,smoothstep(-w,w,el-asin(uCam.z)))+uEnv.z*smoothstep(-.3,1.2,el)+uGlint.z*glint;
}
float lineAt(float x,float at,float width){return exp(-pow((x-at)/width,2.))/(width*1.772);}
void main(){
  vec2 vb=viewOf(gl_FragCoord.xy);
  float px=pixelOf();
  gPx=px/uCam.z;
  float soft=.7*gPx;
  vec3 s=onFloor(vb,uZ.z);
  float wet=inRect(s.xy,uIn,0.,soft);
  vec3 v=towardViewer();
  float add=0.;
  float tint=0.;
  if(wet>.001){
    field(s.xy,1.);
    float lc=max(2.2,1.6*gPx);
    vec4 dw=vec4(s.x-uIn.x,uIn.z-s.x,s.y-uIn.y,uIn.w-s.y);
    vec4 tilt=1.1*exp(-max(dw,0.)/lc);
    vec2 g=-gGrad+vec2(tilt.x-tilt.y,tilt.z-tilt.w);
    vec3 n=normalize(vec3(g,1.));
    float cosI=max(dot(n,v),0.);
    float fres=.02+.98*pow(1.-cosI,5.);
    vec3 r=reflect(-v,n);
    float spread=2.*gPx*(gCurv+dot(tilt,vec4(1.))/lc);
    float refl=fres*envAt(r,spread);
    float flat0=(.02+.98*pow(1.-uCam.z,5.))*envAt(vec3(-v.xy,v.z),0.);
    float rw=max(.45,1.2*soft);
    float rc=uContact+rw;
    float ring=(exp(-pow((length(s.xy-uS1)-rc)/rw,2.))+exp(-pow((length(s.xy-uS2)-rc)/rw,2.)))*(.45/rw);
    float lift=uPale>.5?max(refl-flat0,0.)/.15:refl;
    float sink=uPale>.5?max(flat0-refl,0.)/.74:0.;
    add+=(lift+ring*.12)*wet;
    tint+=(uTint+sink+ring*.05)*wet;
  }
  float pz=px/uCam.w;
  float sz=.7*pz;
  vec3 bx=onWallX(vb,uNear.x);
  vec3 by=onWallY(vb,uNear.y);
  float alongX=window(bx.y,uIn.y,uIn.w,.7*px/uCam.y);
  float alongY=window(by.x,uIn.x,uIn.z,.7*px/uCam.x);
  float bandX=window(bx.z,uFloor,uZ.z,sz)*alongX;
  float bandY=window(by.z,uFloor,uZ.z,sz)*alongY;
  float band=max(bandX,bandY);
  vec3 fx=onWallX(vb,uIn.x);
  vec3 fy=onWallY(vb,uIn.y);
  float glass=max(window(bx.z,uZ.z,uRim,sz)*alongX,window(by.z,uZ.z,uRim,sz)*alongY);
  glass=max(glass,max(window(fx.z,uZ.z,uRim,sz)*window(fx.y,uIn.y,uIn.w,.7*px/uCam.y),window(fy.z,uZ.z,uRim,sz)*window(fy.x,uIn.x,uIn.z,.7*px/uCam.x)));
  add+=glass*uGlass.x;
  tint+=glass*uGlass.y;
  float lw=max(.45,.9*pz);
  float lip=max(lineAt(bx.z,uZ.z+.5,lw)*alongX,lineAt(by.z,uZ.z+.5,lw)*alongY)*1.2;
  float sole=max(lineAt(bx.z,uFloor+.3,lw)*alongX,lineAt(by.z,uFloor+.3,lw)*alongY)*1.2;
  add+=band*uWall.x+(lip+sole*.35)*uWall.z;
  tint+=band*uWall.y+(lip+sole*.35)*uWall.w;
  if(uPale<.5)add+=tonemap(vec3(beamAt(vb)*.16),1.).r*.9;
  add=max(add,0.);tint=clamp(tint,0.,.9);
  float a=add+tint;
  float over=max(a,1.);
  vec3 ink=uPale>.5?vec3(.11,.11,.12):vec3(.02,.025,.03);
  vec3 c=dither(vec3(add)+ink*tint)/over;
  gl_FragColor=vec4(c,a/over)*uLight;
}`;

const FIXED = {
  uReach: D.reach,
  uSlope: D.slope,
  uCaustic: D.optics.caustic,
  uMag: D.optics.magnification,
  uIn: D.inside,
  uBeach: D.beach,
  uZ: D.z,
  uPale: pale ? 1 : 0,
  uBed: D.bed,
  uFloor: D.floor,
  uOut: D.outside,
  uLamp: D.lamp,
  uLampR: D.lampR,
  uBlur: D.blur,
  uPaper: D.paper,
  uBase: D.base,
  uRound: D.round,
  uLegs: D.legs,
  uPost: D.post,
  uBar: D.bar,
  uBeam: D.beam,
  uMotor: D.motor,
  uHigh: D.high,
  uBox: D.box,
  uBoxTop: D.boxTop,
  uNear: D.near,
  uContact: D.contact,
  uSpread: D.spread,
  uFoam: D.foam,
  uLo: PACK.map(([low]) => low),
  uSpan: PACK.map(([low, high]) => high - low),
  uExpo: pale ? [0.3, 0.7, 2.4] : [0.55, 0, 0],
  uEnv: pale ? [0.3, 1, 0, 0.02] : [0.04, 6, 0.3, 0.02],
  uGlass: pale ? [0, 0.05] : [0.02, 0.03],
  uGlint: pale ? [0.035, 0.006, 3] : [0.035, 0.006, 26],
  uRim: D.rim,
  uTop: D.top,
  uTint: pale ? 0.06 : 0.08,
  uWall: pale ? [0, 0.2, 0, 0.45] : [0.05, 0.16, 0.5, 0],
};
const NAMES = ["uS1", "uS2", "uPhase", "uHist", "uIG", "uKNow", "uStep", "uLight"];
const paperLayer = glLayer({ canvas: paperCanvas, fragment: PAPER_GLSL, view: [D.W, D.H], camera: D.camera, uniforms: NAMES, fixed: FIXED, extensions: ["GL_OES_standard_derivatives"] });
const waterLayer = paperLayer ? glLayer({ canvas: waterCanvas, fragment: WATER_GLSL, view: [D.W, D.H], camera: D.camera, uniforms: NAMES, fixed: FIXED }) : null;
const lit = !!(paperLayer && waterLayer);
const bytes = new Uint8Array(HIST * 2 * 4);
const textures = lit ? [paperLayer, waterLayer].map(textureOf) : [];
if (lit) stage.setAttribute("data-gl", "");

const FR = D.range.frequency;
const SR = D.range.spacing;
const freq = { x: D.start.frequency, v: 0 };
const gap = { x: D.start.spacing, v: 0 };
const state = { tf: D.start.frequency, td: D.start.spacing, phase: D.phase, lamp: 0, lampClock: 0, clock: 0, pushed: 0, step: 0 };
let model = stateOf(freq.x, gap.x, D.optics);
const history = { k: new Float64Array(HIST), phi: new Float64Array(HIST), omega: new Float64Array(HIST), decay: new Float64Array(HIST), slow: new Float64Array(HIST), at: new Float64Array(HIST) };
let phi = 0;
let touring = !still.matches;
let tourClock = -1.4;
let tourFrom = { f: freq.x, d: gap.x };
let idleTimer = 0;
let sayTimer = 0;
let visible = false;
let frame = 0;
let last = 0;
let dirty = true;
let drawn = "";

const angleOf = (f) => ((-135 + (270 * (f - FR[0])) / (FR[1] - FR[0])) * Math.PI) / 180;

function tourAt(time) {
  const f = 17 + 8.5 * Math.sin((time * TAU) / 28 - 0.3);
  const d = 50 + 25 * Math.sin((time * TAU) / 16 + 0.2);
  const blend = smooth(0, 1, time / 2.4);
  return { f: tourFrom.f + (f - tourFrom.f) * blend, d: tourFrom.d + (d - tourFrom.d) * blend };
}

function announce() {
  clearTimeout(sayTimer);
  sayTimer = setTimeout(() => {
    const words = wordsOf(stateOf(state.tf, state.td, D.optics));
    if (said.textContent !== words) said.textContent = words;
  }, 700);
}

function record(slot) {
  history.k[slot] = model.kMm;
  history.phi[slot] = phi;
  history.omega[slot] = TAU * model.frequency;
  history.decay[slot] = model.decay;
  history.slow[slot] = 1 / (model.group * 1000);
  history.at[slot] = state.clock;
}

function remember(dt, calm) {
  state.clock += dt;
  phi += TAU * model.frequency * dt;
  if (calm) {
    for (let slot = 0; slot < HIST; slot++) record(slot);
    state.pushed = state.clock;
    return;
  }
  while (state.clock - state.pushed >= D.histDt) {
    state.pushed += D.histDt;
    for (const list of Object.values(history)) list.copyWithin(1, 0, HIST - 1);
  }
  record(0);
}

function put(index, value, [low, high]) {
  const code = Math.round(clampTo((value - low) / (high - low), 0, 1) * 65535);
  bytes[index] = code >> 8;
  bytes[index + 1] = code & 255;
}

function packHistory() {
  for (let slot = 0; slot < HIST; slot++) {
    const lag = phi - history.phi[slot] - history.omega[slot] * (state.clock - history.at[slot]);
    put(slot * 4, history.k[slot], PACK[0]);
    put(slot * 4 + 2, slot ? lag : 0, PACK[1]);
    put((HIST + slot) * 4, history.decay[slot], PACK[2]);
    put((HIST + slot) * 4 + 2, history.slow[slot], PACK[3]);
  }
  for (const [index, layer] of [paperLayer, waterLayer].entries()) {
    const gl = layer.gl;
    gl.bindTexture(gl.TEXTURE_2D, textures[index]);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, HIST, 2, gl.RGBA, gl.UNSIGNED_BYTE, bytes);
  }
}

function drawFallback() {
  if (lit) return;
  const key = `${freq.x.toFixed(2)}|${gap.x.toFixed(1)}`;
  if (key === drawn) return;
  drawn = key;
  const pattern = patternOf(model, D.pattern, (point) => iso(point, D.P), D.phase);
  crestPath.setAttribute("d", pattern.crests);
  nodalPath.setAttribute("d", pattern.nodal);
}

function drawSvg() {
  const half = gap.x / 2;
  const bob = model.amplitude * Math.sin(state.phase);
  farDipper.setAttribute("transform", translateAlong(D.P, [0, -half, bob]));
  nearDipper.setAttribute("transform", translateAlong(D.P, [0, half, bob]));
  bar.setAttribute("transform", translateAlong(D.P, [0, 0, bob]));
  motor.setAttribute("transform", translateAlong(D.P, [0, 0, bob]));
  const a = angleOf(freq.x);
  const [kx, ky] = D.knob;
  pointer.setAttribute("d", lineOnTop([kx + Math.sin(a) * 1.4, ky - Math.cos(a) * 1.4], [kx + Math.sin(a) * 5.6, ky - Math.cos(a) * 5.6], D.boxTop + 8, D.P));
  const [mx, my] = D.meter;
  needle.setAttribute("d", lineOnTop([mx - Math.sin(a) * 2.2, my + Math.cos(a) * 2.2], [mx + Math.sin(a) * 9.6, my - Math.cos(a) * 9.6], D.boxTop + 3.4, D.P));
  const [ex, ez] = D.eccentric;
  eccentric.setAttribute("d", sideRing([ex + 0.05, 2.9 * Math.cos(state.phase), ez + bob + 2.9 * Math.sin(state.phase)], 1.05, "x", D.P, 14));
  stage.style.setProperty("--rt-lamp", state.lamp.toFixed(3));
  drawFallback();
}

function drawGl() {
  if (!lit || !paperCanvas.clientWidth || !paperCanvas.clientHeight) return;
  if (state.lamp <= 0 && !pale) {
    paperLayer.clear();
    waterLayer.clear();
    return;
  }
  packHistory();
  const values = {
    uS1: [D.xs, -gap.x / 2],
    uS2: [D.xs, gap.x / 2],
    uPhase: state.phase,
    uHist: [(state.clock - state.pushed) / D.histDt, D.histDt],
    uIG: 1 / (model.group * 1000),
    uKNow: model.kMm,
    uStep: state.step,
    uLight: state.lamp,
  };
  paperLayer.draw(values, 0);
  if (state.lamp > 0) waterLayer.draw(values, 0);
  else waterLayer.clear();
}

function tick(now) {
  const dt = last ? Math.min((now - last) / 1000, 1 / 30) : 1 / 60;
  last = now;
  const calm = still.matches;
  if (touring && !calm) {
    tourClock += dt;
    if (tourClock >= 0) {
      const aim = tourAt(tourClock);
      state.tf = aim.f;
      state.td = aim.d;
    }
  }
  const before = [freq.x, gap.x];
  settle(freq, state.tf, dt, 0.625, calm);
  settle(gap, state.td, dt, 0.4, calm);
  if (lit) {
    state.lampClock += dt;
    state.lamp = calm ? 1 : smooth(0, 1, (state.lampClock - 0.15) / 0.9);
  }
  if (!calm) state.phase = (state.phase + dt * TAU * D.beat) % TAU;
  state.step = calm ? 0 : dt;
  const changed = dirty || before[0] !== freq.x || before[1] !== gap.x;
  if (changed) model = stateOf(freq.x, gap.x, D.optics);
  remember(dt, calm);
  if (changed || !calm) {
    drawSvg();
    drawGl();
    if (readout.textContent !== readoutOf(model)) readout.textContent = readoutOf(model);
  }
  dirty = false;
  const settled = calm && freq.x === state.tf && gap.x === state.td && (!lit || state.lamp === 1);
  frame = visible && !settled ? requestAnimationFrame(tick) : 0;
}

function run() {
  if (!frame && visible) {
    last = 0;
    frame = requestAnimationFrame(tick);
  }
}

function hold() {
  touring = false;
  clearTimeout(idleTimer);
  idleTimer = setTimeout(() => {
    if (still.matches) return;
    touring = true;
    tourClock = -0.001;
    tourFrom = { f: freq.x, d: gap.x };
    run();
  }, 3600);
}

function steer(f, d) {
  if (touring) {
    freq.v = 0;
    gap.v = 0;
  }
  state.tf = clampTo(f, FR[0], FR[1]);
  state.td = clampTo(d, SR[0], SR[1]);
  dirty = true;
  hold();
  announce();
  run();
}

new IntersectionObserver(
  (entries) => {
    visible = entries[entries.length - 1].isIntersecting;
    if (visible) run();
  },
  { rootMargin: "120px 0px" },
).observe(stage);

function aimOf(event) {
  const box = stage.getBoundingClientRect();
  const fx = clampTo(((event.clientX - box.left) / box.width - 0.12) / 0.76, 0, 1);
  const fy = 1 - clampTo(((event.clientY - box.top) / box.height - 0.16) / 0.66, 0, 1);
  return [FR[0] + fx * (FR[1] - FR[0]), SR[0] + fy * (SR[1] - SR[0])];
}

stage.addEventListener("pointermove", (event) => {
  if (event.pointerType === "touch") {
    if (!event.buttons) return;
    steer(aimOf(event)[0], touring ? gap.x : state.td);
    return;
  }
  const [f, d] = aimOf(event);
  steer(f, d);
});

stage.addEventListener("pointerdown", (event) => {
  const [f, d] = aimOf(event);
  steer(f, d);
});

stage.addEventListener("keydown", (event) => {
  const big = event.shiftKey;
  let f = touring ? freq.x : state.tf;
  let d = touring ? gap.x : state.td;
  if (event.key === "ArrowRight") f += big ? 2 : 0.5;
  else if (event.key === "ArrowLeft") f -= big ? 2 : 0.5;
  else if (event.key === "ArrowUp") d += big ? 10 : 2;
  else if (event.key === "ArrowDown") d -= big ? 10 : 2;
  else if (event.key === "Home") f = FR[0];
  else if (event.key === "End") f = FR[1];
  else return;
  event.preventDefault();
  steer(f, d);
});

still.addEventListener("change", () => {
  touring = !still.matches;
  dirty = true;
  run();
});

if (matchMedia("(hover: none)").matches) {
  const hint = root.querySelector('[data-corner="hint"]');
  if (hint) hint.textContent = "Tap to set f and spacing · drag across for f";
}
window.addEventListener("resize", () => {
  dirty = true;
  run();
});
stage.addEventListener("focus", () => {
  said.textContent = wordsOf(model);
});
for (let slot = 0; slot < HIST; slot++) record(slot);
drawSvg();
drawGl();
readout.textContent = readoutOf(model);
