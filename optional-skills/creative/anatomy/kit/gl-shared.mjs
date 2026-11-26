export const GLSL_ISO = `precision highp float;
uniform vec2 uCanvas;
uniform vec2 uView;
uniform vec2 uOrigin;
uniform float uK;
uniform vec4 uCam;
uniform float uTime;
vec2 viewOf(vec2 frag){return vec2(frag.x,uCanvas.y-frag.y)*(uView.x/uCanvas.x);}
vec2 project(vec3 p){return uOrigin+uK*vec2(p.x*uCam.x-p.y*uCam.y,(p.x*uCam.y+p.y*uCam.x)*uCam.z-p.z*uCam.w);}
vec3 towardViewer(){return vec3(uCam.y*uCam.w,uCam.x*uCam.w,uCam.z);}
vec3 rayOf(vec2 vb){vec2 q=(vb-uOrigin)/uK;return vec3(q.x*uCam.x,-q.x*uCam.y,-q.y/uCam.w);}
float pixelOf(){return (uView.x/uCanvas.x)/uK;}
vec3 lightDir(){return normalize(vec3(-uCam.x,uCam.y,1.2));}
vec3 onFloor(vec2 vb,float h){vec2 q=(vb-uOrigin)/uK;float b=(q.y+h*uCam.w)/uCam.z;return vec3(q.x*uCam.x+b*uCam.y,-q.x*uCam.y+b*uCam.x,h);}
vec3 onWallY(vec2 vb,float y){vec2 q=(vb-uOrigin)/uK;float x=(q.x+y*uCam.y)/uCam.x;return vec3(x,y,((x*uCam.y+y*uCam.x)*uCam.z-q.y)/uCam.w);}
vec3 onWallX(vec2 vb,float x){vec2 q=(vb-uOrigin)/uK;float y=(x*uCam.x-q.x)/uCam.y;return vec3(x,y,((x*uCam.y+y*uCam.x)*uCam.z-q.y)/uCam.w);}
float hash12(vec2 p){vec3 q=fract(vec3(p.xyx)*.1031);q+=dot(q,q.yzx+33.33);return fract((q.x+q.y)*q.z);}
float hash13(vec3 p){p=fract(p*.1031);p+=dot(p,p.zyx+31.32);return fract((p.x+p.y)*p.z);}
float noise2(vec2 p){vec2 i=floor(p);vec2 f=fract(p);f=f*f*(3.-2.*f);return mix(mix(hash12(i),hash12(i+vec2(1.,0.)),f.x),mix(hash12(i+vec2(0.,1.)),hash12(i+vec2(1.,1.)),f.x),f.y);}
float noise3(vec3 p){vec3 i=floor(p);vec3 f=fract(p);f=f*f*(3.-2.*f);
  float a=mix(mix(hash13(i),hash13(i+vec3(1.,0.,0.)),f.x),mix(hash13(i+vec3(0.,1.,0.)),hash13(i+vec3(1.,1.,0.)),f.x),f.y);
  float b=mix(mix(hash13(i+vec3(0.,0.,1.)),hash13(i+vec3(1.,0.,1.)),f.x),mix(hash13(i+vec3(0.,1.,1.)),hash13(i+vec3(1.,1.,1.)),f.x),f.y);
  return mix(a,b,f.z);}
float fbm2(vec2 p){float s=0.;float a=.5;for(int i=0;i<4;i++){s+=a*noise2(p);p=p*2.07+vec2(17.1,9.2);a*=.5;}return s/.9375;}
float fbm3(vec3 p){float s=0.;float a=.5;for(int i=0;i<4;i++){s+=a*noise3(p);p=p*2.07+vec3(17.1,9.2,4.7);a*=.5;}return s/.9375;}
float window(float x,float a,float b,float soft){return smoothstep(a-soft,a+soft,x)*(1.-smoothstep(b-soft,b+soft,x));}
vec3 tonemap(vec3 c,float exposure){return 1.-exp(-c*exposure);}
`;

export const GLSL_TURN = `uniform vec3 uGroupX;
uniform vec3 uGroupY;
uniform vec3 uGroupZ;
uniform vec3 uGroupT;
uniform vec3 uCoverEdge[60];
uniform float uCoverCount[6];
vec3 toGroup(vec3 p){vec3 q=p-uGroupT;return vec3(dot(q,uGroupX),dot(q,uGroupY),dot(q,uGroupZ));}
vec3 toWorld(vec3 g){return uGroupT+g.x*uGroupX+g.y*uGroupY+g.z*uGroupZ;}
vec3 dirToGroup(vec3 d){return vec3(dot(d,uGroupX),dot(d,uGroupY),dot(d,uGroupZ));}
vec3 dirToWorld(vec3 d){return d.x*uGroupX+d.y*uGroupY+d.z*uGroupZ;}
float coverOf(vec2 vb,float soft){float c=0.;for(int p=0;p<6;p++){float n=uCoverCount[p];float inside=-1e4;for(int e=0;e<10;e++){if(float(e)>=n)break;vec3 h=uCoverEdge[p*10+e];inside=max(inside,h.x*vb.x+h.y*vb.y+h.z);}if(n>0.)c=max(c,1.-smoothstep(-soft,soft,inside));}return c;}
`;

export function glslPose(name, prefix = `u${name}`) {
  return `uniform vec3 ${prefix}X;uniform vec3 ${prefix}Y;uniform vec3 ${prefix}Z;uniform vec3 ${prefix}T;
vec3 to${name}(vec3 p){vec3 q=p-${prefix}T;return vec3(dot(q,${prefix}X),dot(q,${prefix}Y),dot(q,${prefix}Z));}
vec3 from${name}(vec3 g){return ${prefix}T+g.x*${prefix}X+g.y*${prefix}Y+g.z*${prefix}Z;}
vec3 dirTo${name}(vec3 d){return vec3(dot(d,${prefix}X),dot(d,${prefix}Y),dot(d,${prefix}Z));}
vec3 dirFrom${name}(vec3 d){return d.x*${prefix}X+d.y*${prefix}Y+d.z*${prefix}Z;}
`;
}

export function glCamera(projection) {
  const a = ((projection.azimuth ?? 45) * Math.PI) / 180;
  const e = ((projection.elevation ?? 30) * Math.PI) / 180;
  return {
    origin: [projection.origin[0], projection.origin[1]],
    k: projection.scale * Math.sqrt(1.6),
    cam: [Math.sin(a), Math.cos(a), Math.sin(e), Math.cos(e)],
  };
}

const VERTEX = "attribute vec2 p;void main(){gl_Position=vec4(p,0.,1.);}";

export function glLayer({ canvas, fragment, view, camera, uniforms = [], fixed = {}, extensions = [], maxRatio = 1.5, untransformed = false, onError = (message) => console.error(message) }) {
  const gl = canvas.getContext("webgl", { premultipliedAlpha: true, alpha: true, antialias: false, preserveDrawingBuffer: false });
  if (!gl) return null;
  const compile = (type, source) => {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (gl.getShaderParameter(shader, gl.COMPILE_STATUS)) return shader;
    onError(gl.getShaderInfoLog(shader));
    return null;
  };
  const vs = compile(gl.VERTEX_SHADER, VERTEX);
  const enabled = extensions.filter((name) => gl.getExtension(name.replace(/^GL_/, "")));
  const fs = compile(gl.FRAGMENT_SHADER, enabled.map((name) => `#extension ${name} : enable\n`).join("") + GLSL_ISO + fragment);
  if (!vs || !fs) return null;
  const program = gl.createProgram();
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    onError(gl.getProgramInfoLog(program));
    return null;
  }
  gl.useProgram(program);
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  const position = gl.getAttribLocation(program, "p");
  gl.enableVertexAttribArray(position);
  gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
  const where = {};
  for (const name of ["uCanvas", "uView", "uOrigin", "uK", "uCam", "uTime", ...uniforms, ...Object.keys(fixed)]) where[name] = gl.getUniformLocation(program, name);
  const types = {};
  const active = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
  for (let index = 0; index < active; index++) {
    const info = gl.getActiveUniform(program, index);
    if (info) types[info.name.replace(/\[0\]$/, "")] = info.type;
  }
  let painted = false;
  const fit = () => {
    const box = untransformed ? { width: canvas.clientWidth, height: canvas.clientHeight } : canvas.getBoundingClientRect();
    const ratio = Math.min(maxRatio, window.devicePixelRatio || 1);
    const width = Math.max(1, Math.round(box.width * ratio));
    const height = Math.max(1, Math.round(box.height * ratio));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
  };
  const set = (name, value) => {
    const at = where[name];
    if (at === undefined || at === null) return;
    if (typeof value === "number") gl.uniform1f(at, value);
    else if (value.length > 4) {
      const data = value instanceof Float32Array ? value : new Float32Array(value);
      const type = types[name];
      if (type === gl.FLOAT_VEC2) gl.uniform2fv(at, data);
      else if (type === gl.FLOAT_VEC3) gl.uniform3fv(at, data);
      else if (type === gl.FLOAT_VEC4) gl.uniform4fv(at, data);
      else gl.uniform1fv(at, data);
    }
    else if (value.length === 1) gl.uniform1f(at, value[0]);
    else if (value.length === 2) gl.uniform2f(at, value[0], value[1]);
    else if (value.length === 3) gl.uniform3f(at, value[0], value[1], value[2]);
    else if (value.length === 4) gl.uniform4f(at, value[0], value[1], value[2], value[3]);
  };
  for (const [name, value] of Object.entries(fixed)) set(name, value);
  return {
    gl,
    extensions: enabled,
    set(values = {}) {
      for (const [name, value] of Object.entries(values)) set(name, value);
    },
    draw(values = {}, time = 0) {
      fit();
      gl.viewport(0, 0, canvas.width, canvas.height);
      set("uCanvas", [canvas.width, canvas.height]);
      set("uView", view);
      set("uOrigin", camera.origin);
      set("uK", camera.k);
      set("uCam", camera.cam);
      set("uTime", time);
      for (const [name, value] of Object.entries(values)) set(name, value);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      painted = true;
    },
    clear() {
      if (!painted) return;
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      painted = false;
    },
  };
}

export const smooth = (a, b, x) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

export const burst = (age, rise, fall) => (age > 0 ? smooth(0, rise, age) * Math.exp(-Math.max(0, age - rise) / fall) : 0);

export function settle(state, target, dt, seconds, calm = false) {
  if (calm) {
    state.x = target;
    state.v = 0;
    return false;
  }
  const w = 1 / seconds;
  for (let index = 0; index < 4; index++) {
    const h = dt / 4;
    state.v += (w * w * (target - state.x) - 2 * w * state.v) * h;
    state.x += state.v * h;
  }
  if (Math.abs(target - state.x) < 1e-4 && Math.abs(state.v) < 1e-4) {
    state.x = target;
    state.v = 0;
    return false;
  }
  return true;
}
