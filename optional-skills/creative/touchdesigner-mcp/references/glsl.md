# GLSL Reference

## Uniforms

```
TouchDesigner          GLSL
─────────────────────────────
vec0name = 'uTime'  →  uniform float uTime;
vec0valuex = 1.0    →  uTime value
```

### Pass Time

```python
glsl_op.par.vec0name = 'uTime'
glsl_op.par.vec0valuex.mode = ParMode.EXPRESSION
glsl_op.par.vec0valuex.expr = 'absTime.seconds'
```

```glsl
uniform float uTime;
void main() { float t = uTime * 0.5; }
```

### Built-in Uniforms (TOP)

```glsl
// Output resolution (always available)
vec2 res = uTDOutputInfo.res.zw;

// Input texture (only when inputs connected)
vec2 inputRes = uTD2DInfos[0].res.zw;
vec4 color = texture(sTD2DInputs[0], vUV.st);

// UV coordinates
vUV.st  // 0-1 texture coords
```

**IMPORTANT:** `uTD2DInfos` requires input textures. For standalone shaders use `uTDOutputInfo`.

## Built-in Utility Functions

```glsl
// Noise
float TDPerlinNoise(vec2/vec3/vec4 v);
float TDSimplexNoise(vec2/vec3/vec4 v);

// Color conversion
vec3 TDHSVToRGB(vec3 c);
vec3 TDRGBToHSV(vec3 c);

// Matrix transforms
mat4 TDTranslate(float x, float y, float z);
mat3 TDRotateX/Y/Z(float radians);
mat3 TDRotateOnAxis(float radians, vec3 axis);
mat3 TDScale(float x, float y, float z);
mat3 TDRotateToVector(vec3 forward, vec3 up);
mat3 TDCreateRotMatrix(vec3 from, vec3 to);  // vectors must be normalized

// Resolution struct
struct TDTexInfo {
  vec4 res;   // (1/width, 1/height, width, height)
  vec4 depth;
};

// Output (always use this — handles sRGB correctly)
fragColor = TDOutputSwizzle(color);

// Instancing (MAT only)
int TDInstanceID();
```

## glslTOP

Docked DATs created automatically:
- `glsl1_pixel` — Pixel shader
- `glsl1_compute` — Compute shader
- `glsl1_info` — Compile info

### Pixel Shader Template

```glsl
out vec4 fragColor;
void main() {
    vec4 color = texture(sTD2DInputs[0], vUV.st);
    fragColor = TDOutputSwizzle(color);
}
```

### Compute Shader Template

```glsl
layout (local_size_x = 8, local_size_y = 8) in;
void main() {
    vec4 color = texelFetch(sTD2DInputs[0], ivec2(gl_GlobalInvocationID.xy), 0);
    TDImageStoreOutput(0, gl_GlobalInvocationID, color);
}
```

### Update Shader

```python
op('/project1/glsl1_pixel').text = shader_code
op('/project1/glsl1').cook(force=True)
# Check errors:
print(op('/project1/glsl1_info').text)
```

## glslMAT

Docked DATs:
- `glslmat1_vertex` — Vertex shader (param: `vdat`)
- `glslmat1_pixel` — Pixel shader (param: `pdat`)
- `glslmat1_info` — Compile info

Note: MAT uses `vdat`/`pdat`, TOP uses `vertexdat`/`pixeldat`.

### Vertex Shader Template

```glsl
uniform float uTime;
void main() {
    vec3 pos = TDPos();
    pos.z += sin(pos.x * 3.0 + uTime) * 0.2;
    vec4 worldSpacePos = TDDeform(pos);
    gl_Position = TDWorldToProj(worldSpacePos);
}
```

## Bayer 8x8 Dither Matrix

Reusable ordered dither function for retro/print aesthetics:

```glsl
float bayer8(vec2 pos) {
    int x = int(mod(pos.x, 8.0)), y = int(mod(pos.y, 8.0)), idx = x + y * 8;
    int b[64] = int[64](
        0,32,8,40,2,34,10,42,48,16,56,24,50,18,58,26,
        12,44,4,36,14,46,6,38,60,28,52,20,62,30,54,22,
        3,35,11,43,1,33,9,41,51,19,59,27,49,17,57,25,
        15,47,7,39,13,45,5,37,63,31,55,23,61,29,53,21
    );
    return float(b[idx]) / 64.0;
}
```

## glslPOP / glsladvancedPOP / glslcopyPOP

All use compute shaders. Docked DATs follow naming convention:
- `glsl1_compute` / `glsladv1_compute`
- `glslcopy1_ptCompute` / `glslcopy1_vertCompute` / `glslcopy1_primCompute`
