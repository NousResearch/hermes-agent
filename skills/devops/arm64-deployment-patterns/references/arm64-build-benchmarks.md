# ARM64 Build Benchmarks

Real build times on Oracle Cloud Free Tier (Ampere A1, 4 ARM cores, 24 GB RAM).

## Docker Build vs Local Build

| Project | Docker Build | Local Build | Speedup |
|---------|-------------|-------------|---------|
| Next.js 16 (medium app) | 35-45 min | 2-3 min | 12x |
| Express + TypeScript | 8-12 min | 30 sec | 16x |
| FastAPI (Python) | 15-20 min | 1-2 min | 10x |
| hireme-agent (web) | 40+ min | ~2 min | 20x |

## Docker Build Phases on ARM64

1. `apt-get install` dependencies: 5-15 min (biggest bottleneck)
2. `npm install`: 3-8 min
3. `npm run build`: 5-15 min
4. Image layering + cleanup: 2-5 min

## Why Docker is Slow on ARM64

- Single-threaded `apt-get` on 4-core ARM
- Docker buildkit parallelism limited on ARM
- No pre-built ARM64 images for many packages
- Emulation overhead for x86-only dependencies

## Recommendation

For Oracle Cloud Free Tier ARM64:
- **Development**: Always build locally (`npm install && npm run build`)
- **Production**: Use multi-stage builds with cached base layers
- **CI**: Consider cross-compilation on x64 runners, push ARM64 images
