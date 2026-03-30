---
name: anima-compiler
description: Deploy ROS2 robotics pipelines using the ANIMA Intelligence Compiler. Compile natural language tasks into validated, GPU-optimized Docker pipelines with 96 AI modules for perception, planning, and control.
version: 1.0.0
author: RobotFlow Labs
license: MIT
metadata:
  hermes:
    tags: [Robotics, ROS2, ANIMA, AI, Perception, Planning, Control, Docker, GPU]
    homepage: https://github.com/RobotFlow-Labs/anima-infra
prerequisites:
  env_vars: []
  commands: []
---

# ANIMA Intelligence Compiler

Compile natural language robotics tasks into validated, deployable ROS2 pipelines.

## What is ANIMA?

ANIMA is an offline robotics intelligence compiler. You describe what you want a robot to do in natural language, and ANIMA:

1. **Translates** your intent into a structured task specification
2. **Validates** through 3 gates (schema, semantic, resolution)
3. **Solves** module selection under constraints (VRAM budget, platform compatibility, licenses)
4. **Generates** deployment artifacts (docker-compose, ROS2 launch, BehaviorTree XML)
5. **Deploys** to GPU-pinned Docker containers with real model inference
6. **Monitors** health continuously with automatic e-stop on safety violations

## Setup

### Option A: ANIMA Compiler API (recommended)

Start the ANIMA compiler server:

```bash
cd anima-infra/anima-agent-ros2
set -a; source .env; set +a
npx tsx src/server/compiler-api.ts
# Listening on http://localhost:3000
```

Set in your `~/.hermes/.env`:

```bash
ANIMA_COMPILER_URL=http://localhost:3000
# Optional: ANIMA_COMPILER_TOKEN=your-token
```

### Option B: ANIMA CLI

```bash
pip install anima-compiler
# CLI automatically connects to local server or runs standalone
```

### Option C: Remote Server

```bash
# Point to your ANIMA server
ANIMA_COMPILER_URL=https://your-anima-server.com:3000
ANIMA_COMPILER_TOKEN=your-bearer-token
```

## When to Use

- "Set up a robot to detect objects and pick up the red ones"
- "Create a perception pipeline with depth estimation and segmentation"
- "Deploy a navigation stack that avoids obstacles"
- "What modules are available for manipulation planning?"
- "Check if my robot pipeline is healthy"
- "Emergency stop the robot NOW"
- "Show me all perception modules in the registry"

## Available Tools

### `anima_compile` — Compile a task into a pipeline

```
Task: "detect objects on the table and estimate their depth for grasping"
Platform: x86_64
VRAM Budget: 23000 MB
```

Returns: docker-compose.yaml, launch.py, pipeline_bt.xml, safety.yaml

### `anima_validate` — Check a task spec before compiling

Validates through 3 gates:
- Gate 1 (Schema): Required fields, types, structure
- Gate 2 (Semantic): Capabilities exist, VRAM fits, platform compatible
- Gate 3 (Resolution): No cycles, types match, all ports connected

### `anima_deploy` — Start the pipeline

Takes the generated docker-compose file and starts all containers with GPU pinning.

### `anima_status` — Monitor running modules

Returns per-module health: status, GPU VRAM, uptime, FPS, latency.

### `anima_stop` — Shut down the pipeline

- Graceful: `anima_stop()` — lets modules finish current inference
- Emergency: `anima_stop(emergency=true)` — kills everything instantly

### `anima_registry` — Browse 96 AI modules

Search by capability or name:
- `perception.vision` — object detection (AZOTH, BESTLA)
- `perception.depth` — depth estimation (GRID, CHRONOS)
- `perception.segmentation` — scene segmentation (MONAD)
- `perception.slam` — SLAM (PRISM, LOCI)
- `planning.manipulation` — grasp planning (DAEDALUS)
- `planning.navigation` — path planning (HERMES)
- `control.diffusion_policy` — learned control (CENTAUR)
- `foundation.world_model` — world understanding (SIBYL)
- `foundation.scene_flow` — 3D motion estimation (MORPHEUS)

## Example Workflow

```
User: "I need a robot that can see objects, estimate depth, and plan grasps"

1. anima_registry(capability="perception.vision")
   → AZOTH (open-vocab detection), BESTLA (YOLO detection)

2. anima_registry(capability="perception.depth")
   → GRID (monocular depth), CHRONOS (stereo depth)

3. anima_compile(task="detect objects, estimate depth, plan grasps for a tabletop scene")
   → Generates pipeline with BESTLA + GRID + DAEDALUS
   → docker-compose.pipeline.yaml
   → safety.yaml (VRAM limits, health monitoring)

4. anima_deploy(compose_file="docker-compose.pipeline.yaml", gpu_ids="0,1,2")
   → 3 containers running on GPUs 0, 1, 2

5. anima_status()
   → bestla: RUNNING (8081, 2.1GB VRAM)
   → grid: RUNNING (8084, 1.8GB VRAM)
   → daedalus: RUNNING (8083, 2.4GB VRAM)

6. anima_stop()
   → Graceful shutdown of all 3 containers
```

## Architecture

```
Natural Language Task
    ↓
Intent Translator (LLM → structured task_spec)
    ↓
3-Gate Validator (schema → semantic → resolution)
    ↓
CSP Constraint Solver (module selection under VRAM/platform/license constraints)
    ↓
DAG Builder (topological sort, type matching)
    ↓
Code Generators
    ├── docker-compose.yaml (GPU-pinned containers)
    ├── pipeline.launch.py (ROS2 launch)
    ├── pipeline_bt.xml (BehaviorTree)
    └── safety.yaml (kill conditions, watchdog)
    ↓
Docker Deploy (containers with real model inference)
    ↓
Safety Supervisor (1Hz health monitoring, auto e-stop)
```

## Module Categories

| Category | Count | Examples |
|----------|-------|---------|
| Perception | 25+ | Detection, depth, segmentation, SLAM, place recognition |
| Planning | 15+ | Manipulation, navigation, task planning |
| Control | 10+ | Diffusion policy, VLA, motor control |
| Foundation | 10+ | World models, scene flow, point clouds |
| Defense | 19 | Adversarial detection, robustness testing |
| Healthcare | 14 | Medical imaging analysis |

## Safety

ANIMA includes a safety supervisor that:
- Monitors all module health at 1Hz via ROS2 topics
- Kills pipelines if VRAM > 90%, latency > 2000ms, or heartbeat lost for 30s
- Provides emergency stop (kills all containers instantly)
- Validates pipelines BEFORE deployment (no unsafe configurations deploy)

## Tips

- Always check `anima_status` after deploying to confirm all modules are healthy
- Use `dry_run=true` with `anima_compile` to preview without generating files
- For safety-critical tasks, always validate before compiling
- Use emergency stop if the robot behaves unexpectedly — it's faster than graceful shutdown
- The registry has 96 modules — search by capability to find what you need
