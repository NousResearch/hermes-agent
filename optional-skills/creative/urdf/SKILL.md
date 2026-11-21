---
name: urdf
description: "URDF robot description authoring, upstream-maintained."
version: 0.7.20
author: Thompson Labs LLC (earthtojake)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [urdf, robotics, ros, kinematics, robot-description, cad]
    category: creative
    related_skills: [cad]
    upstream:
      repo: earthtojake/text-to-cad
      path: skills/urdf
---

# URDF (upstream-maintained)

> **Catalog stub.** This entry is maintained upstream at
> [earthtojake/text-to-cad](https://github.com/earthtojake/text-to-cad): the
> project ships a self-contained skill directory (`skills/urdf/`) with an
> authoring contract and references for frame semantics, inertials, meshes,
> the design ledger and validation. `hermes skills install
> official/creative/urdf` pulls the current tree live from that repo
> (quarantined and scanned like any hub install). This directory holds only
> the catalog metadata.

The skill treats URDF as constrained kinematic modeling rather than XML
writing. The `.urdf` file is the source of truth; every file opens with a
design-ledger comment block (frames, joints, units, assumptions); joint,
link, visual, collision and inertial origins follow URDF frame semantics
exactly; inertia tensors and centres of mass are computed (closed-form or a
throwaway script), never freehanded; and every created or edited file must
pass `cadgen urdf validate` before completion. That validator reports the
robot's root, link and joint counts and total mass, and exits non-zero on
blocking findings such as a joint naming a missing child link.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) on `PATH`; commands run through the same
  pinned `uvx ... --from cadgen==<pin> cadgen` runtime as the `cad` skill.
- `cadgen` telemetry is on by default: `cadgen telemetry off` once, or
  `CADGEN_TELEMETRY=0` per command, unless the user opts in.
- Viewer: Hermes has no `cad_show` tool, so use
  `cadgen viewer --host 127.0.0.1 --json --detach` and return
  `url?file=<absolute path>`; headless sessions report validation output and
  `cadgen urdf snapshot` images instead.
- MoveIt2 semantics (SRDF) and Gazebo worlds (SDF) live in upstream sibling
  skills not yet in the Hermes catalog.

Pair with `cad` when links need STEP/STL meshes.

Full documentation: https://github.com/earthtojake/text-to-cad#readme
