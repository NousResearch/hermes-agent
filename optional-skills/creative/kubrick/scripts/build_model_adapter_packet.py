#!/usr/bin/env python3
"""Build provider-neutral model adapter packets from Kubrick graph/storyboard state."""
from __future__ import annotations
import argparse, json
from pathlib import Path
try:
    import yaml
except ImportError:
    raise SystemExit("pyyaml required")


def load(path: str) -> dict:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.suffix == ".json" else yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def save(path: str, data: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def base_prompt(frame: dict, shared: list[str]) -> str:
    parts = list(shared)
    for field in ("node_states", "motif_ownership", "object_states", "light_states", "material_states"):
        for key, value in sorted((frame.get(field) or {}).items()):
            parts.append(f"{field[:-1].replace('_', ' ')} {key}: {value}")
    parts.extend(f"residue persists: {item}" for item in frame.get("residue", []))
    parts.extend(f"convergence remains observable at {item}" for item in frame.get("convergence_sites", []))
    return "; ".join(str(item).strip() for item in parts if str(item).strip())


def build(graph: dict, storyboard: dict | None, provider: str) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    if graph.get("validation", {}).get("status") != "VALID":
        errors.append("source graph is not VALID")
    graph_id = graph.get("graph_id")
    if not graph_id:
        errors.append("source graph_id missing")
    shared = list(graph.get("surface_output", {}).get("geometry", []))
    shared += list(graph.get("surface_output", {}).get("light", []))
    shared += list(graph.get("surface_output", {}).get("material", []))
    frames = []
    source_frames = (storyboard or {}).get("frames") or [{
        "frame_id": "frame-001",
        "node_states": {n.get("id"): n.get("target_state") for n in graph.get("nodes", []) if n.get("id")},
        "motif_ownership": {},
        "object_states": {},
        "light_states": {},
        "material_states": {},
        "residue": graph.get("residue", []),
        "convergence_sites": [s.get("site_id") for s in graph.get("convergence_sites", []) if s.get("site_id")],
        "prohibited_resets": [],
    }]
    previous = None
    for frame in source_frames:
        continuity = []
        if previous:
            continuity.append(f"preserve graph identity {graph_id}")
            for field in ("motif_ownership", "object_states", "light_states", "material_states"):
                for key, value in sorted((previous.get(field) or {}).items()):
                    continuity.append(f"preserve prior {field[:-1].replace('_', ' ')} {key}: {value}")
            continuity.extend(f"do not remove residue: {item}" for item in previous.get("residue", []))
        frames.append({
            "frame_id": frame.get("frame_id", f"frame-{len(frames)+1:03d}"),
            "prompt": base_prompt(frame, shared),
            "state_constraints": [f"node {k}: {v}" for k, v in sorted((frame.get("node_states") or {}).items())],
            "continuity_from_previous": continuity,
            "provider_parameters": {},
        })
        previous = frame
    if not frames:
        errors.append("no frames available for adaptation")
    return {
        "schema_version": "1.0.0",
        "adapter_id": f"kubrick-{provider}-adapter-v1",
        "provider": provider,
        "source_graph_id": graph_id or "unknown",
        "mode": "storyboard" if storyboard else "single-frame",
        "shared_constraints": {
            "visual_identity": shared,
            "continuity_rules": ["preserve graph identity", "preserve residue", "preserve ownership until explicitly changed"],
            "prohibited_resets": sorted({item for frame in source_frames for item in frame.get("prohibited_resets", [])}),
            "negative_constraints": ["no named esoterica", "no unexplained motif reset", "no continuity-breaking ownership change"],
        },
        "frames": frames,
        "private_state_policy": {
            "graph_mutation_allowed": False,
            "pattern_links_exposed": False,
            "lexicon_links_exposed": False,
        },
        "validation": {"status": "VALID" if not errors else "INVALID", "errors": errors, "warnings": warnings},
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--graph", required=True)
    p.add_argument("--storyboard")
    p.add_argument("--provider", default="generic")
    p.add_argument("--output")
    args = p.parse_args()
    packet = build(load(args.graph), load(args.storyboard) if args.storyboard else None, args.provider)
    rendered = yaml.safe_dump(packet, sort_keys=False)
    if args.output:
        save(args.output, packet)
    else:
        print(rendered)
    raise SystemExit(0 if packet["validation"]["status"] == "VALID" else 1)


if __name__ == "__main__":
    main()
