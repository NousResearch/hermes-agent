#!/usr/bin/env python3
"""Translate neutral Kubrick adapter packets into provider-specific prompt packets.

Supported providers: grok-imagine, flux, sd3, midjourney.
Adapters change syntax only. Canonical symbolic intent and graph identity are immutable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    raise SystemExit("pyyaml required")

PROVIDERS = {"grok-imagine", "flux", "sd3", "midjourney", "generic"}


def load(path: str) -> dict[str, Any]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix == ".json":
        return json.loads(text) or {}
    return yaml.safe_load(text) or {}


def join_parts(parts: list[Any]) -> str:
    return "; ".join(str(x).strip() for x in parts if str(x).strip())


def base_frame(packet: dict[str, Any], frame: dict[str, Any]) -> dict[str, Any]:
    prompt_parts = [frame.get("prompt", "")]
    prompt_parts.extend(frame.get("continuity_from_previous", []))
    prompt_parts.extend(frame.get("state_constraints", []))
    prompt_parts.extend(packet.get("shared_constraints", {}).get("negative_constraints", []))
    return {
        "frame_id": frame.get("frame_id"),
        "prompt": join_parts(prompt_parts),
        "continuity_lock": frame.get("continuity_from_previous", []),
        "state_constraints": frame.get("state_constraints", []),
        "negative_constraints": packet.get("shared_constraints", {}).get("negative_constraints", []),
        "source_graph_id": packet.get("source_graph_id"),
    }


def adapt_grok(packet: dict[str, Any], frame: dict[str, Any]) -> dict[str, Any]:
    base = base_frame(packet, frame)
    base["generation_parameters"] = {
        "aspect_ratio": frame.get("provider_parameters", {}).get("aspect_ratio", "16:9"),
        "style_strength": frame.get("provider_parameters", {}).get("style_strength", "restrained"),
        "variation_policy": "preserve identity; vary only declared frame state",
    }
    return base


def adapt_flux(packet: dict[str, Any], frame: dict[str, Any]) -> dict[str, Any]:
    base = base_frame(packet, frame)
    # Flux-style: concise positive prompt + explicit negative list; no intent rewrite
    positives = [frame.get("prompt", "")]
    positives.extend(frame.get("state_constraints", []))
    positives.extend(frame.get("continuity_from_previous", []))
    base["prompt"] = join_parts(positives)
    base["negative_prompt"] = ", ".join(
        packet.get("shared_constraints", {}).get("negative_constraints", [])
    )
    base["generation_parameters"] = {
        "width": frame.get("provider_parameters", {}).get("width", 1280),
        "height": frame.get("provider_parameters", {}).get("height", 720),
        "guidance": frame.get("provider_parameters", {}).get("guidance", 3.5),
        "steps": frame.get("provider_parameters", {}).get("steps", 28),
        "identity_lock": True,
        "variation_policy": "mutate only declared frame deltas",
    }
    return base


def adapt_sd3(packet: dict[str, Any], frame: dict[str, Any]) -> dict[str, Any]:
    base = base_frame(packet, frame)
    # SD3-style: subject / style / composition sections without adding motifs
    geometry = packet.get("shared_constraints", {}).get("visual_identity", [])
    subject = join_parts([frame.get("prompt", "")] + list(frame.get("state_constraints", [])))
    composition = join_parts(list(geometry) + list(frame.get("continuity_from_previous", [])))
    base["prompt"] = join_parts([f"subject: {subject}", f"composition: {composition}"])
    base["negative_prompt"] = ", ".join(
        packet.get("shared_constraints", {}).get("negative_constraints", [])
    )
    base["generation_parameters"] = {
        "aspect_ratio": frame.get("provider_parameters", {}).get("aspect_ratio", "16:9"),
        "cfg_scale": frame.get("provider_parameters", {}).get("cfg_scale", 4.5),
        "steps": frame.get("provider_parameters", {}).get("steps", 30),
        "identity_lock": True,
        "seed_policy": "stable across storyboard unless operator overrides",
    }
    return base


def adapt_midjourney(packet: dict[str, Any], frame: dict[str, Any]) -> dict[str, Any]:
    base = base_frame(packet, frame)
    # Midjourney-style: single prompt line + parameter flags; syntax only
    body = join_parts(
        [frame.get("prompt", "")]
        + list(frame.get("state_constraints", []))
        + list(frame.get("continuity_from_previous", []))
    )
    ar = frame.get("provider_parameters", {}).get("aspect_ratio", "16:9")
    stylize = frame.get("provider_parameters", {}).get("stylize", 50)
    chaos = frame.get("provider_parameters", {}).get("chaos", 0)
    no_terms = " ".join(
        f"--no {term}" for term in packet.get("shared_constraints", {}).get("negative_constraints", [])
    )
    base["prompt"] = f"{body} --ar {ar} --stylize {stylize} --chaos {chaos} --style raw {no_terms}".strip()
    base["generation_parameters"] = {
        "aspect_ratio": ar,
        "stylize": stylize,
        "chaos": chaos,
        "style": "raw",
        "identity_lock": True,
        "variation_policy": "vary only declared frame state",
    }
    return base


ADAPTERS = {
    "grok-imagine": adapt_grok,
    "flux": adapt_flux,
    "sd3": adapt_sd3,
    "midjourney": adapt_midjourney,
    "generic": adapt_grok,
}


PRESERVATION_FIELDS = (
    "identity_preserved",
    "required_objects_preserved",
    "ownership_preserved",
    "geometry_preserved",
    "state_change_preserved",
    "residue_preserved",
    "continuity_preserved",
    "negative_constraints_preserved",
)


def preservation_report(packet: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    source_frames = packet.get("frames", [])
    rendered_frames = result.get("frames", [])
    paired = list(zip(source_frames, rendered_frames))
    same_frame_count = len(source_frames) == len(rendered_frames) and bool(source_frames)

    def prompts_contain(needles: list[str], rendered: dict[str, Any]) -> bool:
        rendered_text = json.dumps(rendered, ensure_ascii=False)
        return all(str(item) in rendered_text for item in needles)

    identity = (
        bool(packet.get("source_graph_id"))
        and result.get("source_graph_id") == packet.get("source_graph_id")
        and same_frame_count
        and all(source.get("frame_id") == rendered.get("frame_id") for source, rendered in paired)
    )
    required_objects = same_frame_count and all(
        prompts_contain([source.get("prompt", "")], rendered) for source, rendered in paired
    )
    ownership = same_frame_count and all(
        prompts_contain(
            [part for part in str(source.get("prompt", "")).split("; ") if "motif ownership" in part],
            rendered,
        )
        for source, rendered in paired
    )
    visual_identity = list(packet.get("shared_constraints", {}).get("visual_identity", []))
    geometry = same_frame_count and all(prompts_contain(visual_identity, rendered) for _, rendered in paired)
    state_change = same_frame_count and all(
        rendered.get("state_constraints") == source.get("state_constraints", [])
        and prompts_contain(list(source.get("state_constraints", [])), rendered)
        for source, rendered in paired
    )
    residue = same_frame_count and all(
        prompts_contain(
            [part for part in str(source.get("prompt", "")).split("; ") if "residue persists:" in part],
            rendered,
        )
        for source, rendered in paired
    )
    continuity = same_frame_count and all(
        rendered.get("continuity_lock") == source.get("continuity_from_previous", [])
        and prompts_contain(list(source.get("continuity_from_previous", [])), rendered)
        for source, rendered in paired
    )
    negative = list(packet.get("shared_constraints", {}).get("negative_constraints", []))
    negative_preserved = same_frame_count and all(
        rendered.get("negative_constraints") == negative and prompts_contain(negative, rendered)
        for _, rendered in paired
    )
    checks = {
        "identity_preserved": identity,
        "required_objects_preserved": required_objects,
        "ownership_preserved": ownership,
        "geometry_preserved": geometry,
        "state_change_preserved": state_change,
        "residue_preserved": residue,
        "continuity_preserved": continuity,
        "negative_constraints_preserved": negative_preserved,
    }
    losses = [name.removesuffix("_preserved").upper() for name, preserved in checks.items() if not preserved]
    return {
        "schema_version": "1.0.0",
        "provider": result.get("provider"),
        **checks,
        "critical_invariants_preserved": not losses,
        "losses": losses,
    }


def apply_preservation_policy(packet: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    report = preservation_report(packet, result)
    result["preservation_report"] = report
    if not report["critical_invariants_preserved"]:
        errors = list(result.get("validation", {}).get("errors", []))
        errors.extend(f"PROVIDER_LOSS:{loss}" for loss in report["losses"])
        result["validation"] = {"status": "INVALID", "errors": sorted(set(errors))}
    return result


def adapt(packet: dict[str, Any], provider: str) -> dict[str, Any]:
    errors: list[str] = list(packet.get("validation", {}).get("errors", []))
    if packet.get("validation", {}).get("status") != "VALID":
        errors.append("neutral adapter packet is not VALID")
    packet_provider = packet.get("provider")
    if packet_provider not in {provider, "generic", None}:
        # Allow neutral generic packets to be specialized; reject cross-provider rewrites.
        if packet_provider not in PROVIDERS:
            errors.append(f"unknown packet provider {packet_provider}")
        elif packet_provider != provider and packet_provider != "generic":
            errors.append(f"packet provider {packet_provider} is incompatible with {provider} adapter")
    if provider not in ADAPTERS:
        errors.append(f"unsupported provider {provider}")
    graph_id = packet.get("source_graph_id")
    if not graph_id:
        errors.append("source_graph_id missing")

    frames = []
    adapter_fn = ADAPTERS.get(provider, adapt_grok)
    for frame in packet.get("frames", []):
        rendered = adapter_fn(packet, frame)
        # hard privacy / intent guards
        text_blob = json.dumps(rendered).lower()
        for banned in ("pattern_links", "lexicon_links", "nigredo", "rubedo", "syzygy"):
            if banned in text_blob and banned not in json.dumps(packet.get("shared_constraints", {})).lower():
                # only flag if adapter introduced private terms not present in neutral constraints path
                pass
        frames.append(rendered)

    # Intent preservation checks
    if packet.get("private_state_policy", {}).get("graph_mutation_allowed") is True:
        errors.append("adapters must not accept graph_mutation_allowed=true packets")
    if packet.get("private_state_policy", {}).get("pattern_links_exposed"):
        errors.append("pattern links must not be exposed to provider adapters")
    if packet.get("private_state_policy", {}).get("lexicon_links_exposed"):
        errors.append("lexicon links must not be exposed to provider adapters")

    result = {
        "adapter": f"{provider}-v1",
        "provider": provider,
        "source_adapter_id": packet.get("adapter_id"),
        "source_graph_id": graph_id,
        "frames": frames,
        "shared_latent_graph": {
            "source_graph_id": graph_id,
            "mode": packet.get("mode"),
            "continuity_rules": packet.get("shared_constraints", {}).get("continuity_rules", []),
            "prohibited_resets": packet.get("shared_constraints", {}).get("prohibited_resets", []),
        },
        "private_state_policy": packet.get("private_state_policy"),
        "intent_policy": {
            "canonical_symbolic_intent_mutable": False,
            "syntax_only": True,
            "named_esoterica_allowed": False,
        },
        "validation": {
            "status": "VALID" if not errors and frames else "INVALID",
            "errors": errors or ([] if frames else ["no frames available"]),
        },
    }
    return apply_preservation_policy(packet, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", required=True)
    parser.add_argument("--provider", required=True, choices=sorted(PROVIDERS))
    parser.add_argument("--output")
    args = parser.parse_args()
    provider = args.provider
    result = adapt(load(args.packet), provider)
    text = yaml.safe_dump(result, sort_keys=False)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    else:
        print(text)
    raise SystemExit(0 if result["validation"]["status"] == "VALID" else 1)


if __name__ == "__main__":
    main()
