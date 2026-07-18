"""Sol food transport plugin (owner-private; Health owns the commit)."""

from __future__ import annotations

import os
from pathlib import Path


_PARSER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["candidates"],
    "properties": {
        "candidates": {
            "type": "array",
            "minItems": 1,
            "maxItems": 4,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["label", "items"],
                "properties": {
                    "label": {"type": "string", "minLength": 1, "maxLength": 120},
                    "items": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 24,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["plant_key", "is_plant"],
                            "properties": {
                                "plant_key": {"type": "string", "minLength": 1},
                                "is_plant": {"type": "boolean"},
                            },
                        },
                    },
                },
            },
        }
    },
}


def _parser_for(llm):
    """Build the bounded host-model parser; no provider/profile override."""
    from agent.plugin_llm import PluginLlmImageInput, PluginLlmTextInput
    from plugins.sol_food.proposal import Candidate, validate_candidates

    async def _parse(text, image_path):
        inputs = []
        if text:
            inputs.append(PluginLlmTextInput(text=text))
        if image_path is not None:
            image_bytes = image_path.read_bytes()
            if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
                mime = "image/png"
            elif image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
                mime = "image/webp"
            else:
                mime = "image/jpeg"
            inputs.append(
                PluginLlmImageInput(data=image_bytes, mime_type=mime)
            )
        result = await llm.acomplete_structured(
            instructions=(
                "Extract up to four plausible meal candidates. Use short normalized "
                "plant_key identifiers and mark whether each item is a plant. Do not "
                "add commentary or infer health advice."
            ),
            input=inputs,
            json_schema=_PARSER_SCHEMA,
            schema_name="sol_food_candidates_v1",
            max_tokens=1200,
            purpose="sol_food_candidate_parse",
        )
        data = result.parsed
        if not isinstance(data, dict):
            raise ValueError("sol_food_parser_invalid")
        candidates = [
            Candidate(
                label=entry["label"],
                items=tuple(dict(item) for item in entry["items"]),
            )
            for entry in data.get("candidates", [])
        ]
        validate_candidates(candidates)
        return candidates

    return _parse


def _build_topic_hook(llm=None, *, profile: str = "sol"):
    """Build the Sol hook only after explicit strict-route activation.

    Plugin discovery itself remains inert: no credential is read and no state
    directory is created until ``topic_routing.hooks`` contains ``sol``.
    """
    from hermes_cli.profiles import get_profile_dir
    from hermes_constants import get_default_hermes_root
    from plugins.sol_food.health_client import HealthClientError, HealthFoodClient
    from plugins.sol_food.hook import SolFoodHook

    endpoint = os.getenv("HEALTH_FOOD_COMMIT_URL", "")
    token = os.getenv("HEALTH_FOOD_COMMIT_TOKEN", "")
    if not endpoint or not token:
        raise RuntimeError("sol_food_configuration_missing")
    try:
        health_client = HealthFoodClient(endpoint, token)
    except HealthClientError as exc:
        # Stable reason code only; never echo the endpoint or credential.
        raise RuntimeError(exc.reason_code) from None
    # The hook factory is instantiated by the default multiplex gateway, not
    # inside the routed profile's process scope. Resolve its state and legacy
    # single-writer guard from the explicitly registered route profile.
    profile_home = Path(get_profile_dir(profile))
    default_home = Path(get_default_hermes_root())
    return SolFoodHook(
        state_dir=profile_home / "state" / "sol-food",
        hermes_home=default_home,
        additional_legacy_guard_homes=(profile_home,),
        health_client=health_client,
        parser=_parser_for(llm) if llm is not None else None,
    )


def register(ctx) -> None:
    """Advertise the Sol hook through the generic lazy factory seam."""
    # Capture the context, not the facade: discovery remains inert and the
    # host-owned LLM is constructed only when strict config activates Sol.
    ctx.register_topic_hook_factory(
        "sol", lambda: _build_topic_hook(ctx.llm, profile="sol")
    )
