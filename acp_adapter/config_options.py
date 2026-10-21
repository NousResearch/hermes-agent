"""ACP session config options: the model, reasoning-effort and edit-mode pickers.

ACP's ``configOptions`` (and editors following it, e.g. Zed) render the composer's model and
thought-level pickers from typed ``select`` options with a semantic ``category``; the unstable
``models`` field alone renders nothing there. The same choices keep riding ``models`` / ``modes``
for clients that read those. Every value round-trips: the ``model`` option's values are the
``provider:model`` picker ids ``session/set_model`` already accepts, the ``reasoning_effort``
values are the owner's ``session.mutate reasoning`` levels, and the ``edit_approval_policy``
values are the edit approval policies ``session/set_mode`` sets.
"""
from __future__ import annotations

from acp.schema import SessionConfigOptionSelect, SessionConfigSelectOption, SessionModelState

from acp_adapter.modes import EDIT_APPROVAL_POLICY_CONFIG_ID, MODE_TO_EDIT_APPROVAL_POLICY, MODES

MODEL_CONFIG_ID = "model"
REASONING_CONFIG_ID = "reasoning_effort"
# ``default`` hands the level back to the profile (``/reasoning reset``); ``none`` turns it off.
REASONING_DEFAULT = "default"


def reasoning_choices() -> tuple[str, ...]:
    from hermes_constants import VALID_REASONING_EFFORTS
    return (REASONING_DEFAULT, "none", *VALID_REASONING_EFFORTS)


def reasoning_payload(value) -> dict | None:
    """The owner's ``reasoning`` mutation payload for a picked value, or None when not a level."""
    if value == REASONING_DEFAULT:
        return {"effort": None}
    if isinstance(value, str) and value in reasoning_choices():
        return {"effort": value}
    return None


def _model_option(models: SessionModelState | None) -> SessionConfigOptionSelect | None:
    if models is None or not models.current_model_id:
        return None
    choices = [SessionConfigSelectOption(value=m.model_id, name=m.name, description=m.description)
               for m in models.available_models]
    if models.current_model_id not in {choice.value for choice in choices}:
        choices.insert(0, SessionConfigSelectOption(value=models.current_model_id, name=models.current_model_id))
    return SessionConfigOptionSelect(
        id=MODEL_CONFIG_ID, name="Model", category="model", type="select",
        description="The model this session's next turn runs on.",
        current_value=models.current_model_id, options=choices)


def _reasoning_option(effort: str | None, provider: str | None, model: str | None) -> SessionConfigOptionSelect:
    from agent.reasoning_effort import effort_display_label
    current = effort if effort in reasoning_choices() else REASONING_DEFAULT

    def name(value):
        if value == REASONING_DEFAULT:
            return "Default (profile)"
        if value == "none":
            return "Off"
        # A Hermes-internal step the route clamps reads "ultra (sends max on this route)", as /reasoning.
        return effort_display_label(value, provider, model)
    return SessionConfigOptionSelect(
        id=REASONING_CONFIG_ID, name="Reasoning effort", category="thought_level", type="select",
        description="How hard the model thinks before answering, for this session. Default follows the profile.",
        current_value=current,
        options=[SessionConfigSelectOption(value=v, name=name(v)) for v in reasoning_choices()])


def _mode_option(mode_id: str) -> SessionConfigOptionSelect:
    return SessionConfigOptionSelect(
        id=EDIT_APPROVAL_POLICY_CONFIG_ID, name="Edits", category="mode", type="select",
        description="When file edits ask before they are written.",
        current_value=MODE_TO_EDIT_APPROVAL_POLICY.get(mode_id, MODE_TO_EDIT_APPROVAL_POLICY["default"]),
        options=[SessionConfigSelectOption(value=policy, name=name, description=description)
                 for policy, name, description in MODES.values()])


def config_options(models: SessionModelState | None, info: dict, mode_id: str) -> list[SessionConfigOptionSelect]:
    """The session's full option list, in the order editors lay them out: mode, model, effort.

    *info* is the owner's ``session.info``: ``reasoning_effort`` is the level the next turn runs at
    (a session pin or the profile's); empty or unknown shows as ``default``."""
    options = [_mode_option(mode_id)]
    if (model := _model_option(models)) is not None:
        options.append(model)
    options.append(_reasoning_option(info.get("reasoning_effort"), info.get("provider"), info.get("model")))
    return options
