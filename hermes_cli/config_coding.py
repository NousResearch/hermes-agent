"""Validation for the optional, profile-local desktop workspace preferences."""


def validate_coding_preferences(config: dict) -> None:
    desktop = config.get("desktop")
    if not isinstance(desktop, dict) or "coding" not in desktop:
        return
    coding = desktop["coding"]
    if not isinstance(coding, dict):
        raise ValueError("desktop.coding must be a mapping")
    if "show_controls" in coding and not isinstance(coding["show_controls"], bool):
        raise ValueError("desktop.coding.show_controls must be a boolean")
    if "default_checkout" in coding and coding["default_checkout"] not in ("worktree", "current"):
        raise ValueError("desktop.coding.default_checkout must be worktree or current")
