"""Skills configuration for Hermes Agent. `hermes skills` enters this module."""
from typing import List, Optional, Set

from hermes_cli.config import cfg_get, load_config, save_config
from hermes_cli.colors import Colors, color
from hermes_cli.platforms import PLATFORMS as _PLATFORMS

# {key: label} view of the messaging platforms (``PLATFORMS.items()`` / ``.get(key)`` below).
PLATFORMS = {k: info.label for k, info in _PLATFORMS.items() if k != "api_server"}


def _normalize_skill_names(values) -> Set[str]:
    """Config value -> set of skill names (mirrors ``agent.skill_utils._normalize_string_set``):
    ``None`` (YAML null) is empty and a bare scalar is a single-item list, NOT its characters.

    See #13026.
    """
    if values is None:
        return set()
    if isinstance(values, str):
        values = [values]
    try:
        return {str(v).strip() for v in values if str(v).strip()}
    except TypeError:
        return set()


def get_disabled_skills(config: dict, platform: Optional[str] = None) -> Set[str]:
    """Disabled skill names: the global list unioned with the platform list when given (globally
    disabled stays disabled everywhere; mirrors ``agent.skill_utils.get_disabled_skill_names``)."""
    skills_cfg = config.get("skills") or {}
    if not isinstance(skills_cfg, dict):
        return set()
    from agent.skill_utils import ESSENTIAL_SKILLS
    disabled = _normalize_skill_names(skills_cfg.get("disabled"))
    if platform is not None:
        platform_disabled = cfg_get(skills_cfg, "platform_disabled", platform)
        if platform_disabled is not None:
            disabled = disabled | _normalize_skill_names(platform_disabled)
    return disabled - ESSENTIAL_SKILLS


def save_disabled_skills(config: dict, disabled: Set[str], platform: Optional[str] = None):
    """Persist disabled skill names to config; essential skills (e.g. ``hermes-agent``) are
    silently dropped — they cannot be disabled from any surface."""
    from agent.skill_utils import ESSENTIAL_SKILLS
    disabled = set(disabled) - ESSENTIAL_SKILLS
    config.setdefault("skills", {})
    if platform is None:
        config["skills"]["disabled"] = sorted(disabled)
    else:
        config["skills"].setdefault("platform_disabled", {})
        config["skills"]["platform_disabled"][platform] = sorted(disabled)
    save_config(config)


def _list_all_skills() -> List[dict]:
    """Return all installed skills (ignoring disabled state)."""
    try:
        from tools.skills_tool import _find_all_skills
        return _find_all_skills(skip_disabled=True)
    except Exception:
        return []


def _get_categories(skills: List[dict]) -> List[str]:
    """Return sorted unique category names (None -> 'uncategorized')."""
    return sorted({s["category"] or "uncategorized" for s in skills})


def _select_platform() -> Optional[str]:
    """Ask which platform to configure; None means global."""
    options = [("global", "All platforms (global default)")] + list(PLATFORMS.items())
    print()
    print(color("  Configure skills for:", Colors.BOLD))
    for i, (key, label) in enumerate(options, 1):
        print(f"  {i}. {label}")
    print()
    try:
        raw = input(color("  Select [1]: ", Colors.YELLOW)).strip()
    except (KeyboardInterrupt, EOFError):
        return None
    try:
        idx = int(raw) - 1  # empty input -> ValueError -> global
    except ValueError:
        return None
    if 0 <= idx < len(options) and options[idx][0] != "global":
        return options[idx][0]
    return None


def _toggle_by_category(skills: List[dict], disabled: Set[str]) -> Set[str]:
    """Toggle all skills in a category at once."""
    from hermes_cli.curses_ui import curses_checklist
    categories = _get_categories(skills)
    cat_skills = [{s["name"] for s in skills if (s["category"] or "uncategorized") == cat}
                  for cat in categories]
    cat_labels = [f"{cat} ({len(names)} skills)" for cat, names in zip(categories, cat_skills)]
    # A category is "enabled" (checked) when NOT all its skills are disabled
    pre_selected = {i for i, names in enumerate(cat_skills)
                    if not all(s in disabled for s in names)}
    chosen = curses_checklist("Categories — toggle entire categories",
                              cat_labels, pre_selected, cancel_returns=pre_selected)
    new_disabled = set(disabled)
    for i, names in enumerate(cat_skills):
        if i in chosen:
            new_disabled -= names  # category enabled → remove from disabled
        else:
            new_disabled |= names  # category disabled → add to disabled
    return new_disabled


def skills_command(args=None):
    """Entry point for `hermes skills`."""
    from hermes_cli.curses_ui import curses_checklist
    config = load_config()
    skills = _list_all_skills()
    if not skills:
        print(color("  No skills installed.", Colors.DIM))
        return

    platform = _select_platform()
    platform_label = PLATFORMS.get(platform, "All platforms") if platform else "All platforms"
    print()
    print(color(f"  Configure for: {platform_label}", Colors.DIM))
    print()
    print("  1. Toggle individual skills")
    print("  2. Toggle by category")
    print()
    try:
        mode = input(color("  Select [1]: ", Colors.YELLOW)).strip() or "1"
    except (KeyboardInterrupt, EOFError):
        return

    disabled = get_disabled_skills(config, platform)
    if mode == "2":
        new_disabled = _toggle_by_category(skills, disabled)
    else:
        labels = [f"{s['name']}  ({s['category'] or 'uncategorized'})  —  {s['description'][:55]}"
                  for s in skills]
        # "selected" = enabled (not disabled) — matches the [✓] convention
        pre_selected = {i for i, s in enumerate(skills) if s["name"] not in disabled}
        chosen = curses_checklist(f"Skills for {platform_label}",
                                  labels, pre_selected, cancel_returns=pre_selected)
        new_disabled = {skills[i]["name"] for i in range(len(skills)) if i not in chosen}

    if new_disabled == disabled:
        print(color("  No changes.", Colors.DIM))
        return

    save_disabled_skills(config, new_disabled, platform)
    enabled_count = len(skills) - len(new_disabled)
    print(color(f"✓ Saved: {enabled_count} enabled, {len(new_disabled)} disabled ({platform_label}).", Colors.GREEN))
