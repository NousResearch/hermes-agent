"""Skills configuration for Hermes Agent. `hermes skills` enters this module."""
from typing import Iterable, List, Optional, Set

from hermes_cli.config import load_config, save_config
from hermes_cli.colors import Colors, color
from hermes_cli.platforms import PLATFORMS as _PLATFORMS

# {key: label} view of the messaging platforms (``PLATFORMS.items()`` / ``.get(key)`` below).
PLATFORMS = {k: info.label for k, info in _PLATFORMS.items() if k != "api_server"}


def get_disabled_skills(config: dict, platform: Optional[str] = None) -> Set[str]:
    """Disabled skill names: the global list unioned with the platform list when given (globally
    disabled stays disabled everywhere) — parsed by the same reader the agent uses."""
    from agent.skill_utils import disabled_skill_names_from
    return disabled_skill_names_from(config.get("skills"), platform)


def allowlist_hidden_skills(config: dict, names: Iterable[str], platform: Optional[str] = None) -> Set[str]:
    """Of *names*, those ``skills.enabled`` / ``platform_enabled`` or an ``external_dirs`` filter hides.
    A ``skills.disabled`` toggle cannot change them, so a UI must neither persist them as disabled
    (that would outlive a later allowlist edit) nor report them as enabled."""
    from agent.skill_utils import skill_visibility_from
    skills_cfg = config.get("skills") if isinstance(config.get("skills"), dict) else {}
    rules = skill_visibility_from({**skills_cfg, "disabled": [], "platform_disabled": {}}, platform)
    return {name for name in names if rules.hides(name)}


def managed_locked_skills(names: Iterable[str], platform: Optional[str] = None) -> Set[str]:
    """Of *names*, those the administrator's managed scope decides: hidden by its pinned ``skills``
    rules, or all of them when the list a toggle would write is itself pinned. UIs show them locked
    and never write them — ``save_config`` would drop the write, and the pin wins at load anyway."""
    from agent.skill_utils import skill_visibility_from
    from hermes_cli import managed_scope
    names = set(names)
    target = "skills.disabled" if platform is None else f"skills.platform_disabled.{platform}"
    if managed_scope.is_key_managed(target):
        return names
    pinned = skill_visibility_from(managed_scope.load_managed_config().get("skills"), platform)
    return {name for name in names if pinned.hidden_reason(name) in ("disabled", "not_enabled")}


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


def _toggle_by_category(skills: List[dict], visible: Set[str]) -> Set[str]:
    """Toggle all skills in a category at once; returns the names left on."""
    from hermes_cli.curses_ui import curses_checklist
    categories = _get_categories(skills)
    cat_skills = [{s["name"] for s in skills if (s["category"] or "uncategorized") == cat}
                  for cat in categories]
    cat_labels = [f"{cat} ({len(names)} skills)" for cat, names in zip(categories, cat_skills)]
    # A category is "enabled" (checked) while any of its skills is visible
    pre_selected = {i for i, names in enumerate(cat_skills) if names & visible}
    chosen = curses_checklist("Categories — toggle entire categories",
                              cat_labels, pre_selected, cancel_returns=pre_selected)
    return set().union(*(names for i, names in enumerate(cat_skills) if i in chosen))


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

    from agent.skill_utils import skill_visibility_from
    disabled = get_disabled_skills(config, platform)
    visibility = skill_visibility_from(config.get("skills"), platform)
    visible = {s["name"] for s in skills if not visibility.hides(s["name"])}
    locked = managed_locked_skills((s["name"] for s in skills), platform)
    if mode == "2":
        chosen_on = _toggle_by_category(skills, visible)
    else:
        labels = [f"{s['name']}  ({s['category'] or 'uncategorized'})  —  {s['description'][:55]}"
                  + ("  [locked by administrator]" if s["name"] in locked else "") for s in skills]
        # "selected" = visible — matches the [✓] convention
        pre_selected = {i for i, s in enumerate(skills) if s["name"] in visible}
        chosen = curses_checklist(f"Skills for {platform_label}",
                                  labels, pre_selected, cancel_returns=pre_selected)
        chosen_on = {skills[i]["name"] for i in chosen}
    # Managed-scope skills keep their state: the administrator's pin wins over any write.
    if touched := sorted((chosen_on ^ visible) & locked):
        print(color(f"  Locked by your administrator (managed scope), left unchanged: {', '.join(touched)}",
                    Colors.YELLOW))
    turned_on = (chosen_on - locked) | (visible & locked)

    # Only visible skills the user unchecked join skills.disabled: one an allowlist already hides is
    # not the user's toggle, and persisting it would outlive a later edit of the allowlist.
    new_disabled = (disabled - turned_on) | (visible - turned_on)
    held_back = sorted(allowlist_hidden_skills(config, turned_on, platform))
    if held_back:
        print(color(f"  Still hidden by skills.enabled / platform_enabled or an external_dirs filter "
                    f"(edit config.yaml): {', '.join(held_back)}", Colors.YELLOW))
    if new_disabled == disabled:
        print(color("  No changes.", Colors.DIM))
        return

    save_disabled_skills(config, new_disabled, platform)
    enabled_count = len(turned_on) - len(held_back)
    print(color(f"✓ Saved: {enabled_count} enabled, {len(skills) - enabled_count} disabled ({platform_label}).",
                Colors.GREEN))
