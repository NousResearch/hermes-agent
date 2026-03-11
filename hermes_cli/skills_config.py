"""
Skills configuration for Hermes Agent.
`hermes skills` enters this module.

Toggle individual skills or categories on/off, globally or per-platform.
Config stored in ~/.hermes/config.yaml under:

  skills:
    disabled: [skill-a, skill-b]          # global disabled list
    platform_disabled:                    # per-platform overrides
      telegram: [skill-c]
      cli: []
"""
from typing import Dict, List, Set, Optional
from hermes_cli.config import load_config, save_config
from hermes_cli.colors import Colors, color

PLATFORMS = {
    "cli":      "🖥️  CLI",
    "telegram": "📱 Telegram",
    "discord":  "💬 Discord",
    "slack":    "💼 Slack",
    "whatsapp": "📱 WhatsApp",
}

# ─── Config Helpers ───────────────────────────────────────────────────────────

def get_disabled_skills(config: dict, platform: Optional[str] = None) -> Set[str]:
    """Return disabled skill names. Platform-specific list falls back to global."""
    skills_cfg = config.get("skills", {})
    global_disabled = set(skills_cfg.get("disabled", []))
    if platform is None:
        return global_disabled
    platform_disabled = skills_cfg.get("platform_disabled", {}).get(platform)
    if platform_disabled is None:
        return global_disabled
    return set(platform_disabled)


def save_disabled_skills(config: dict, disabled: Set[str], platform: Optional[str] = None):
    """Persist disabled skill names to config."""
    config.setdefault("skills", {})
    if platform is None:
        config["skills"]["disabled"] = sorted(disabled)
    else:
        config["skills"].setdefault("platform_disabled", {})
        config["skills"]["platform_disabled"][platform] = sorted(disabled)
    save_config(config)


# ─── Skill Discovery ──────────────────────────────────────────────────────────

def _list_all_skills_unfiltered() -> List[dict]:
    """Return all installed skills ignoring disabled state."""
    try:
        from tools.skills_tool import SKILLS_DIR, _parse_frontmatter, skill_matches_platform, _get_category_from_path, MAX_NAME_LENGTH, MAX_DESCRIPTION_LENGTH
        skills = []
        if not SKILLS_DIR.exists():
            return skills
        for skill_md in SKILLS_DIR.rglob("SKILL.md"):
            if any(part in ('.git', '.github', '.hub') for part in skill_md.parts):
                continue
            skill_dir = skill_md.parent
            try:
                content = skill_md.read_text(encoding='utf-8')
                frontmatter, body = _parse_frontmatter(content)
                if not skill_matches_platform(frontmatter):
                    continue
                name = frontmatter.get('name', skill_dir.name)[:MAX_NAME_LENGTH]
                description = frontmatter.get('description', '')
                if not description:
                    for line in body.strip().split('\n'):
                        line = line.strip()
                        if line and not line.startswith('#'):
                            description = line
                            break
                if len(description) > MAX_DESCRIPTION_LENGTH:
                    description = description[:MAX_DESCRIPTION_LENGTH - 3] + "..."
                category = _get_category_from_path(skill_md)
                skills.append({"name": name, "description": description, "category": category})
            except Exception:
                continue
        return skills
    except Exception:
        return []


def _get_categories(skills: List[dict]) -> List[str]:
    """Return sorted unique category names (None -> 'uncategorized')."""
    cats = set()
    for s in skills:
        cats.add(s["category"] or "uncategorized")
    return sorted(cats)


# ─── Checklist UI ─────────────────────────────────────────────────────────────

def _prompt_checklist(title: str, items: List[str], disabled_items: Set[str]) -> Set[str]:
    """Generic curses multi-select. Returns set of DISABLED item names."""
    pre_disabled = {i for i, item in enumerate(items) if item in disabled_items}

    try:
        import curses
        selected = set(pre_disabled)
        result_holder = [None]

        def _curses_ui(stdscr):
            curses.curs_set(0)
            if curses.has_colors():
                curses.start_color()
                curses.use_default_colors()
                curses.init_pair(1, curses.COLOR_GREEN, -1)
                curses.init_pair(2, curses.COLOR_YELLOW, -1)
                curses.init_pair(3, curses.COLOR_RED, -1)
            cursor = 0
            scroll_offset = 0
            while True:
                stdscr.clear()
                max_y, max_x = stdscr.getmaxyx()
                try:
                    hattr = curses.A_BOLD | (curses.color_pair(2) if curses.has_colors() else 0)
                    stdscr.addnstr(0, 0, title, max_x - 1, hattr)
                    stdscr.addnstr(1, 0, "  ↑↓ navigate  SPACE toggle  ENTER confirm  ESC cancel", max_x - 1,
                                   curses.color_pair(3) if curses.has_colors() else curses.A_DIM)
                    stdscr.addnstr(2, 0, "  [✓] enabled   [✗] disabled", max_x - 1, curses.A_DIM)
                except curses.error:
                    pass
                visible_rows = max_y - 4
                if cursor < scroll_offset:
                    scroll_offset = cursor
                elif cursor >= scroll_offset + visible_rows:
                    scroll_offset = cursor - visible_rows + 1
                for draw_i, i in enumerate(range(scroll_offset, min(len(items), scroll_offset + visible_rows))):
                    y = draw_i + 4
                    if y >= max_y - 1:
                        break
                    is_disabled = i in selected
                    check = "✗" if is_disabled else "✓"
                    arrow = "→" if i == cursor else " "
                    line = f" {arrow} [{check}] {items[i]}"
                    attr = curses.A_NORMAL
                    if i == cursor:
                        attr = curses.A_BOLD | (curses.color_pair(1) if curses.has_colors() else 0)
                    elif is_disabled and curses.has_colors():
                        attr = curses.color_pair(3)
                    try:
                        stdscr.addnstr(y, 0, line, max_x - 1, attr)
                    except curses.error:
                        pass
                stdscr.refresh()
                key = stdscr.getch()
                if key in (curses.KEY_UP, ord('k')):
                    cursor = (cursor - 1) % len(items)
                elif key in (curses.KEY_DOWN, ord('j')):
                    cursor = (cursor + 1) % len(items)
                elif key == ord(' '):
                    if cursor in selected:
                        selected.discard(cursor)
                    else:
                        selected.add(cursor)
                elif key in (curses.KEY_ENTER, 10, 13):
                    result_holder[0] = {items[i] for i in selected}
                    return
                elif key in (27, ord('q')):
                    result_holder[0] = disabled_items
                    return

        curses.wrapper(_curses_ui)
        return result_holder[0] if result_holder[0] is not None else disabled_items

    except Exception:
        return _numbered_toggle(title, items, disabled_items)


def _numbered_toggle(title: str, items: List[str], disabled: Set[str]) -> Set[str]:
    """Fallback text-based toggle."""
    current = set(disabled)
    while True:
        print()
        print(color(f"{title}", Colors.BOLD))
        for i, item in enumerate(items, 1):
            mark = "✗" if item in current else "✓"
            print(f"  {i:3}. [{mark}] {item}")
        print()
        print(color("  Number to toggle, 's' save, 'q' cancel:", Colors.DIM))
        try:
            raw = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            return disabled
        if raw.lower() == 's':
            return current
        if raw.lower() == 'q':
            return disabled
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(items):
                name = items[idx]
                if name in current:
                    current.discard(name)
                    print(color(f"  ✓ {name} enabled", Colors.GREEN))
                else:
                    current.add(name)
                    print(color(f"  ✗ {name} disabled", Colors.DIM))
        except ValueError:
            print(color("  Invalid input", Colors.DIM))


# ─── Platform Selection ───────────────────────────────────────────────────────

def _select_platform() -> Optional[str]:
    """Ask user which platform to configure, or global."""
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
    if not raw:
        return None  # global
    try:
        idx = int(raw) - 1
        if 0 <= idx < len(options):
            key = options[idx][0]
            return None if key == "global" else key
    except ValueError:
        pass
    return None


# ─── Category Toggle ──────────────────────────────────────────────────────────

def _toggle_by_category(skills: List[dict], disabled: Set[str]) -> Set[str]:
    """Toggle all skills in a category at once."""
    categories = _get_categories(skills)
    cat_items = []
    cat_disabled = set()
    for cat in categories:
        cat_skills = [s["name"] for s in skills if (s["category"] or "uncategorized") == cat]
        cat_items.append(f"{cat} ({len(cat_skills)} skills)")
        if all(s in disabled for s in cat_skills):
            cat_disabled.add(f"{cat} ({len(cat_skills)} skills)")

    new_cat_disabled = _prompt_checklist("Categories — disable entire categories", cat_items, cat_disabled)

    new_disabled = set(disabled)
    for i, cat in enumerate(categories):
        label = cat_items[i]
        cat_skills = [s["name"] for s in skills if (s["category"] or "uncategorized") == cat]
        if label in new_cat_disabled:
            new_disabled.update(cat_skills)
        else:
            new_disabled -= set(cat_skills)
    return new_disabled


# ─── Entry Point ──────────────────────────────────────────────────────────────

def skills_command(args=None):
    """Entry point for `hermes skills`."""
    config = load_config()
    skills = _list_all_skills_unfiltered()

    if not skills:
        print(color("  No skills installed.", Colors.DIM))
        return

    # Step 1: Select platform
    platform = _select_platform()
    platform_label = PLATFORMS.get(platform, "All platforms") if platform else "All platforms"

    # Step 2: Select mode — individual or by category
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
        skill_items = [
            f"{s['name']}  ({s['category'] or 'uncategorized'})  —  {s['description'][:55]}"
            for s in skills
        ]
        disabled_labels = {
            f"{s['name']}  ({s['category'] or 'uncategorized'})  —  {s['description'][:55]}"
            for s in skills if s["name"] in disabled
        }
        new_disabled_labels = _prompt_checklist(
            f"Skills for {platform_label}  —  space=toggle, enter=confirm",
            skill_items,
            disabled_labels
        )
        # Map labels back to skill names
        label_to_name = {
            f"{s['name']}  ({s['category'] or 'uncategorized'})  —  {s['description'][:55]}": s["name"]
            for s in skills
        }
        new_disabled = {label_to_name[l] for l in new_disabled_labels if l in label_to_name}

    if new_disabled == disabled:
        print(color("  No changes.", Colors.DIM))
        return

    save_disabled_skills(config, new_disabled, platform)
    enabled_count = len(skills) - len(new_disabled)
    print(color(f"✓ Saved: {enabled_count} enabled, {len(new_disabled)} disabled ({platform_label}).", Colors.GREEN))
