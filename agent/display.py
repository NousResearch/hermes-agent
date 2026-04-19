_ANSI_RESET = "\x1b[0m"

# Diff colors — resolved lazily from the skin engine so they adapt
# to light/dark themes.  Falls back to sensible defaults on import
# failure.  We cache after first resolution for performance.
_diff_colors_cached: dict[str, str] | None = None


def _diff_ansi() -> dict[str, str]:
    """Return ANSI escapes for diff display, resolved from the active skin."""
    global _diff_colors_cached
    if _diff_colors_cached is not None:
        return _diff_colors_cached

    # Defaults that work on dark terminals
    dim = "\x1b[38;2;150;150;150m"
    file_c = "\x1b[38;2;180;160;255m"
    hunk = "\x1b[38;2;120;120;140m"
    minus = "\x1b[38;2;255;255;255;48;2;120;20;20m"
    plus = "\x1b[38;2;255;255;255;48;2;20;90;20m"

    try:
        from hermes_cli.skin_engine import get_active_skin
        skin = get_active_skin()

        def _hex_fg(key: str, fallback_rgb: tuple[int, int, int]) -> str:
            h = skin.get_color(key, "")
            if h and len(h) == 7 and h[0] == "#":
                r, g, b = int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)
                return f"\x1b[38;2;{r};{g};{b}m"
            r, g, b = fallback_rgb
            return f"\x1b[38;2;{r};{g};{b}m"

        dim = _hex_fg("banner_dim", (150, 150, 150))
        file_c = _hex_fg("session_label", (180, 160, 255))
        hunk = _hex_fg("session_border", (120, 120, 140))
        # minus/plus use background colors — derive from ui_error/ui_ok
        err_h = skin.get_color("ui_error", "#ef5350")
        ok_h = skin.get_color("ui_ok", "#4caf50")
        if err_h and len(err_h) == 7:
            er, eg, eb = int(err_h[1:3], 16), int(err_h[3:5], 16), int(err_h[5:7], 16)
            # Use a dark tinted version as background
            minus = f"\x1b[38;2;255;255;255;48;2;{max(er//2,20)};{max(eg//4,10)};{max(eb//4,10)}m"
        if ok_h and len(ok_h) == 7:
            or_, og, ob = int(ok_h[1:3], 16), int(ok_h[3:5], 16), int(ok_h[5:7], 16)
            plus = f"\x1b[38;2;255;255;255;48;2;{max(or_//4,10)};{max(og//2,20)};{max(ob//4,10)}m"
    except Exception:
        pass

    _diff_colors_cached = {
        "dim": dim, "file": file_c, "hunk": hunk,
        "minus": minus, "plus": plus,
    }
    return _diff_colors_cached


# Module-level helpers — each call resolves from the active skin lazily.
def _diff_dim():   return _diff_ansi()["dim"]
def _diff_file():  return _diff_ansi()["file"]
def _diff_hunk():  return _diff_ansi()["hunk"]
def _diff_minus(): return _diff_ansi()["minus"]
def _diff_plus():  return _diff_ansi()["plus"]

_ANSI_LNUM  = "\x1b[38;2;100;100;110m"
_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")
def _render_inline_unified_diff(diff: str) -> list[str]:
    """Render unified diff lines with a line-number gutter and Claude Code-style colors."""
    rendered: list[str] = []
    from_file = to_file = None
    old_ln = new_ln = 0
    max_ln = 0
    for line in diff.splitlines():
        m = _HUNK_RE.match(line)
        if m:
            max_ln = max(max_ln, int(m.group(1)), int(m.group(2)))
    width = max(len(str(max_ln)), 3)

    for raw_line in diff.splitlines():
        if raw_line.startswith("--- "):
            from_file = raw_line[4:].strip()
            continue
        if raw_line.startswith("+++ "):
            to_file = raw_line[4:].strip()
            continue
        m = _HUNK_RE.match(raw_line)
        if m:
            old_ln = int(m.group(1))
            new_ln = int(m.group(2))
            rendered.append(f"{_diff_hunk()}{raw_line}{_ANSI_RESET}")
            continue
        if raw_line.startswith("-"):
            lnum = str(old_ln).rjust(width)
            old_ln += 1
            rendered.append(f"{_ANSI_LNUM}{lnum} {_ANSI_RESET}{_diff_minus()}{raw_line}{_ANSI_RESET}")
            continue
        if raw_line.startswith("+"):
            lnum = str(new_ln).rjust(width)
            new_ln += 1
            rendered.append(f"{_ANSI_LNUM}{lnum} {_ANSI_RESET}{_diff_plus()}{raw_line}{_ANSI_RESET}")
            continue
        if raw_line.startswith(" "):
            lnum = str(new_ln).rjust(width)
            old_ln += 1
            new_ln += 1
            rendered.append(f"{_ANSI_LNUM}{lnum} {_ANSI_RESET}{_diff_dim()}{raw_line}{_ANSI_RESET}")
            continue
        if raw_line:
            rendered.append(raw_line)

    return rendered
