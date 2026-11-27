"""``hermes plugins config <name>``: edit a plugin's declared settings in the terminal.

The fields are the ones the Desktop Plugins page renders (:func:`plugin_settings_fields`), plus the
plugin's ``requires_env`` variables, so a plugin gets a settings screen from its manifest alone.
Settings save through :func:`save_plugin_settings` (validated against the schema, same writer as
``ctx.set_config``); secrets and env variables save through the credential route the Desktop's
``PUT /api/env`` uses, so they land in the profile's ``.env`` and never in ``config.yaml``.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from hermes_cli.plugins_settings.fields import _manifest_config_schema, plugin_settings_fields, save_plugin_settings

_UNCHANGED = object()
# Same words and spellings as the Desktop form's ``humanizeSettingKey`` (plugin-settings-form.tsx), so a
# key reads the same on both surfaces.
_KEY_ACRONYMS = frozenset({"ai", "api", "cpu", "css", "dns", "gpu", "html", "http", "https", "id", "ip", "json",
                           "llm", "mcp", "oauth", "sql", "ssh", "ssl", "tls", "tts", "ui", "uri", "url", "xml"})
_ACRONYM_SPELLING = {"ids": "IDs", "oauth": "OAuth", "urls": "URLs"}


def humanize_key(key: str) -> str:
    """``maps_api_key`` → "Maps API key", ``webhookURL`` → "Webhook URL"."""
    spaced = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", key))
    words = [w for w in re.split(r"[\s_.\-/]+", spaced) if w]

    def _word(index: int, word: str) -> str:
        lower = word.lower()
        if lower in _ACRONYM_SPELLING:
            return _ACRONYM_SPELLING[lower]
        if lower in _KEY_ACRONYMS:
            return lower.upper()
        return lower.capitalize() if index == 0 else lower
    return " ".join(_word(i, w) for i, w in enumerate(words))


def _env_fields(manifest: Mapping[str, Any], declared: set[str]) -> list[dict[str, Any]]:
    """``requires_env`` entries as fields (skipping names a ``secret`` setting already covers). Masking
    uses the platform setup's own rule (``password``/``secret``, or a token-shaped name)."""
    from hermes_cli.config import _platform_manifest_env_entries, get_env_value
    out = []
    for name, is_secret, spec in _platform_manifest_env_entries(dict(manifest), optional=False):
        if name in declared:
            continue
        current = get_env_value(name)
        out.append({"key": name, "type": "env", "label": name, "env": name,
                    "description": str(spec.get("description") or ""), "url": str(spec.get("url") or ""),
                    "masked": is_secret, "has_value": bool(current),
                    "value": None if is_secret else current})
    return out


def collect_fields(plugin_id: str, plugin_dir: Optional[Path], manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every field the editor shows: schema settings first, then the plugin's env variables."""
    fields = plugin_settings_fields(plugin_id, plugin_dir)
    for field in fields:
        if field["type"] == "secret":
            field["masked"] = True
        if field["label"] == field["key"]:
            field["label"] = humanize_key(field["key"])
    declared = {f["env"] for f in fields if f.get("env")}
    return fields + _env_fields(manifest, declared)


def display_value(field: Mapping[str, Any]) -> str:
    """One-line current value: stored secrets are never shown, only whether they are set."""
    if field.get("masked"):
        return "set" if field.get("has_value") else "not set"
    value = field.get("value")
    if value is None or value == "":
        return "not set"
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _row(field: Mapping[str, Any]) -> str:
    flag = "  (required)" if field.get("required") else ""
    return f"{field['label']}: {display_value(field)}{flag}"


def _print_field_help(field: Mapping[str, Any]) -> None:
    print()
    print(f"  {field['label']}")
    if field.get("description"):
        print(f"  {field['description']}")
    if field.get("url"):
        print(f"  Get one at: {field['url']}")


def _ask_text(question: str, current: str) -> Any:
    from hermes_cli.cli_output import line_input
    hint = f" [{current}]" if current else ""
    try:
        raw = line_input(f"  {question}{hint} (Enter keeps it): ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return _UNCHANGED
    return raw if raw else _UNCHANGED


def _edit_string(field, spec) -> Any:
    current = field.get("value")
    return _ask_text("New value", "" if current is None else str(current))


def _edit_number(field, spec) -> Any:
    raw = _ask_text("New number", display_value(field) if field.get("value") is not None else "")
    if raw is _UNCHANGED:
        return raw
    integer = str(spec.get("type") or "").lower() in ("int", "integer")
    try:
        return int(raw) if integer or not any(c in raw for c in ".eE") else float(raw)
    except ValueError:
        raise ValueError(f"'{raw}' is not {'a whole number' if integer else 'a number'}") from None


def _pick(field, labels: list[Any], selected: int) -> Optional[int]:
    from hermes_cli.curses_ui import curses_radiolist
    idx = curses_radiolist(field["label"], labels, selected=selected, cancel_returns=-1,
                           description=field.get("description") or None)
    return None if idx < 0 else idx


def _edit_enum(field, spec) -> Any:
    choices = list(field.get("choices") or [])
    current = field.get("value")
    idx = _pick(field, choices, choices.index(current) if current in choices else 0)
    return _UNCHANGED if idx is None else choices[idx]


def _edit_boolean(field, spec) -> Any:
    idx = _pick(field, ["On", "Off"], 0 if field.get("value") else 1)
    return _UNCHANGED if idx is None else idx == 0


def parse_json_value(raw: str) -> Any:
    """A list/mapping typed as JSON or YAML, restricted to JSON values (an unquoted YAML date or
    timestamp would save, then break every later listing)."""
    import hermes_yaml
    try:
        value = hermes_yaml.safe_load(raw)
    except hermes_yaml.YAMLError:
        raise ValueError("that is not valid JSON or YAML (check the brackets and quotes)") from None
    if not isinstance(value, (list, dict)):
        raise ValueError("expected a list like [a, b] or a mapping like {key: value}")
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError):
        raise ValueError("use only text, numbers, true/false and nested lists or mappings "
                         "(quote dates and times)") from None


def _edit_json(field, spec) -> Any:
    raw = _ask_text("New value as JSON or YAML (for example [a, b] or {key: value})", display_value(field))
    return raw if raw is _UNCHANGED else parse_json_value(raw)


def _edit_env(field, spec) -> Any:
    from hermes_cli.secret_prompt import masked_secret_prompt
    if not field.get("masked"):
        return _ask_text("New value", display_value(field) if field.get("has_value") else "")
    try:
        raw = masked_secret_prompt("  New value (Enter keeps the current one): ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return _UNCHANGED
    return raw if raw else _UNCHANGED


_EDITORS: dict[str, Callable[[Mapping[str, Any], Mapping[str, Any]], Any]] = {
    "string": _edit_string, "number": _edit_number, "enum": _edit_enum, "boolean": _edit_boolean,
    "json": _edit_json, "secret": _edit_env, "env": _edit_env,
}


def save_field(plugin_id: str, plugin_dir: Optional[Path], field: Mapping[str, Any], value: Any) -> str:
    """Persist one edited value; returns where it went. Raises ``ValueError``/``PermissionError``."""
    from hermes_constants import display_hermes_home
    if field["type"] in ("secret", "env"):
        from hermes_cli.credential_lifecycle import save_provider_env_credential
        save_provider_env_credential(field["env"], str(value))
        return f"{display_hermes_home()}/.env"
    save_plugin_settings(plugin_id, plugin_dir, {field["key"]: value})
    return f"{display_hermes_home()}/config.yaml"


def _edit_one(plugin_id: str, plugin_dir: Optional[Path], field: Mapping[str, Any], spec: Mapping[str, Any]) -> str:
    """Edit and save one field; returns the outcome line the menu shows above the list (the full-screen
    menu clears anything printed before it, so a printed "Saved" would never be seen)."""
    if field["type"] not in ("enum", "boolean"):
        _print_field_help(field)
    try:
        value = _EDITORS[field["type"]](field, spec)
        if value is _UNCHANGED:
            return f"{field['label']}: unchanged"
        where = save_field(plugin_id, plugin_dir, field, value)
    except (ValueError, PermissionError) as exc:
        return f"✗ {field['label']} not saved: {exc}"
    return f"✓ Saved {field['label']} to {where}"


def print_fields(name: str, plugin_id: str, fields: list[dict[str, Any]]) -> None:
    """Plain listing for a terminal-less run (scripts, pipes)."""
    print(f"{name} settings:")
    for field in fields:
        print(f"  {_row(field)}")
        where = (f".env {field['env']}" if field.get("env")
                 else f"plugins.entries.{plugin_id}.settings.{field['key']}")
        print(f"    {where}" + (f" — {field['description']}" if field.get("description") else ""))
    print(f"Run `hermes plugins config {name}` in a terminal to change them.")


def run_editor(name: str, plugin_id: str, plugin_dir: Optional[Path], manifest: Mapping[str, Any]) -> None:
    """The interactive loop: pick a setting, edit it, repeat until Done (or Esc). The last edit's outcome
    rides in the title row: the full-screen menu clears anything printed before it."""
    from hermes_cli.curses_ui import curses_single_select
    schema = _manifest_config_schema(plugin_dir)
    cursor, status = 0, "Enter edits a setting"
    while True:
        fields = collect_fields(plugin_id, plugin_dir, manifest)
        idx = curses_single_select(f"{name} settings · {status}", [_row(f) for f in fields],
                                   default_index=cursor, cancel_label="Done")
        if idx is None:
            return
        cursor = idx
        status = _edit_one(plugin_id, plugin_dir, fields[idx], schema.get(fields[idx]["key"], {}))


def _missing_required(fields: list[dict[str, Any]]) -> list[str]:
    """Labels of required settings with no value (a stored secret counts as a value)."""
    return [f["label"] for f in fields if f.get("required") and not (
        f.get("has_value") if f.get("masked") or f["type"] == "env" else f.get("value") not in (None, ""))]


def offer_settings_after_install(console, name: str, plugin_dir: Optional[Path], *, brief: bool = False) -> None:
    """Last step of an install: show the plugin's settings and offer the editor.

    Only a person at a terminal is asked (a TTY, not ``HERMES_NONINTERACTIVE``, not a *brief*
    setup-wizard install); everyone else gets the one-line pointer. The default answer is No unless
    a required setting is still empty. Runs after everything else, so a Ctrl-C here leaves a complete install.
    """
    from hermes_cli import plugins_cmd as pc
    from hermes_cli.setup import is_noninteractive
    if plugin_dir is None or not _manifest_config_schema(plugin_dir) or (match := pc._find_plugin_entry(name)) is None:
        return
    pname, _version, _desc, _source, _dir, key = match
    if brief or is_noninteractive() or not pc._is_tty():
        console.print(f"[dim]Change its settings any time: hermes plugins config {pname}[/dim]")
        return
    manifest = pc._read_manifest(plugin_dir)
    fields = collect_fields(key, plugin_dir, manifest)
    console.print(f"\n[bold]{pname}[/bold] settings:")
    for field in fields:
        console.print(f"  {_row(field)}", markup=False, highlight=False)
    missing = _missing_required(fields)
    if missing:
        console.print(f"[yellow]Still needed: {', '.join(missing)}[/yellow]")
    question = "Edit these now? [Y/n]: " if missing else "Edit these now? [y/N]: "
    try:
        answer = input(question).strip().lower()
    except (EOFError, KeyboardInterrupt):
        console.print()
        answer = "n"
    if (answer in ("y", "yes")) or (missing and answer == ""):
        run_editor(pname, key, plugin_dir, manifest)
    else:
        console.print(f"[dim]Change them any time: hermes plugins config {pname}[/dim]")


def cmd_config(name: str) -> None:
    """``hermes plugins config <name>``."""
    from hermes_cli import plugins_cmd as pc
    console = pc._console()
    match = pc._find_plugin_entry(name)
    if match is None:
        pc._fail(console, f"[red]Plugin '{name}' not found.[/red]\n[dim]List installed plugins:[/dim] hermes plugins list")
    pname, _version, _desc, _source, dir_path, key = match
    plugin_dir = Path(str(dir_path)) if dir_path else None
    manifest = pc._read_manifest(plugin_dir) if plugin_dir else {}
    fields = collect_fields(key, plugin_dir, manifest)
    if not fields:
        console.print(f"{pname} has no settings to configure. Its README describes any setup it needs.")
        return
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        print_fields(pname, key, fields)
        return
    run_editor(pname, key, plugin_dir, manifest)
