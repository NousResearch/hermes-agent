"""Post-migration section-skip logic and the OpenClaw first-run migration flow (setup.py
re-exports the names it still uses; setup helpers are imported lazily so test patches apply)."""

import importlib.util
import logging
import sys
from functools import partial
from pathlib import Path
from typing import Optional

from hermes_constants import get_optional_skills_dir

logger = logging.getLogger("hermes_cli.setup")

PROJECT_ROOT = Path(__file__).parent.parent.resolve()


# ── Post-Migration Section Skip Logic ──

_OPENROUTER_ENV_VARS = ("OPENROUTER_API_KEY", "OPENAI_API_KEY")


def _model_section_has_credentials(config: dict) -> bool:
    """True when any known inference provider has usable credentials: ``active_provider`` in the
    auth store (OAuth providers), ``PROVIDER_REGISTRY`` ``api_key_env_vars``, or the legacy
    OpenRouter aggregator env vars (``OPENAI_API_KEY`` / ``OPENROUTER_API_KEY``)."""
    from hermes_cli.setup import get_env_value
    try:
        from hermes_cli.auth import get_active_provider
        if get_active_provider():
            return True
    except Exception:
        pass
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY
    except Exception:
        PROVIDER_REGISTRY = {}  # type: ignore[assignment]

    def _has_key(pconfig) -> bool:
        # CLAUDE_CODE_OAUTH_TOKEN is set by Claude Code itself, not by the user —
        # mirrors is_provider_explicitly_configured in auth.py.
        return any(get_env_value(v) for v in pconfig.api_key_env_vars if v != "CLAUDE_CODE_OAUTH_TOKEN")

    any_openrouter_key = any(get_env_value(v) for v in _OPENROUTER_ENV_VARS)

    # Prefer the provider declared in config.yaml, avoids false positives from stray
    # env vars (GH_TOKEN, etc.) when the user has already picked a different provider.
    model_cfg = config.get("model") if isinstance(config, dict) else None
    if isinstance(model_cfg, dict):
        provider_id = (model_cfg.get("provider") or "").strip().lower()
        if provider_id in PROVIDER_REGISTRY and _has_key(PROVIDER_REGISTRY[provider_id]):
            return True
        if provider_id == "openrouter" and any_openrouter_key:
            return True

    # OpenRouter aggregator fallback (no provider declared in config).
    if any_openrouter_key:
        return True
    # Skip copilot in auto-detect: GH_TOKEN / GITHUB_TOKEN are commonly set for git tooling.
    # Mirrors resolve_provider in auth.py.
    return any(_has_key(pconfig) for pid, pconfig in PROVIDER_REGISTRY.items() if pid != "copilot")


def _model_summary(config: dict) -> Optional[str]:
    if not _model_section_has_credentials(config):
        return None
    model = config.get("model")
    if isinstance(model, str) and model.strip():
        return model.strip()
    if isinstance(model, dict):
        return str(model.get("default") or model.get("model") or "configured")
    return "configured"


def _cfg_summary(config: dict, section: str, key: str, default, prefix: str) -> str:
    from hermes_cli.setup import cfg_get
    return f"{prefix}{cfg_get(config, section, key, default=default)}"


def _gateway_summary(config: dict) -> Optional[str]:
    from hermes_cli.gateway import _all_platforms, _platform_status
    # Any non-empty status other than "not configured" counts — WhatsApp ("enabled, not paired"),
    # Matrix ("configured + E2EE"), Signal ("partially configured") mean setup already started.
    configured = [
        # Trailing parenthetical qualifiers are stripped from the label.
        plat["label"].split("(", 1)[0].strip() or plat["label"]
        for plat in _all_platforms()
        if _platform_status(plat) and _platform_status(plat) != "not configured"
    ]
    return ", ".join(configured) if configured else None


_TOOL_ENV_LABELS = (
    ("ELEVENLABS_API_KEY", "TTS/ElevenLabs"),
    ("BROWSERBASE_API_KEY", "Browser"),
    ("FIRECRAWL_API_KEY", "Firecrawl"),
)


def _tools_summary(config: dict) -> Optional[str]:
    from hermes_cli.setup import get_env_value
    tools = [label for env_var, label in _TOOL_ENV_LABELS if get_env_value(env_var)]
    return ", ".join(tools) if tools else None


_SECTION_SUMMARIES = {
    "model": _model_summary,
    "terminal": partial(_cfg_summary, section="terminal", key="backend", default="local", prefix="backend: "),
    "agent": partial(_cfg_summary, section="agent", key="max_turns", default=90, prefix="max turns: "),
    "gateway": _gateway_summary,
    "tools": _tools_summary,
}


def _get_section_config_summary(config: dict, section_key: str) -> Optional[str]:
    """Short summary if a setup section is already configured (post-OpenClaw-migration skip
    detection), else None. ``get_env_value`` is reached through hermes_cli.setup so test patches
    on ``setup_mod.get_env_value`` apply."""
    summarize = _SECTION_SUMMARIES.get(section_key)
    return summarize(config) if summarize else None


def _skip_configured_section(config: dict, section_key: str, label: str) -> bool:
    """Show an already-configured section summary and offer to skip; True when the user skips."""
    from hermes_cli.setup import print_success, prompt_yes_no
    summary = _get_section_config_summary(config, section_key)
    if not summary:
        return False
    print()
    print_success(f"  {label}: {summary}")
    return not prompt_yes_no(f"  Reconfigure {label.lower()}?", default=False)


# ── OpenClaw Migration ──

_OPENCLAW_SCRIPT = (
    get_optional_skills_dir(PROJECT_ROOT / "optional-skills")
    / "migration" / "openclaw-migration" / "scripts" / "openclaw_to_hermes.py"
)


def _load_openclaw_migration_module():
    """Load the openclaw_to_hermes migration script as a module; None if it can't be loaded."""
    if not _OPENCLAW_SCRIPT.exists():
        return None
    spec = importlib.util.spec_from_file_location("openclaw_to_hermes", _OPENCLAW_SCRIPT)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    # Registered in sys.modules so @dataclass can resolve the module (Python 3.11+ requirement).
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return mod


# Item kinds that warrant explicit warnings: gateway tokens/channels hijack the old agent's
# platforms; config values and instruction/context .md files may not map 1:1 to Hermes.
_HIGH_IMPACT_KIND_KEYWORDS = {
    "gateway": "⚠ Gateway/messaging — this will configure Hermes to use your OpenClaw messaging channels",
    "telegram": "⚠ Telegram — this will point Hermes at your OpenClaw Telegram bot",
    "slack": "⚠ Slack — this will point Hermes at your OpenClaw Slack workspace",
    "discord": "⚠ Discord — this will point Hermes at your OpenClaw Discord bot",
    "whatsapp": "⚠ WhatsApp — this will point Hermes at your OpenClaw WhatsApp connection",
    "config": "⚠ Config values — OpenClaw settings may not map 1:1 to Hermes equivalents",
    "soul": "⚠ Instruction file — may contain OpenClaw-specific setup/restart procedures",
    "memory": "⚠ Memory/context file — may reference OpenClaw-specific infrastructure",
    "context": "⚠ Context file — may contain OpenClaw-specific instructions",
}

_MIGRATION_WARNING_NOTES = (
    "  Note: OpenClaw config values may have different semantics in Hermes.",
    "  For example, OpenClaw's tool_call_execution: \"auto\" ≠ Hermes's yolo mode.",
    "  Instruction files (.md) from OpenClaw may contain incompatible procedures.",
)


def _migrated_row(item: dict, kind: str) -> str:
    dest = item.get("destination", "")
    if dest:
        return f"      {kind:<22s} → {str(dest).replace(str(Path.home()), '~')}"
    return f"      {kind}"


def _reason_row(default_reason: str, item: dict, kind: str) -> str:
    return f"      {kind:<22s}  {item.get('reason', default_reason)}"


def _print_migration_preview(report: dict):
    """Dry-run preview grouped by status, with warnings for high-impact items (gateway takeover,
    config semantics)."""
    from hermes_cli.setup import color, Colors, print_info
    items = report.get("items", [])
    if not items:
        print_info("Nothing to migrate.")
        return
    groups = (
        ("migrated", "  Would import:", Colors.GREEN, _migrated_row),
        ("conflict", "  Would overwrite (conflicts with existing Hermes config):", Colors.YELLOW,
         partial(_reason_row, "already exists")),
        ("skipped", "  Would skip:", Colors.DIM, partial(_reason_row, "")),
    )
    warnings_shown = set()
    for status, header, col, row in groups:
        group = [i for i in items if i.get("status") == status]
        if not group:
            continue
        print(color(header, col))
        for item in group:
            kind = item.get("kind", "unknown")
            print(row(item, kind))
            if status == "migrated":  # collect warnings for high-impact items
                kind_lower, dest_lower = kind.lower(), str(item.get("destination", "")).lower()
                warnings_shown.update(
                    w for kw, w in _HIGH_IMPACT_KIND_KEYWORDS.items() if kw in kind_lower or kw in dest_lower)
        print()
    if warnings_shown:
        print(color("  ── Warnings ──", Colors.YELLOW))
        for warning in sorted(warnings_shown):
            print(color(f"    {warning}", Colors.YELLOW))
        print()
        for line in _MIGRATION_WARNING_NOTES:
            print(color(line, Colors.YELLOW))
        print()


def _run_migrator(mod, openclaw_dir: Path, hermes_home: Path, selected, *, execute: bool, overwrite: bool):
    """Run a Migrator with the fixed first-time-setup options and return its report."""
    return mod.Migrator(
        source_root=openclaw_dir.resolve(), target_root=hermes_home.resolve(), execute=execute,
        workspace_target=None, overwrite=overwrite, migrate_secrets=True, output_dir=None,
        selected_options=selected, preset_name="full",
    ).migrate()


_FAILED = object()


def _migration_step(label: str, log_label: str, fn):
    """Run one migration phase; on failure warn ``"<label>: <exc>"``, log the trace and return
    ``_FAILED`` (the caller aborts)."""
    from hermes_cli.setup import print_warning
    try:
        return fn()
    except Exception as e:
        print_warning(f"{label}: {e}")
        logger.debug(log_label, exc_info=True)
        return _FAILED


def _offer_openclaw_migration(hermes_home: Path) -> bool:
    """Detect ~/.openclaw and offer to migrate during first-time setup: dry-run preview first,
    execute only after explicit confirmation. Returns True iff migration ran successfully."""
    from hermes_cli.setup import (
        get_config_path, _info, load_config, print_header, print_info, print_success, print_warning, prompt_yes_no,
        save_config
    )
    openclaw_dir = Path.home() / ".openclaw"
    if not openclaw_dir.is_dir() or not _OPENCLAW_SCRIPT.exists():
        return False
    print_header("OpenClaw Installation Detected", gap=True)
    _info(f"Found OpenClaw data at {openclaw_dir}",
          "Hermes can preview what would be imported before making any changes.", None)
    if not prompt_yes_no("Would you like to see what can be imported?", default=True):
        print_info("Skipping migration. You can run it later with: hermes claw migrate --dry-run")
        return False

    # Ensure config.yaml exists before migration tries to read it
    if not get_config_path().exists():
        save_config(load_config())
    mod = _migration_step("Could not load migration script", "OpenClaw migration module load error",
                          _load_openclaw_migration_module)
    if mod is None:
        print_warning("Could not load migration script.")
    if mod is None or mod is _FAILED:
        return False

    # ── Phase 1: Dry-run preview (overwrite=True shows everything, including conflicts) ──
    def _preview():
        selected = mod.resolve_selected_options(None, None, preset="full")
        return selected, _run_migrator(mod, openclaw_dir, hermes_home, selected, execute=False, overwrite=True)

    previewed = _migration_step("Migration preview failed", "OpenClaw migration preview error", _preview)
    if previewed is _FAILED:
        return False
    selected, preview_report = previewed
    preview_count = preview_report.get("summary", {}).get("migrated", 0)
    if preview_count == 0:
        _info(None, "Nothing to import from OpenClaw.")
        return False
    print_header(f"Migration Preview — {preview_count} item(s) would be imported", gap=True)
    _info("No changes have been made yet. Review the list below:", None)
    _print_migration_preview(preview_report)

    # ── Phase 2: Confirm and execute ──
    if not prompt_yes_no("Proceed with migration?", default=False):
        _info("Migration cancelled. You can run it later with: hermes claw migrate",
              "Use --dry-run to preview again, or --preset minimal for a lighter import.")
        return False

    # overwrite=False so existing Hermes configs are preserved. The user saw the preview;
    # conflicts are skipped by default.
    report = _migration_step("Migration failed", "OpenClaw migration error", lambda: _run_migrator(
        mod, openclaw_dir, hermes_home, selected, execute=True, overwrite=False))
    if report is _FAILED:
        return False
    summary = report.get("summary", {})
    print()
    for key, printer, text in (
        ("migrated", print_success, "Imported {n} item(s) from OpenClaw."),
        ("conflict", print_info,
         "Skipped {n} item(s) that already exist in Hermes (use hermes claw migrate --overwrite to force)."),
        ("skipped", print_info, "Skipped {n} item(s) (not found or unchanged)."),
        ("error", print_warning, "{n} item(s) had errors — check the migration report."),
    ):
        count = summary.get(key, 0)
        if count:
            printer(text.format(n=count))
    output_dir = report.get("output_dir")
    if output_dir:
        print_info(f"Full report saved to: {output_dir}")
    print_success("Migration complete! Continuing with setup...")
    return True
