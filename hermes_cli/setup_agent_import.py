"""First-time ``hermes setup``'s "Coming from another agent?" step (it replaces core's old OpenClaw
migration offer).

Detection is a few ``is_dir`` checks and needs nothing installed. Importing lives in the
``agent-importers`` catalog plugin (``hermes import-agent <source>``): only after the user says yes
does this step install it through the normal catalog install and run its import in a child process,
which owns the preview, the confirmation and what gets imported. Setup helpers are imported lazily
from hermes_cli.setup so test patches there apply.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import NamedTuple, Optional

PLUGIN = "agent-importers"


class ImportSource(NamedTuple):
    name: str                 # ``hermes import-agent <name>``
    label: str
    dirs: tuple[str, ...]     # dirs under ``~`` the plugin reads; the first one found wins
    extra_args: tuple[str, ...] = ()


# The plugin's importers, with the dirs its own auto-detection checks.
IMPORT_SOURCES: tuple[ImportSource, ...] = (
    # Core's setup step migrated OpenClaw's allowlisted API keys and bot tokens after the preview, so
    # the provider and messaging sections that follow find them already set; kept.
    ImportSource("openclaw", "OpenClaw", (".openclaw", ".clawdbot", ".moltbot"), ("--migrate-secrets",)),
    ImportSource("claude-code", "Claude Code", (".claude",)),
    ImportSource("codex", "Codex CLI", (".codex",)),
    ImportSource("cursor", "Cursor", (".cursor",)),
)


def detect_import_sources() -> list[tuple[ImportSource, Path]]:
    """The sources with a setup on this computer, each with the dir found."""
    found = []
    for source in IMPORT_SOURCES:
        home = next((p for p in (Path.home() / d for d in source.dirs) if p.is_dir()), None)
        if home is not None:
            found.append((source, home))
    return found


def offer_agent_import(hermes_home: Path) -> str:
    """When another agent's setup is on this computer, offer to import it: pick which (when several),
    install the plugin if it is missing and run its import for each pick. Returns the labels whose
    import changed this home's config.yaml or .env (``""`` when none did). Asks nothing, and so never
    installs anything, when nothing is found or no one can answer the prompts."""
    from hermes_cli.setup import _info, is_noninteractive, print_header, print_info, prompt_yes_no
    found = [] if is_noninteractive() else detect_import_sources()
    if not found:
        return ""
    user_home = str(Path.home())
    print_header("Coming from another agent?", gap=True)
    context = ("Found on this computer: " + ", ".join(
        f"{s.label} ({str(p).replace(user_home, '~', 1)})" for s, p in found) + ".",
        "Hermes can bring over its settings, memories, skills and MCP servers.",
        "You'll see a preview and confirm before anything changes.")
    _info(*context)
    names = " or ".join(s.label for s, _ in found) if len(found) <= 2 else "any of them"
    # The menu clears the screen, so the question carries what it refers to ("any of them").
    if not prompt_yes_no(f"Import your setup from {names}?", default=True,
                         description="\n".join(f"  {w}" for line in context for w in textwrap.wrap(line, 76))):
        print_info(f"Skipped. Import later with: hermes plugins install {PLUGIN}, then hermes import-agent")
        return ""
    picked = _pick_sources(found)
    if not picked or not _ensure_plugin(hermes_home):
        return ""
    return ", ".join(s.label for s in picked if _run_import(s, hermes_home))


def _pick_sources(found: list[tuple[ImportSource, Path]]) -> list[ImportSource]:
    """Only one found: that one (the question named it). Several: a checklist, all checked."""
    from hermes_cli.setup import print_info, prompt_checklist
    if len(found) == 1:
        return [found[0][0]]
    picked = prompt_checklist("Which agents should Hermes import from?", [s.label for s, _ in found],
                              list(range(len(found))))
    if not picked:
        print_info("Nothing selected, so nothing was imported.")
    return [found[i][0] for i in picked]


def _ensure_plugin(hermes_home: Path) -> bool:
    """The plugin installed and enabled in *hermes_home*: the catalog install (reviewed pin, the
    usual install checks) when it is missing, else just an enable."""
    from hermes_cli.left_core_migration import plugin_present
    from hermes_cli.plugins_admission import AdmissionRefused
    from hermes_cli.plugins_cmd import cmd_enable
    from hermes_cli.plugins_cmd_install import cmd_install
    from hermes_cli.setup import _info, print_warning
    present = plugin_present(PLUGIN, hermes_home)
    if not present:
        _info(None, f"Importing is done by the '{PLUGIN}' plugin from the Hermes plugin catalog. "
                    "Installing it now...")
    try:
        if present:
            cmd_enable(PLUGIN)
        else:
            cmd_install(PLUGIN, enable=True, brief=True)  # the line above already said what and why
    # Both print their own error first: a refusal exits, an enable whose dependencies PM cannot
    # admit raises. Either way the import is skipped and the rest of setup goes on.
    except (SystemExit, AdmissionRefused):
        pass
    else:
        if plugin_present(PLUGIN, hermes_home):
            return True
    verb = "enabled" if present else "installed"
    print_warning(f"The '{PLUGIN}' plugin could not be {verb}, so nothing was imported. "
                  f"Try again later with: hermes plugins {'enable' if present else 'install'} {PLUGIN}")
    return False


def _snapshot(hermes_home: Path) -> tuple[Optional[bytes], ...]:
    return tuple(p.read_bytes() if p.is_file() else None  # config-reader: ok — child-run change snapshot
                 for p in (hermes_home / "config.yaml", hermes_home / ".env"))


def _run_import(source: ImportSource, hermes_home: Path) -> bool:
    """``hermes import-agent <source>`` in a child process, so the freshly installed plugin's command
    is wired; it previews and asks before changing anything (and exits 0 on a no too). True when the
    run changed config.yaml or .env, which the setup sections after it read."""
    from hermes_cli.setup import print_header, print_warning
    from tools.environments.local import served_profile_child_env
    print_header(f"Importing from {source.label}", gap=True)
    before = _snapshot(hermes_home)
    # A home that is not <root>/profiles/<name> is the root: without a selector the child's
    # _apply_profile_override would follow the sticky active_profile into another profile.
    selector = [] if hermes_home.resolve().parent.name == "profiles" else ["--profile", "default"]
    # --from-setup: the plugin's import speaks as a step of this setup (no "run hermes setup" pointers,
    # OpenClaw's skipped items folded into one line).
    argv = [sys.executable, "-P", "-m", "hermes_cli.main", *selector, "import-agent", source.name, *source.extra_args,
            "--from-setup"]
    env = served_profile_child_env(target_home=hermes_home)
    if os.environ.get("PYTHONPATH"):  # this same interpreter running Hermes: keep the source tree
        env["PYTHONPATH"] = os.environ["PYTHONPATH"]  # a checkout launched by path, which it strips
    # health: allow HX006 -- interactive child: the user reads its preview and answers its prompts
    if subprocess.run(argv, env=env, check=False).returncode:
        # An install from before the plugin gained this importer, or --from-setup, refuses it (argparse, exit 2).
        print_warning(f"The {source.label} import did not run. If the error names an invalid choice or "
                      f"unrecognized arguments, update the plugin first: hermes plugins update {PLUGIN}")
    return _snapshot(hermes_home) != before
