"""Best-effort refresh of companions already installed in PM's Node prefix."""
import json
import subprocess


def update_installed_codex() -> None:
    """Never provision Node or Codex; use only the host-safe managed npm."""
    from hermes_cli.main_install_repair import _resolve_node_runtime_npm
    from hermes_constants import with_hermes_node_path

    npm = _resolve_node_runtime_npm()
    if not npm:
        return
    env = with_hermes_node_path()
    try:
        result = subprocess.run(
            [npm, "list", "-g", "@openai/codex", "--depth=0", "--json"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            env=env, check=False, timeout=30,
        )
        listing = json.loads(result.stdout or "{}")
        packages = listing.get("dependencies", {}) if isinstance(listing, dict) else {}
        if not isinstance(packages, dict) or "@openai/codex" not in packages:
            return
        print("→ Updating installed Codex CLI...")
        result = subprocess.run(
            [npm, "install", "-g", "@openai/codex@latest"],
            env=env, check=False, timeout=180,
        )
        if result.returncode:
            print(f"  ⚠ Codex CLI update failed (exit {result.returncode}; non-fatal)")
        else:
            print("  ✓ Codex CLI updated")
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        print(f"  ⚠ Codex CLI update unavailable (non-fatal): {exc}")
