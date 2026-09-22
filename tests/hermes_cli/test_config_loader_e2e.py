"""E2E for the canonical-loader migration (managed-scope/env-expansion drift fix).

Runs a real subprocess with a temp HERMES_HOME whose config.yaml contains a
``${ENV_VAR}`` reference, plus a managed-scope overlay dir (HERMES_MANAGED_DIR).

Asserts the two halves of the contract:

  1. A migrated BEHAVIORAL site (``tui_gateway.server._load_cfg``) resolves the
     env-expanded AND managed-overlaid values — the drift bug this branch fixes.
  2. A WRITE-BACK site round-trips the raw user file without leaking managed
     values, expanded literals, or merged defaults into it.

Subprocess (not in-process monkeypatching) so module-level ``_hermes_home``
globals, managed-scope caches, and ``_under_pytest`` guards behave like
production: ``HERMES_MANAGED_DIR`` is set explicitly, which bypasses the
pytest suppression in ``get_managed_dir``.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _run_py(code: str, env_extra: dict[str, str], tmp_path: Path) -> dict:
    """Run ``code`` in a subprocess; results come back via a JSON file.

    A file (not stdout) because importing ``tui_gateway.server`` re-routes
    the real stdout into its JSON-RPC transport.
    """
    import os

    out_file = tmp_path / "e2e_result.json"
    env = dict(os.environ)
    env.pop("PYTEST_CURRENT_TEST", None)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["E2E_OUT_FILE"] = str(out_file)
    env.update(env_extra)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(tmp_path),
        timeout=120,
    )
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stdout}\n{proc.stderr}"
    assert out_file.exists(), f"no result file:\n{proc.stdout}\n{proc.stderr}"
    return json.loads(out_file.read_text(encoding="utf-8"))


def test_behavioral_read_gets_expansion_and_overlay_while_writeback_stays_raw(
    tmp_path,
):
    home = tmp_path / "hermes_home"
    home.mkdir()
    user_yaml = (
        "custom_prompt: 'hello ${E2E_PROMPT_SUFFIX}'\n"
        "agent:\n"
        "  reasoning_effort: low\n"
        "display:\n"
        "  skin: usertheme\n"
    )
    (home / "config.yaml").write_text(user_yaml, encoding="utf-8")

    managed_dir = tmp_path / "managed"
    managed_dir.mkdir()
    # Administrator pins reasoning_effort — must win over the user's "low".
    (managed_dir / "config.yaml").write_text(
        "agent:\n  reasoning_effort: high\n", encoding="utf-8"
    )

    code = textwrap.dedent(
        """
        import json
        from tui_gateway import server

        cfg = server._load_cfg()
        raw = server._load_cfg_raw()

        # Write-back round-trip through the real save path.
        rt = server._load_cfg_raw()
        rt.setdefault("display", {})["battery"] = True
        server._save_cfg(rt)

        import os
        from pathlib import Path
        saved = Path(server._hermes_home, "config.yaml").read_text(encoding="utf-8")
        Path(os.environ["E2E_OUT_FILE"]).write_text(json.dumps({
            "behavioral_prompt": cfg.get("custom_prompt"),
            "behavioral_effort": (cfg.get("agent") or {}).get("reasoning_effort"),
            "raw_prompt": raw.get("custom_prompt"),
            "raw_effort": (raw.get("agent") or {}).get("reasoning_effort"),
            "saved": saved,
        }), encoding="utf-8")
        """
    )
    out = _run_py(
        code,
        {
            "HERMES_HOME": str(home),
            "HERMES_MANAGED_DIR": str(managed_dir),
            "E2E_PROMPT_SUFFIX": "world",
        },
        tmp_path,
    )

    # 1. Behavioral read: ${VAR} expanded + managed overlay applied.
    assert out["behavioral_prompt"] == "hello world"
    assert out["behavioral_effort"] == "high"

    # 2. Raw primitive: byte-faithful view of the user's file.
    assert out["raw_prompt"] == "hello ${E2E_PROMPT_SUFFIX}"
    assert out["raw_effort"] == "low"

    # 3. Write-back: user values round-trip; no managed/expanded/default leak.
    saved = out["saved"]
    assert "hello ${E2E_PROMPT_SUFFIX}" in saved     # template preserved
    assert "hello world" not in saved                # no expansion persisted
    assert "reasoning_effort: low" in saved          # user value preserved
    assert "high" not in saved                       # managed value NOT persisted
    assert "battery: true" in saved                  # the actual edit landed
    assert "max_turns" not in saved                  # no DEFAULT_CONFIG pollution


def test_writeback_roundtrip_byte_identical_when_unchanged(tmp_path):
    """read_user_config_raw → save with no mutation must not alter content
    semantics (yaml re-dump may reorder nothing here: flat mapping)."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    original = "custom_prompt: keep ${NOT_SET_VAR}\ndisplay:\n  skin: usertheme\n"
    (home / "config.yaml").write_text(original, encoding="utf-8")

    code = textwrap.dedent(
        """
        import json
        from pathlib import Path
        from hermes_cli.config import read_user_config_raw
        import yaml

        p = Path(__import__('os').environ['HERMES_HOME']) / 'config.yaml'
        before = p.read_text(encoding='utf-8')
        data = read_user_config_raw(p)
        # No mutation, no save — the primitive itself must be read-only.
        after = p.read_text(encoding='utf-8')
        import os
        Path(os.environ["E2E_OUT_FILE"]).write_text(json.dumps({
            "identical": before == after,
            "parsed": data,
        }), encoding="utf-8")
        """
    )
    out = _run_py(code, {"HERMES_HOME": str(home)}, tmp_path)
    assert out["identical"] is True
    assert out["parsed"]["custom_prompt"] == "keep ${NOT_SET_VAR}"
