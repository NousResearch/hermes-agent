"""Behavior-parity check for the image-gen FAL plugin migration (#26241).

Spawns one subprocess per (version, scenario) cell — pinned to either
``origin/main`` (legacy in-tree FAL fall-through + ``configured == "fal"``
skip in ``_dispatch_to_plugin_provider``) or this PR's worktree (FAL is
itself a plugin and the dispatcher routes every set provider through
the registry). Each subprocess clears all FAL-related env vars + writes
a ``config.yaml``, then asks the dispatcher how it would route an
``image_generate`` call. The emitted shape tuple is
``{dispatch_kind, provider_name, model}``:

* ``dispatch_kind`` ∈ ``{"legacy_fal", "plugin", "error", None}`` —
  whether the call would go straight to the in-tree pipeline,
  through ``_dispatch_to_plugin_provider``, raise an explicit
  provider-not-registered error, or fall through silently.
* ``provider_name`` — when ``dispatch_kind == "plugin"``, the
  resolved provider name. ``None`` otherwise.
* ``model`` — the resolved FAL model id when applicable.

The parent process diffs the shapes per scenario. A diff means the
migration introduced an observable behaviour change vs origin/main —
likely a real regression for users on the existing config keys.

Run from the PR worktree:

    python tests/plugins/image_gen/check_parity_vs_main.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


# Pin one path to current main, one to the PR worktree.
# ``REPO_ROOT`` is ``.../.worktrees/<name>``; the main checkout lives
# two levels up. When running directly from a regular clone (no
# worktree), ``MAIN_DIR`` falls back to a sibling ``hermes-agent-main``
# checkout if one exists.
def _resolve_main_dir() -> Path:
    candidate = REPO_ROOT.parent.parent
    if (candidate / "tools" / "image_generation_tool.py").exists() and candidate != REPO_ROOT:
        return candidate
    sibling = REPO_ROOT.parent / "hermes-agent-main"
    if (sibling / "tools" / "image_generation_tool.py").exists():
        return sibling
    return REPO_ROOT


MAIN_DIR = _resolve_main_dir()
PR_DIR = REPO_ROOT
assert (PR_DIR / "tools" / "image_generation_tool.py").exists(), (
    f"PR_DIR={PR_DIR} doesn't look like a hermes-agent checkout"
)


SUBPROCESS_SCRIPT = r"""
import json, os, sys, tempfile
sys.path.insert(0, sys.argv[1])

# Isolated HERMES_HOME so the config write is hermetic.
home = tempfile.mkdtemp()
os.environ["HERMES_HOME"] = home

# Clear FAL-related env so dispatch decisions are config-driven.
for k in (
    "FAL_KEY", "FAL_QUEUE_GATEWAY_URL",
    "TOOL_GATEWAY_DOMAIN", "TOOL_GATEWAY_USER_TOKEN",
    "FAL_IMAGE_MODEL",
):
    os.environ.pop(k, None)

scenario_env = json.loads(sys.argv[2])
os.environ.update(scenario_env)

config_yaml = sys.argv[3]
config_path = os.path.join(home, "config.yaml")
with open(config_path, "w") as f:
    f.write(config_yaml)

# Fresh import — must not have anything cached.
for name in list(sys.modules):
    if (name.startswith("tools.")
            or name.startswith("agent.")
            or name.startswith("plugins.")
            or name.startswith("hermes_cli.")):
        sys.modules.pop(name, None)

import tools.image_generation_tool as image_tool

dispatch_kind = None
provider_name = None
model = None
error_text = None

try:
    raw = image_tool._dispatch_to_plugin_provider("ping", "landscape")
    if raw is None:
        dispatch_kind = "legacy_fal"
    else:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(parsed, dict):
            if parsed.get("error_type") == "provider_not_registered":
                dispatch_kind = "error"
                error_text = parsed.get("error")
            else:
                dispatch_kind = "plugin"
                provider_name = parsed.get("provider")
                model = parsed.get("model")
        else:
            dispatch_kind = "unknown_payload"

    if model is None:
        # _resolve_fal_model still returns the active FAL model id even
        # when dispatch goes to a non-FAL plugin — used for the diff
        # only when applicable.
        try:
            model_id, _meta = image_tool._resolve_fal_model()
            if dispatch_kind == "legacy_fal":
                model = model_id
        except Exception:
            pass
except Exception as exc:
    dispatch_kind = "exception"
    error_text = repr(exc)

shape = {
    "dispatch_kind": dispatch_kind,
    "provider_name": provider_name,
    "model": model,
    "error_present": error_text is not None,
}
print(json.dumps(shape))
"""


SCENARIOS: list[tuple[str, str, dict[str, str]]] = [
    # (label, config.yaml body, extra env vars)
    ("no-config-no-env", "", {}),
    (
        "explicit-fal-no-creds",
        "image_gen:\n  provider: fal\n",
        {},
    ),
    (
        "explicit-fal-with-creds",
        "image_gen:\n  provider: fal\n",
        {"FAL_KEY": "test-key"},
    ),
    (
        "explicit-fal-with-model",
        "image_gen:\n  provider: fal\n  model: fal-ai/flux-2-pro\n",
        {"FAL_KEY": "test-key"},
    ),
    (
        "explicit-typo-provider",
        "image_gen:\n  provider: not-a-real-backend\n",
        {"FAL_KEY": "test-key"},
    ),
    (
        "managed-gateway-only",
        "",
        {
            "TOOL_GATEWAY_DOMAIN": "nousresearch.com",
            "TOOL_GATEWAY_USER_TOKEN": "nous-token",
        },
    ),
]


def _run_scenario(repo_path: Path, label: str, config_yaml: str, env: dict) -> dict:
    venv_python = repo_path / ".venv" / "bin" / "python"
    if not venv_python.exists():
        venv_python = MAIN_DIR / ".venv" / "bin" / "python"
    if not venv_python.exists():
        venv_python = Path("python3")

    out = subprocess.run(
        [
            str(venv_python),
            "-c",
            SUBPROCESS_SCRIPT,
            str(repo_path),
            json.dumps(env),
            config_yaml,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if out.returncode != 0:
        return {
            "error": "subprocess failed",
            "stdout": out.stdout[-500:],
            "stderr": out.stderr[-500:],
        }
    try:
        return json.loads(out.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return {"error": f"could not parse output: {exc}", "stdout": out.stdout}


def _reduce(shape: dict) -> dict:
    """Reduce to the parts that matter for user-visible parity.

    On origin/main, ``explicit-fal-*`` scenarios short-circuit to
    ``legacy_fal`` because of the ``configured == "fal"`` skip. On the
    PR, those same scenarios route through the plugin and emit
    ``dispatch_kind == "plugin"`` with ``provider_name == "fal"``.

    Both shapes are functionally equivalent — the plugin's ``generate()``
    re-enters the same in-tree pipeline via ``_it`` indirection — but
    we want the diff to be visible so reviewers can sign off on the
    intentional behaviour delta.
    """
    return {
        "dispatch_kind": shape.get("dispatch_kind"),
        "provider_name": shape.get("provider_name"),
        "model": shape.get("model"),
        "error_present": shape.get("error_present"),
    }


def main() -> int:
    print(f"main:    {MAIN_DIR}")
    print(f"pr:      {PR_DIR}")
    print()

    if MAIN_DIR == PR_DIR:
        print(
            "WARN: MAIN_DIR == PR_DIR — diffs will be trivially identical.\n"
            "      Set up a sibling 'hermes-agent-main' checkout pinned to "
            "origin/main to get real parity coverage."
        )
        print()

    failures: list[str] = []
    errors: list[str] = []
    intentional_diffs: list[tuple[str, dict, dict]] = []
    for label, config_yaml, env in SCENARIOS:
        main_shape = _run_scenario(MAIN_DIR, label, config_yaml, env)
        pr_shape = _run_scenario(PR_DIR, label, config_yaml, env)

        if "error" in main_shape or "error" in pr_shape:
            print(f"  [ERR ] {label}: subprocess failed")
            print(f"    main: {main_shape}")
            print(f"    pr:   {pr_shape}")
            errors.append(label)
            continue

        main_reduced = _reduce(main_shape)
        pr_reduced = _reduce(pr_shape)

        if main_reduced == pr_reduced:
            print(f"  [OK]   {label}: {main_reduced}")
            continue

        # On main, "explicit-fal-*" returns legacy_fal; on PR, plugin
        # dispatch. That's the only acceptable diff — flag everything
        # else as a regression.
        legacy_to_plugin_fal = (
            main_reduced.get("dispatch_kind") == "legacy_fal"
            and pr_reduced.get("dispatch_kind") == "plugin"
            and pr_reduced.get("provider_name") == "fal"
        )
        if legacy_to_plugin_fal:
            print(f"  [DIFF] {label}: legacy_fal → plugin (fal) — expected")
            intentional_diffs.append((label, main_reduced, pr_reduced))
        else:
            print(f"  [FAIL] {label}")
            print(f"    main: {main_reduced}")
            print(f"    pr:   {pr_reduced}")
            failures.append(label)

    print()
    if errors:
        print(f"SUBPROCESS ERRORS in {len(errors)} scenario(s):")
        for e in errors:
            print(f"  - {e}")
    if failures:
        print(f"BEHAVIOUR REGRESSION in {len(failures)} scenario(s):")
        for f in failures:
            print(f"  - {f}")
    if intentional_diffs:
        print(
            f"INTENTIONAL DIFFS ({len(intentional_diffs)}): "
            f"legacy_fal → plugin dispatch for explicit FAL paths."
        )
    if failures or errors:
        return 1
    print(f"PARITY OK across {len(SCENARIOS)} scenarios.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
