"""Regression test: ``hermes pm`` must see the home dotenv before it dispatches.

The ``pm`` subcommand dispatches from ``hermes_cli/main.py`` at a point that
used to sit ABOVE the module-level ``load_hermes_dotenv()`` call. Every other
entry point (the CLI itself, the desktop update probe) loaded the home dotenv
first, but ``hermes pm`` did not — so ``pm`` ran with whatever the parent shell
happened to export and nothing else.

That is invisible until the one thing ``pm`` reads from the environment is a
credential. ``pm.update._github_headers()`` lifts the api.github.com quota from
the anonymous 60 requests/hour **per exit IP** to 5000 requests/hour **per
account** using ``GITHUB_TOKEN`` / ``GH_TOKEN``. On a shared egress (office
NAT, Clash/VPN node) the anonymous bucket is spent by every other client
behind the same address, so every GitHub-sourced package failed to resolve:

    ffmpeg   resolve failed: HTTP Error 403: rate limit exceeded
    gh       resolve failed: HTTP Error 403: rate limit exceeded
    git      resolve failed: HTTP Error 403: rate limit exceeded
    ripgrep  resolve failed: HTTP Error 403: rate limit exceeded
    uv       resolve failed: HTTP Error 403: rate limit exceeded

The token was present the whole time in ``<HERMES_HOME>/.env`` — the file the
rest of the codebase reads configuration from. It simply had not been loaded
yet when ``pm`` started.

This test drives the REAL entrypoint in a subprocess (the only way to observe
module import order) against a temp ``HERMES_HOME`` whose ``.env`` carries a
token, and asserts the value is visible in ``os.environ`` by the time ``pm``
runs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# A dotenv value whose presence in os.environ is unambiguous. The real
# GITHUB_TOKEN is a secret and must never be echoed; this marker stands in for
# it, and the assertion is about ORDERING, not about the token's value.
_DOTENV_MARKER = "hermes-pm-dotenv-ordering-marker"


def _run_pm_probe(env_home: Path) -> dict:
    """Run ``hermes pm`` in a subprocess and report what pm saw in os.environ.

    ``pm.cli.main`` is monkeypatched through a sitecustomize-free probe: the
    child imports ``pm.cli``, whose ``main`` is replaced before
    ``hermes_cli.main`` dispatches it. The probe prints one JSON line with the
    environment snapshot taken *inside* the pm dispatch, i.e. exactly what the
    pm code path could see.
    """
    probe = (
        "import json, os, sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "import pm.cli as _pm_cli\n"
        "def _probe_main(argv):\n"
        "    sys.stdout.write('HERMES_PM_PROBE:' + json.dumps({\n"
        "        'token': os.environ.get('GITHUB_TOKEN'),\n"
        "        'gh_token': os.environ.get('GH_TOKEN'),\n"
        "    }) + '\\n')\n"
        "    return 0\n"
        "_pm_cli.main = _probe_main\n"
        "import hermes_cli.main\n"
    )
    child_env = dict(os.environ)
    child_env["HERMES_HOME"] = str(env_home)
    # A clean slate: the marker must arrive via the dotenv, never via the
    # parent's environment.
    child_env.pop("GITHUB_TOKEN", None)
    child_env.pop("GH_TOKEN", None)
    child_env["HERMES_TEST_SANDBOX_HOME"] = str(env_home)
    child_env["PYTHONPATH"] = str(ROOT)

    completed = subprocess.run(
        [sys.executable, "-c", probe, "pm", "--help"],
        capture_output=True,
        text=True,
        env=child_env,
        cwd=str(ROOT),
        timeout=180,
        check=False,
    )
    for line in completed.stdout.splitlines():
        if line.startswith("HERMES_PM_PROBE:"):
            return json.loads(line[len("HERMES_PM_PROBE:"):])
    raise AssertionError(
        "pm dispatch probe never ran.\n"
        f"exit={completed.returncode}\n"
        f"stdout tail: {completed.stdout[-800:]}\n"
        f"stderr tail: {completed.stderr[-800:]}"
    )


def test_pm_dispatch_sees_home_dotenv(tmp_path):
    """``hermes pm`` must observe GITHUB_TOKEN from the home dotenv."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    (home / ".env").write_text(f"GITHUB_TOKEN={_DOTENV_MARKER}\n", encoding="utf-8")

    seen = _run_pm_probe(home)

    assert seen["token"] == _DOTENV_MARKER, (
        "the pm subcommand dispatched before load_hermes_dotenv() ran, so it "
        "never saw GITHUB_TOKEN from the home dotenv. On a shared proxy exit "
        "this silently downgrades pm's api.github.com calls to the anonymous "
        "60/hour per-IP bucket, which every other client behind the same "
        "address also spends."
    )


def test_pm_dispatch_does_not_need_the_token(tmp_path):
    """The ordering fix must not make a token REQUIRED to run pm.

    A machine with no token in its dotenv must still be able to run
    ``hermes pm`` — it simply gets the anonymous quota, as before. This is the
    regression guard on the fix itself: publishing the dotenv early must not
    turn a missing credential into a hard failure.
    """
    home = tmp_path / "hermes-home-empty"
    home.mkdir()
    (home / ".env").write_text("# no credentials here\n", encoding="utf-8")

    seen = _run_pm_probe(home)

    assert seen["token"] is None, (
        "a GITHUB_TOKEN appeared in os.environ without the dotenv defining "
        "one — the ordering fix must not synthesize or leak a credential"
    )
