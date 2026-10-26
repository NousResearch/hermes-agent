"""``providers.discovery_in_progress()`` must keep lazy readers from latching.

Companion to ``test_lazy_canonical_providers.py``. That file proves the
auto-extend is lazy; this one proves the *other half* of the contract:

``providers._discovered`` is set to ``True`` at the TOP of
``_discover_providers()`` (so re-entrant ``list_providers()`` calls don't
recurse). That means a plugin imported by that very pass sees a registry that
is only half-populated. If such a plugin reads ``CANONICAL_PROVIDERS`` during
its own import, the lazy auto-extend would run against the partial registry
and cache the result as final — permanently dropping every provider whose
plugin sorts AFTER the reader.

``discovery_in_progress()`` exists so the lazy readers can tell that case
apart and decline to latch. This test pins that: an early-sorting plugin that
reads ``CANONICAL_PROVIDERS`` at import must not hide a later-sorting plugin's
provider.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Sorts FIRST in the plugin directory listing: reads CANONICAL_PROVIDERS while
# discovery is still walking the directory.
READER_INIT = textwrap.dedent(
    """
    import builtins
    from hermes_cli.models import CANONICAL_PROVIDERS
    builtins._MID_DISCOVERY_READ_N = len(CANONICAL_PROVIDERS)
    """
)

# Sorts LAST: registers a provider that must still reach CANONICAL_PROVIDERS.
LATE_INIT = textwrap.dedent(
    """
    from providers import register_provider
    from providers.base import ProviderProfile

    register_provider(ProviderProfile(
        name="zz-late-registrant",
        display_name="Late Registrant",
        description="Late Registrant (test fixture)",
        base_url="https://late.invalid/v1",
        env_vars=("ZZ_LATE_REGISTRANT_KEY",),
    ))
    """
)

DRIVER = textwrap.dedent(
    """
    import builtins, json

    import hermes_cli.models as models
    import providers

    providers.list_providers()

    out = {
        "mid_discovery_read_n": getattr(builtins, "_MID_DISCOVERY_READ_N", None),
        "registry_has_late": any(
            p.name == "zz-late-registrant" for p in providers.list_providers()
        ),
        "canonical_slugs": [p.slug for p in models.CANONICAL_PROVIDERS],
        "late_in_canonical": "zz-late-registrant" in [
            p.slug for p in models.CANONICAL_PROVIDERS
        ],
        # Every provider in the registry that the auto-extend is supposed to
        # inject (api_key auth) must be present — an invariant, not a count.
        "registry_api_key_names": [
            p.name for p in providers.list_providers() if p.auth_type == "api_key"
        ],
    }
    print("PROBE_JSON " + json.dumps(out))
    """
)


def test_mid_discovery_read_does_not_drop_later_plugins(tmp_path):
    home = tmp_path / "hermes_home"
    root = home / "plugins" / "model-providers"

    reader = root / "aa-mid-discovery-reader"
    reader.mkdir(parents=True)
    (reader / "__init__.py").write_text(READER_INIT)
    (reader / "plugin.yaml").write_text(
        "name: aa-mid-discovery-reader\nkind: model-provider\nversion: 0.0.1\n"
    )

    late = root / "zz-late-registrant"
    late.mkdir(parents=True)
    (late / "__init__.py").write_text(LATE_INIT)
    (late / "plugin.yaml").write_text(
        "name: zz-late-registrant\nkind: model-provider\nversion: 0.0.1\n"
    )

    driver = tmp_path / "driver.py"
    driver.write_text(DRIVER)

    env = dict(os.environ)
    env["HERMES_HOME"] = str(home)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.run(
        [sys.executable, str(driver)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("PROBE_JSON ")]
    assert lines, (
        f"driver produced no verdict (rc={proc.returncode})\n"
        f"--- stdout ---\n{proc.stdout[-4000:]}\n--- stderr ---\n{proc.stderr[-4000:]}"
    )
    out = json.loads(lines[-1][len("PROBE_JSON "):])

    # Non-vacuity: the reader must actually have run mid-discovery, and the
    # late plugin must actually have registered. Without both, the assertion
    # below proves nothing.
    assert out["mid_discovery_read_n"] is not None, (
        "the mid-discovery reader plugin never ran — test is vacuous"
    )
    assert out["registry_has_late"] is True, (
        "the late plugin never registered — test is vacuous"
    )

    assert out["late_in_canonical"] is True, (
        "a plugin that read CANONICAL_PROVIDERS mid-discovery caused the lazy "
        "auto-extend to latch against a half-populated registry; the "
        "later-sorting plugin's provider was dropped permanently"
    )
    # Generalised form of the same invariant: NO api_key provider in the
    # registry may be missing from CANONICAL_PROVIDERS. A count assertion
    # would pass while an arbitrary subset silently went missing.
    missing = [
        name for name in out["registry_api_key_names"]
        if name not in out["canonical_slugs"]
    ]
    assert not missing, (
        f"registered api_key providers absent from CANONICAL_PROVIDERS: {missing}"
    )
