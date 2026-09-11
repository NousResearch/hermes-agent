"""The packaged update scanner must be generated from its single source module.

``apps/desktop/resources/update-scanner/scan-venv-blockers.py`` is the only
copy the Desktop ever runs. It used to be hand-synced with
``hermes_cli/_scan_venv_blockers.py``, drifted by 160 lines, and shipped a
``_redact_sensitive_cmdline`` stub that blanked every command line — which
silently disabled holder classification on the only build that ships
(#104687 B3/B5).

These tests run ``scripts/desktop-update/sync_scan_venv_blockers.py --check``
in-process, so ordinary Python CI enforces parity with no extra workflow and no
subprocess.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR_PATH = REPO_ROOT / "scripts" / "desktop-update" / "sync_scan_venv_blockers.py"


def _load_generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_sync_scan_venv_blockers", GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


@pytest.fixture(scope="module")
def generator() -> ModuleType:
    return _load_generator()


def test_generator_script_exists() -> None:
    assert GENERATOR_PATH.is_file(), (
        "the carrier is generated; scripts/desktop-update/sync_scan_venv_blockers.py "
        "is the source of the substitution list"
    )


def test_committed_carrier_matches_generated_output(generator: ModuleType) -> None:
    """The whole point: no hand-edit of the carrier can survive CI."""
    drift = generator.carrier_drift()

    assert drift == "", (
        "apps/desktop/resources/update-scanner/scan-venv-blockers.py is stale. Run "
        "`python scripts/desktop-update/sync_scan_venv_blockers.py --write`.\n" + drift
    )


def test_every_substitution_matches_exactly_once(generator: ModuleType) -> None:
    """A moved source region must fail loudly rather than silently no-op."""
    source = generator.SOURCE_PATH.read_text(encoding="utf-8")

    for substitution in generator.SUBSTITUTIONS:
        assert source.count(substitution.old) == 1, (
            f"substitution {substitution.name!r} no longer matches exactly once in "
            f"{generator.SOURCE_PATH.name}"
        )


def test_substitution_list_is_documented_in_both_docstrings(generator: ModuleType) -> None:
    """The docstrings must not claim a count the substitution list contradicts.

    The pre-fix carrier said "five substitutions" and listed four.
    """
    assert len(generator.SUBSTITUTIONS) == 6

    carrier = generator.CARRIER_PATH.read_text(encoding="utf-8")
    assert "Exactly six deliberate" in carrier

    source = generator.SOURCE_PATH.read_text(encoding="utf-8")
    assert "sync_scan_venv_blockers.py" in source
    assert "test_scan_venv_blockers_parity.py" in source


def test_redaction_is_not_substituted(generator: ModuleType) -> None:
    """B3: the carrier must redact exactly as the module does, not blank everything."""
    substituted = "".join(substitution.old for substitution in generator.SUBSTITUTIONS)
    assert "_redact_sensitive_cmdline" not in substituted

    carrier = generator.CARRIER_PATH.read_text(encoding="utf-8")
    source = generator.SOURCE_PATH.read_text(encoding="utf-8")

    for text, label in ((carrier, "carrier"), (source, "module")):
        assert "_SECRET_TOKEN_RE" in text, f"{label} lost the inline secret matcher"
        assert "def _mask_secret_values" in text, f"{label} lost the inline masker"

    marker = "def _redact_sensitive_cmdline"
    carrier_body = carrier[carrier.index(marker) : carrier.index("def _classify_local_preview_args")]
    source_body = source[source.index(marker) : source.index("def _classify_local_preview_args")]
    assert carrier_body == source_body


def test_carrier_imports_nothing_from_the_mutable_checkout(generator: ModuleType) -> None:
    """The carrier runs under the target venv against the root being rewritten."""
    carrier = generator.CARRIER_PATH.read_text(encoding="utf-8")
    forbidden = (
        "agent.redact",
        "hermes_cli.process_identity",
        "hermes_cli.update_cmd",
        "hermes_constants",
        "hermes_mcp_update_gate",
    )

    for name in forbidden:
        assert f"import {name}" not in carrier, f"carrier imports {name} from the target checkout"
        assert f"from {name} import" not in carrier


def test_render_carrier_rejects_a_missing_region(generator: ModuleType) -> None:
    """`--check` must fail loudly, not silently emit an unsubstituted carrier."""
    source = generator.SOURCE_PATH.read_text(encoding="utf-8")
    mangled = source.replace(generator.SUBSTITUTIONS[1].old, "", 1)

    with pytest.raises(SystemExit):
        generator.render_carrier(mangled)
