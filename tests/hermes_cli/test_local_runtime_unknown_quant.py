"""Regression for #123247: ternary-style quant types must not brick the managed runtime.

A GGUF whose tensor types sit above the pinned engine's type table (ggml type 142 in the
field report) is structurally valid — fork engines load and serve it. The header reader must
therefore parse the file, and the launch policy must record an explicit refusal instead of
leaving the model without a preset section (which made the router 400 on the model name and
kept ``_presets_stale()`` true forever, so every boot stopped and replaced the server).
"""

from __future__ import annotations

import struct

from hermes_cli.local_runtime.estimator import HardwareBudget
from hermes_cli.local_runtime.gguf import read_gguf_header
from hermes_cli.local_runtime.presets import generate_presets, read_preset_decisions


def _gguf_with_tensor_type(path, ttype: int) -> None:
    name = b"output.weight"
    path.write_bytes(
        b"GGUF"
        + struct.pack("<IQQ", 3, 1, 0)
        + struct.pack("<Q", len(name)) + name
        + struct.pack("<IQIQ", 1, 64, ttype, 0)
    )


def test_header_parses_forward_compat_quant_type_and_flags_it(tmp_path):
    """ggml type 142 (ternary PQ) parses; the flag marks the file not runnable on stock."""
    gguf = tmp_path / "Ternary-Bonsai-2-27B-PQ2_0.gguf"
    _gguf_with_tensor_type(gguf, 142)
    header = read_gguf_header(gguf)
    assert header.has_unknown_quant_types is True


def test_known_types_leave_the_flag_off(tmp_path):
    gguf = tmp_path / "normal.gguf"
    _gguf_with_tensor_type(gguf, 39)  # MXFP4, priced in the table
    header = read_gguf_header(gguf)
    assert header.has_unknown_quant_types is False
    assert header.tensor_bytes == 34


def test_unknown_type_below_the_table_is_still_a_hard_error(tmp_path):
    """A type inside the historic range but absent from the table is corruption, not a fork quant."""
    import pytest

    gguf = tmp_path / "corrupt.gguf"
    _gguf_with_tensor_type(gguf, 63)
    with pytest.raises(ValueError, match="unknown ggml tensor type 63"):
        read_gguf_header(gguf)


def test_preset_records_refusal_instead_of_omitting_the_model(tmp_path, monkeypatch):
    """The refusal must reach the decision record: an omission is what 400'd the router and
    kept presets permanently stale (#123247)."""
    from hermes_cli.local_runtime.bootstrap import _presets_stale

    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    mdir = home / "models"
    mdir.mkdir(parents=True)
    ternary = mdir / "Ternary-Bonsai-2-27B-PQ2_0.gguf"
    healthy = mdir / "healthy.gguf"
    _gguf_with_tensor_type(ternary, 142)
    _gguf_with_tensor_type(healthy, 39)

    ini = home / "runtimes" / "llamacpp" / "presets.ini"
    budget = HardwareBudget(8 << 30, 8 << 30, 8 << 30)
    entries = generate_presets(mdir, budget, ini)

    by_id = {e.model_id: e for e in entries}
    assert by_id["Ternary-Bonsai-2-27B-PQ2_0"].refusal
    assert by_id["Ternary-Bonsai-2-27B-PQ2_0"].window == 0
    assert by_id["healthy"].keys is not None

    decisions = read_preset_decisions(ini)
    assert decisions["healthy"].keys["model"] == str(healthy)
    # The ternary refusal stays in the record (picker/log can say WHY), but no launch section.
    assert decisions["Ternary-Bonsai-2-27B-PQ2_0"].refusal
    assert not decisions["Ternary-Bonsai-2-27B-PQ2_0"].keys

    # The boot-adopt invariant: a generated policy covers every staged model.
    assert _presets_stale() is False


def test_unpriceable_unknown_quant_model_never_shrinks_residency(tmp_path, monkeypatch):
    """Its weights are unpriced, so the file must not feed the residency arithmetic."""
    from hermes_cli.local_runtime.estimator import HardwareBudget
    from hermes_cli.local_runtime.presets import admitted_residency_count

    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    mdir = home / "models"
    mdir.mkdir(parents=True)
    _gguf_with_tensor_type(mdir / "ternary.gguf", 142)
    budget = HardwareBudget(24 << 30, 24 << 30, 16 << 30)
    assert admitted_residency_count(mdir, budget, 2) == 2
