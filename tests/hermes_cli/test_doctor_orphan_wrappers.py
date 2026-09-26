"""Regression: orphan-alias check must ignore shell-variable profile targets."""

from __future__ import annotations

from types import SimpleNamespace

from hermes_cli import doctor_state, profiles


def test_orphan_alias_only_flags_literal_missing_profiles(tmp_path, monkeypatch):
    """`hermes -p "$PROFILE"` is not an orphan; `hermes -p <missing>` still is."""
    (tmp_path / "browser-mode").write_text(
        '#!/bin/sh\nPROFILE="trabajo"\ncfg() { hermes -p "$PROFILE" config "$@"; }\n'
        'hermes -p $PROFILE x; hermes -p ${PROFILE} y; hermes -p "$1" z\n'
    )
    (tmp_path / "viejo").write_text("#!/bin/sh\nexec hermes -p fantasma chat\n")
    (tmp_path / "sano").write_text("#!/bin/sh\nexec hermes -p 'trabajo' chat\n")

    named = SimpleNamespace(
        is_default=False, name="trabajo", path=tmp_path / "trabajo",
        gateway_running=False, model=None,
    )
    monkeypatch.setattr(profiles, "list_profiles", lambda: [named])
    monkeypatch.setattr(profiles, "_get_wrapper_dir", lambda: tmp_path)
    monkeypatch.setattr(profiles, "profile_exists", lambda name: name == "trabajo")
    warnings: list[str] = []
    monkeypatch.setattr(doctor_state, "check_warn", lambda msg, *a, **k: warnings.append(msg))
    monkeypatch.setattr(doctor_state, "check_ok", lambda *a, **k: None)
    monkeypatch.setattr(doctor_state, "_section", lambda *a, **k: None)

    doctor_state._check_profiles(False)

    orphans = [w for w in warnings if w.startswith("Orphan alias")]
    assert len(orphans) == 1
    assert "viejo" in orphans[0] and "fantasma" in orphans[0]
