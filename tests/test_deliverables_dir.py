"""User-facing deliverables must survive the gateway media sweep (#126419).

``cache/scratch`` is the transient scratch space: entries idle for
``SCRATCH_MAX_IDLE_HOURS`` (24h) are pruned. Deliverable Mode pointed users at
that directory for files they are meant to keep — the chart, the PDF report —
so the deliverable silently disappeared a day after its last write.

``get_deliverables_dir()`` is the durable home: nothing prunes it, and the
gateway's media-delivery allowlist vouches for files under it, so a day-old
deliverable still attaches to a chat on a strict gateway with recency trust off.

RED on main: there is no ``get_deliverables_dir``, and the doc points at
``cache/scratch``, which the prune deletes.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from gateway.platforms import base
from hermes_constants import get_deliverables_dir, prune_scratch_dir


def _backdate_hours(path, hours: float = 25.0) -> None:
    old = time.time() - hours * 3600
    os.utime(path, (old, old))


def test_deliverables_dir_is_created_owner_only(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    deliverables = get_deliverables_dir()

    assert deliverables == tmp_path / "home" / "deliverables"
    assert deliverables.is_dir()

    if os.name != "nt":
        assert deliverables.stat().st_mode & 0o777 == 0o700


def test_deliverable_survives_the_scratch_prune(tmp_path, monkeypatch):
    """The whole point: a deliverable is still there a day later."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    deliverables = get_deliverables_dir()
    report = deliverables / "q3-report.pdf"
    report.write_bytes(b"%PDF-1.4 deliverable")

    _backdate_hours(report)

    # Scratch pruning is what used to delete deliverables; it must not touch
    # this tree. Run it against the scratch dir so the prune path is exercised.
    from hermes_constants import get_scratch_dir
    prune_scratch_dir(get_scratch_dir(prune=False))

    assert report.exists(), "the deliverable was swept by the scratch prune"
    assert report.read_bytes() == b"%PDF-1.4 deliverable"


def test_deliverable_is_deliverable_on_a_strict_gateway(tmp_path, monkeypatch):
    """A day-old deliverable must still attach to a chat.

    Strict mode with recency trust off accepts a file only when an allowlist
    root vouches for it, so a deliverable that is not allowlisted would fail to
    attach exactly when the user asks for it again the next day.

    A profile home is used on purpose: the static safe roots are built at import
    time, while the per-profile roots are enumerated at check time — so this
    also covers the profile-scoped deliverables dir.
    """
    home = tmp_path / "profiles" / "p"
    home.mkdir(parents=True)
    monkeypatch.setattr(base, "_HERMES_ROOT", tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    deliverables = get_deliverables_dir()
    report = deliverables / "q3-report.pdf"
    report.write_bytes(b"%PDF-1.4 deliverable")
    _backdate_hours(report)

    monkeypatch.setenv("HERMES_MEDIA_DELIVERY_STRICT", "1")
    monkeypatch.setenv("HERMES_MEDIA_TRUST_RECENT_FILES", "0")

    assert base.validate_media_delivery_path(str(report)) == str(report.resolve())


def test_deliverables_dir_is_not_a_scratch_entry(tmp_path, monkeypatch):
    """The deliverables tree must not live under the swept scratch dir."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    deliverables = get_deliverables_dir()

    assert "cache" not in deliverables.parts
    assert deliverables.name != "scratch"


def test_doc_points_at_the_durable_directory():
    """Deliverable Mode must not send users back to the swept scratch dir.

    The kanban ``artifacts=[...]`` example is deliberately left alone: a kanban
    artifact is copied into the task's durable attachments only when it lives in
    that task's own workspace, so the workspace path is the right answer there
    and a separate change owns it.
    """
    doc = Path(__file__).resolve().parents[1] / "website" / "docs" / "user-guide" / "features" / "deliverable-mode.md"
    text = doc.read_text(encoding="utf-8")

    assert "~/.hermes/deliverables/" in text, "the doc never names the deliverables dir"
    assert "~/.hermes/deliverables/q3-revenue.png" in text, (
        "the doc's worked example still points at the swept scratch dir"
    )

    # Outside the kanban block, no deliverable path may name scratch.
    before_kanban = text.split("## Kanban: artifacts ride completion notifications")[0]
    assert "~/.hermes/cache/scratch/" not in before_kanban, (
        "the doc still points deliverables at the 24h-pruned scratch dir"
    )
