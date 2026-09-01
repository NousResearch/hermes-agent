"""Runtime install progress: the slow legs must tick, never hang silently.

The incident: 'Installing runtime…' sat frozen for minutes on a slow
line — ensure_runtime_installed downloaded and extracted with no
progress stream, so both the quickstart hero and the pane's install row
showed a dead bar. Contract: _download and _extract tick per chunk /
per member, ensure_runtime_installed forwards a staged stream, and the
router's hook translates it into live job fields."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import hermes_cli.local_runtime.binaries as binaries
from hermes_cli.web_routers.local_models import _job, _runtime_progress_hook


def _make_zip(path: Path, names_sizes: dict[str, int]) -> None:
    with zipfile.ZipFile(path, "w") as z:
        for name, size in names_sizes.items():
            z.writestr(name, b"x" * size)


def test_extract_ticks_per_member(tmp_path):
    archive = tmp_path / "runtime.zip"
    _make_zip(archive, {"a.bin": 1000, "b.bin": 3000, "c.bin": 500})
    ticks: list[tuple[int, int]] = []
    binaries._extract(archive, tmp_path / "out",
                      progress=lambda d, t: ticks.append((d, t)))
    assert len(ticks) == 3
    total = 4500
    assert all(t == total for _, t in ticks)
    assert [d for d, _ in ticks] == sorted(d for d, _ in ticks)
    assert ticks[-1][0] == total
    assert (tmp_path / "out" / "b.bin").stat().st_size == 3000


def test_download_ticks_with_content_length(tmp_path, monkeypatch):
    payload = b"y" * (3 << 20)  # 3 MiB -> several 1 MiB chunks

    class _Resp(io.BytesIO):
        headers = {"Content-Length": str(len(payload))}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(binaries.urllib.request, "urlopen",
                        lambda url, timeout=0: _Resp(payload))
    ticks: list[tuple[int, int]] = []
    dest = tmp_path / "asset.zip"
    binaries._download("http://x/asset.zip", dest,
                       progress=lambda d, t: ticks.append((d, t)))
    assert dest.read_bytes() == payload
    assert len(ticks) >= 3
    assert ticks[-1] == (len(payload), len(payload))


def test_ensure_runtime_installed_forwards_staged_progress(tmp_path, monkeypatch):
    """The full install path emits download -> verify -> extract stages
    (per asset) and a final verify, all through one callback."""
    monkeypatch.setattr(binaries, "runtimes_root", lambda: tmp_path)

    class _Plan:
        assets = ["a.zip", "b.zip"]
        backend = "cuda"
        install_dir = tmp_path / "b1" / "cuda"

    monkeypatch.setattr(binaries, "resolve_assets", lambda tag, backend: _Plan())
    monkeypatch.setattr(binaries, "verify_install", lambda d, t: "ok")

    def _fake_download(url, dest, progress=None):
        _make_zip(dest, {"f.bin": 2048})
        if progress is not None:
            progress(1024, 2048)
            progress(2048, 2048)

    monkeypatch.setattr(binaries, "_download", _fake_download)

    events: list[tuple[str, str]] = []
    binaries.ensure_runtime_installed(
        "b1", "cuda",
        progress=lambda stage, d, t, label: events.append((stage, label)))

    stages = [s for s, _ in events]
    assert "download" in stages and "extract" in stages and "verify" in stages
    # Two assets -> per-asset labels on the slow stages.
    assert ("download", "1/2") in events and ("download", "2/2") in events
    assert ("extract", "1/2") in events and ("extract", "2/2") in events
    # Stage order per asset: download before extract.
    assert stages.index("download") < stages.index("extract")


def test_progress_hook_translates_stages_to_job_fields():
    job = _job("quickstart", "Test Model")
    hook = _runtime_progress_hook(job)

    hook("download", 5 << 20, 100 << 20, "1/2")
    assert job["phase"] == "downloading-runtime"
    assert "1/2" in job["detail"]
    assert job["done_bytes"] == 5 << 20
    assert job["total_bytes"] == 100 << 20

    # Rapid second tick inside the throttle window is dropped...
    hook("download", 6 << 20, 100 << 20, "1/2")
    assert job["done_bytes"] == 5 << 20
    # ...but a terminal tick (done == total) always lands.
    hook("download", 100 << 20, 100 << 20, "1/2")
    assert job["done_bytes"] == 100 << 20

    hook("extract", 10, 100, "")
    assert job["phase"] in ("downloading-runtime", "unpacking-runtime")

    job2 = _job("runtime-install", "x")
    hook2 = _runtime_progress_hook(job2)
    hook2("extract", 100, 100, "")
    assert job2["phase"] == "unpacking-runtime"
    hook2("verify", 0, 0, "")
    assert job2["phase"] == "verifying-runtime"
    assert job2["total_bytes"] is None  # indeterminate bar, not a stuck 0%


def test_progress_hook_accumulates_across_assets(monkeypatch):
    """A two-asset engine reads as ONE growing download: the second asset's
    bytes stack on the first's instead of restarting the bar at zero, and
    unpack/verify leave the finished download's counters standing."""
    # Drive the throttle's clock so every tick lands (the real hook drops
    # sub-250ms non-terminal ticks; this test is about arithmetic, not
    # pacing — pacing has its own assertions above).
    from hermes_cli.web_routers import local_models as lm

    clock = {"now": 0.0}

    def fake_monotonic():
        clock["now"] += 1.0
        return clock["now"]

    monkeypatch.setattr(lm.time, "monotonic", fake_monotonic)

    job = _job("runtime-install", "engine")
    hook = _runtime_progress_hook(job)

    hook("download", 40 << 20, 40 << 20, "1/2")
    assert job["done_bytes"] == 40 << 20
    assert job["total_bytes"] == 40 << 20

    # Second asset starts: counters continue from the first asset's total.
    hook("download", 0, 60 << 20, "2/2")
    assert job["done_bytes"] == 40 << 20
    assert job["total_bytes"] == 100 << 20

    hook("download", 60 << 20, 60 << 20, "2/2")
    assert job["done_bytes"] == 100 << 20
    assert job["total_bytes"] == 100 << 20

    # Unpack and verify narrate without rewinding the finished bar.
    hook("extract", 1, 100, "2/2")
    assert job["phase"] == "unpacking-runtime"
    assert job["done_bytes"] == 100 << 20
    hook("verify", 0, 0, "")
    assert job["phase"] == "verifying-runtime"
    assert job["done_bytes"] == 100 << 20
