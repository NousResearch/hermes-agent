"""Every desktop bundle that carries an agent payload plants the prebuilt Webapp beside the TUI and
the dashboard: the payload has no sources to build one from, so `hermes webapp` in the app serves it."""
from __future__ import annotations

import contextlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.bundles import desktop


def _compile(argv: list[str], **_kwargs) -> None:
    """Stands in for the product compilers: each writes its product, marked with its builder's name."""
    script = Path(argv[1]).name
    if script not in {"tui.mjs", "web.mjs", "webapp.mjs"}:
        return
    out = Path(argv[argv.index("--out") + 1])
    if script == "tui.mjs":
        (out / "dist").mkdir(parents=True)
        (out / "dist/entry.js").write_text(script, encoding="utf-8")
        (out / "package.json").write_text('{"type": "module"}', encoding="utf-8")
    else:
        out.mkdir(parents=True)
        (out / "index.html").write_text(script, encoding="utf-8")


@pytest.mark.parametrize("variant", ["bundled", "store"])
def test_every_payload_bundle_plants_the_prebuilt_webapp(tmp_path, monkeypatch, variant):
    from scripts.build.agent import plant_surfaces
    from scripts.bundles import desktop_inputs, desktop_prepare, native

    payload = tmp_path / "payload" / "hermes-agent"
    request = SimpleNamespace(source=tmp_path / "checkout", target="host", channel_request=None,
                              release_epoch=None, version="1.2.3", commit="c" * 40, tag=None, workspaces=list)
    prepared = SimpleNamespace(request=request, validate=lambda: None, node=Path("node"), icon_python=Path("python"),
                               packager=tmp_path / "packager", native=tmp_path / "native",
                               native_toolchain="fixture", payload=tmp_path / "payload.prepared.json")

    def finish(prepared_payload, frontends):
        assert prepared_payload == prepared.payload
        plant_surfaces(payload, frontends)
        return 0

    monkeypatch.setattr(desktop, "run", _compile)
    monkeypatch.setattr(desktop, "flavored_assets", lambda *_a: contextlib.nullcontext())
    monkeypatch.setattr(native, "finish_native", finish)
    monkeypatch.setattr(desktop_prepare, "require_source", lambda *_a: None)
    monkeypatch.setattr(desktop_inputs, "validate_builder_identity", lambda *_a: None)
    monkeypatch.setattr(desktop_inputs, "select_variant", lambda _prepared, chosen: chosen)
    monkeypatch.setattr(desktop_inputs, "build_environment", lambda *_a: {})
    monkeypatch.setattr(desktop_inputs, "packaging_environment", lambda *_a: {})

    desktop._build_prepared(prepared, [], variant)

    assert (payload / "apps/desktop/dist-webapp/index.html").read_text(encoding="utf-8") == "webapp.mjs"
    assert (payload / "hermes_cli/web_dist/index.html").read_text(encoding="utf-8") == "web.mjs"
    assert (payload / "hermes_cli/tui_dist/entry.js").read_text(encoding="utf-8") == "tui.mjs"


def test_source_bundle_staging_plants_the_prebuilt_webapp(tmp_path, monkeypatch):
    """`pnpm run dist:bundled` stages through scripts/bundles/stage.py, whose native child parses its own argv."""
    from scripts.build.agent import plant_surfaces
    from scripts.bundles import native, stage

    payload = tmp_path / "out" / "hermes-agent"

    def staged(args):
        plant_surfaces(payload, args.frontends)
        return 0

    def run(argv, **kwargs):
        if argv[0] == "/prepared/node":
            return _compile(argv, **kwargs)
        assert argv[1:4] == ["-B", "-m", "scripts.bundles.native"]
        monkeypatch.setattr(sys, "argv", ["native", *argv[4:]])
        return subprocess.CompletedProcess(argv, native.main())

    monkeypatch.setattr(stage.shutil, "which", lambda _name: "/prepared/node")
    monkeypatch.setattr("scripts.bundles.payload.snapshot", lambda _root, _ref, destination: destination.mkdir())
    monkeypatch.setattr("scripts.build.icon_environment.prepare_icon_environment", lambda *_a: sys.executable)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(native, "_stage_native", staged)

    assert stage.main(["--out", str(tmp_path / "out")]) == 0

    assert (payload / "apps/desktop/dist-webapp/index.html").read_text(encoding="utf-8") == "webapp.mjs"
    assert (payload / "hermes_cli/web_dist/index.html").read_text(encoding="utf-8") == "web.mjs"
