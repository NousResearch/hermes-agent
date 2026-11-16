"""Models staged under the old per-profile ``<profile home>/models`` layout are moved into the
machine-scoped ``<root>/models`` so they keep loading."""
from hermes_cli.local_runtime import bootstrap


def _profile(root, name, *, identity=True):
    home = root / "profiles" / name
    (home / "models").mkdir(parents=True)
    if identity:
        (home / "config.yaml").write_bytes(b"{}\n")
    return home


def test_legacy_models_move_into_the_machine_dir_without_clobbering(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    new_dir = bootstrap.models_dir()
    new_dir.mkdir(parents=True)
    (new_dir / "shared.gguf").write_bytes(b"current")
    work = _profile(tmp_path, "work")
    split = ["big-00001-of-00002.gguf", "big-00002-of-00002.gguf"]
    for name in (*split, "shared.gguf"):
        (work / "models" / name).write_bytes(f"legacy {name}".encode())
    (work / "models" / "assets").mkdir()
    (work / "models" / "assets" / "mmproj.gguf").write_bytes(b"proj")
    play = _profile(tmp_path, "play")
    (play / "models" / "small.gguf").write_bytes(b"small")
    ghost = _profile(tmp_path, "ghost", identity=False)  # marker-less shell: not a profile
    (ghost / "models" / "ghost.gguf").touch()

    bootstrap.adopt_legacy_models()

    assert {p.name for p in bootstrap.staged_models()} == {"shared.gguf", "big-00001-of-00002.gguf", "small.gguf"}
    assert all((new_dir / name).read_bytes() == f"legacy {name}".encode() for name in split)
    assert (bootstrap.assets_dir() / "mmproj.gguf").read_bytes() == b"proj"
    assert (new_dir / "shared.gguf").read_bytes() == b"current"  # never clobbered...
    assert (work / "models" / "shared.gguf").read_bytes() == b"legacy shared.gguf"  # ...and not lost
    assert not (play / "models").exists()  # emptied legacy dir is cleaned up
    assert (ghost / "models" / "ghost.gguf").exists()
    assert bootstrap.adopt_legacy_models() == []  # idempotent


def test_boot_and_status_both_adopt_legacy_models(tmp_path, monkeypatch):
    from hermes_cli.local_runtime import binaries
    from hermes_cli.web_routers import local_models as lm

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(binaries, "installed_engine", lambda backend="auto": None)
    work = _profile(tmp_path, "work")

    (work / "models" / "booted.gguf").touch()
    bootstrap.ensure_local_runtime({"local_runtime": {"enabled": True}})
    assert (bootstrap.models_dir() / "booted.gguf").exists()

    (work / "models").mkdir()  # the boot's adoption emptied and removed it
    (work / "models" / "listed.gguf").touch()
    monkeypatch.setattr(lm, "_runtime_section", lambda: {"enabled": False})
    monkeypatch.setattr(lm, "_state_endpoint", lambda: None)
    assert "listed" in {row["id"] for row in lm.local_models_status()["models"]}
