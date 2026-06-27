from pathlib import Path


def test_read_config_model_uses_lightweight_model_block(tmp_path):
    from hermes_cli import profiles

    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    (profile_dir / "config.yaml").write_text(
        "model:\n"
        "  default: gpt-test\n"
        "  provider: codex-test\n"
        "providers:\n"
        "  ignored: true\n"
    )

    assert profiles._read_config_model(profile_dir) == ("gpt-test", "codex-test")


def test_scan_profile_aliases_skips_large_extensionless_binaries(tmp_path, monkeypatch):
    from hermes_cli import profiles

    wrapper_dir = tmp_path / "bin"
    wrapper_dir.mkdir()
    (wrapper_dir / "uv").write_bytes(b"x" * (profiles._WRAPPER_SCAN_MAX_BYTES + 1))
    (wrapper_dir / "ana").write_text("#!/bin/sh\nexec hermes -p ana-creative-producer \"$@\"\n")
    (wrapper_dir / "ana-creative-producer").write_text(
        "#!/bin/sh\nexec hermes -p ana-creative-producer \"$@\"\n"
    )

    monkeypatch.setattr(profiles, "_get_wrapper_dir", lambda: wrapper_dir)

    assert profiles._scan_profile_aliases()["ana-creative-producer"] == "ana"
    assert profiles.find_alias_for_profile("ana-creative-producer") == "ana"


def test_list_profiles_scans_alias_wrappers_once(tmp_path, monkeypatch):
    from hermes_cli import profiles

    hermes_home = tmp_path / ".hermes"
    profile_root = hermes_home / "profiles"
    profile_root.mkdir(parents=True)
    for name in ["alpha", "beta", "gamma"]:
        p = profile_root / name
        p.mkdir()
        (p / "config.yaml").write_text("model:\n  default: test-model\n  provider: test-provider\n")

    wrapper_dir = tmp_path / "bin"
    wrapper_dir.mkdir()
    (wrapper_dir / "beta-alias").write_text("#!/bin/sh\nexec hermes -p beta \"$@\"\n")

    scan_count = 0
    original_scan = profiles._scan_profile_aliases

    def counted_scan():
        nonlocal scan_count
        scan_count += 1
        return original_scan()

    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: hermes_home)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: profile_root)
    monkeypatch.setattr(profiles, "_get_wrapper_dir", lambda: wrapper_dir)
    monkeypatch.setattr(profiles, "_scan_profile_aliases", counted_scan)
    monkeypatch.setattr(profiles, "_check_gateway_running", lambda _path: False)
    monkeypatch.setattr(profiles, "_count_skills", lambda _path: 0)
    monkeypatch.setattr(profiles, "_read_distribution_meta", lambda _path: (None, None, None))
    monkeypatch.setattr(profiles, "read_profile_meta", lambda _path: {"description": "", "description_auto": False})

    listed = profiles.list_profiles()
    by_name = {p.name: p for p in listed}

    assert scan_count == 1
    assert by_name["beta"].alias_name == "beta-alias"
    assert by_name["alpha"].alias_name is None
    assert by_name["gamma"].alias_name is None
