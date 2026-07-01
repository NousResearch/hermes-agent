def test_openviking_backup_paths_include_named_ovcli_profiles(tmp_path, monkeypatch):
    import plugins.memory.openviking as openviking_module
    from plugins.memory.openviking import OpenVikingMemoryProvider

    openviking_home = tmp_path / ".openviking"
    openviking_home.mkdir()
    active_path = openviking_home / "ovcli.conf"
    backup_path = openviking_home / "ovcli.conf.bak"
    saved_path = openviking_home / "ovcli.conf.VPS"
    active_path.write_text("{}", encoding="utf-8")
    backup_path.write_text("{}", encoding="utf-8")
    saved_path.write_text("{}", encoding="utf-8")
    (openviking_home / "other.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(openviking_module.Path, "home", staticmethod(lambda: tmp_path))

    paths = OpenVikingMemoryProvider().backup_paths()

    assert paths == [str(active_path), str(backup_path), str(saved_path)]

