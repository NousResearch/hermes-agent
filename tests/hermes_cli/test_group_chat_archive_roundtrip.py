"""Keep pending group-sync transactions recoverable across existing archive paths."""
from pathlib import Path
import zipfile

import hermes_yaml as yaml


def test_pending_group_transaction_round_trips_named_profile_export_and_full_backup(tmp_path, monkeypatch):
    import tui_gateway.server as srv
    import utils
    from hermes_cli.backup import create_pre_update_backup
    from hermes_cli.profiles import export_profile, import_profile
    from tui_gateway.group_chat_storage import migrate_group_chat_state

    root = tmp_path / ".hermes"
    source = root / "profiles" / "helper"
    source.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    key = "hermes-bots-groups"
    state = Path("state/group-chat.yaml")
    (root / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    (source / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    (source / "profile.yaml").write_text("display_name: Example\n", encoding="utf-8")
    real_write = utils.atomic_yaml_write

    def fail_profile(path, data, **kwargs):
        if Path(path) == source / "profile.yaml":
            raise OSError("synthetic metadata failure")
        return real_write(path, data, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(utils, "atomic_yaml_write", fail_profile)
        result = srv._methods["profiles.configure"]("write", {
            "name": "helper", "ui_meta": {key: {"rooms": {}}, "example-settings": {"color": "green"}},
            "ui_meta_expected_revisions": {key: 0, "example-settings": 0},
        })
        assert result["result"]["applied"]["ui_meta"] is True
    assert "pending_ui_meta" in yaml.safe_load((source / state).read_text())

    archive = export_profile("helper", str(tmp_path / "profile.tar.gz"))
    restored = import_profile(str(archive), name="restored")
    migrate_group_chat_state(restored)
    assert yaml.safe_load((restored / "profile.yaml").read_text())["ui_meta"] == {
        "example-settings": {"color": "green"}}
    assert yaml.safe_load((restored / state).read_text()) == {"revision": 1, "snapshot": {"rooms": {}}}

    backup = create_pre_update_backup(hermes_home=source)
    assert backup is not None
    restored_backup = tmp_path / "restored-backup"
    restored_backup.mkdir()
    # The archive is created solely from this synthetic fixture, not untrusted input.
    with zipfile.ZipFile(backup) as zf:
        for rel in ("config.yaml", "profile.yaml", str(state)):
            target = restored_backup / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(rel))
    migrate_group_chat_state(restored_backup)
    assert yaml.safe_load((restored_backup / "profile.yaml").read_text())["ui_meta"] == {
        "example-settings": {"color": "green"}}
    assert (restored_backup / state).read_bytes() == (restored / state).read_bytes()
