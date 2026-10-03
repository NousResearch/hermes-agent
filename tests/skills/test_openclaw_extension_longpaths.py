from pathlib import Path
import importlib.util
import json
import sys
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "optional-skills/migration/openclaw-migration/scripts/openclaw_to_hermes.py"

@pytest.mark.windows_only
def test_extension_archive_preserves_dependencies_beyond_max_path(tmp_path):
    spec = importlib.util.spec_from_file_location("claw_longpath", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    source = tmp_path / "source"
    source.mkdir()
    (source / "openclaw.json").write_text(json.dumps({"plugins": {"entries": {"demo": {"enabled": True}}}}))
    relative = Path("demo/node_modules") / ("nested-" + "x" * 70) / ("nested-" + "y" * 70) / "payload.txt"
    payload = b"dependency content retained\x00\xff"
    def extended(path):
        return Path("\\\\?\\" + str(path.resolve()))
    original = extended(source / "extensions" / relative)
    original.parent.mkdir(parents=True)
    original.write_bytes(payload)
    migrator = module.Migrator(source, tmp_path / ("target-" + "z" * 60), True, None, False, False, None, selected_options={"plugins-config"})
    migrator.migrate_plugins_config()
    destination = migrator.archive_dir / "extensions" / relative
    assert len(str(destination)) > 260
    assert extended(destination).read_bytes() == original.read_bytes() == payload
    assert json.loads((migrator.archive_dir / "plugins-config.json").read_text())["entries"]["demo"]["enabled"] is True
    migrator.migrate_plugins_config()
    assert extended(destination).read_bytes() == payload
