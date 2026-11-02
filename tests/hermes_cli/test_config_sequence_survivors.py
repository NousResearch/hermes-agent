"""Public full-state config writes preserve surviving list entries (#147594)."""

import hermes_yaml as yaml

from hermes_cli.config import atomic_config_write


def test_remove_and_restore_keeps_survivor_formatting_and_unrelated_state(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "# user header\n"
        "marker: keep\n"
        "skills:\n"
        "  external_dirs:\n"
        '    - "skills/design"  # design\n'
        '    - "skills/pilot"  # pilot\n'
        "    - 'skills/third'  # third\n",
        encoding="utf-8",
    )
    for dirs in (
        ["skills/pilot", "skills/third"],
        ["skills/design", "skills/pilot", "skills/third"],
    ):
        state = {"marker": "keep", "skills": {"external_dirs": dirs}}
        atomic_config_write(config_path, state)

        written = config_path.read_text(encoding="utf-8-sig")
        lines = [" ".join(line.split()) for line in written.splitlines()]
        assert yaml.safe_load(written) == state
        assert lines[0] == "# user header"
        assert '- "skills/pilot" # pilot' in lines
        assert "- 'skills/third' # third" in lines
    stable_bytes = config_path.read_bytes()
    atomic_config_write(config_path, state)
    assert config_path.read_bytes() == stable_bytes
