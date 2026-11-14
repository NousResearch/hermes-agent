"""Public full-state config writes preserve surviving list entries (#147594)."""

import hermes_yaml as yaml
import pytest

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


@pytest.mark.parametrize("value", [1, 1.0])
@pytest.mark.parametrize("shape", ["scalar", "mapping", "sequence"])
def test_survivors_keep_numeric_kinds_and_their_own_comments(tmp_path, value, shape):
    config_path = tmp_path / "config.yaml"
    if shape == "mapping":
        entries, after = ["{value: 0}", "{value: 1}", "{value: 1.0}"], [{"value": value}]
    elif shape == "sequence":
        entries, after = ["[0]", "[1]", "[1.0]"], [[value]]
    else:
        entries, after = ["0", "1", "1.0"], [value]
    config_path.write_text(
        "items:\n" + "".join(
            f"  - {entry}  # {kind}\n"
            for entry, kind in zip(entries, ["zero", "integer", "float"])
        ),
        encoding="utf-8",
    )
    state = {"items": after}
    atomic_config_write(config_path, state)

    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))["items"][0]
    if shape == "mapping":
        loaded = loaded["value"]
    elif shape == "sequence":
        loaded = loaded[0]
    assert loaded == value
    assert type(loaded) is type(value)
    written = config_path.read_text(encoding="utf-8-sig")
    assert ("# float" if isinstance(value, float) else "# integer") in written
    stable_bytes = config_path.read_bytes()
    atomic_config_write(config_path, state)
    assert config_path.read_bytes() == stable_bytes
