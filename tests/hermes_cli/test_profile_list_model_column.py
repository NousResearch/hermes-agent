"""``hermes profile list`` prints each profile's model id whole (#159994).

The table is the only profile roster the CLI prints, so its Model column gets parsed and handed
back to ``chat -m``: an id cut at 26 characters names a model that does not exist.
"""

import argparse
from pathlib import Path

from hermes_cli import profile_cmd

LONG_MODEL = "some-provider/a-model-id-longer-than-the-old-column"


def test_a_long_model_id_is_printed_whole_and_the_columns_stay_aligned(tmp_path, monkeypatch, capsys):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(f"model:\n  default: {LONG_MODEL}\n", encoding="utf-8")

    profile_cmd._profile_list(argparse.Namespace())

    lines = capsys.readouterr().out.splitlines()
    header = next(line for line in lines if "Gateway" in line)
    row = next(line for line in lines if "default" in line)
    assert LONG_MODEL in row.split()
    assert row.index(LONG_MODEL) == header.index("Model")
    assert row.index(" stopped") + 1 == header.index("Gateway")
