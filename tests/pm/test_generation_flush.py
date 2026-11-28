"""The macOS publication barrier is scoped to the generation's volume."""
from pathlib import Path

import pytest

from pm.environments import flush_before_selecting


@pytest.mark.platforms("macos")
def test_flush_waits_for_the_generation_volume_without_global_sync(tmp_path, monkeypatch):
    def refuse_global_sync():
        pytest.fail("a dependency publication must not sync unrelated macOS volumes")

    monkeypatch.setattr("os.sync", refuse_global_sync)
    generation = tmp_path / "neue Umgebung"
    generation.mkdir()
    (generation / "pyvenv.cfg").write_text("ready\n")

    flush_before_selecting(generation)


@pytest.mark.platforms("macos")
def test_missing_generation_fails_the_publication_barrier(tmp_path):
    generation = tmp_path / "missing"
    with pytest.raises(FileNotFoundError) as error:
        flush_before_selecting(generation)
    assert Path(error.value.filename) == generation
