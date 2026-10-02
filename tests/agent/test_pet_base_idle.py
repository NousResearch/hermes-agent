"""Hatching reuses the approved base for resting poses without paid idle calls."""
from PIL import Image, ImageDraw, UnidentifiedImageError
import pytest

from agent.pet.generate import orchestrate, atlas
from agent.pet.generate.imagegen import GenerationError


@pytest.fixture
def hatch_fixture(tmp_path, monkeypatch):
    base = tmp_path / "base.png"
    sprite = Image.new("RGBA", (80, 100))
    ImageDraw.Draw(sprite).ellipse((20, 15, 60, 90), fill=(100, 70, 200, 255))
    sprite.save(base)
    calls = []
    provider = object()

    def generate(prompt, *, prefix, provider: object, **kwargs):
        calls.append((prefix, provider))
        state = prefix.removeprefix("pet_row_")
        count = next(count for name, _, count in atlas.ROW_SPECS if name == state)
        strip = Image.new("RGBA", (80 * count, 100))
        for index in range(count):
            strip.alpha_composite(sprite, (80 * index, 0))
        path = tmp_path / f"{prefix}.png"
        strip.save(path)
        return [path]

    monkeypatch.setattr(orchestrate.imagegen, "generate", generate)
    return base, calls, provider


def test_hatch_uses_approved_base_without_paid_idle(hatch_fixture):
    base, calls, provider = hatch_fixture
    progress = []
    result = orchestrate.hatch_pet(
        base_image=base, slug="base-idle", provider=provider,
        on_progress=lambda *event: progress.append(event),
    )
    row_progress = [detail for event, detail in progress if event == "row"]
    assert row_progress[0] == f"idle:1:{len(atlas.ROW_SPECS)}"
    assert row_progress[-1].endswith(f":{len(atlas.ROW_SPECS)}:{len(atlas.ROW_SPECS)}")
    assert "pet_row_idle" not in [prefix for prefix, _ in calls]
    assert all(actual is provider for _, actual in calls)
    assert result.spritesheet.is_file()
    assert result.validation["ok"]
    assert {"idle", "running-right", "running-left", "waving"} <= set(result.states)


def test_cancelled_hatch_does_not_generate_or_save(hatch_fixture):
    base, calls, provider = hatch_fixture
    with pytest.raises(GenerationError, match="cancelled"):
        orchestrate.hatch_pet(base_image=base, slug="cancelled", provider=provider, is_cancelled=lambda: True)
    assert calls == []


def test_corrupt_base_fails_before_paid_work(hatch_fixture):
    base, calls, provider = hatch_fixture
    base.write_bytes(b"not an image")
    with pytest.raises(UnidentifiedImageError):
        orchestrate.hatch_pet(base_image=base, slug="corrupt", provider=provider)
    assert calls == []


@pytest.mark.parametrize("failure", ["empty", "exception"])
def test_paid_row_failure_does_not_install_idle_only_pet(hatch_fixture, monkeypatch, failure):
    base, calls, provider = hatch_fixture

    def fail(*args, prefix, **kwargs):
        calls.append((prefix, provider))
        if failure == "exception":
            raise RuntimeError("offline fixture failure")
        return []

    monkeypatch.setattr(orchestrate.imagegen, "generate", fail)
    with pytest.raises(GenerationError):
        orchestrate.hatch_pet(base_image=base, slug="failed-rows", provider=provider)
    assert calls
    assert "pet_row_idle" not in [prefix for prefix, _ in calls]

