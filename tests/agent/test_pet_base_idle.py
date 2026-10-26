"""Hatching reuses the approved base for resting poses without paid idle calls."""
from PIL import Image, ImageDraw, UnidentifiedImageError
import pytest

from agent.pet.generate import orchestrate, atlas
from agent.pet.generate.imagegen import GenerationError
from agent.pet import store
from hermes_constants import get_hermes_home


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
    restored = store.load_pet(result.slug)
    assert restored is not None and restored.spritesheet == result.spritesheet
    assert restored.directory.is_relative_to(get_hermes_home())
    with Image.open(restored.spritesheet) as saved:
        idle_row = saved.crop((0, 0, atlas.ATLAS_WIDTH, atlas.CELL_HEIGHT))
        assert idle_row.getbbox() is not None
        # Static idle contains the base frame, not an expensive generated strip.
        assert idle_row.crop((atlas.CELL_WIDTH, 0, atlas.ATLAS_WIDTH, atlas.CELL_HEIGHT)).getbbox() is None


@pytest.mark.parametrize("color", [(100, 70, 200, 255), (0, 0, 0, 0)])
def test_empty_derived_idle_uses_generated_row(hatch_fixture, color):
    base, calls, provider = hatch_fixture
    Image.new("RGBA", (80, 100), color).save(base)
    progress = []
    result = orchestrate.hatch_pet(
        base_image=base, slug="idle-rescue", provider=provider,
        on_progress=lambda *event: progress.append(event),
    )
    assert "pet_row_idle" in [prefix for prefix, _ in calls]
    assert all(actual is provider for _, actual in calls)
    assert "idle" in result.states
    assert store.load_pet(result.slug) is not None
    row_progress = [detail for event, detail in progress if event == "row"]
    assert sorted(int(detail.split(":")[1]) for detail in row_progress) == list(range(1, len(atlas.ROW_SPECS) + 1))
    with Image.open(result.spritesheet) as saved:
        assert saved.crop((0, 0, atlas.ATLAS_WIDTH, atlas.CELL_HEIGHT)).getbbox() is not None


@pytest.mark.parametrize("failure", ["empty", "exception"])
def test_failed_idle_rescue_does_not_install_pet(hatch_fixture, monkeypatch, failure):
    base, calls, provider = hatch_fixture
    Image.new("RGBA", (80, 100), (100, 70, 200, 255)).save(base)
    generate = orchestrate.imagegen.generate

    def fail_idle(*args, prefix, **kwargs):
        if prefix != "pet_row_idle":
            return generate(*args, prefix=prefix, **kwargs)
        calls.append((prefix, kwargs["provider"]))
        if failure == "exception":
            raise RuntimeError("idle fixture failure")
        return []

    monkeypatch.setattr(orchestrate.imagegen, "generate", fail_idle)
    with pytest.raises(GenerationError, match="missing required animation row.*idle"):
        orchestrate.hatch_pet(base_image=base, slug="failed-idle", provider=provider)
    assert store.load_pet("failed-idle") is None
    assert "pet_row_idle" in [prefix for prefix, _ in calls]


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
    assert store.load_pet("failed-rows") is None
    assert calls
    assert "pet_row_idle" not in [prefix for prefix, _ in calls]
