"""Profile configs opting into inheritance from the root profile.

A named profile normally owns its config outright: what is not in its own
``config.yaml`` falls back to ``DEFAULT_CONFIG``, never to the root profile.
That isolation is deliberate, so inheritance is opt-in per profile via
``inherit: true`` and the root is only ever consulted for keys the profile
did not set itself.
"""

import os
from pathlib import Path

import pytest
import yaml

from hermes_cli.config import load_config


@pytest.fixture
def hermes_root(tmp_path, monkeypatch):
    """A Hermes home with a root config and an empty profiles/ dir.

    Yields a helper that writes a profile config and loads it the way a
    running profile would — through ``HERMES_HOME`` pointed at the profile
    directory, which is what makes the root config a *parent* rather than
    just another file on disk.
    """
    root = tmp_path / "hermes"
    (root / "profiles").mkdir(parents=True)

    def write_root(config: dict) -> None:
        (root / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    def load_profile(name: str, config: dict | None) -> dict:
        profile_dir = root / "profiles" / name
        profile_dir.mkdir(parents=True, exist_ok=True)
        if config is not None:
            (profile_dir / "config.yaml").write_text(
                yaml.safe_dump(config), encoding="utf-8"
            )
        monkeypatch.setenv("HERMES_HOME", str(profile_dir))
        _clear_config_cache()
        return load_config()

    yield write_root, load_profile


def _clear_config_cache() -> None:
    """Drop the (mtime, size)-keyed cache between loads in one test process.

    Two writes inside the same test can land in the same mtime granularity,
    so the cache would serve the first result for the second read.
    """
    from hermes_cli import config as config_module

    config_module._LOAD_CONFIG_CACHE.clear()


def model_key(cfg: dict, key: str):
    """Read one key out of ``model``, which is not always a mapping.

    A config that names no model normalizes to an empty string rather than an
    empty dict, so indexing it directly raises instead of reporting "unset".
    """
    model = cfg.get("model")
    return model.get(key) if isinstance(model, dict) else None


class TestInheritanceOptIn:
    def test_profile_without_inherit_does_not_see_root_model(self, hermes_root):
        """Isolation stays the default: no opt-in, no inheritance."""
        write_root, load_profile = hermes_root
        write_root({"model": {"provider": "anthropic", "default": "claude-opus-5"}})

        cfg = load_profile("standalone", {"agent": {"max_turns": 7}})

        assert model_key(cfg, "default") != "claude-opus-5"

    def test_inherit_true_pulls_root_model(self, hermes_root):
        write_root, load_profile = hermes_root
        write_root({"model": {"provider": "anthropic", "default": "claude-opus-5"}})

        cfg = load_profile("heir", {"inherit": True})

        assert model_key(cfg, "default") == "claude-opus-5"
        assert model_key(cfg, "provider") == "anthropic"

    def test_inherit_false_is_explicit_isolation(self, hermes_root):
        write_root, load_profile = hermes_root
        write_root({"model": {"provider": "anthropic", "default": "claude-opus-5"}})

        cfg = load_profile("opted-out", {"inherit": False, "agent": {"max_turns": 7}})

        assert model_key(cfg, "default") != "claude-opus-5"


class TestProfileWins:
    def test_profile_value_overrides_inherited_one(self, hermes_root):
        """Inheritance fills gaps; it never overwrites a profile's own choice."""
        write_root, load_profile = hermes_root
        write_root({"model": {"provider": "anthropic", "default": "claude-opus-5"}})

        cfg = load_profile(
            "override", {"inherit": True, "model": {"default": "gpt-5.6-sol"}}
        )

        assert model_key(cfg, "default") == "gpt-5.6-sol"

    def test_merge_is_per_leaf_not_per_block(self, hermes_root):
        """Setting one key under model must not drop the sibling keys."""
        write_root, load_profile = hermes_root
        write_root(
            {
                "model": {
                    "provider": "anthropic",
                    "default": "claude-opus-5",
                    "base_url": "https://api.anthropic.com",
                }
            }
        )

        cfg = load_profile(
            "partial", {"inherit": True, "model": {"default": "gpt-5.6-sol"}}
        )

        assert model_key(cfg, "default") == "gpt-5.6-sol"
        assert model_key(cfg, "base_url") == "https://api.anthropic.com"


class TestRootEditsTakeEffect:
    def test_editing_the_root_updates_an_inheriting_profile(self, hermes_root):
        """Changing the root model must reach profiles that inherit it.

        The load cache keys on the profile file's (mtime, size). An inheriting
        profile also depends on a file that key does not watch, so without the
        parent in the signature a root edit stays invisible until something
        else happens to touch the profile's own config.
        """
        write_root, load_profile = hermes_root
        write_root({"model": {"provider": "anthropic", "default": "model-before"}})

        first = load_profile("follower", {"inherit": True})
        assert model_key(first, "default") == "model-before"

        write_root({"model": {"provider": "anthropic", "default": "model-after"}})

        assert model_key(load_config(), "default") == "model-after"


class TestRootProfileUnaffected:
    def test_root_config_never_inherits_from_itself(self, tmp_path, monkeypatch):
        """The root has no parent; ``inherit`` there is a no-op, not a loop."""
        root = tmp_path / "hermes"
        root.mkdir(parents=True)
        (root / "config.yaml").write_text(
            yaml.safe_dump({"inherit": True, "model": {"default": "claude-opus-5"}}),
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(root))
        _clear_config_cache()

        cfg = load_config()

        assert model_key(cfg, "default") == "claude-opus-5"


class TestDegradesSafely:
    def test_missing_root_config_leaves_profile_working(self, tmp_path, monkeypatch):
        """A profile asking to inherit from a root that has no config still loads."""
        root = tmp_path / "hermes"
        profile_dir = root / "profiles" / "orphan"
        profile_dir.mkdir(parents=True)
        (profile_dir / "config.yaml").write_text(
            yaml.safe_dump({"inherit": True, "agent": {"max_turns": 7}}),
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(profile_dir))
        _clear_config_cache()

        cfg = load_config()

        assert cfg["agent"]["max_turns"] == 7

    def test_unreadable_root_config_does_not_break_the_profile(
        self, hermes_root, monkeypatch
    ):
        """Broken YAML in the root must not take a working profile down with it."""
        write_root, load_profile = hermes_root
        root_dir = None

        cfg = load_profile("resilient", {"inherit": True, "agent": {"max_turns": 7}})
        assert cfg["agent"]["max_turns"] == 7

        # Corrupt the root, then reload the same profile.
        profile_home = Path(os.environ["HERMES_HOME"])
        root_dir = profile_home.parent.parent
        (root_dir / "config.yaml").write_text("model: [unclosed", encoding="utf-8")
        _clear_config_cache()

        cfg = load_config()

        assert cfg["agent"]["max_turns"] == 7

    def test_a_nested_inherit_is_reported_instead_of_ignored(
        self, hermes_root, capsys
    ):
        """`inherit` inside a section does nothing, so it must not do it quietly.

        The misplaced key leaves the section empty, and the user meets a missing
        model rather than a misplaced line. Naming the key is the difference
        between a two-second fix and a debugging session.
        """
        write_root, load_profile = hermes_root
        write_root({"model": {"default": "root-model"}})

        cfg = load_profile("nested", {"model": {"inherit": True}})

        assert cfg.get("model", {}).get("default") != "root-model"
        assert "model.inherit" in capsys.readouterr().err

    def test_a_top_level_inherit_warns_about_nothing(self, hermes_root, capsys):
        """The correct spelling must stay silent."""
        write_root, load_profile = hermes_root
        write_root({"model": {"default": "root-model"}})

        cfg = load_profile("correct", {"inherit": True})

        assert cfg["model"]["default"] == "root-model"
        assert "has no effect" not in capsys.readouterr().err

    def test_an_unreachable_root_is_reported_not_swallowed(self, tmp_path, monkeypatch, capsys):
        """A profile outside `<root>/profiles/<name>/` cannot inherit — say so.

        `_resolve_root_config_path` walks a fixed two levels up, so a profile
        nested deeper finds no root. Failing silently there would recreate the
        exact bug inheritance exists to fix: a config that reads as empty with
        nothing on screen explaining why.
        """
        root = tmp_path / "hermes"
        deep = root / "profiles" / "team" / "member"
        deep.mkdir(parents=True)
        (root / "config.yaml").write_text(
            yaml.safe_dump({"model": {"default": "root-model"}}), encoding="utf-8"
        )
        (deep / "config.yaml").write_text(
            yaml.safe_dump({"inherit": True}), encoding="utf-8"
        )
        monkeypatch.setenv("HERMES_HOME", str(deep))
        _clear_config_cache()

        cfg = load_config()

        # The profile is left with no model at all — the symptom the warning
        # exists to explain.
        assert not cfg.get("model")
        assert "has no effect here" in capsys.readouterr().err
