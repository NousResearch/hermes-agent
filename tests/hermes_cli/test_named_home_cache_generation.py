"""A reused directory identity must not skip named-profile initialization."""

import os
import shutil
import stat
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import config
from hermes_cli.profile_incarnation import write_fresh_profile_incarnation
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture
def named_home(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    home = root / "profiles" / "worker"
    home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(config, "is_managed", lambda: False)
    monkeypatch.setattr(config, "_HERMES_HOME_ENSURED", {})
    override = set_hermes_home_override(home)
    try:
        yield home
    finally:
        reset_hermes_home_override(override)


@pytest.mark.parametrize("with_marker", (False, True), ids=("legacy", "marked"))
def test_recreated_named_home_initializes_when_stat_identity_is_reused(
    named_home, monkeypatch, with_marker,
):
    home = named_home
    if with_marker:
        write_fresh_profile_incarnation(home)

    # Force the reuse seen on Linux without depending on inode allocation or
    # filesystem clock precision. All other stat fields describe the real path.
    original_stat = Path.stat
    first_stat = home.stat()

    def reused_stat(path, *args, **kwargs):
        value = original_stat(path, *args, **kwargs)
        if path != home:
            return value
        fields = {name: getattr(value, name) for name in dir(value) if name.startswith("st_")}
        fields.update(
            st_dev=first_stat.st_dev,
            st_ino=first_stat.st_ino,
            st_ctime_ns=first_stat.st_ctime_ns,
        )
        return SimpleNamespace(**fields)

    monkeypatch.setattr(Path, "stat", reused_stat)
    config.ensure_hermes_home()
    assert all((home / subdir).is_dir() for subdir in config._HERMES_HOME_SUBDIRS)
    shutil.rmtree(home)
    home.mkdir()
    soul = home / "SOUL.md"
    soul.write_text("Replacement personality.", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME_MODE", "700")
    if os.name == "posix":
        home.chmod(0o755)
    if with_marker:
        write_fresh_profile_incarnation(home)

    config.ensure_hermes_home()

    assert all((home / subdir).is_dir() for subdir in config._HERMES_HOME_SUBDIRS)
    assert soul.read_text(encoding="utf-8") == "Replacement personality."
    if os.name == "posix":
        assert stat.S_IMODE(home.stat().st_mode) == 0o700


@pytest.mark.parametrize("initially_marked", (False, True), ids=("legacy", "marked"))
def test_initialization_tail_does_not_cache_replacement_generation(
    named_home, monkeypatch, initially_marked,
):
    if initially_marked:
        write_fresh_profile_incarnation(named_home)
    real_seed = config._ensure_default_soul_md
    replaced = False

    def replace_after_seed(home):
        nonlocal replaced
        real_seed(home)
        if replaced:
            return
        assert all((home / subdir).is_dir() for subdir in config._HERMES_HOME_SUBDIRS)
        shutil.rmtree(home)
        home.mkdir()
        write_fresh_profile_incarnation(home)
        replaced = True

    monkeypatch.setattr(config, "_ensure_default_soul_md", replace_after_seed)
    config.ensure_hermes_home()
    assert replaced, "the replacement must occur after directory initialization"
    assert not (named_home / "hooks").exists()

    config.ensure_hermes_home()

    assert all((named_home / subdir).is_dir() for subdir in config._HERMES_HOME_SUBDIRS)
    assert (named_home / "SOUL.md").is_file()


def test_stable_marked_home_keeps_fast_path_without_lifecycle_lock(named_home, monkeypatch):
    from hermes_cli import config_home, profile_incarnation

    write_fresh_profile_incarnation(named_home)
    config.ensure_hermes_home()

    def unexpected_call(*args, **kwargs):
        pytest.fail("Stable home must not initialize or take the lifecycle lock")

    monkeypatch.setattr(config_home, "initialize_home", unexpected_call)
    monkeypatch.setattr(profile_incarnation, "_profile_mutation_lease", unexpected_call)
    config.ensure_hermes_home()


def test_legacy_home_initialization_never_waits_for_a_busy_lease(
    named_home, monkeypatch, busy_profile_lease,
):
    from hermes_cli import profile_incarnation, profile_lifecycle

    def unexpected_lease(*args, **kwargs):
        pytest.fail("Config initialization must not wait on a busy lifecycle lease")

    monkeypatch.setattr(profile_incarnation, "_profile_mutation_lease", unexpected_lease)
    # A regression that waits gives up after this, not the production 120 s.
    monkeypatch.setattr(profile_lifecycle, "_PROFILE_LIFECYCLE_LOCK_TIMEOUT_SECONDS", 5.0)
    busy_profile_lease(named_home)
    started = time.monotonic()
    config.ensure_hermes_home()
    config.ensure_hermes_home()
    waited = time.monotonic() - started

    assert waited < 2, f"config initialization waited {waited:.1f}s on the lifecycle lease"
    assert all((named_home / subdir).is_dir() for subdir in config._HERMES_HOME_SUBDIRS)
    assert not (named_home / profile_incarnation.PROFILE_INCARNATION_FILENAME).exists()


def test_legacy_named_home_initializes_once_yet_retirement_still_raises(named_home, monkeypatch):
    """Regression: a marker-less named home re-ran initialize_home on every load_config()."""
    from hermes_cli import config_home
    from hermes_cli.profile_lifecycle import mark_profile_deleting

    (named_home / "config.yaml").write_text("model:\n  provider: auto\n", encoding="utf-8")
    initialized = []
    initialize = config_home.initialize_home

    def counted(*args):
        initialized.append(args[0])
        initialize(*args)

    monkeypatch.setattr(config_home, "initialize_home", counted)
    for _ in range(5):
        config.load_config()
    assert initialized == [named_home]

    mark_profile_deleting(named_home)
    with pytest.raises(FileNotFoundError, match="missing or being deleted"):
        config.load_config()


@pytest.mark.platforms("posix")
def test_warm_legacy_homes_revalidate_without_directory_mutations(
    named_home, monkeypatch, busy_profile_lease,
):
    from hermes_cli import profile_incarnation

    def unexpected_lease(*args, **kwargs):
        pytest.fail("Config initialization must not wait on a busy lifecycle lease")

    monkeypatch.setattr(profile_incarnation, "_profile_mutation_lease", unexpected_lease)
    monkeypatch.setenv("HERMES_MANAGED", "false")
    monkeypatch.setenv("HERMES_HOME_MODE", "700")
    monkeypatch.setenv("HERMES_UID", str(os.getuid()))
    monkeypatch.setenv("HERMES_GID", str(os.getgid()))
    sibling = named_home.with_name("sibling")
    sibling.mkdir()
    homes = (named_home, sibling)
    # A busy lease keeps both homes tokenless: the revalidation path is what's under test.
    busy_profile_lease(*homes)

    def ensure(home):
        token = set_hermes_home_override(home)
        try:
            config.ensure_hermes_home()
        finally:
            reset_hermes_home_override(token)

    for home in homes:
        ensure(home)
    mutations = []

    def observe(name):
        original = getattr(os, name)

        def observed(path, *args, **kwargs):
            if not isinstance(path, int):
                target = Path(path)
                if any(target == home or home in target.parents for home in homes):
                    mutations.append((name, target))
            return original(path, *args, **kwargs)

        monkeypatch.setattr(os, name, observed)

    for name in ("mkdir", "chmod", "chown"):
        observe(name)
    for home in (named_home, sibling, named_home):
        ensure(home)

    assert mutations == []
    for home in homes:
        assert all((home / subdir).is_dir() for subdir in config._HERMES_HOME_SUBDIRS)
        assert stat.S_IMODE(home.stat().st_mode) == 0o700
        assert not (home / profile_incarnation.PROFILE_INCARNATION_FILENAME).exists()
        assert str(home) not in config._HERMES_HOME_ENSURED


def test_malformed_marker_cannot_reuse_cached_initialization(named_home):
    from hermes_cli.profile_incarnation import PROFILE_INCARNATION_FILENAME

    write_fresh_profile_incarnation(named_home)
    config.ensure_hermes_home()
    (named_home / PROFILE_INCARNATION_FILENAME).write_text("invalid", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Invalid profile incarnation marker"):
        config.ensure_hermes_home()


@pytest.mark.parametrize("with_marker", (False, True), ids=("legacy", "marked"))
def test_deleted_named_home_is_not_recreated_by_config(named_home, with_marker):
    if with_marker:
        write_fresh_profile_incarnation(named_home)
    config.ensure_hermes_home()
    shutil.rmtree(named_home)

    with pytest.raises(FileNotFoundError, match="missing or being deleted"):
        config.ensure_hermes_home()
    assert not named_home.exists()


@pytest.mark.parametrize("managed", (False, True))
def test_recreated_home_keeps_customized_soul(named_home, monkeypatch, managed):
    config.ensure_hermes_home()
    shutil.rmtree(named_home)
    named_home.mkdir()
    for subdir in ("cron", "sessions", "logs", "memories"):
        (named_home / subdir).mkdir()
    soul = named_home / "SOUL.md"
    soul.write_text("User-written personality.", encoding="utf-8")
    monkeypatch.setattr(config, "is_managed", lambda: managed)

    config.ensure_hermes_home()

    assert (named_home / "logs" / "curator").is_dir()
    assert soul.read_text(encoding="utf-8") == "User-written personality."
    if managed:
        assert not (named_home / "skills").exists()
    else:
        assert (named_home / "skills").is_dir()


def test_managed_recreation_requires_operator_owned_directories(named_home, monkeypatch):
    from hermes_cli.config_home import HomeInitializationError

    config.ensure_hermes_home()
    shutil.rmtree(named_home)
    named_home.mkdir()
    monkeypatch.setattr(config, "is_managed", lambda: True)

    with pytest.raises(HomeInitializationError, match="Required directory does not exist"):
        config.ensure_hermes_home()
    assert not (named_home / "cron").exists()


@pytest.mark.parametrize("reader", [config.load_config, config.load_config_readonly])
def test_cached_config_cannot_bypass_named_profile_retirement(named_home, reader):
    from hermes_cli.profile_lifecycle import mark_profile_deleting

    write_fresh_profile_incarnation(named_home)
    (named_home / "config.yaml").write_text("model:\n  provider: auto\n", encoding="utf-8")
    reader()
    # Retirement fences the still-present files before deleting them; the YAML
    # signature remains a cache hit, but this generation is no longer usable.
    mark_profile_deleting(named_home)

    with pytest.raises(FileNotFoundError, match="missing or being deleted"):
        reader()
