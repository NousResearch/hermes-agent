"""Tests for the per-path load_env_file() memo in agent.secret_scope.

`build_profile_secret_scope()` sits on the gateway's hot paths (every turn,
every cron job, every MCP/browser lifecycle adoption, the 60s housekeeping
delivery drain) and each call used to re-read and re-parse the whole profile
`.env`. The consolidated readers now share this tokenizer, including
`hermes_cli.config.load_env()`, which keeps its own outer memo.

The freshness contract is the point of these tests, because a stale credential
map is a far worse failure than a slow one. Every call still OPENS the file and
keys on the descriptor's `fstat`, which is what makes three things true: an
unopenable file is never cached, a repointed symlink cannot file one file's
contents under another's identity, and the open still drives close-to-open
revalidation on network filesystems.

Files are written as bytes throughout: `Path.write_text` translates "\\n" to
"\\r\\n" on Windows, which would break every size assertion below.
"""

from __future__ import annotations

import os
import sys
import threading

import pytest

import agent.secret_scope as ss


@pytest.fixture(autouse=True)
def _clear_cache():
    ss.invalidate_env_file_cache()
    yield
    ss.invalidate_env_file_cache()


def _write(path, text: str | bytes, *, mtime=None) -> None:
    """Write with literal LF, so sizes are identical on every platform."""
    path.write_bytes(text.encode("utf-8") if isinstance(text, str) else text)
    if mtime is not None:
        os.utime(path, ns=(mtime, mtime))


# Comfortably past the coarsest mtime resolution in common use (FAT32 truncates
# to two seconds), so the bump cannot round back onto the previous value.
_MTIME_STEP_NS = 10_000_000_000


def _write_advancing_mtime(path, text: str | bytes) -> None:
    """Rewrite and force the mtime far enough forward to be unambiguous.

    Tests that rely on "the timestamp moved" must not depend on the host
    filesystem resolving two immediate writes differently, or this asserts a
    stricter contract than the implementation promises.
    """
    previous = path.stat().st_mtime_ns
    _write(path, text)
    bumped = max(previous + _MTIME_STEP_NS, path.stat().st_mtime_ns + _MTIME_STEP_NS)
    os.utime(path, ns=(bumped, bumped))
    if path.stat().st_mtime_ns == previous:
        pytest.skip("filesystem will not advance the mtime; cannot test this path")


def _count_parses(monkeypatch) -> list:
    """Record every real parse so a test can assert the memo was reached."""
    calls: list = []
    real = ss._parse_env_text
    monkeypatch.setattr(ss, "_parse_env_text", lambda t: (calls.append(t), real(t))[1])
    return calls


class TestCacheHits:
    def test_unchanged_file_is_parsed_once(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        _write(env, "A=1\nB=2\n")
        calls = _count_parses(monkeypatch)

        first = ss.load_env_file(env)
        for _ in range(5):
            assert ss.load_env_file(env) == first
        assert first == {"A": "1", "B": "2"}
        assert len(calls) == 1, f"expected one parse, got {len(calls)}"

    def test_returns_a_fresh_dict_each_call(self, tmp_path):
        env = tmp_path / ".env"
        _write(env, "A=1\n")

        first = ss.load_env_file(env)
        first["INJECTED"] = "should-not-persist"
        second = ss.load_env_file(env)

        assert second == {"A": "1"}
        assert first is not second

    def test_build_profile_secret_scope_does_not_poison_the_cache(self, tmp_path, monkeypatch):
        """build_profile_secret_scope layers external secrets onto what
        load_env_file returns, so the returned dict must not be the cached one."""
        env = tmp_path / ".env"
        _write(env, "PROFILE_KEY=from-env\n")
        monkeypatch.setattr(
            "hermes_cli.env_loader.get_secret_source_values",
            lambda home: {"EXTERNAL_KEY": "from-vault"},
        )

        scope = ss.build_profile_secret_scope(tmp_path)
        assert scope["PROFILE_KEY"] == "from-env"
        assert scope["EXTERNAL_KEY"] == "from-vault"
        assert ss.load_env_file(env) == {"PROFILE_KEY": "from-env"}

    def test_build_profile_secret_scope_reaches_the_memo(self, tmp_path, monkeypatch):
        """The motivating path: repeated scope builds must parse the file once.

        Pins the benefit against a future refactor that reads the .env some
        other way and silently reintroduces a parse per turn.
        """
        env = tmp_path / ".env"
        _write(env, "PROFILE_KEY=v\n")
        monkeypatch.setattr("hermes_cli.env_loader.get_secret_source_values", lambda home: {})
        calls = _count_parses(monkeypatch)

        for _ in range(10):
            assert ss.build_profile_secret_scope(tmp_path)["PROFILE_KEY"] == "v"
        assert len(calls) == 1, f"expected one parse across 10 builds, got {len(calls)}"

    def test_separate_paths_get_separate_entries(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        a.mkdir()
        b.mkdir()
        _write(a / ".env", "WHO=a\n")
        _write(b / ".env", "WHO=b\n")

        assert ss.load_env_file(a / ".env") == {"WHO": "a"}
        assert ss.load_env_file(b / ".env") == {"WHO": "b"}
        assert ss.load_env_file(a / ".env") == {"WHO": "a"}

    def test_genuinely_empty_file_is_cached_as_empty(self, tmp_path, monkeypatch):
        """An empty .env is a real answer, unlike an unreadable one below."""
        env = tmp_path / ".env"
        _write(env, "# only a comment\n")
        calls = _count_parses(monkeypatch)

        assert ss.load_env_file(env) == {}
        assert ss.load_env_file(env) == {}
        assert len(calls) == 1


class TestDecoding:
    @pytest.mark.parametrize("encoding", ["utf-8", "latin-1"])
    def test_bom_and_unprefixed_files_cache_the_same_values(
        self, tmp_path, monkeypatch, encoding
    ):
        text = 'KEY=caf\u00e9\r\nexport OTHER="x # y" # note\r\n'
        expected = {"KEY": "caf\u00e9", "OTHER": "x # y"}

        for name, prefix in [("plain.env", b""), ("bom.env", b"\xef\xbb\xbf")]:
            env = tmp_path / name
            _write(env, prefix + text.encode(encoding))
            assert ss._parse_env_file(env) == expected

            with monkeypatch.context() as patch:
                calls = _count_parses(patch)
                assert ss.load_env_file(env) == expected
                assert ss.load_env_file(env) == expected
                assert len(calls) == 1, "decoded contents did not reach the memo"

    @pytest.mark.parametrize("prefix", [b"", b"\xef\xbb\xbf"], ids=["plain", "bom"])
    def test_encoding_switch_reparses_same_size_writes(
        self, tmp_path, monkeypatch, prefix
    ):
        """Encoding changes must invalidate by mtime even when size and inode stay put."""
        env = tmp_path / ".env"
        _write(env, prefix + b"KEY=\xc3\xa9\n")
        size_before = env.stat().st_size
        calls = _count_parses(monkeypatch)

        for raw, expected in [
            (None, {"KEY": "\u00e9"}),
            (b"KEY=\xe9x\n", {"KEY": "\u00e9x"}),
            (b"KEY=\xc3\xb1\n", {"KEY": "\u00f1"}),
        ]:
            if raw is not None:
                _write_advancing_mtime(env, prefix + raw)
            assert env.stat().st_size == size_before

            before = len(calls)
            assert ss.load_env_file(env) == expected
            assert ss.load_env_file(env) == expected
            assert len(calls) == before + 1, "each write needs exactly one cached parse"
            assert ss._parse_env_file(env) == expected
            assert len(calls) == before + 2, "the reference path must stay uncached"


class TestUnreadableIsNeverCached:
    """A file we could not open or read must never become a cached answer.

    Memoising the fail-soft ``{}`` turns a momentary EACCES into "this profile
    has no secrets" for the rest of the process lifetime.
    """

    @pytest.mark.skipif(os.geteuid() == 0 if hasattr(os, "geteuid") else False,
                        reason="root ignores the permission bits this test sets")
    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX mode bits")
    def test_transient_permission_error_is_not_remembered(self, tmp_path):
        env = tmp_path / ".env"
        _write(env, "REAL_API_KEY=sk-live\n")
        assert ss.load_env_file(env) == {"REAL_API_KEY": "sk-live"}

        ss.invalidate_env_file_cache()
        os.chmod(env, 0o000)
        try:
            assert ss.load_env_file(env) == {}
        finally:
            os.chmod(env, 0o600)

        # Restoring the mode changes no fingerprint field, so a memoised {}
        # would survive here.
        assert ss.load_env_file(env) == {"REAL_API_KEY": "sk-live"}

    @pytest.mark.skipif(os.geteuid() == 0 if hasattr(os, "geteuid") else False,
                        reason="root ignores the permission bits this test sets")
    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX mode bits")
    def test_warm_cache_stops_serving_when_read_is_revoked(self, tmp_path):
        env = tmp_path / ".env"
        _write(env, "K=v\n")
        assert ss.load_env_file(env) == {"K": "v"}

        os.chmod(env, 0o000)
        try:
            assert ss.load_env_file(env) == {}
        finally:
            os.chmod(env, 0o600)

    def test_missing_file_is_not_cached(self, tmp_path):
        env = tmp_path / ".env"
        assert ss.load_env_file(env) == {}

        _write(env, "APPEARED=1\n")
        assert ss.load_env_file(env) == {"APPEARED": "1"}

    def test_deleted_file_drops_its_entry(self, tmp_path):
        env = tmp_path / ".env"
        _write(env, "SECRET=abc\n")
        assert ss.load_env_file(env) == {"SECRET": "abc"}

        env.unlink()
        assert ss.load_env_file(env) == {}
        assert str(env) not in ss._ENV_FILE_CACHE, "credentials outlived the file"


class TestInvalidation:
    def test_edit_changing_size_is_seen(self, tmp_path):
        env = tmp_path / ".env"
        _write(env, "KEY=old\n")
        assert ss.load_env_file(env) == {"KEY": "old"}

        _write(env, "KEY=a-much-longer-new-value\nEXTRA=1\n")
        assert ss.load_env_file(env) == {"KEY": "a-much-longer-new-value", "EXTRA": "1"}

    def test_same_size_edit_is_seen(self, tmp_path):
        """Identical length, different content: mtime_ns must catch it."""
        env = tmp_path / ".env"
        _write(env, "KEY=aaa\n")
        size_before = env.stat().st_size
        assert ss.load_env_file(env) == {"KEY": "aaa"}

        _write_advancing_mtime(env, "KEY=bbb\n")
        assert env.stat().st_size == size_before
        assert ss.load_env_file(env) == {"KEY": "bbb"}

    def test_atomic_replace_with_identical_stat_is_seen(self, tmp_path):
        """Same size AND forced-identical mtime: only the inode differs.

        An atomic tmp-plus-rename is how credential writers replace .env, so
        this is the case (mtime, size) keying alone would get wrong.
        """
        env = tmp_path / ".env"
        _write(env, "KEY=aaa\n")
        assert ss.load_env_file(env) == {"KEY": "aaa"}
        size_before = env.stat().st_size
        mtime_before = env.stat().st_mtime_ns
        ino_before = env.stat().st_ino

        replacement = tmp_path / ".env.tmp"
        _write(replacement, "KEY=bbb\n", mtime=mtime_before)
        os.replace(replacement, env)

        assert env.stat().st_size == size_before
        assert env.stat().st_mtime_ns == mtime_before
        if env.stat().st_ino == ino_before:
            pytest.skip("filesystem reused the inode; no field left to distinguish")
        assert ss.load_env_file(env) == {"KEY": "bbb"}

    def test_write_between_the_two_fstats_is_not_cached(self, tmp_path, monkeypatch):
        """The post-read re-fstat guard, fired inside the window it protects.

        The write lands after the key was taken but before the read, so the
        bytes parsed never matched that fingerprint. Storing them would serve
        the intermediate contents if the file were later restored to the
        original stat. The size is held identical so a size mismatch cannot be
        what saves us.
        """
        env = tmp_path / ".env"
        _write(env, "KEY=aaa\n")
        real_fingerprint = ss._fd_fingerprint
        fired = {"done": False}

        def fingerprint_then_race(fileno):
            taken = real_fingerprint(fileno)
            if not fired["done"]:
                fired["done"] = True
                # Same length, different bytes, timestamp forced forward so a
                # coarse-resolution filesystem cannot mask the change.
                _write_advancing_mtime(env, "KEY=bbb\n")
            return taken

        monkeypatch.setattr(ss, "_fd_fingerprint", fingerprint_then_race)
        ss.load_env_file(env)
        monkeypatch.setattr(ss, "_fd_fingerprint", real_fingerprint)

        assert str(env) not in ss._ENV_FILE_CACHE, "stored a parse the file outran"
        assert ss.load_env_file(env) == {"KEY": "bbb"}

    def test_symlink_repointed_between_open_and_read_serves_the_opened_file(
        self, tmp_path, monkeypatch
    ):
        """Descriptor identity, not pathname identity, decides what is read.

        The swap lands after the open and before the read, which is exactly the
        window a path-based implementation gets wrong: it would stat A, then
        re-read the path and get B. Reading through the descriptor must return
        A, and the following call must then see B because the path now resolves
        there.
        """
        a, b = tmp_path / "A.env", tmp_path / "B.env"
        _write(a, "WHO=A\n")
        _write(b, "WHO=BB\n")  # different length, so a path read is unmistakable
        link = tmp_path / ".env"
        try:
            link.symlink_to(a)
        except (OSError, NotImplementedError) as exc:  # Windows without the privilege
            pytest.skip(f"cannot create symlinks here: {exc}")

        real_fingerprint = ss._fd_fingerprint
        fired = {"done": False}

        def swap_right_after_open(fileno):
            if not fired["done"]:
                fired["done"] = True
                link.unlink()
                link.symlink_to(b)
            return real_fingerprint(fileno)

        monkeypatch.setattr(ss, "_fd_fingerprint", swap_right_after_open)
        got = ss.load_env_file(link)
        monkeypatch.setattr(ss, "_fd_fingerprint", real_fingerprint)

        assert got == {"WHO": "A"}, "read followed the path instead of the descriptor"
        assert ss.load_env_file(link) == {"WHO": "BB"}

    def test_in_flight_reader_cannot_undo_an_invalidation(self, tmp_path, monkeypatch):
        """A reader that started before an invalidation must not repopulate it.

        Without the generation check the slow reader's older contents would land
        in the cache after the writer had already cleared it.
        """
        env = tmp_path / ".env"
        _write(env, "K=old\n")
        real = ss._parse_env_text
        reached_parse = threading.Event()
        may_finish = threading.Event()

        def slow_parse(text):
            out = real(text)
            if "old" in text:
                reached_parse.set()
                # Event.wait() returning False means it TIMED OUT, and a timeout
                # would let the reader finish before the invalidation, making the
                # whole test vacuous. Raise so read_once() records it.
                if not may_finish.wait(30):
                    raise AssertionError("reader was never released; race not exercised")
            return out

        errors: list = []

        def read_once():
            try:
                ss.load_env_file(env)
            except Exception as exc:  # pragma: no cover - surfaced below
                errors.append(exc)

        monkeypatch.setattr(ss, "_parse_env_text", slow_parse)
        reader = threading.Thread(target=read_once)
        reader.start()
        try:
            assert reached_parse.wait(10), "reader never reached the parse"
            ss.invalidate_env_file_cache()
            may_finish.set()
            reader.join(30)
        finally:
            may_finish.set()
            reader.join(30)
        monkeypatch.setattr(ss, "_parse_env_text", real)

        # slow_parse raises if it was never released in time, so an empty
        # `errors` is what proves the reader really was still in flight when
        # the invalidation ran.
        assert not errors, errors
        assert not reader.is_alive(), "reader thread did not finish"
        assert str(env) not in ss._ENV_FILE_CACHE

    def test_invalidate_one_path_leaves_others(self, tmp_path, monkeypatch):
        a, b = tmp_path / "a", tmp_path / "b"
        a.mkdir()
        b.mkdir()
        _write(a / ".env", "WHO=a\n")
        _write(b / ".env", "WHO=b\n")
        ss.load_env_file(a / ".env")
        ss.load_env_file(b / ".env")

        ss.invalidate_env_file_cache(a / ".env")
        calls = _count_parses(monkeypatch)
        ss.load_env_file(a / ".env")
        ss.load_env_file(b / ".env")
        assert len(calls) == 1

    def test_invalidate_all(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        _write(env, "KEY=value\n")
        ss.load_env_file(env)

        ss.invalidate_env_file_cache()
        calls = _count_parses(monkeypatch)
        ss.load_env_file(env)
        assert len(calls) == 1

    def test_config_invalidate_env_cache_clears_this_memo_too(self, tmp_path, monkeypatch):
        from hermes_cli.config import invalidate_env_cache

        env = tmp_path / ".env"
        _write(env, "KEY=value\n")
        ss.load_env_file(env)

        invalidate_env_cache()
        calls = _count_parses(monkeypatch)
        ss.load_env_file(env)
        assert len(calls) == 1

    def test_save_env_value_is_seen_by_a_warmed_profile_scope(self, tmp_path, monkeypatch):
        """End to end through a real writer, not just the invalidation helper.

        Scope note, so this is not read as more than it is: save_env_value()
        rewrites the file, which moves the fingerprint, so this would also pass
        with the invalidation link removed. It guards the integration (a warmed
        profile scope sees a rotated key) rather than the link itself;
        test_config_invalidate_env_cache_clears_this_memo_too is what pins the
        link, by holding the file still and checking only the memo.
        """
        from hermes_cli import config as config_mod

        env = tmp_path / ".env"
        _write(env, "EXISTING=1\n")
        monkeypatch.setattr(config_mod, "get_env_path", lambda: env)
        monkeypatch.setattr(config_mod, "ensure_hermes_home", lambda: None)
        monkeypatch.setattr(config_mod, "_secure_file", lambda _p: None)
        monkeypatch.setattr(config_mod, "is_managed", lambda: False)
        monkeypatch.setattr("hermes_cli.env_loader.get_secret_source_values", lambda home: {})

        assert ss.build_profile_secret_scope(tmp_path) == {"EXISTING": "1"}

        try:
            config_mod.save_env_value("ROTATED_KEY", "sk-new")
            scope = ss.build_profile_secret_scope(tmp_path)
            assert scope.get("ROTATED_KEY") == "sk-new"

            assert config_mod.remove_env_value("ROTATED_KEY") is True
            assert "ROTATED_KEY" not in ss.build_profile_secret_scope(tmp_path)
        finally:
            monkeypatch.delenv("ROTATED_KEY", raising=False)


class TestUnavailableMetadata:
    """fstat failing means "cannot cache", never "cannot read".

    CPython tolerates fstat failures on some filesystems (VirtualBox shared
    folders among them). A file whose bytes read fine must still be parsed and
    returned, not downgraded to an empty credential map.
    """

    def test_fingerprint_failure_still_returns_the_parsed_file(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        _write(env, "K=value\n")

        def always_fails(_fileno):
            raise OSError("fstat unsupported on this filesystem")

        monkeypatch.setattr(ss, "_fd_fingerprint", always_fails)
        assert ss.load_env_file(env) == {"K": "value"}
        assert str(env) not in ss._ENV_FILE_CACHE

    def test_second_fingerprint_failure_does_not_discard_the_read(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        _write(env, "K=value\n")
        real = ss._fd_fingerprint
        calls = {"n": 0}

        def fails_on_second(fileno):
            calls["n"] += 1
            if calls["n"] == 2:
                raise OSError("fstat unsupported on this filesystem")
            return real(fileno)

        monkeypatch.setattr(ss, "_fd_fingerprint", fails_on_second)
        assert ss.load_env_file(env) == {"K": "value"}
        assert str(env) not in ss._ENV_FILE_CACHE

    def test_matches_the_uncached_path_when_metadata_is_unavailable(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        _write(env, 'A="x # y"  # note\nB=plain\n')
        expected = ss._parse_env_file(env)

        monkeypatch.setattr(
            ss, "_fd_fingerprint", lambda _f: (_ for _ in ()).throw(OSError("nope"))
        )
        assert ss.load_env_file(env) == expected


class TestBounds:
    def test_cache_is_bounded(self, tmp_path):
        for i in range(ss._ENV_FILE_CACHE_MAX + 10):
            home = tmp_path / f"h{i}"
            home.mkdir()
            _write(home / ".env", f"KEY={i}\n")
            ss.load_env_file(home / ".env")
        assert len(ss._ENV_FILE_CACHE) <= ss._ENV_FILE_CACHE_MAX

    def test_eviction_is_least_recently_used(self, tmp_path):
        homes = []
        for i in range(ss._ENV_FILE_CACHE_MAX):
            home = tmp_path / f"h{i}"
            home.mkdir()
            _write(home / ".env", f"KEY={i}\n")
            ss.load_env_file(home / ".env")
            homes.append(home)

        ss.load_env_file(homes[0] / ".env")  # touch the oldest

        extra = tmp_path / "extra"
        extra.mkdir()
        _write(extra / ".env", "KEY=extra\n")
        ss.load_env_file(extra / ".env")

        assert str(homes[0] / ".env") in ss._ENV_FILE_CACHE
        assert str(homes[1] / ".env") not in ss._ENV_FILE_CACHE

    def test_an_evicted_entry_still_reads_correctly(self, tmp_path):
        first = tmp_path / "first"
        first.mkdir()
        _write(first / ".env", "KEY=first\n")
        ss.load_env_file(first / ".env")

        for i in range(ss._ENV_FILE_CACHE_MAX + 5):
            home = tmp_path / f"h{i}"
            home.mkdir()
            _write(home / ".env", f"KEY={i}\n")
            ss.load_env_file(home / ".env")

        assert str(first / ".env") not in ss._ENV_FILE_CACHE
        assert ss.load_env_file(first / ".env") == {"KEY": "first"}


class TestConcurrency:
    def test_threads_get_correct_independent_dicts(self, tmp_path):
        env = tmp_path / ".env"
        _write(env, "SHARED=value\n")
        errors = []
        done = []

        def worker():
            try:
                for _ in range(40):
                    got = ss.load_env_file(env)
                    assert got == {"SHARED": "value"}
                    got["LOCAL"] = "mutation"
                done.append(True)
            except Exception as exc:  # pragma: no cover - surfaced via errors
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, errors
        assert len(done) == 8
        assert ss.load_env_file(env) == {"SHARED": "value"}

    def test_readers_racing_a_writer_never_see_a_torn_map(self, tmp_path):
        """Readers may see the old or the new value, never a blend or a crash."""
        env = tmp_path / ".env"
        _write(env, "K=" + "a" * 20 + "\n")
        seen = []
        errors = []
        stop = threading.Event()
        observed = threading.Event()
        ready = threading.Barrier(5, timeout=10)

        def reader():
            try:
                ready.wait()
                while not stop.is_set():
                    got = ss.load_env_file(env)
                    if got:
                        seen.append(got["K"])
                        observed.set()
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=reader) for _ in range(4)]
        for t in threads:
            t.start()
        try:
            ready.wait()
            # Do not start rewriting until a reader has seen the initial file,
            # so an empty `seen` cannot be mistaken for a pass.
            assert observed.wait(10), "no reader observed the file before writes began"
            for i in range(30):
                _write(env, "K=" + ("b" if i % 2 else "a") * 20 + "\n")
                ss.invalidate_env_file_cache(env)
        finally:
            stop.set()
            for t in threads:
                t.join(10)

        assert not errors, errors
        assert all(not t.is_alive() for t in threads), "reader threads did not finish"
        assert seen, "readers never observed a value"
        assert set(seen) <= {"a" * 20, "b" * 20}, set(seen)


class TestParsingUnchanged:
    @pytest.mark.parametrize(
        "contents, expected",
        [
            ("export KEY=value\n", {"KEY": "value"}),
            ("# comment only\n", {}),
            ("KEY=value # trailing\n", {"KEY": "value"}),
            ("KEY=foo#bar\n", {"KEY": "foo#bar"}),
            ('KEY="quoted value"\n', {"KEY": "quoted value"}),
            ("﻿KEY=value\n", {"KEY": "value"}),
            ("NOEQUALS\nKEY=value\n", {"KEY": "value"}),
        ],
    )
    def test_semantics_survive_the_memo(self, tmp_path, contents, expected):
        env = tmp_path / ".env"
        _write(env, contents)
        assert ss.load_env_file(env) == expected
        assert ss.load_env_file(env) == expected  # again, from the cache

    def test_cached_and_uncached_paths_agree(self, tmp_path):
        """load_env_file() and the uncached reference path must not diverge."""
        env = tmp_path / ".env"
        _write(
            env,
            'export A="x # y"  # note\nB=plain\n# skip\nC=foo#bar\n\nD=\n',
        )
        assert ss.load_env_file(env) == ss._parse_env_file(env)
