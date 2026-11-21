# /// script
# requires-python = ">=3.10"
# dependencies = ["pytest>=8.0"]
# ///
"""Tests for git_evidence.py — measurement over a real temp git repo.

Run: uv run scripts/tests/test_git_evidence.py
 or: uv run --with pytest -m pytest scripts/tests/test_git_evidence.py
"""

import json
import os
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "git_evidence.py"

# The fixture commit identity, shared by both git helpers below.
_IDENT = {
    "GIT_AUTHOR_NAME": "T",
    "GIT_AUTHOR_EMAIL": "t@t",
    "GIT_COMMITTER_NAME": "T",
    "GIT_COMMITTER_EMAIL": "t@t",
}


def _git_env(repo):
    """The environment every fixture git runs under, layered outward.

    Inherit the real environment (PATH above all: git lives in /opt/homebrew,
    /usr/local, or a nix store as readily as /usr/bin, and an env holding only
    GIT_* vars sends execvp to os.defpath), then strip every ambient GIT_* var
    -- GIT_DIR, GIT_WORK_TREE and GIT_CONFIG_COUNT would each silently redirect
    or reconfigure the fixture -- and pin identity plus every source git reads
    for settings, so nothing on the developer's machine can reach the fixture:

    - gitconfig (commit.gpgsign, core.autocrlf, core.hooksPath,
      init.defaultBranch). GIT_CONFIG_NOSYSTEM/GIT_CONFIG_GLOBAL cover
      git >= 2.32; HOME and XDG_CONFIG_HOME cover older git, and are set to the
      repo's parent directory -- always a per-test directory under pytest's
      tmp_path -- so nothing is ever planted inside the working tree.
    - gitattributes, a separate source GIT_CONFIG_NOSYSTEM does not cover: a
      system `* -diff` rule would make numstat call every path binary and take
      the churn assertions down with it. GIT_ATTR_NOSYSTEM shuts it out.
    - the locale. LC_ALL/LANG are pinned to C, matching _run/_proc's existing
      pin, so fixture git's text output cannot vary with the developer's
      locale. Inheriting the environment is what makes this pin necessary:
      the old four-variable env had no locale in it to inherit.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(_IDENT)
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_ATTR_NOSYSTEM"] = "1"
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["HOME"] = env["XDG_CONFIG_HOME"] = str(Path(repo).parent)
    env["LC_ALL"] = env["LANG"] = "C"
    return env


def _json(proc):
    """Parse the JSON-only stdout contract, surfacing a crash instead of hiding
    it behind a JSONDecodeError."""
    assert proc.stdout, f"empty stdout; stderr was: {proc.stderr}"
    assert "Traceback" not in proc.stderr, proc.stderr
    return json.loads(proc.stdout)


def _run(*args):
    # LC_ALL=C keeps git's error strings in English so assertions on them
    # are stable across locales.
    proc = subprocess.run(
        ["uv", "run", str(SCRIPT), *args],
        capture_output=True,
        text=True,
        env={**os.environ, "LC_ALL": "C", "LANG": "C"},
    )
    return proc.returncode, _json(proc)


def _proc(*args, env=None):
    """Run the script and return the raw process, so a test can assert on the
    exit code and stderr together — and so `env` can carry an overlay (a fake
    `git` earlier on PATH) that `_run` has no way to pass."""
    overlay = {"LC_ALL": "C", "LANG": "C"}
    if env:
        overlay.update(env)
    return subprocess.run(
        ["uv", "run", str(SCRIPT), *args],
        capture_output=True,
        text=True,
        env={**os.environ, **overlay},
    )


def _git(repo, *args):
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        env=_git_env(repo),
    )


def _git_unchecked(repo, *args):
    """`git` that tolerates a non-zero exit — the conflicting merge in
    `_merge_repo` is supposed to fail, and the hand resolution comes after it.
    Same environment as `_git`, which runs with check=True."""
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        env=_git_env(repo),
    )


def _make_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "a.py").write_text("one\ntwo\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "epic-1-1 initial a")
    (repo / "a.py").write_text("one\ntwo\nthree\nfour\n")
    (repo / "b.py").write_text("x\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "epic-1-2 grow a, add b")
    return repo


def test_no_range_returns_empty(tmp_path):
    repo = _make_repo(tmp_path)
    code, out = _run("--repo", str(repo))
    assert code == 0
    assert out["range"] is None
    assert out["commits"] == [] and out["files"] == []


def test_measures_commits_and_files_with_attribution(tmp_path):
    repo = _make_repo(tmp_path)
    code, out = _run(
        "--repo", str(repo), "--range", "HEAD~1..HEAD", "--stories", "1-2,1-1"
    )
    assert code == 0
    assert out["range"] == "HEAD~1..HEAD"
    assert out["commit_count"] == 1
    # The single commit in range is the second one; attributed to story "1-2".
    assert out["commits"][0]["stories"] == ["1-2"]
    files = {f["path"]: f for f in out["files"]}
    # a.py grew by two lines, b.py added one — measured, not judged.
    assert files["a.py"]["added"] == 2 and files["a.py"]["net"] == 2
    assert files["b.py"]["added"] == 1


def test_story_attribution_respects_word_boundary(tmp_path):
    # Story id "1-2" must NOT match a commit subject mentioning "11-2".
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "f.py").write_text("a\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    (repo / "f.py").write_text("a\nb\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "epic-11-2 unrelated story")
    code, out = _run("--repo", str(repo), "--range", "HEAD~1..HEAD", "--stories", "1-2")
    assert code == 0
    assert out["commits"][0]["stories"] == []


def test_bad_range_errors_as_json(tmp_path):
    repo = _make_repo(tmp_path)
    code, out = _run("--repo", str(repo), "--range", "nope..alsonope")
    assert code == 1
    assert out["ok"] is False and out["error"]


def test_single_rev_range_rejected(tmp_path):
    # A single rev is not a range: git would log ALL history up to it and the
    # script would report the whole repo as the epic's evidence.
    repo = _make_repo(tmp_path)
    code, out = _run("--repo", str(repo), "--range", "HEAD")
    assert code == 2
    assert out["ok"] is False and "invalid --range" in out["error"]


def test_pathspec_range_rejected(tmp_path):
    # A path that exists must not be silently consumed as a pathspec.
    repo = _make_repo(tmp_path)
    code, out = _run("--repo", str(repo), "--range", "a.py")
    assert code == 2
    assert out["ok"] is False and "invalid --range" in out["error"]


def test_range_shaped_pathspec_forced_to_rev_parse(tmp_path):
    # A committed file literally named "a..b" passes the REV..REV shape check;
    # without the trailing "--" in the git argv, git silently logs that FILE's
    # history with exit 0. The "--" forces rev interpretation, so this must
    # error instead of measuring the decoy.
    repo = _make_repo(tmp_path)
    (repo / "a..b").write_text("decoy\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "add decoy file named like a range")
    code, out = _run("--repo", str(repo), "--range", "a..b")
    assert code == 1
    assert out["ok"] is False and "bad revision" in out["error"]


def test_option_like_range_rejected(tmp_path):
    # A range starting with "-" must never reach git, where it would be
    # consumed as an option (e.g. --output=... writes an arbitrary file).
    repo = _make_repo(tmp_path)
    code, out = _run("--repo", str(repo), "--range=--output=evil.txt")
    assert code == 2
    assert out["ok"] is False and "invalid --range" in out["error"]
    assert not (repo / "evil.txt").exists()


def test_degenerate_range_shapes_rejected(tmp_path):
    # Shapes that contain ".." but are not REV..REV: git would silently
    # default an empty endpoint to HEAD ("..", "a..", "..HEAD"), a leading
    # dash must never reach git even when dots are present ("-3..HEAD"),
    # unstripped values must not slip past the dash guard, and a three-dot
    # range is a symmetric difference — git would measure commits reachable
    # from either endpoint but not both, a different evidence set entirely.
    repo = _make_repo(tmp_path)
    for bad in (
        "..",
        "a..",
        "..HEAD",
        "-3..HEAD",
        " HEAD~1..HEAD",
        "HEAD~1...HEAD",
        "a...b",
    ):
        code, out = _run("--repo", str(repo), f"--range={bad}")
        assert code == 2, f"accepted {bad!r}"
        assert out["ok"] is False and "invalid --range" in out["error"], bad


def test_malformed_args_emit_json_not_usage(tmp_path):
    # An unknown flag must still land on the JSON contract, not argparse's
    # plain usage text on stderr.
    code, out = _run("--bogus-flag")
    assert code != 0
    assert out["ok"] is False and out["error"]


def test_help_flags_emit_json_not_usage():
    # argparse's built-in help action bypasses the error() override entirely --
    # it prints usage text on stdout and exits 0, which breaks the JSON-only
    # contract for a machine consumer. add_help=False demotes -h to an ordinary
    # unrecognized argument, which error() already handles.
    for flag in ("-h", "--help"):
        proc = _proc(flag)
        # Exit 2 specifically: the module docstring reserves 2 for argument
        # errors and 1 for git/I-O failures, so collapsing them must fail here.
        assert proc.returncode == 2, flag
        assert "usage:" not in proc.stdout, flag
        out = _json(proc)
        assert out["ok"] is False and out["error"], flag


# --- helpers for the fixtures below -----------------------------------------


def _rev(repo, ref):
    return _git_unchecked(repo, "rev-parse", ref).stdout.strip()


def _fake_git(tmp_path, body):
    """Write a `git` shim and return the PATH overlay that puts it ahead of the
    real binary for the script's own subprocesses (never for the fixtures,
    which build their repos through `_git`'s own environment)."""
    bindir = tmp_path / "fakebin"
    bindir.mkdir()
    shim = bindir / "git"
    shim.write_text(body)
    os.chmod(shim, 0o755)
    return {"PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}"}


def _merge_repo(tmp_path):
    """Two story branches merged into the mainline; the second merge conflicts
    and is resolved by hand, adding a line neither branch had. Returns
    (repo, base_sha) — `base_sha..HEAD` is the epic range."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "s.py").write_text("l1\nl2\nl3\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    base = _rev(repo, "HEAD")

    _git(repo, "checkout", "-q", "-b", "s1")
    (repo / "s.py").write_text("l1\nA\nl3\n")
    _git(repo, "commit", "-qam", "epic-1-2 story one")
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--no-ff", "s1", "-m", "merge story 1-2")

    _git(repo, "checkout", "-q", "-b", "s2", base)
    (repo / "s.py").write_text("l1\nB\nl3\n")
    _git(repo, "commit", "-qam", "epic-1-3 story two")
    _git(repo, "checkout", "-q", "main")
    conflicted = _git_unchecked(repo, "merge", "--no-ff", "s2", "-m", "merge story 1-3")
    assert conflicted.returncode != 0, "fixture expected a merge conflict"
    (repo / "s.py").write_text("l1\nAB\nl3\nl4\n")  # hand resolution
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "merge story 1-3")
    return repo, base


def test_rename_yields_two_openable_paths(tmp_path):
    # git's default rename detection emits "src/{mod.py => renamed.py}" — an
    # unopenable pseudo-path that also splits one file's churn across keys.
    # --no-renames makes the rename an honest delete + add.
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "src").mkdir()
    (repo / "src" / "mod.py").write_text("l1\nl2\nl3\nl4\nl5\nl6\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    base = _rev(repo, "HEAD")
    _git(repo, "mv", "src/mod.py", "src/renamed.py")
    _git(repo, "commit", "-qm", "rename mod")
    (repo / "src" / "renamed.py").write_text("l1\nl2\nl3\nl4\nl5\nl6\nl7\n")
    _git(repo, "commit", "-qam", "add a line after the rename")

    out = _json(_proc("--repo", str(repo), "--range", f"{base}..HEAD"))
    files = {f["path"]: f for f in out["files"]}
    assert not any("=>" in path for path in files), sorted(files)
    assert files["src/mod.py"]["added"] == 0
    assert files["src/mod.py"]["deleted"] == 6
    assert files["src/renamed.py"]["added"] == 7
    assert files["src/renamed.py"]["deleted"] == 0


def _accented_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "src").mkdir()
    (repo / "src" / "café.py").write_text("ca\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    base = _rev(repo, "HEAD")
    (repo / "src" / "café.py").write_text("ca\ncb\n")
    _git(repo, "commit", "-qam", "touch the accented file")
    return repo, base


def _assert_accented_path(repo, out):
    paths = [f["path"] for f in out["files"]]
    assert len(paths) == 1
    assert "\\" not in paths[0] and '"' not in paths[0], paths
    assert unicodedata.normalize("NFC", paths[0]) == "src/café.py"
    # The reported path is a real path: it opens under --repo.
    assert (repo / paths[0]).read_text() == "ca\ncb\n"


def test_non_ascii_path_is_a_real_string(tmp_path):
    # Without core.quotePath=false git emits "src/caf\303\251.py" — quoted and
    # octal-escaped, so the documented "open the ranked files" step cannot.
    repo, base = _accented_repo(tmp_path)
    _assert_accented_path(
        repo, _json(_proc("--repo", str(repo), "--range", f"{base}..HEAD"))
    )


def test_non_ascii_path_survives_a_non_utf8_locale(tmp_path):
    # Pins the explicit encoding="utf-8" on the subprocess. Modern CPython's
    # UTF-8 mode hides its absence even under LC_ALL=C, so the pin only bites
    # with UTF-8 mode and C-locale coercion both off — where the interpreter
    # default is US-ASCII and git's UTF-8 path bytes fail to decode, taking the
    # whole measurement down with them.
    repo, base = _accented_repo(tmp_path)
    out = _json(
        _proc(
            "--repo",
            str(repo),
            "--range",
            f"{base}..HEAD",
            env={"PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0"},
        )
    )
    _assert_accented_path(repo, out)


def test_merge_churn_is_measured_and_counted(tmp_path):
    repo, base = _merge_repo(tmp_path)
    out = _json(_proc("--repo", str(repo), "--range", f"{base}..HEAD"))
    assert out["commit_count"] == 4
    assert out["merge_count"] == 2
    assert out["merges_measured"] == 2
    # Both merges' first-parent churn: 1/1 for the clean merge, 2/1 for the
    # hand-resolved one (the resolution added a line neither branch had).
    merge_files = {f["path"]: f for f in out["merge_files"]}
    assert merge_files["s.py"]["added"] == 3
    assert merge_files["s.py"]["deleted"] == 2
    assert merge_files["s.py"]["net"] == 1
    assert merge_files["s.py"]["commit_count"] == 2


def test_merge_commits_listed_but_excluded_from_files(tmp_path):
    repo, base = _merge_repo(tmp_path)
    out = _json(_proc("--repo", str(repo), "--range", f"{base}..HEAD"))
    by_subject = {c["subject"]: c for c in out["commits"]}
    assert by_subject["merge story 1-2"]["is_merge"] is True
    assert by_subject["merge story 1-3"]["is_merge"] is True
    assert by_subject["epic-1-2 story one"]["is_merge"] is False
    # `files` is the two story commits only — 1/1 each. Folding the merges in
    # would double count: their diff restates the churn they merged.
    files = {f["path"]: f for f in out["files"]}
    assert files["s.py"]["added"] == 2
    assert files["s.py"]["deleted"] == 2
    assert files["s.py"]["commit_count"] == 2


def test_story_attribution_survives_merges(tmp_path):
    repo, base = _merge_repo(tmp_path)
    out = _json(
        _proc(
            "--repo", str(repo), "--range", f"{base}..HEAD", "--stories", "1-2,1-3"
        )
    )
    by_subject = {c["subject"]: c for c in out["commits"]}
    assert by_subject["epic-1-2 story one"]["stories"] == ["1-2"]
    assert by_subject["epic-1-3 story two"]["stories"] == ["1-3"]


def test_off_spine_merge_is_counted_but_not_measured(tmp_path):
    # A back-merge of the mainline into a story branch is a merge in the range,
    # but its first-parent diff would restate unrelated mainline content as
    # epic churn. It stays in merge_count and out of merges_measured, so the
    # gap between the two says plainly that a merge went unmeasured.
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "f.txt").write_text("base\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    base = _rev(repo, "HEAD")
    (repo / "f.txt").write_text("base\nmain1\n")
    _git(repo, "commit", "-qam", "mainline work")

    _git(repo, "checkout", "-q", "-b", "s1", base)
    (repo / "s1.txt").write_text("s1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "epic-1-2 story one")
    _git(repo, "merge", "-q", "--no-ff", "main", "-m", "back-merge main into s1")
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--no-ff", "s1", "-m", "merge story 1-2")

    out = _json(_proc("--repo", str(repo), "--range", f"{base}..HEAD"))
    assert out["merge_count"] == 2
    assert out["merges_measured"] == 1
    merge_files = {f["path"]: f for f in out["merge_files"]}
    assert set(merge_files) == {"s1.txt"}


def test_merge_pass_survives_a_hostile_log_diffmerges_config(tmp_path):
    # `-m` means "whatever log.diffMerges says", so a user or repo config of
    # `off` makes the merge pass emit no file rows: merge_files comes back
    # empty beside a non-zero merges_measured and reads as "the merges changed
    # nothing". The command-line -c pin beats the config.
    repo, base = _merge_repo(tmp_path)
    _git(repo, "config", "log.diffMerges", "off")
    out = _json(_proc("--repo", str(repo), "--range", f"{base}..HEAD"))
    assert out["merges_measured"] == 2
    merge_files = {f["path"]: f for f in out["merge_files"]}
    assert merge_files["s.py"]["added"] == 3
    assert merge_files["s.py"]["deleted"] == 2


def test_distinct_non_utf8_paths_stay_distinct(tmp_path):
    # errors="replace" maps every invalid byte to the same U+FFFD, collapsing
    # two different files into one `files` key with their churn summed —
    # measurement corruption with nothing in the output admitting to it.
    overlay = _fake_git(
        tmp_path,
        "#!/bin/sh\n"
        "printf 'aaaa\\037\\037subj\\n\\n1\\t0\\tsrc/caf\\351.py\\n"
        "2\\t0\\tsrc/caf\\377.py\\n'\n",
    )
    out = _json(_proc("--repo", str(tmp_path), "--range", "a..b", env=overlay))
    files = {f["path"]: f for f in out["files"]}
    assert len(files) == 2, files
    assert sorted(f["added"] for f in files.values()) == [1, 2]


def test_repeated_merge_headers_are_counted_once(tmp_path):
    # Pins the dedupe guard in _parse_log. git before 2.31 does not honour
    # --first-parent for `-m`'s diff format, so a merge's header repeats once
    # per parent with a diff block under each. Without the guard that doubles
    # the merge churn and pushes merges_measured above merge_count, inverting
    # the invariant the reference documents. Inert on modern git, so it needs
    # pre-2.31-shaped output to be exercised at all.
    real_git = shutil.which("git")
    assert real_git, "git must be on PATH"
    repo, base = _merge_repo(tmp_path)
    overlay = _fake_git(
        tmp_path,
        "#!/bin/sh\n"
        'for a in "$@"; do\n'
        '  if [ "$a" = "--min-parents=2" ]; then\n'
        "    printf 'aaaa\\037p1 p2\\037merge story 1-3\\n\\n2\\t1\\ts.py\\n"
        "aaaa\\037p1 p2\\037merge story 1-3\\n\\n5\\t4\\ts.py\\n'\n"
        "    exit 0\n"
        "  fi\n"
        "done\n"
        f'exec "{real_git}" "$@"\n',
    )
    out = _json(_proc("--repo", str(repo), "--range", f"{base}..HEAD", env=overlay))
    # One merge sha, however many blocks git printed for it.
    assert out["merges_measured"] == 1
    assert out["merges_measured"] <= out["merge_count"]
    merge_files = {f["path"]: f for f in out["merge_files"]}
    # The first block is the first-parent diff on every git version; the
    # repeat must not be added on top of it.
    assert merge_files["s.py"]["added"] == 2
    assert merge_files["s.py"]["deleted"] == 1
    assert merge_files["s.py"]["commit_count"] == 1


def test_valid_range_with_no_commits_keeps_the_full_shape(tmp_path):
    # A mis-specified epic range is a valid, empty range. It must still answer
    # with every documented key rather than a differently-shaped stub.
    repo = _make_repo(tmp_path)
    proc = _proc("--repo", str(repo), "--range", "HEAD..HEAD")
    out = _json(proc)
    assert proc.returncode == 0
    assert set(out) == {
        "range",
        "commit_count",
        "merge_count",
        "merges_measured",
        "commits",
        "files",
        "merge_files",
        "stories_supplied",
    }
    assert out["commit_count"] == 0
    assert out["merge_count"] == 0
    assert out["merges_measured"] == 0
    assert out["commits"] == [] and out["files"] == [] and out["merge_files"] == []


def test_linear_history_reports_no_merges(tmp_path):
    repo = _make_repo(tmp_path)
    out = _json(_proc("--repo", str(repo), "--range", "HEAD~1..HEAD"))
    assert out["merge_count"] == 0
    assert out["merges_measured"] == 0
    assert out["merge_files"] == []


def _recording_git(tmp_path, calls):
    """A `git` shim that records each invocation as one \\x1f-delimited record
    (one field per argument, so argument boundaries survive) and then execs the
    real git."""
    real_git = shutil.which("git")
    assert real_git, "git must be on PATH"
    return _fake_git(
        tmp_path,
        "#!/bin/sh\n"
        f"( printf '%s\\037' \"$@\"; printf '\\n' ) >> \"{calls}\"\n"
        f'exec "{real_git}" "$@"\n',
    )


def _numstat_invocations(calls):
    """The recorded `git log --numstat` invocations, each as an argument list."""
    out = []
    for line in calls.read_text().splitlines():
        argv = [field for field in line.split("\x1f") if field]
        if "--numstat" in argv:
            out.append(argv)
    return out


def test_second_pass_runs_only_when_the_range_has_merges(tmp_path):
    # The merge pass is skipped outright on linear history, so the common case
    # still costs exactly one `git log` — and the two passes must differ in
    # exactly the arguments the design depends on.
    calls = tmp_path / "calls.log"
    overlay = _recording_git(tmp_path, calls)
    merge_args = {"-m", "--first-parent", "--min-parents=2"}

    (tmp_path / "linear").mkdir()
    (tmp_path / "merged").mkdir()

    linear = _make_repo(tmp_path / "linear")
    _json(_proc("--repo", str(linear), "--range", "HEAD~1..HEAD", env=overlay))
    logs = _numstat_invocations(calls)
    assert len(logs) == 1, logs

    calls.write_text("")
    merged, base = _merge_repo(tmp_path / "merged")
    _json(_proc("--repo", str(merged), "--range", f"{base}..HEAD", env=overlay))
    logs = _numstat_invocations(calls)
    assert len(logs) == 2, logs
    # Pass 1 keeps full topology: none of the merge-pass arguments may reach it,
    # or the story-branch commits drop out of the listing and attribution dies.
    assert merge_args.isdisjoint(logs[0]), logs[0]
    assert merge_args.issubset(logs[1]), logs[1]


def test_multi_story_subject_attributes_to_every_match(tmp_path):
    # First-match-wins silently dropped the second story from the attribution.
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "f.py").write_text("a\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    (repo / "f.py").write_text("a\nb\n")
    _git(repo, "commit", "-qam", "fix seam between 1-2 and 1-3")

    out = _json(
        _proc(
            "--repo", str(repo), "--range", "HEAD~1..HEAD", "--stories", "1-3,1-2"
        )
    )
    # Every match, in --stories order — not whichever id was passed first.
    assert out["commits"][0]["stories"] == ["1-3", "1-2"]

    # A repeated id must not list the commit twice: any per-story total built
    # from `stories` would count it twice.
    repeated = _json(
        _proc(
            "--repo", str(repo), "--range", "HEAD~1..HEAD", "--stories", "1-3,1-2,1-3"
        )
    )
    assert repeated["commits"][0]["stories"] == ["1-3", "1-2"]
    assert repeated["stories_supplied"] == ["1-3", "1-2"]


def test_subject_naming_no_story_gets_an_empty_list(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "f.py").write_text("a\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    (repo / "f.py").write_text("a\nb\n")
    _git(repo, "commit", "-qam", "chore: tidy imports")

    out = _json(
        _proc(
            "--repo", str(repo), "--range", "HEAD~1..HEAD", "--stories", "1-2,1-3"
        )
    )
    assert out["commits"][0]["stories"] == []


def test_git_failure_with_empty_stderr_reports_the_exit_code(tmp_path):
    # A quiet git failure (signal kill, empty stderr) must not leave the caller
    # with `"error": ""` and nothing to report.
    overlay = _fake_git(tmp_path, "#!/bin/sh\nexit 3\n")
    proc = _proc("--repo", str(tmp_path), "--range", "HEAD~1..HEAD", env=overlay)
    out = _json(proc)
    assert proc.returncode == 1
    assert out["ok"] is False
    assert out["error"] == "git exited 3"


def _binary_repo(tmp_path):
    """A binary-only path plus a path that is binary twice and text once."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "keep.txt").write_text("keep\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    base = _rev(repo, "HEAD")

    (repo / "src").mkdir()
    (repo / "src" / "x.py").write_bytes(b"\x00\x01bin\n")
    (repo / "blob.bin").write_bytes(b"\x00\x01blob\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "add binary content")
    (repo / "src" / "x.py").write_text("one\ntwo\n")  # binary -> text: still binary
    _git(repo, "commit", "-qam", "x.py becomes text")
    (repo / "src" / "x.py").write_text("one\ntwo\nthree\n")  # text -> text: measured
    _git(repo, "commit", "-qam", "grow x.py")
    (repo / "blob.bin").write_bytes(b"\x00\x02blob\n")
    _git(repo, "commit", "-qam", "churn the blob")
    return repo, base


def test_binary_revisions_no_longer_erase_measured_text_churn(tmp_path):
    repo, base = _binary_repo(tmp_path)
    out = _json(_proc("--repo", str(repo), "--range", f"{base}..HEAD"))
    x = {f["path"]: f for f in out["files"]}["src/x.py"]
    assert x["added"] == 1 and x["deleted"] == 0 and x["net"] == 1
    assert x["binary_revisions"] == 2
    assert x["commit_count"] == 3


def test_binary_only_path_reports_zero_sums_and_its_revision_count(tmp_path):
    repo, base = _binary_repo(tmp_path)
    out = _json(_proc("--repo", str(repo), "--range", f"{base}..HEAD"))
    blob = {f["path"]: f for f in out["files"]}["blob.bin"]
    assert blob["added"] == 0 and blob["deleted"] == 0 and blob["net"] == 0
    assert blob["binary_revisions"] == 2
    assert blob["commit_count"] == 2


def test_success_shape_carries_every_documented_key(tmp_path):
    repo = _make_repo(tmp_path)
    out = _json(_proc("--repo", str(repo), "--range", "HEAD~1..HEAD"))
    assert set(out) >= {
        "range",
        "commit_count",
        "merge_count",
        "merges_measured",
        "commits",
        "files",
        "merge_files",
        "stories_supplied",
    }
    assert set(out["commits"][0]) == {"sha", "subject", "stories", "is_merge"}
    assert set(out["files"][0]) == {
        "path",
        "added",
        "deleted",
        "net",
        "commit_count",
        "binary_revisions",
    }


def test_explicit_repo_ignores_ambient_git_dir(tmp_path):
    repo = _make_repo(tmp_path)
    proc = _proc(
        "--repo",
        str(repo),
        "--range",
        "HEAD~1..HEAD",
        env={"GIT_DIR": str(tmp_path / "wrong-git-dir")},
    )
    assert _json(proc)["commit_count"] == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
