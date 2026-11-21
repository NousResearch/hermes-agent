# /// script
# requires-python = ">=3.10"
# ///
"""Measure git commit and file-change evidence over a revision range.

Prints ONLY JSON to stdout. Errors are emitted as JSON to stdout with a
non-zero exit code: 2 for invalid arguments (rejected before git runs),
1 for git or I/O failures. This script only MEASURES — it never judges
acceleration or violations. The model interprets the numbers.

Two git passes. The first lists every commit in the range (merges included)
and sums the per-file churn of the non-merge commits, which is what `files`
reports. The second runs only when the range contains merges and measures
those merges alone, reported separately as `merge_files` — never folded into
`files`, because a merge's diff against its first parent restates the churn
of the commits it merged in, which the first pass already counted.
"""

import argparse
import json
import os
import re
import subprocess
import sys

UNIT_SEP = "\x1f"
# sha, space-separated parents (empty for a root commit), subject.
LOG_FORMAT = f"--format=%H{UNIT_SEP}%P{UNIT_SEP}%s"


def _emit(obj, code=0):
    sys.stdout.write(json.dumps(obj))
    sys.exit(code)


class JsonArgumentParser(argparse.ArgumentParser):
    """Emit argparse failures on the JSON-only stdout contract, not usage text.

    The parser is constructed with ``add_help=False``. The override below covers
    ``error()``, but ``-h`` never reaches it: the built-in help action calls
    ``print_help()`` and ``exit(0)`` directly, which would put plain usage text
    on stdout with a zero exit and break the JSON-only contract. Removing the
    action instead of intercepting it routes ``-h`` through the already-tested
    ``error()`` path as an ordinary unrecognized argument. The cost is that the
    ``help=`` strings are unreachable from the CLI; the skill's references carry
    the usage a human needs.
    """

    def error(self, message):
        _emit({"ok": False, "error": f"argument error: {message}"}, 2)


def _parse_numstat_line(line):
    # numstat lines: "<added>\t<deleted>\t<path>"; binary files use "-".
    parts = line.split("\t")
    if len(parts) < 3:
        return None
    added_raw, deleted_raw, path = parts[0], parts[1], "\t".join(parts[2:])
    added = None if added_raw == "-" else int(added_raw)
    deleted = None if deleted_raw == "-" else int(deleted_raw)
    return added, deleted, path


def _git_log(repo, extra_args, rng):
    """Run one `git log --numstat` pass over `rng` and return its stdout.

    `core.quotePath=false` keeps non-ASCII paths as real UTF-8 strings instead
    of octal escapes, and `--no-renames` makes a rename an honest delete + add
    instead of an unopenable "src/{a => b}" pseudo-path that splits one file's
    churn across several keys. Both matter for every pass, so both live here.

    `log.diffMerges=separate` is pinned on the command line because it is what
    `-m` means: a user or repo config setting it to `off` makes pass 2 emit no
    file rows at all, so `merge_files` would come back empty beside a non-zero
    `merges_measured` and read as "the merges changed nothing".
    """
    cmd = [
        "git",
        "-c",
        "core.quotePath=false",
        "-c",
        "log.diffMerges=separate",
        "-C",
        repo,
        "log",
        "--numstat",
        "--no-renames",
        *extra_args,
        LOG_FORMAT,
        rng,
        "--",  # terminate rev parsing so the range can never match a pathspec
    ]
    try:
        # Decode explicitly: git emits UTF-8 path bytes regardless of the
        # caller's locale, and a C locale would otherwise decode them as ASCII.
        # surrogateescape, not replace: replace maps every invalid byte to the
        # same U+FFFD, so two distinct non-UTF-8 paths would collapse into one
        # `files` key with their churn silently summed. Lone surrogates survive
        # json.dumps (escaped as \udcXX under ensure_ascii) and json.loads.
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="surrogateescape",
            env={k: v for k, v in os.environ.items() if not k.startswith("GIT_")},
        )
    except Exception as exc:  # noqa: BLE001
        _emit({"ok": False, "error": str(exc)}, 1)

    if proc.returncode != 0:
        # stderr can be empty (a signal kill, a quiet failure); the exit code is
        # then the only thing left to report, so never emit an empty error.
        _emit(
            {
                "ok": False,
                "error": proc.stderr.strip() or f"git exited {proc.returncode}",
            },
            1,
        )
    return proc.stdout


def _parse_log(output, stories):
    """Turn one pass's log output into (commits, files_map). Shared by both."""
    commits = []
    files = {}  # path -> {path, _added, _deleted, binary_revisions, commit_count}
    seen = set()
    counting = True

    for raw in output.splitlines():
        if UNIT_SEP in raw:
            sha, parents, subject = raw.split(UNIT_SEP, 2)
            # Under -m, git repeats a merge's header once per parent unless it
            # also honours --first-parent (git 2.31+). Count only the first
            # block for a sha — git emits parents in order, so that block is
            # the first-parent diff either way, and no churn is double counted.
            counting = sha not in seen
            if not counting:
                continue
            seen.add(sha)
            commits.append(
                {
                    "sha": sha,
                    "subject": subject,
                    # Every id the subject names, in --stories order: a commit
                    # spanning two stories belongs to both. Word-boundary match
                    # so a story id like "1-2" does not also match "11-2".
                    "stories": [
                        sid
                        for sid in stories
                        if re.search(rf"\b{re.escape(sid)}\b", subject)
                    ],
                    "is_merge": len(parents.split()) > 1,
                }
            )
            continue

        if not counting or not raw.strip():
            continue

        parsed = _parse_numstat_line(raw)
        if parsed is None:
            continue
        added, deleted, path = parsed

        entry = files.get(path)
        if entry is None:
            # _added/_deleted are running sums over the path's text revisions.
            entry = {
                "path": path,
                "_added": 0,
                "_deleted": 0,
                "binary_revisions": 0,
                "commit_count": 0,
            }
            files[path] = entry

        entry["commit_count"] += 1
        if added is None or deleted is None:
            # A binary revision is unmeasurable, not zero — count it alongside
            # the sums instead of nulling the path's real measured churn.
            entry["binary_revisions"] += 1
        else:
            entry["_added"] += added
            entry["_deleted"] += deleted

    return commits, files


def _file_list(files):
    return [
        {
            "path": entry["path"],
            "added": entry["_added"],
            "deleted": entry["_deleted"],
            "net": entry["_added"] - entry["_deleted"],
            "commit_count": entry["commit_count"],
            "binary_revisions": entry["binary_revisions"],
        }
        for entry in files.values()
    ]


def main(argv=None):
    parser = JsonArgumentParser(
        description=(
            "Measure commit and per-file change evidence over a git revision "
            "range. Measures only; does not judge."
        ),
        add_help=False,
    )
    parser.add_argument("--repo", default=".", help="Path to the git repo (default: .)")
    parser.add_argument("--range", dest="range", help="Revision range REV..REV")
    parser.add_argument(
        "--stories",
        help="Comma-separated story ids to match against commit subjects.",
    )
    args = parser.parse_args(argv)

    stories = []
    if args.stories:
        # dict.fromkeys dedupes while keeping the caller's order: a repeated id
        # would otherwise land twice in a commit's `stories`, double counting
        # that commit in any per-story total built from the output.
        stories = list(
            dict.fromkeys(s.strip() for s in args.stories.split(",") if s.strip())
        )

    if not args.range:
        _emit(
            {
                "range": None,
                "note": "no range supplied",
                "commits": [],
                "files": [],
            }
        )

    # Accept only an explicit REV..REV range. Anything else silently measures
    # the wrong thing: a leading "-" is consumed by git as an option, a single
    # rev logs all history up to it, a bare pathspec logs by path, an empty
    # endpoint ("..", "a..", "..b") makes git default that side to HEAD, and a
    # three-dot "A...B" is a symmetric difference — a different commit set
    # entirely. partition splits at the FIRST "..", so any of those extra-dot
    # shapes leaves `right` empty or dot-prefixed.
    left, _, right = args.range.partition("..")
    if (
        args.range != args.range.strip()
        or args.range.startswith("-")
        or not left
        or not right
        or right.startswith(".")
    ):
        _emit(
            {
                "ok": False,
                "error": f"invalid --range {args.range!r}: expected a revision range like REV..REV",
            },
            2,
        )

    # Pass 1 — the listing. No extra args, so full topology: every commit in
    # the range including merges, which is what per-story attribution reads.
    # Merges contribute no numstat rows here, so `files` is non-merge churn.
    commits, files = _parse_log(_git_log(args.repo, [], args.range), stories)
    merge_count = sum(1 for commit in commits if commit["is_merge"])

    # Pass 2 — merge churn, only when there is any. `-m --first-parent
    # --min-parents=2` walks the range head's first-parent spine and emits
    # exactly one diff-against-first-parent block per merge sitting on it.
    # Merges off that spine are counted in merge_count and never measured,
    # which is precisely why merges_measured is a separate key: the gap
    # between the two is a visible statement that some merges went
    # unmeasured. This never folds into `files` — a merge's first-parent diff
    # restates the churn of the commits it merged in, which pass 1 already
    # counted, so adding it in would double count.
    merge_commits, merge_files = [], {}
    if merge_count:
        merge_commits, merge_files = _parse_log(
            _git_log(
                args.repo,
                ["-m", "--first-parent", "--min-parents=2"],
                args.range,
            ),
            stories,
        )

    _emit(
        {
            "range": args.range,
            "commit_count": len(commits),
            "merge_count": merge_count,
            "merges_measured": len(merge_commits),
            "commits": commits,
            "files": _file_list(files),
            "merge_files": _file_list(merge_files),
            "stories_supplied": stories,
        }
    )


if __name__ == "__main__":
    if sys.platform == "win32":
        # Piped output on Windows defaults to a legacy code page, not UTF-8.
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    main()
