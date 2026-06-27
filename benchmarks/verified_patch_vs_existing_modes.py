"""Compact benchmark for verified anchored patch behavior.

Models an editor applying edits from a stale file snapshot. `hashline_baseline`
is an in-script strict whole-file hash guard, not a public tool mode.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import verified_patch_core as vp
from tools.file_operations import PatchResult, ShellFileOperations


class LocalEnv:
    def __init__(self, cwd: str):
        self.cwd = cwd

    def execute(self, command, cwd=None, timeout=None, stdin_data=None, **_kwargs):
        proc = subprocess.run(
            command,
            shell=True,
            cwd=cwd or self.cwd,
            capture_output=True,
            text=True,
            input=stdin_data,
            timeout=timeout,
        )
        return {"output": proc.stdout + proc.stderr, "returncode": proc.returncode}


def scenarios(n: int):
    templates = ("value = {v}", "MAX_RETRIES = {v}", 'settings["limit"] = {v}', "configure(retries={v})")
    for i in range(n):
        old = templates[i % len(templates)].format(v=i)
        new = templates[i % len(templates)].format(v=i + 2000)
        stale = f'header = "case-{i}"\n{old}\ntail = {i}\n'
        yield stale, stale, 2, old, new, True
        yield stale, stale.replace(old, old.replace(str(i), str(i + 1000))), 2, old, new, False
        yield stale, stale.replace("case-", "changed-"), 2, old, new, True
        layout = old.replace(" = ", "    = ", 1) if " = " in old else old.replace("retries=", " retries = ", 1)
        yield stale, stale.replace(old, layout), 2, old, new, True
        ws_snapshot = f'header = "case-{i}"\nmessage = "a b-{i}"\ntail = {i}\n'
        ws_actual = f'header = "case-{i}"\nmessage = "ab-{i}"\ntail = {i}\n'
        yield ws_snapshot, ws_actual, 2, f'message = "a b-{i}"', f'message = "updated-{i}"', False
        dup = f'scope = "first-{i}"\nflag = True\nscope = "second-{i}"\nflag = True\ntail = {i}\n'
        yield dup, dup, 4, "flag = True", "flag = False", True


def expected(actual: str, line_no: int, new: str) -> str:
    lines = actual.splitlines()
    lines[line_no - 1] = new
    return "\n".join(lines) + "\n"


def v4a_patch(path: Path, stale: str, line_no: int, old: str, new: str) -> str:
    lines = stale.splitlines()
    i = line_no - 1
    hunk = ([" " + lines[i - 1]] if i else []) + ["-" + old, "+" + new]
    if i + 1 < len(lines):
        hunk.append(" " + lines[i + 1])
    return "*** Begin Patch\n" f"*** Update File: {path}\n" + "\n".join(hunk) + "\n*** End Patch"


def strict_tag(content: str) -> str:
    return hashlib.blake2b(content.encode(), digest_size=2).hexdigest()


def apply(method: str, ops: ShellFileOperations, path: Path, stale: str, line_no: int, old: str, new: str) -> PatchResult:
    if method == "replace":
        return ops.patch_replace(str(path), old, new)
    if method == "v4a":
        return ops.patch_v4a(v4a_patch(path, stale, line_no, old, new))
    if method == "hashline_baseline":
        if strict_tag(path.read_text(encoding="utf-8")) != strict_tag(stale):
            return PatchResult(error="stale whole-file tag")
        lines = path.read_text(encoding="utf-8").splitlines()
        lines[line_no - 1] = new
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return PatchResult(success=True, files_modified=[str(path)])
    try:
        op = vp.make_replace_operation(stale, line_no, line_no, [new], context=1)
        path.write_text(
            vp.apply_operations(path.read_text(encoding="utf-8"), [op]),
            encoding="utf-8",
        )
        return PatchResult(success=True, files_modified=[str(path)])
    except vp.VerifiedPatchError as e:
        return PatchResult(error=str(e))


def classify(result: PatchResult, should_apply: bool, output: str, want: str) -> str:
    if result.success:
        return "correct" if should_apply and output == want else "silent_wrong"
    return "false_reject" if should_apply else "safe_reject"


def run(per_category: int) -> dict[str, Counter]:
    totals = {name: Counter() for name in ("replace", "v4a", "hashline_baseline", "verified")}
    with tempfile.TemporaryDirectory() as tmp:
        ops = ShellFileOperations(LocalEnv(tmp), cwd=tmp)
        for index, case in enumerate(scenarios(per_category)):
            stale, actual, line_no, old, new, should_apply = case
            want = expected(actual, line_no, new)
            for method in totals:
                path = Path(tmp) / f"{index}_{method}.py"
                path.write_text(actual)
                result = apply(method, ops, path, stale, line_no, old, new)
                totals[method][classify(result, should_apply, path.read_text(), want)] += 1
    return totals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-category", type=int, default=25)
    totals = run(parser.parse_args().per_category)
    print("| Mode | Correct | Safe reject | False reject | Silent wrong write |")
    print("|---|---:|---:|---:|---:|")
    for method, counts in totals.items():
        print(
            f"| {method} | {counts['correct']} | {counts['safe_reject']} | "
            f"{counts['false_reject']} | {counts['silent_wrong']} |"
        )


if __name__ == "__main__":
    main()
