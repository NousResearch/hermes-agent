"""End-to-end check for the write guard and the read_file continuation offset.

Run with the hermes venv outside pytest (the pytest home-IO guard blocks this install's
real terminal backend, so a pytest version of these assertions cannot run here):

    ./venv/Scripts/python.exe tests/manual/write_guard_paged_read_check.py

Exit code 0 means every assertion held. Covers, against the real tool:
  * a truncated read reports next_offset, and coverage stops where the page stopped;
  * paging to the last page authorizes an overwrite (was: refused, forever);
  * one page alone still refuses (safety preserved);
  * an unread file still refuses; a small read file still writes;
  * a path respelled with different case still authorizes (Windows/macOS).
"""

import json
import os
import sys
import tempfile
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
os.environ.setdefault("HERMES_HOME", tempfile.mkdtemp(prefix="hg-check-"))

from tools.file_tools import read_file_tool, write_file_tool

BASE = pathlib.Path(tempfile.mkdtemp(prefix="wg-check-"))
failures = []


def check(name, condition, detail=""):
    print(("PASS  " if condition else "FAIL  ") + name + (f"  [{detail}]" if detail and not condition else ""))
    if not condition:
        failures.append(name)


def err(result):
    return json.loads(result).get("error")


def big(name, lines=3000):
    path = BASE / name
    path.write_text("\n".join(f"{i:04d} " + "x" * 60 for i in range(1, lines + 1)) + "\n",
                    encoding="utf-8")
    return path


p = big("partial.md")
page = json.loads(read_file_tool(str(p), 1, 2000, task_id="c1"))
check("truncated read reports next_offset", isinstance(page.get("next_offset"), int), page.get("hint", ""))
check("one page still refuses the overwrite", "Refusing to overwrite" in (err(write_file_tool(str(p), "x\n", task_id="c1")) or ""))

p = big("paged.md")
offset, pages = 1, 0
while offset and pages < 20:
    offset = json.loads(read_file_tool(str(p), offset, 2000, task_id="c2")).get("next_offset")
    pages += 1
check("fixture needs several pages", pages > 1, str(pages))
check("paging to the end authorizes the overwrite", err(write_file_tool(str(p), "rewritten\n", task_id="c2")) is None)
check("the overwrite landed", p.read_text(encoding="utf-8") == "rewritten\n")

p = BASE / "small.md"
p.write_text("a\nb\n", encoding="utf-8")
read_file_tool(str(p), 1, 2000, task_id="c3")
check("small read file still writes", err(write_file_tool(str(p), "c\n", task_id="c3")) is None)

p = BASE / "unread.md"
p.write_text("existing\n", encoding="utf-8")
check("unread file still refuses", "Refusing to overwrite" in (err(write_file_tool(str(p), "x\n", task_id="c4")) or ""))
check("refused file untouched", p.read_text(encoding="utf-8") == "existing\n")

p = BASE / "CaseMixed.md"
p.write_text("a\nb\n", encoding="utf-8")
read_file_tool(str(p), 1, 2000, task_id="c5")
check("case-variant path authorizes the overwrite", err(write_file_tool(str(p).upper(), "rewritten\n", task_id="c5")) is None)

print()
print(f"{len(failures)} failure(s)")
sys.exit(1 if failures else 0)
