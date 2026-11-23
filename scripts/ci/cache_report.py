"""Cache archive observability from ordinary actions/cache logs (no inventory scans).

Sizes are compressed archive bytes, never unpacked directory or per-file changes.
A progress total is evidence of size, not evidence that a restore/write succeeded.
"""

from __future__ import annotations

from datetime import datetime
from html import escape
import re


def _record(name: str, action: str, step_number: int | None) -> dict:
    return {
        "name": name, "step_number": step_number, "action": action, "requested_key": None, "matched_key": None,
        "restore_status": "not-attempted" if action.endswith("/save") else "unknown",
        "restored_bytes": None, "save_status": "not-attempted", "saved_bytes": None,
        "saved_key": None, "warnings": [], "restore_warnings": [], "paths": [],
    }


def _message(raw: str) -> tuple[str | None, str]:
    raw = re.sub(r"\x1b\[[0-9;]*m", "", raw)
    match = re.match(r"^(\d{4}-\d\d-\d\dT[\d:.]+(?:Z|[+-][\d:]+)) (.*)$", raw)
    return (match[1], match[2]) if match else (None, raw)


def _stable_label(name: str) -> str:
    # GitHub uses the pinned action reference as the default top-level step name.
    return re.sub(r"(actions/cache(?:/restore|/save)?)@\S+", r"\1", name.removeprefix("Post "))


class _CacheLogParser:
    def __init__(self, steps: list[dict]):
        self.entries: list[dict] = []
        self.phase: dict | None = None
        self.marker: str | None = None
        self.inputs = False
        self.path_input = False
        self.steps = [
            (datetime.fromisoformat(s["started_at"]).timestamp(),
             datetime.fromisoformat(s["completed_at"]).timestamp() + 1, s["name"], s.get("number"))
            for s in steps if s.get("started_at") and s.get("completed_at")
            and s.get("conclusion") != "skipped"
        ]

    def _context(self, timestamp: str | None, action: str) -> tuple[str, int | None]:
        step, number = None, None
        if timestamp:
            instant = datetime.fromisoformat(timestamp).timestamp()
            step, number = next(((name, number) for start, end, name, number in reversed(self.steps)
                                 if start <= instant < end), (None, None))
        labels = [_stable_label(label) for label in (step, self.marker) if label]
        return " / ".join(dict.fromkeys(labels)) or action, number

    def _start(self, timestamp: str | None, action: str, save: bool):
        name, number = self._context(timestamp, action)
        self.phase = {
            "entry": _record(name, action, number), "save": save,
            "size": None, "received": None, "sent": None, "restored": False,
        }

    def _save_target(self, entry: dict) -> dict | None:
        key = entry["saved_key"] or entry["requested_key"]
        candidates = [e for e in reversed(self.entries) if e["save_status"] == "not-attempted"]
        if key:
            exact = next((e for e in candidates if e["requested_key"] == key), None)
            if exact is not None:
                return exact
        # Content-keyed saves compute a new key after the build. Their identical
        # transport paths identify which restored archive the write replaces.
        if entry["paths"]:
            return next((e for e in candidates if e["paths"] == entry["paths"]), None)
        if key:
            return None
        # Missing-path and other pre-upload failures can have no key in the post log.
        return next((e for e in candidates if e["name"] == entry["name"]), None)

    def _finish(self):
        phase, self.phase = self.phase, None
        self.inputs = False
        self.path_input = False
        if phase is None:
            return
        entry = phase["entry"]
        if phase["save"]:
            if entry["save_status"] == "not-attempted":
                return
            if entry["save_status"] == "saved":
                entry["saved_bytes"] = phase["sent"] if phase["sent"] is not None else phase["size"]
            target = self._save_target(entry)
            if target:
                for field in ("saved_key", "save_status", "saved_bytes"):
                    target[field] = entry[field]
                target["warnings"].extend(entry["warnings"])
                return
        elif phase["restored"]:
            entry["restore_status"] = (
                "exact" if entry["requested_key"] == entry["matched_key"] else "fallback"
            )
            entry["restored_bytes"] = phase["size"] if phase["size"] is not None else phase["received"]
        elif entry["restore_status"] == "unknown" and entry["warnings"]:
            entry["restore_status"] = "failed"
        self.entries.append(entry)

    def _warning(self, phase: dict, line: str) -> bool:
        match = re.match(r"^(?:##\[(?:warning|error)\]|\[(?:warning|error)\])(.+)$", line)
        message = match[1] if match else line
        failure = message.startswith(("Failed to save:", "Failed to restore:", "Path Validation Error:"))
        if not match and not failure:
            return False
        phase["entry"]["warnings"].append(message)
        if not phase["save"]:
            phase["entry"]["restore_warnings"].append(message)
        # Unrelated actions also emit warnings during Post job cleanup.
        if phase["save"] and failure:
            phase["entry"]["save_status"] = "failed"
            key = re.search(r"with key (.+?)(?:,|$)", message)
            if key:
                phase["entry"]["requested_key"] = key[1]
        return True

    def _event(self, phase: dict, line: str):
        if self._warning(phase, line):
            return
        size = re.fullmatch(r"Cache Size: ~[\d.]+ MB \((\d+) B\)", line)
        progress = re.fullmatch(r"(Received|Sent) \d+ of (\d+) \([\d.]+%\), [\d.]+ MBs/sec", line)
        if size:
            phase["size"] = int(size[1])
        if progress:
            phase["received" if progress[1] == "Received" else "sent"] = int(progress[2])
            if phase["save"] and phase["entry"]["save_status"] == "not-attempted":
                phase["entry"]["save_status"] = "unknown"
        if phase["save"]:
            self._save_event(phase, line)
        else:
            self._restore_event(phase, line)

    def _save_event(self, phase: dict, line: str):
        entry = phase["entry"]
        saved = re.fullmatch(r"Cache saved with key: (.+)", line)
        skipped = re.fullmatch(r"Cache hit occurred on the primary key (.+), not saving cache\.", line)
        if saved or skipped:
            entry["saved_key"] = (saved or skipped)[1]
            entry["save_status"] = "saved" if saved else "skipped-exact-hit"

    def _restore_event(self, phase: dict, line: str):
        entry = phase["entry"]
        hit = re.fullmatch(r"Cache hit for(?: restore-key)?: (.+)", line)
        restored = re.fullmatch(r"Cache restored from key: (.+)", line)
        miss = re.fullmatch(r"Cache not found for input keys: (.+)", line)
        lookup = re.fullmatch(r"Cache found and can be restored from key: (.+)", line)
        if hit or restored or lookup:
            entry["matched_key"] = (hit or restored or lookup)[1]
            if entry["requested_key"] is None and hit and "restore-key" not in line:
                entry["requested_key"] = hit[1]
        if restored or line == "Cache restored successfully":
            phase["restored"] = True
        if miss:
            entry["restore_status"] = "miss"
            if entry["requested_key"] is None:
                entry["requested_key"] = miss[1].split(", ")[0]
        if lookup or line == "Lookup only - skipping download":
            entry["restore_status"] = "lookup-only"

    def feed(self, timestamp: str | None, line: str):
        start = re.fullmatch(r"##\[start-action display=(.*);id=[^\]]+\]", line)
        if start:
            self._finish()
            self.marker = start[1]
            return
        if line.startswith("##[end-action"):
            self._finish()
            self.marker = None
            return
        if line == "Post job cleanup.":
            self._finish()
            self._start(timestamp, "actions/cache", save=True)
            return
        if line.startswith("##[group]Run "):
            self._finish()
            action = re.fullmatch(r"##\[group\]Run (actions/cache(?:/restore|/save)?)@\S+", line)
            if action:
                self._start(timestamp, action[1], save=action[1].endswith("/save"))
                self.inputs = True
            else:
                self.marker = None
            return
        if line == "##[endgroup]":
            self.inputs = False
            return
        if self.phase is None:
            return
        if self.inputs:
            path = re.fullmatch(r"  path: ?(.*)", line)
            if path:
                self.phase["entry"]["paths"] = [path[1]] if path[1] else []
                self.path_input = True
            elif re.match(r"  [\w-]+:", line) or line in ("with:", "env:"):
                self.path_input = False
            elif self.path_input and line:
                self.phase["entry"]["paths"].append(line)
            key = re.fullmatch(r"  key: (.+)", line)
            if key:
                self.phase["entry"]["requested_key"] = key[1]
            return
        self._event(self.phase, line)


def parse_cache_log(log: str, steps: list[dict] | None = None) -> list[dict]:
    """Parse invocation-local cache receipts; absent archive sizes remain ``None``.

    Raw GitHub API steps supply top-level labels when composite markers are absent.
    Identities are label + occurrence, independent of content keys and post UUIDs.
    Restore-only and explicit-save actions with the same primary key are paired,
    without applying the combined action's exact-hit save guard to explicit saves.
    """
    parser = _CacheLogParser(steps or [])
    for raw in log.splitlines():
        parser.feed(*_message(raw))
    parser._finish()
    occurrences: dict[str, int] = {}
    for entry in parser.entries:
        # PRs use restore-only PM actions where main uses combined cache actions.
        labels = [re.sub(r"^(?:Cache|Restore|Save) ", "", label).removesuffix(" without saving")
                  for label in entry["name"].split(" / ")]
        name = " / ".join(labels)
        occurrences[name] = occurrences.get(name, 0) + 1
        entry["occurrence"] = occurrences[name]
        entry["identity"] = f"{name} #{entry['occurrence']}"
    return parser.entries


def _bytes(value: int | None, *, signed: bool = False) -> str:
    if value is None:
        return "unknown"
    exact = f"{value:+,}" if signed else f"{value:,}"
    if abs(value) < 1024:
        return f"{exact} B"
    mib = f"{value / (1024 * 1024):+.2f}" if signed else f"{value / (1024 * 1024):.2f}"
    return f"{mib} MiB"


def _archive_size(entry: dict | None) -> int | None:
    if entry is None:
        return None
    if entry["save_status"] == "saved":
        return entry["saved_bytes"]
    if entry["restore_status"] in ("exact", "fallback"):
        return entry["restored_bytes"]
    return None


def _bucket(entry: dict) -> str:
    status = entry["restore_status"]
    if status in ("exact", "fallback"):
        return "hits"
    if status == "miss" and not entry.get("restore_warnings", entry["warnings"]):
        return "misses"
    if status == "not-attempted":
        return "saves without restore"
    return "unavailable"


def _view(timings: dict, baseline: dict | None) -> dict:
    data = timings["cache_data"]
    is_main = timings.get("is_main", False)
    references = {(e.get("job", ""), e["identity"]): e
                  for e in (baseline or {}).get("cache_data", {}).get("entries", [])}
    groups: dict[str, list[dict]] = {name: [] for name in ("hits", "misses", "unavailable", "saves without restore")}
    missed = []
    for entry in data.get("entries", []):
        group = _bucket(entry)
        groups[group].append(entry)
        if entry["restore_status"] not in ("not-attempted", "lookup-only") and group != "hits":
            reference = entry if is_main else references.get((entry.get("job", ""), entry["identity"]))
            missed.append(_archive_size(reference))
    return {"groups": groups, "attempts": len(groups["hits"]) + len(missed), "missed": missed,
            "references": references, "is_main": is_main, "data": data,
            "has_baseline": "cache_data" in (baseline or {})}


def _amount(values: list[int | None], label: str) -> str:
    known = [value for value in values if value is not None]
    unknown = len(values) - len(known)
    result = f"{_bytes(sum(known))} {label}" if known or not values else f"? bytes {label}"
    return result + (f" (+{unknown} unknown sizes)" if unknown else "")


def _summary(view: dict) -> str:
    groups = view["groups"]
    result = (f"{len(groups['hits'])} / {view['attempts']} cache hits ({len(groups['misses'])} miss{'es' if len(groups['misses']) != 1 else ''})"
    + "\n" 
              + _amount([e["restored_bytes"] for e in groups["hits"]], "restored")
              + "\n" + _amount(view["missed"], "not restored"))
    if view["is_main"]:
        result += "; " + _amount([e["saved_bytes"] for rows in groups.values() for e in rows
                                 if e["save_status"] == "saved"], "saved")
    return result + "."


def _notes(view: dict) -> list[str]:
    source = "newly saved archive sizes" if view["is_main"] else "the main-size reference"
    notes = [f"Compressed archive bytes. Not-restored bytes use {source}; missing sizes remain unknown."]
    if not view["is_main"] and not view["has_baseline"]:
        notes.append("No main archive reference yet; deltas and missed sizes are unknown.")
    if view["data"].get("unavailable_jobs"):
        notes.append("Unavailable job logs: " + ", ".join(view["data"]["unavailable_jobs"]))
    lookups = sum(e["restore_status"] == "lookup-only" for e in view["groups"]["unavailable"])
    if lookups:
        notes.append(f"{lookups} lookup-only checks excluded from restore attempts.")
    return notes


def _link(entry: dict) -> str:
    url = entry.get("job_url", "")
    if not url.startswith(("https://", "http://")):
        return "unavailable"
    number = entry.get("step_number")
    label = "view step" if number is not None else "view job"
    if number is not None:
        url += f"#step:{number}:1"
    return f'<a href="{escape(url, quote=True)}">{label}</a>'


def _saved(entry: dict) -> str:
    status = entry["save_status"]
    if status == "saved":
        return _bytes(entry["saved_bytes"])
    if status == "skipped-exact-hit":
        return _bytes(0) + " (exact-hit skip)"
    return status.replace("-", " ")


def _headers(group: str, is_main: bool) -> list[str]:
    headers = ["Job", "Step", "Logs"]
    if group == "hits":
        headers.append("Restored")
        if not is_main:
            headers.append("Δ vs main")
    elif group == "unavailable":
        headers.append("Status")
    if is_main:
        headers.append("Saved")
    return headers


def _cells(entry: dict, group: str, view: dict) -> list[str]:
    cells = [entry.get("job", ""), entry["name"]]
    if group == "hits":
        cells.append(_bytes(entry["restored_bytes"]))
        if not view["is_main"]:
            reference = view["references"].get((entry.get("job", ""), entry["identity"]))
            size = _archive_size(reference)
            current = entry["restored_bytes"]
            cells.append(_bytes(current - size, signed=True) if current is not None and size is not None else "unknown")
    elif group == "unavailable":
        cells.append(entry["restore_status"])
    if view["is_main"]:
        cells.append(_saved(entry))
    return cells


def _table(rows: list[dict], group: str, view: dict) -> str:
    headers = "".join(f"<th>{escape(h)}</th>" for h in _headers(group, view["is_main"]))
    lines = [f"<table><thead><tr>{headers}</tr></thead><tbody>"]
    for entry in rows:
        cells = [escape(cell) for cell in _cells(entry, group, view)]
        cells.insert(2, _link(entry))
        lines.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    lines.append("</tbody></table>")
    return "\n".join(lines)


def _details(view: dict) -> str:
    lines = ["<details>", "<summary>per-job cache results</summary>", ""]
    for group, rows in view["groups"].items():
        if group not in ("hits", "misses") and (not rows or (group == "saves without restore" and not view["is_main"])):
            continue
        lines += [f"<h3>{group}</h3>", ""]
        lines += [_table(rows, group, view) if rows else "none", ""]
    for note in _notes(view):
        lines.append(f"<p>{escape(note)}</p>")
    for rows in view["groups"].values():
        for entry in rows:
            for warning in entry["warnings"]:
                text = escape(f"{entry.get('job', '')} / {entry['name']}: {warning}")
                lines.append(f"<p>{text}</p>")
    lines += ["", "</details>"]
    return "\n".join(lines)


def render_cache_summary(timings: dict, baseline: dict | None) -> str:
    if "cache_data" not in timings:
        return ""
    view = _view(timings, baseline)
    return "### CI caches\n\n" + _summary(view) + "\n\n" + _details(view)


def render_cache_html(timings: dict, baseline: dict | None) -> str:
    if "cache_data" not in timings:
        return ""
    view = _view(timings, baseline)
    return ('<section id="ci-caches"><h2>CI caches</h2><p>' + escape(_summary(view)) + '</p>'
            + _details(view) + '</section>')


def cache_review_status(timings: dict, baseline: dict | None, report_url: str | None = None) -> list[dict]:
    if "cache_data" not in timings:
        return []
    view = _view(timings, baseline)
    result = {"kind": "info", "title": "CI caches", "summary": _summary(view), "detail": _details(view)}
    if report_url:
        result.update(link=report_url, link_label="View report")
    return [{"source": "ci caches", "results": [result]}]
