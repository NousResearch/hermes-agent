# /// script
# requires-python = ">=3.10"
# dependencies = ["ruamel.yaml>=0.18"]
# ///
"""Detect the current retrospective epic and surgically update sprint-status.yaml.

Prints ONLY JSON to stdout. Errors are emitted as JSON to stdout with a non-zero
exit code. The ``update`` subcommand round-trips the YAML to preserve all comments
and formatting, writes atomically (temp file + ``os.replace``), and restores the
original file bytes on any validation failure.
"""

import argparse
import hashlib
import io
import json
import os
import re
import stat
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping
from datetime import datetime

from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import DoubleQuotedScalarString

STORY_RE = re.compile(r"^(\d+)-\d+[a-z]?-")  # trailing [a-z]? matches split-story keys like 2-6a-...
DATE_FORMAT = "%m-%d-%Y %H:%M"
# The authoritative action-item vocabulary, mirrored from bmad-sprint-planning's
# SKILL.md. Anything outside it would render as unknown in the status dashboard.
ACTION_STATUSES = ("open", "in-progress", "done")
# The retro-document frontmatter vocabulary. --verdict is only echoed back, but
# orchestrators branch on the echo, so a free-spelled value ("accepted with open
# items") would silently fall through every branch they write.
VERDICTS = ("accepted", "accepted-with-open-items", "rejected")


def _load_yaml(path):
    yaml = YAML(typ="rt")
    yaml.preserve_quotes = True
    # Pin the emitter to the indentation the sprint-status template ships with.
    # Without this, ruamel re-dumps block sequences at its own default offset and
    # every write silently de-indents pre-existing, untouched action_items.
    yaml.indent(mapping=2, sequence=4, offset=2)
    # Pin the dump encoding too: `_dump_bytes` serializes into a BytesIO, so the
    # emitter -- not this module -- encodes the bytes that land in the user's
    # file. utf-8 is ruamel's current default, but the file is read back as
    # utf-8 unconditionally, so state it rather than inherit it.
    yaml.encoding = "utf-8"
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.load(fh)
    return yaml, data


def _emit(obj, code=0):
    sys.stdout.write(json.dumps(obj))
    sys.exit(code)


def _emit_error(message, code=1, restored=None):
    """Emit a failure on the JSON-only contract.

    ``restored`` is included only when the caller can speak to the state of the
    target file; ``retro-document.md`` teaches callers to read it, so a write-path
    failure must never omit it and a read-only subcommand must never invent it.
    """
    payload = {"ok": False, "error": message}
    if restored is not None:
        payload["restored"] = restored
    _emit(payload, code)


class JsonArgumentParser(argparse.ArgumentParser):
    """Emit argparse failures on the JSON-only stdout contract, not usage text.

    Every parser built from this class is constructed with ``add_help=False``.
    The override below covers ``error()``, but ``-h`` never reaches it: the
    built-in help action calls ``print_help()`` and ``exit(0)`` directly, which
    would put plain usage text on stdout with a zero exit and break the
    JSON-only contract for the machine consumer this script exists to serve.
    Removing the action instead of intercepting it keeps the fix to one keyword
    per parser and routes ``-h`` through the already-tested ``error()`` path as
    an ordinary unrecognized argument. The cost is that the ``help=`` strings
    are unreachable from the CLI; the skill's references carry the usage a
    human needs.
    """

    def error(self, message):
        _emit({"ok": False, "error": f"argument error: {message}"}, 2)


def _slugify(text, maxlen=40):
    text = str(text)
    # Unicode-aware: a non-Latin action must keep its own characters in the id
    # rather than collapsing to a single placeholder shared by every item.
    slug = re.sub(r"[^\w]+", "-", text.lower(), flags=re.UNICODE).strip("-")
    slug = slug[:maxlen].strip("-")
    if not slug:
        # Nothing sluggable (punctuation/emoji only): a short content hash keeps
        # the id deterministic and distinct instead of a bare "item".
        slug = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    return slug


def _selector_label(entry):
    """Human-readable form of a --set-action-status selector, for error text.

    An entry carrying both forms is described by its ``id``, because that is the
    form resolution actually uses.
    """
    if isinstance(entry.get("id"), str):
        return f"id={entry['id']!r}"
    return f"epic={entry.get('epic')!r} action={entry.get('action')!r}"


def _match_action_items(entry, items):
    """Indices in ``items`` that the selector ``entry`` resolves to.

    ``id`` wins whenever it is present: the epic/action pair is the fallback for
    legacy items written before ids existed, so a caller that copied a whole item
    through gets the precise match rather than a text comparison. Matching is
    exact equality -- no normalization -- so a file that spells its epic as a
    string simply does not match and the caller gets a "no match" error instead of
    a silent write to the wrong item. ``bool`` is excluded on the file side too,
    since ``True == 1`` in Python.
    """
    item_id = entry.get("id")
    if isinstance(item_id, str):
        return [
            idx
            for idx, item in enumerate(items)
            if isinstance(item, Mapping) and item.get("id") == item_id
        ]
    epic_value = entry.get("epic")
    action_value = entry.get("action")
    return [
        idx
        for idx, item in enumerate(items)
        if isinstance(item, Mapping)
        and not isinstance(item.get("epic"), bool)
        and item.get("epic") == epic_value
        and item.get("action") == action_value
    ]


def _comment_counts(text):
    """Multiset of the comment lines in ``text``, indentation included.

    Keyed by the whole line so that a re-indented comment counts as a loss too:
    the guarantee callers are given is comments *and formatting*, and ruamel
    re-emits comments at their original column even when the block around them
    is re-indented, so an exact key costs nothing in practice.
    """
    return Counter(
        line for line in text.splitlines() if line.lstrip().startswith("#")
    )


def _load_document(path, restored=None):
    """Load and shape-check the document, reporting every failure as JSON.

    Returns ``(yaml, data, dev)``. ``dev`` is the live ``development_status``
    mapping when the key exists, otherwise a detached empty mapping -- the key is
    never inserted into the document as a side effect of loading.
    """
    try:
        yaml, data = _load_yaml(path)
    except UnicodeDecodeError as exc:
        _emit_error(f"{path} is not valid UTF-8: {exc}", 1, restored)
    except OSError as exc:
        _emit_error(str(exc), 1, restored)
    except Exception as exc:  # noqa: BLE001 - report any parse error as JSON
        _emit_error(str(exc), 1, restored)

    if data is not None and not isinstance(data, Mapping):
        _emit_error("root document is not a mapping", 1, restored)

    dev = data.get("development_status") if data is not None else None
    if dev is None:
        dev = {}
    elif not isinstance(dev, Mapping):
        _emit_error("development_status is not a mapping", 1, restored)

    return yaml, data, dev


def _retro_status(dev, retro_key, restored=None):
    status_value = dev.get(retro_key)
    if status_value is not None and not isinstance(status_value, str):
        _emit_error(
            f"{retro_key} status must be a string or null",
            1,
            restored,
        )
    return status_value


def _dump_bytes(yaml, data):
    """Serialize the document to bytes before any file is touched, so a dump
    failure cannot leave a partial file anywhere."""
    buf = io.BytesIO()
    yaml.dump(data, buf)
    return buf.getvalue()


def _atomic_write(path, payload, mode=None):
    """Replace ``path``'s contents with ``payload`` atomically.

    The bytes land in a temp file alongside the target, are fsynced, take the
    target's permission bits (mkstemp creates 0600, which would silently narrow
    the file), and only then rename over it -- so a kill or a full disk leaves
    the original file intact rather than truncated. ``path`` is resolved through
    symlinks first: renaming onto a symlink would detach the link and leave the
    real file stale while reporting success. The directory is fsynced too --
    best-effort, see below -- so the rename survives a power loss and not just
    the bytes.
    """
    path = os.path.realpath(path)
    directory = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(
        prefix=".sprint-status-", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        if mode is not None:
            os.chmod(tmp_path, mode)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    # The directory sync sits outside the try because once os.replace has
    # returned, the new bytes ARE the file: a failure past that point must not
    # propagate as a write failure, or the caller would report the original
    # "restored" about a write that in fact landed. Skipping it only risks the
    # rename not surviving a hard power loss.
    try:
        dir_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass


def cmd_detect_epic(args):
    # detect-epic never writes, so it reports no "restored" key.
    _, _, dev = _load_document(args.file)

    done_stories = []
    max_epic = None
    # Every story key with its epic, in document order, so the pending list can
    # be scoped to whichever epic detection lands on without a second pass over
    # the mapping. Non-story keys (epic-2, epic-2-retrospective, ...) never enter
    # here, because STORY_RE does not match them.
    story_keys = []
    for key, value in dev.items():
        m = STORY_RE.match(str(key))
        if not m:
            continue
        epic_num = int(m.group(1))
        story_keys.append((epic_num, key, value))
        if value == "done":
            done_stories.append(key)
            if max_epic is None or epic_num > max_epic:
                max_epic = epic_num

    # Optional --epic aims the gate at a supplied number (the -H <epic> path)
    # instead of auto-picking the highest epic with a done story. Without it,
    # behavior is unchanged: detect, then scope pending_stories to that epic.
    if args.epic is not None:
        if args.epic < 1:
            _emit_error(
                f"invalid --epic {args.epic} (expected a positive integer)",
                1,
            )
        selected = args.epic
    else:
        selected = max_epic

    if selected is None:
        # Uniform shape: pending_stories is always present, even with no epic to
        # scope it to, so a caller can read it without branching on epic first.
        _emit(
            {
                "epic": None,
                "story_count": 0,
                "done_stories": done_stories,
                "pending_stories": [],
                "retro_key": None,
                "retro_status": None,
            }
        )

    # Scoped to the selected epic only -- deliberately unlike done_stories, which
    # spans the whole file. A pending story in some *other* epic is not this
    # retrospective's business.
    selected_keys = [
        (key, value) for epic_num, key, value in story_keys if epic_num == selected
    ]
    pending_stories = [key for key, value in selected_keys if value != "done"]

    retro_key = f"epic-{selected}-retrospective"
    retro_status = _retro_status(dev, retro_key)
    _emit(
        {
            "epic": selected,
            # An epic the file has never heard of returns the same empty
            # pending_stories as a finished one; story_count is the key that
            # separates "complete" from "nonexistent" (a typo'd --epic), so the
            # unfinished-story gate can refuse to read silence as done.
            "story_count": len(selected_keys),
            "done_stories": done_stories,
            "pending_stories": pending_stories,
            "retro_key": retro_key,
            "retro_status": retro_status,
        }
    )


def cmd_update(args):
    # Every failure below happens before the write is attempted, so the file is
    # untouched and "restored": true is the honest report.
    untouched = True

    # 0. Validate the inputs before anything is mutated or written.
    if args.epic < 1:
        _emit_error(
            f"invalid --epic {args.epic} (expected a positive integer)", 1, untouched
        )

    if args.date is not None:
        try:
            parsed_date = datetime.strptime(args.date, DATE_FORMAT)
        except (ValueError, TypeError):
            _emit_error(
                f'invalid --date {args.date!r} (expected "MM-DD-YYYY HH:MM")',
                1,
                untouched,
            )
        # Normalize: strptime also accepts unpadded spellings like
        # "1-2-2026 9:05", and writing those through would defeat the point of
        # validating the format at all.
        last_updated = parsed_date.strftime(DATE_FORMAT)
    else:
        last_updated = datetime.now().strftime(DATE_FORMAT)

    if args.verdict is not None and args.verdict not in VERDICTS:
        _emit_error(
            f"invalid --verdict {args.verdict!r} (allowed: {', '.join(VERDICTS)})",
            1,
            untouched,
        )

    actions = []
    if args.add_action:
        try:
            actions = json.loads(args.add_action)
        except json.JSONDecodeError as exc:
            _emit_error(f"invalid --add-action JSON: {exc}", 1, untouched)
        if not isinstance(actions, list):
            _emit_error("--add-action must be a JSON array", 1, untouched)
        for item in actions:
            if not isinstance(item, dict):
                _emit_error(
                    "each --add-action item must be an object", 1, untouched
                )
            action_value = item.get("action")
            if not isinstance(action_value, str) or not action_value.strip():
                # A JSON null/number/object would otherwise be str()'d into a
                # literal "None"/"{...}" and written as a real action item.
                _emit_error(
                    "each --add-action item must have a non-empty string action",
                    1,
                    untouched,
                )

    status_updates = []
    if args.set_action_status:
        try:
            status_updates = json.loads(args.set_action_status)
        except json.JSONDecodeError as exc:
            _emit_error(f"invalid --set-action-status JSON: {exc}", 1, untouched)
        if not isinstance(status_updates, list):
            _emit_error("--set-action-status must be a JSON array", 1, untouched)
        for entry in status_updates:
            if not isinstance(entry, dict):
                _emit_error(
                    "each --set-action-status entry must be an object", 1, untouched
                )
            status_value = entry.get("status")
            if not isinstance(status_value, str) or status_value not in ACTION_STATUSES:
                _emit_error(
                    f"invalid --set-action-status status {status_value!r} "
                    f"(allowed: {', '.join(ACTION_STATUSES)})",
                    1,
                    untouched,
                )
            item_id = entry.get("id")
            if item_id is not None:
                # Present but unusable is an input error, not a silent fallback to
                # the epic/action form -- the caller meant to select by id.
                if not isinstance(item_id, str) or not item_id.strip():
                    _emit_error(
                        "each --set-action-status id must be a non-empty string",
                        1,
                        untouched,
                    )
                continue
            epic_value = entry.get("epic")
            action_value = entry.get("action")
            if isinstance(epic_value, bool) or not isinstance(epic_value, int):
                _emit_error(
                    "each --set-action-status entry must have a non-empty string id, "
                    "or an integer epic and a non-empty string action",
                    1,
                    untouched,
                )
            if not isinstance(action_value, str) or not action_value.strip():
                _emit_error(
                    "each --set-action-status entry must have a non-empty string id, "
                    "or an integer epic and a non-empty string action",
                    1,
                    untouched,
                )

    # 1. Keep original bytes for restore-on-failure, and the mode to write back
    #    with -- taken from the open handle so an unlink mid-run cannot leave the
    #    replacement silently narrowed to mkstemp's 0600.
    try:
        with open(args.file, "rb") as fh:
            original_bytes = fh.read()
            original_mode = stat.S_IMODE(os.fstat(fh.fileno()).st_mode)
    except OSError as exc:
        _emit_error(str(exc), 1, untouched)

    try:
        original_text = original_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        _emit_error(f"{args.file} is not valid UTF-8: {exc}", 1, untouched)

    # Every comment line in the file, not just the leading block: the template
    # ships one above action_items, and losing it corrupts the document just the
    # same as losing the header.
    original_comments = _comment_counts(original_text)

    yaml, data, dev = _load_document(args.file, restored=untouched)

    if data is None:
        _emit_error("empty or invalid YAML document", 1, untouched)

    epic = args.epic
    retro_key = f"epic-{epic}-retrospective"

    # null distinguishes "the flag was not passed" from "the key was absent",
    # which is the only case retro-document.md assigns "false" to.
    retro_key_found = None
    retro_status_before = None
    retro_status_after = None

    # 2. Optionally set the retrospective status to done (only if key exists).
    if args.set_retro_done:
        retro_key_found = retro_key in dev
        if retro_key_found:
            retro_status_before = _retro_status(dev, retro_key, restored=untouched)
            dev[retro_key] = "done"
            retro_status_after = "done"

    # 3. Take the action_items sequence as loaded and shape-check it once; both
    #    of the steps below operate on this same list.
    existing_actions = data.get("action_items")
    if existing_actions is not None and not isinstance(existing_actions, list):
        # A hand-corrupted file must still fail on the JSON contract, not crash.
        _emit_error("action_items in file is not a list", 1, untouched)
    items_added = 0
    original_action_len = len(existing_actions) if existing_actions is not None else 0

    # 4. Optionally transition the status of items already in the file. Selectors
    #    resolve against action_items *as loaded* and strictly before the
    #    --add-action append below, which is what makes an item appended in the
    #    same invocation unaddressable in that run. Every selector is resolved
    #    before any is applied, so a rejected batch never leaves a partial edit --
    #    and since nothing has been written yet, the file is still untouched.
    status_targets = []
    if status_updates:
        pool = existing_actions if isinstance(existing_actions, list) else []
        claimed = {}
        for entry in status_updates:
            label = _selector_label(entry)
            matches = _match_action_items(entry, pool)
            if not matches:
                _emit_error(f"no action item matches {label}", 1, untouched)
            if len(matches) > 1:
                _emit_error(
                    f"ambiguous --set-action-status selector {label}: "
                    f"{len(matches)} matches",
                    1,
                    untouched,
                )
            idx = matches[0]
            if idx in claimed:
                # Applying both would overcount action_items_updated, and a
                # conflicting pair would surface as a confusing post-write
                # validation failure instead of the input error it is.
                _emit_error(
                    f"duplicate --set-action-status targets: {label} and "
                    f"{claimed[idx]} resolve to the same action item",
                    1,
                    untouched,
                )
            claimed[idx] = label
            status_targets.append((idx, entry["status"]))

        for idx, new_status in status_targets:
            # A plain assignment keeps the item's own scalar style: ruamel's
            # CommentedMap re-applies the existing key's style on overwrite, for
            # every ScalarString subclass. Pinned by the style tests.
            pool[idx]["status"] = new_status

    # 5. Optionally append action items.
    if actions:
        seq = data.get("action_items")
        if seq is None:
            seq = []
            data["action_items"] = seq

        for item in actions:
            # Stable identity for orchestrator consumers: an id that lets a
            # re-run dedupe against prior items, and a ref back to the sourced
            # finding in the retro document. Both accept an explicit override.
            seq_num = len(seq) + 1
            action_text = str(item.get("action", ""))
            item_id = item.get("id") or (
                f"epic-{int(epic)}-retro-item-{seq_num}-{_slugify(action_text)}"
            )
            ref = item.get("ref") or (args.ref or "")
            entry = {
                "id": DoubleQuotedScalarString(str(item_id)),
                "epic": int(epic),
                "action": DoubleQuotedScalarString(action_text),
                "owner": DoubleQuotedScalarString(str(item.get("owner", ""))),
                "status": "open",
                "ref": DoubleQuotedScalarString(str(ref)),
            }
            seq.append(entry)
            items_added += 1

    # 6. Update last_updated.
    data["last_updated"] = last_updated

    # 7. Serialize, then swap the file atomically.
    try:
        _atomic_write(args.file, _dump_bytes(yaml, data), original_mode)
    except Exception as exc:  # noqa: BLE001
        # The target is only ever touched by the final rename, so if the write
        # raised, the original is still on disk byte-for-byte. Calling _restore
        # here would rewrite a file that was never modified -- the one write in
        # the program with nothing to gain and a truncated file to lose.
        _emit({"ok": False, "error": f"write failed: {exc}", "restored": True}, 1)

    # 8. Validate the written file; restore on any failure.
    def _fail(msg):
        restored = _restore(args.file, original_bytes, original_mode)
        _emit({"ok": False, "error": msg, "restored": restored}, 1)

    try:
        _, reloaded = _load_yaml(args.file)
    except Exception as exc:  # noqa: BLE001
        _fail(f"re-parse failed after write: {exc}")

    if reloaded is None:
        _fail("re-parse produced empty document after write")

    if not isinstance(reloaded, Mapping):
        _fail("re-parse produced a non-mapping document after write")

    rdev = reloaded.get("development_status") or {}
    if args.set_retro_done and retro_key_found:
        if not isinstance(rdev, Mapping) or rdev.get(retro_key) != "done":
            _fail(f"validation: {retro_key} not set to done after write")

    new_action_len = 0
    if reloaded.get("action_items") is not None:
        new_action_len = len(reloaded.get("action_items"))
    if new_action_len != original_action_len + items_added:
        _fail(
            "validation: action_items length mismatch "
            f"(expected {original_action_len + items_added}, got {new_action_len})"
        )

    if status_targets:
        # The recorded indices are still valid: the only other mutation to the
        # sequence is an append, and the length check above just confirmed it.
        reloaded_actions = reloaded.get("action_items")
        if not isinstance(reloaded_actions, list):
            _fail("validation: action_items is not a list after write")
        for idx, new_status in status_targets:
            reloaded_item = reloaded_actions[idx]
            if (
                not isinstance(reloaded_item, Mapping)
                or reloaded_item.get("status") != new_status
            ):
                _fail(
                    f"validation: action item at index {idx} is not "
                    f"{new_status!r} after write"
                )

    try:
        with open(args.file, "r", encoding="utf-8") as fh:
            new_text = fh.read()
    except (OSError, UnicodeDecodeError) as exc:
        _fail(f"re-read failed after write: {exc}")

    # Loss-only: a comment may legitimately move or be added (a long quoted value
    # can wrap onto a line that begins with '#'), but none may disappear.
    lost = original_comments - _comment_counts(new_text)
    if lost:
        first = next(
            (line for line in original_text.splitlines() if line in lost), None
        )
        _fail(f"validation: comment line lost after write: {first!r}")

    _emit(
        {
            "ok": True,
            "retro_key_found": retro_key_found,
            "retro_status_before": retro_status_before,
            "retro_status_after": retro_status_after,
            "action_items_added": items_added,
            "action_items_updated": len(status_targets),
            "last_updated": last_updated,
            "verdict": args.verdict,
        }
    )


def _restore(path, original_bytes, mode=None):
    """Best-effort restore of the original bytes. Returns True on success so a
    caller can surface a restore failure instead of hiding a half-written file.

    Atomic for the same reason the primary write is: a truncating rewrite that
    dies halfway destroys the very bytes it was trying to put back.
    """
    try:
        _atomic_write(path, original_bytes, mode)
        return True
    except Exception as exc:  # noqa: BLE001 - best-effort restore
        sys.stderr.write(f"restore failed: {exc}\n")
        return False


def build_parser():
    parser = JsonArgumentParser(
        description=(
            "Detect the current retrospective epic and surgically update "
            "sprint-status.yaml while preserving comments and formatting."
        ),
        add_help=False,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_detect = sub.add_parser(
        "detect-epic",
        help=(
            "Find the highest epic with a done story and its retrospective "
            "status, or aim the same pending_stories gate at --epic N."
        ),
        add_help=False,
    )
    p_detect.add_argument("--file", required=True, help="Path to sprint-status.yaml")
    p_detect.add_argument(
        "--epic",
        type=int,
        default=None,
        help=(
            "Optional. Scope the response to this epic number instead of "
            "auto-detecting the highest epic with a done story. Orchestrators "
            "passing -H <epic> should pass the same number here so pending_stories "
            "covers the epic they are about to retro."
        ),
    )
    p_detect.set_defaults(func=cmd_detect_epic)

    p_update = sub.add_parser(
        "update",
        help="Surgically update retro status and/or action items.",
        add_help=False,
    )
    p_update.add_argument("--file", required=True, help="Path to sprint-status.yaml")
    p_update.add_argument("--epic", required=True, type=int, help="Epic number")
    p_update.add_argument(
        "--set-retro-done",
        action="store_true",
        help="Set epic-<N>-retrospective to done if the key exists.",
    )
    p_update.add_argument(
        "--add-action",
        help='JSON array of {"action":str,"owner":str,"id"?:str,"ref"?:str} to append.',
    )
    p_update.add_argument(
        "--set-action-status",
        help=(
            "JSON array of status transitions for action items already in the file. "
            'Select each by id -- {"id":str,"status":"open|in-progress|done"} -- or, '
            'for legacy items with no id, by epic plus exact action text: '
            '{"epic":int,"action":str,"status":...}. An entry carrying both uses the '
            "id. Every selector must match exactly one item; any failure aborts the "
            "whole invocation and leaves the file untouched."
        ),
    )
    p_update.add_argument(
        "--ref",
        help="Reference (e.g. the retro document path) recorded on each appended action item.",
    )
    p_update.add_argument(
        "--verdict",
        help=(
            "Acceptance verdict echoed back in the JSON result for orchestrator "
            f"consumers. One of: {', '.join(VERDICTS)}."
        ),
    )
    p_update.add_argument(
        "--date",
        help='Value for last_updated (default: now as "MM-DD-YYYY HH:MM").',
    )
    p_update.set_defaults(func=cmd_update)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    if sys.platform == "win32":
        # Piped output on Windows defaults to a legacy code page, not UTF-8.
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    main()
