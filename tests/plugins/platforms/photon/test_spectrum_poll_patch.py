"""Regression tests for Hermes' Spectrum poll-schema workaround.

Covers the bug where tapping a native iMessage poll dropped the vote: spectrum-ts
8.x's `pollSchema.title` was `z.string().nonempty()`, but the poll metadata the
cloud returns for an inbound vote carries an empty/missing title, so
`toCachedPoll` -> `asPoll(...)` threw a ZodError and the vote was silently lost.
Mirrors photon-hq/spectrum-ts#91 (closed, not merged) for the pinned 8.0.0 SDK.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


_PATCHER = Path("plugins/platforms/photon/sidecar/patch-spectrum-poll-schema.mjs")


def _tabify(src: str) -> str:
    """Convert two-space indentation to the tab indentation spectrum-ts ships in
    `@spectrum-ts/core/dist`, so the patch anchors (which match tabs) apply
    exactly as they do against a real install."""
    out = []
    for line in src.split("\n"):
        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)
        out.append("\t" * (indent // 2) + " " * (indent % 2) + stripped)
    return "\n".join(out)


# A faithful, *executable* slice of spectrum-ts 8.x's poll schemas plus a tiny
# inline `z` shim with just enough behavior (string/nonempty/max/optional,
# object/literal/boolean/array+min+max, superRefine) to make `asPoll` run
# without any external dependency. The schemas mirror the compiled
# `@spectrum-ts/core/dist` output (tab-indented via `_tabify`), so the patch
# anchors exercise the real code shape, and exporting `asPoll` lets the test
# assert runtime behavior (empty-title poll) rather than only string shape.
_SPECTRUM_POLL_FIXTURE = """
const z = (() => {
  const str = () => ({ kind: "string", nonEmpty: false, max: null, isOptional: false, min: null });
  const withMethods = (s) => Object.assign(s, {
    nonempty() { this.nonEmpty = true; return this; },
    max(n) { this.max = n; return this; },
    optional() { this.isOptional = true; return this; },
    min(n) { this.min = n; return this; },
    superRefine() { return this; },
    parse(v) { return parse(this, v); },
  });
  const parse = (s, v) => {
    if (s.isOptional && v === undefined) return undefined;
    switch (s.kind) {
      case "string": {
        if (typeof v !== "string") throw new Error("expected string");
        if (s.nonEmpty && v.length < 1) throw new Error("too small: expected string to have >=1 characters");
        if (s.max !== null && v.length > s.max) throw new Error("too big");
        return v;
      }
      case "boolean": return v;
      case "literal": if (v !== s.v) throw new Error("invalid literal"); return v;
      case "array": {
        if (!Array.isArray(v)) throw new Error("expected array");
        if (s.min !== null && v.length < s.min) throw new Error("too few");
        if (s.max !== null && v.length > s.max) throw new Error("too many");
        return v.map((x) => parse(s.elem, x));
      }
      case "object": {
        const out = {};
        for (const [k, sub] of Object.entries(s.shape)) out[k] = parse(sub, v[k]);
        return out;
      }
    }
  };
  return {
    string: () => withMethods(str()),
    boolean: () => ({ kind: "boolean" }),
    literal: (v) => ({ kind: "literal", v }),
    array: (elem, min, max) => withMethods({ kind: "array", elem, min: min ?? null, max: max ?? null }),
    object: (shape) => withMethods({ kind: "object", shape }),
  };
})();
const pollChoiceSchema = z.object({ title: z.string().nonempty() });
const pollSchema = z.object({
  type: z.literal("poll"),
  title: z.string().nonempty().max(300),
  options: z.array(pollChoiceSchema).min(2).max(10)
});
const pollOptionSchema = z.object({
  type: z.literal("poll_option"),
  option: pollChoiceSchema,
  poll: pollSchema,
  selected: z.boolean(),
  title: z.string().nonempty()
}).superRefine((value, ctx) => { if (value.title !== value.option.title) ctx.addIssue({}); });
const asPoll = (input) => pollSchema.parse({ type: "poll", ...input });
const asPollOption = (input) => pollOptionSchema.parse({ type: "poll_option", ...input, title: input.option.title });
export { asPoll, asPollOption };
"""


def _write_core_fixture(tmp_path: Path) -> Path:
    core = tmp_path / "node_modules" / "@spectrum-ts" / "core"
    dist = core / "dist"
    dist.mkdir(parents=True)
    (core / "package.json").write_text('{"type":"module"}\n', encoding="utf-8")
    chunk = dist / "index.js"
    chunk.write_text(_tabify(_SPECTRUM_POLL_FIXTURE), encoding="utf-8")
    return chunk


def _run_as_poll(chunk: Path, title: str) -> str:
    """Execute the patched/unpatched fixture and report asPoll() with a title."""
    url = chunk.resolve().as_uri()
    script = (
        f'import {{ asPoll }} from "{url}";'
        f'try {{ const out = asPoll({{ title: {title!r}, options: [{{ title: "A" }}, {{ title: "B" }}] }});'
        f'console.log("PARSED:" + out.title); }}'
        f'catch (e) {{ console.log("THREW:" + e.message); }}'
    )
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _run_as_poll_option(chunk: Path, option_title: str, poll_title: str | None) -> str:
    """Execute the patched/unpatched fixture and report asPollOption().

    Mirrors the agent-bound vote path: the chosen option title is read from
    ``input.option.title`` and re-emitted as ``title`` on the parsed payload.
    A parent poll with an empty/absent title is the production shape that
    previously dropped the vote.
    """
    url = chunk.resolve().as_uri()
    poll_field = (
        "{ type: 'poll', options: [{ title: 'A' }, { title: 'B' }] }"
        if poll_title is None
        else f"{{ type: 'poll', title: {repr(poll_title)}, options: [{{ title: 'A' }}, {{ title: 'B' }}] }}"
    )
    script = (
        f'import {{ asPollOption }} from "{url}";'
        f'try {{ const out = asPollOption({{ option: {{ title: {option_title!r} }}, '
        f'poll: {poll_field}, '
        f'selected: true }});'
        f'console.log("PARSED:" + out.title); }}'
        f'catch (e) {{ console.log("THREW:" + e.message); }}'
    )
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_spectrum_poll_schema_patch_accepts_empty_title(tmp_path: Path) -> None:
    """The poll-schema patch must let an inbound poll with an empty title reach
    the agent (the vote is no longer dropped), while keeping normal polls parsed."""
    chunk = _write_core_fixture(tmp_path)

    # Regression baseline: unpatched, an empty-title poll throws (what dropped
    # the real vote in production).
    pre = _run_as_poll(chunk, "")
    assert pre.startswith("THREW:too small"), pre
    # A normal title still parses on the unpatched schema.
    pre_ok = _run_as_poll(chunk, "Lunch?")
    assert pre_ok == "PARSED:Lunch?", pre_ok

    # Apply the Hermes patch.
    result = subprocess.run(
        ["node", str(_PATCHER), str(tmp_path)],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    patched = chunk.read_text(encoding="utf-8")
    assert "Relax spectrum-ts poll schemas" in patched
    # The three title schemas are now lenient on the wire...
    assert "const pollChoiceSchema = z.object({ title: z.string() });" in patched
    assert "title: z.string().max(300).optional()," in patched
    # ...and no `.nonempty()` survives on a title field.
    assert "title: z.string().nonempty()" not in patched

    # Runtime regression: empty-title poll now parses (vote no longer dropped).
    post = _run_as_poll(chunk, "")
    assert post == "PARSED:", post
    # Normal polls still parse with their title preserved.
    post_ok = _run_as_poll(chunk, "Lunch?")
    assert post_ok == "PARSED:Lunch?", post_ok

    # Re-running the patch is a no-op (idempotent self-heal on sidecar start).
    again = subprocess.run(
        ["node", str(_PATCHER), str(tmp_path)],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert again.returncode == 0, again.stderr
    assert chunk.read_text(encoding="utf-8") == patched


def test_spectrum_poll_schema_patch_accepts_empty_parent_title_for_vote(tmp_path: Path) -> None:
    """The poll-schema patch must also let an inbound *poll_option* (agent-bound
    vote path) with an empty/absent parent poll title round-trip, preserving the
    chosen option title — not just the bare asPoll() poll shape."""
    chunk = _write_core_fixture(tmp_path)

    # Regression baseline: unpatched, an empty parent-poll title throws on the
    # embedded pollSchema (what dropped the real vote on the agent-bound path).
    pre_empty = _run_as_poll_option(chunk, option_title="A", poll_title="")
    assert pre_empty.startswith("THREW:too small"), pre_empty
    # An absent poll title (undefined) also throws on the non-optional schema.
    pre_absent = _run_as_poll_option(chunk, option_title="A", poll_title=None)
    assert pre_absent.startswith("THREW:expected"), pre_absent
    # A titled poll still parses on the unpatched schema with the option kept.
    pre_ok = _run_as_poll_option(chunk, option_title="A", poll_title="Lunch?")
    assert pre_ok == "PARSED:A", pre_ok

    # Apply the Hermes patch.
    result = subprocess.run(
        ["node", str(_PATCHER), str(tmp_path)],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    # Runtime: empty and absent parent titles both round-trip, chosen option kept.
    post_empty = _run_as_poll_option(chunk, option_title="A", poll_title="")
    assert post_empty == "PARSED:A", post_empty
    post_absent = _run_as_poll_option(chunk, option_title="A", poll_title=None)
    assert post_absent == "PARSED:A", post_absent
    # Titled parent poll unaffected.
    post_ok = _run_as_poll_option(chunk, option_title="A", poll_title="Lunch?")
    assert post_ok == "PARSED:A", post_ok
