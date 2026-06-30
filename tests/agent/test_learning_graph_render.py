"""Behavior contracts for the terminal Star Map renderer.

Asserts invariants of the timeline layout, the ported age gradient + palette, and
the constellation scrubber — never a cell snapshot, which would be a
change-detector against layout tuning.
"""

from __future__ import annotations

from agent import learning_graph_render as render

LEAD_IN = render.LEAD_IN


def _payload(skills: int = 8, memories: int = 3, *, base_ts: int = 1_700_000_000):
    nodes = []
    for i in range(skills):
        nodes.append(
            {
                "id": f"skill{i}",
                "label": f"skill{i}",
                "kind": "skill",
                "timestamp": base_ts + i * 86400 * 20,
                "category": "devops" if i % 2 else "research",
                "useCount": i,
            }
        )
    for j in range(memories):
        nodes.append(
            {
                "id": f"memory:memory:{j}",
                "label": f"mem {j}",
                "kind": "memory",
                "timestamp": base_ts + (skills + j) * 86400 * 20,
                "category": "memory",
            }
        )
    edges = [{"source": "skill0", "target": "skill1"}] if skills > 1 else []
    return {
        "nodes": nodes,
        "edges": edges,
        "clusters": [{"category": "devops", "count": skills}, {"category": "memory", "count": memories}],
        "stats": {
            "learned_skills": skills,
            "memory_nodes": memories,
            "related_edges": len(edges),
            "memory_skill_edges": 0,
        },
    }


def _flatten(grid):
    return "".join(run[0] for row in grid for run in row)


def _styles(grid):
    return {run[1] for row in grid for run in row}


def test_recency_is_timed_and_bounded():
    rec = render.compute_recency(_payload()["nodes"])
    assert rec["timed"] is True
    for ratio in rec["rec"].values():
        assert LEAD_IN - 1e-9 <= ratio <= 1 + 1e-9
    assert abs(min(rec["rec"].values()) - LEAD_IN) < 1e-9
    assert abs(max(rec["rec"].values()) - 1.0) < 1e-9


def test_recency_ink_follows_age_gradient():
    # Old quiet → recent bright (constants.ts AGE_GRADIENT), monotonic in between.
    assert abs(render.recency_ink(0.0) - render.AGE_OLD_INK) < 1e-6
    assert abs(render.recency_ink(1.0) - render.AGE_NEW_INK) < 1e-6
    samples = [render.recency_ink(x / 10) for x in range(11)]
    assert samples == sorted(samples)


def test_undated_graph_falls_back_to_ordinal():
    nodes = [{"id": f"n{i}", "kind": "skill"} for i in range(5)]
    rec = render.compute_recency(nodes)
    assert rec["timed"] is False
    assert len(set(rec["rec"].values())) == len(nodes)


def test_grid_runs_are_text_style_alpha():
    # Runs are [text, style, alpha] with an optional 4th hex override for
    # category-colored bars.
    frame = render.render_graph(_payload(), cols=60, rows=20)
    for row in frame["grid"]:
        for run in row:
            assert 3 <= len(run) <= 4
            assert isinstance(run[0], str) and isinstance(run[1], str)
            assert isinstance(run[2], (int, float)) and 0.0 <= run[2] <= 1.0
            assert run[0] != ""
            if len(run) == 4:
                assert run[3] is None or isinstance(run[3], str)


def test_bars_render_skills_and_memories():
    frame = render.render_graph(_payload(skills=10, memories=4), cols=72, rows=18, reveal=1.0)
    flat = _flatten(frame["grid"])
    # Skills draw as comet trails (━), memories anchor on diamonds (◆).
    assert "━" in flat
    assert render.MEMORY_GLYPH in flat
    styles = _styles(frame["grid"])
    assert render.STYLE_SKILL in styles
    assert render.STYLE_MEMORY in styles


def test_run_alpha_follows_age_for_lit_stars():
    # An all-skill, dated graph at full reveal: the newest star is brighter ink
    # than the oldest (age gradient carried in the run alpha).
    payload = _payload(skills=12, memories=0)
    frame = render.render_graph(payload, cols=80, rows=20, reveal=1.0)
    alphas = [run[2] for row in frame["grid"] for run in row if run[1] == render.STYLE_SKILL]
    assert max(alphas) > min(alphas)


def test_reveal_monotonically_builds_up():
    payload = _payload(skills=12, memories=5)
    counts = [render.render_graph(payload, cols=60, rows=20, reveal=r)["visible"] for r in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert counts == sorted(counts)
    assert counts[-1] == len(payload["nodes"])


def test_empty_payload_renders_placeholder():
    frame = render.render_graph({"nodes": []}, cols=40, rows=12)
    assert frame["visible"] == 0
    assert "no learning yet" in _flatten(frame["grid"])


def test_grid_fits_within_row_budget():
    # The chart is a timeline of dated buckets + a trajectory row; it fills up to
    # the row budget but never overflows it.
    frame = render.render_graph(_payload(), cols=60, rows=14, reveal=1.0)
    assert 0 < len(frame["grid"]) <= 14


def test_legend_counts_and_glyphs():
    payload = _payload(skills=9, memories=4)
    legend = render.build_legend(payload)
    labels = {item["label"] for item in legend}
    assert "skills (9)" in labels
    assert "memories (4)" in labels
    glyphs = {item["glyph"] for item in legend}
    assert render.SKILL_GLYPH in glyphs and render.MEMORY_GLYPH in glyphs


def test_axis_labels_present_when_dated():
    axis = render.axis_labels(_payload())
    assert axis["start"] != "oldest"  # dated → real dates
    assert axis["end"] != "now"


def test_frames_play_through_grows_visibility():
    payload = _payload(skills=10, memories=4)
    out = render.render_frames(payload, cols=50, rows=16, frames=12)
    assert out["count"] == len(payload["nodes"])
    assert len(out["frames"]) == 12
    assert out["frames"][0]["visible"] <= out["frames"][-1]["visible"]
    assert out["frames"][-1]["visible"] == len(payload["nodes"])
    assert "axis" in out
    for fr in out["frames"]:
        assert fr["grid"]


def test_frames_count_is_clamped():
    payload = _payload(skills=3, memories=1)
    assert len(render.render_frames(payload, cols=40, rows=12, frames=1)["frames"]) == 2
    assert len(render.render_frames(payload, cols=40, rows=12, frames=9999)["frames"]) == 240


def test_format_date_handles_missing():
    assert render.format_date(None) == "unknown"
    assert render.format_date(0) == "unknown"
    assert render.format_date(1_700_000_000) != "unknown"


def test_derive_palette_distinct_memory_hue():
    pal = render.derive_palette("#FFD700", dark=True)
    assert pal["skill"].startswith("#") and pal["memory"].startswith("#")
    # Memory is a complement of the gold primary → clearly different ink.
    assert pal["memory"].lower() != pal["skill"].lower()


def test_summary_reports_learning_totals():
    lines = render.build_summary(_payload(skills=7, memories=2))
    assert any("7 learned skills" in line and "2 memories" in line for line in lines)
