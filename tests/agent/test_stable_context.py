from agent.stable_context import (
    ContextSegment,
    ContextSegmentKind,
    StableContextBuilder,
)


def test_stable_segments_render_before_dynamic_segments():
    builder = StableContextBuilder()
    result = builder.build(
        [
            ContextSegment(
                name="current_task",
                kind=ContextSegmentKind.CURRENT_TASK,
                content="Fix bug",
                stable=False,
            ),
            ContextSegment(
                name="system",
                kind=ContextSegmentKind.SYSTEM,
                content="You are Hermes.",
                stable=True,
            ),
        ]
    )
    assert result.index('name="system"') < result.index('name="current_task"')


def test_builder_preserves_order_within_groups():
    builder = StableContextBuilder(include_hashes=False)
    result = builder.build(
        [
            ContextSegment("dynamic-a", ContextSegmentKind.CURRENT_TASK, "d1", stable=False),
            ContextSegment("stable-a", ContextSegmentKind.SYSTEM, "s1", stable=True),
            ContextSegment("dynamic-b", ContextSegmentKind.TOOL_RESULT, "d2", stable=False),
            ContextSegment("stable-b", ContextSegmentKind.SKILLS, "s2", stable=True),
        ]
    )
    assert result.index('name="stable-a"') < result.index('name="stable-b"')
    assert result.index('name="dynamic-a"') < result.index('name="dynamic-b"')


def test_builder_ignores_empty_segments():
    builder = StableContextBuilder()
    result = builder.build(
        [
            ContextSegment("blank", ContextSegmentKind.SYSTEM, "   "),
            ContextSegment("system", ContextSegmentKind.SYSTEM, "hello"),
        ]
    )
    assert 'name="blank"' not in result
    assert 'name="system"' in result


def test_hash_rendering_can_be_disabled():
    builder = StableContextBuilder(include_hashes=False)
    result = builder.build([
        ContextSegment("system", ContextSegmentKind.SYSTEM, "hello")
    ])
    assert 'hash=' not in result


def test_plain_builder_returns_plain_content():
    builder = StableContextBuilder()
    result = builder.build_plain([
        ContextSegment("system", ContextSegmentKind.SYSTEM, "alpha"),
        ContextSegment("task", ContextSegmentKind.CURRENT_TASK, "beta", stable=False),
    ])
    assert result == 'alpha\n\nbeta'


def test_hashes_are_deterministic():
    seg1 = ContextSegment("system", ContextSegmentKind.SYSTEM, "same")
    seg2 = ContextSegment("system", ContextSegmentKind.SYSTEM, "same")
    seg3 = ContextSegment("system", ContextSegmentKind.SYSTEM, "different")
    assert seg1.content_hash == seg2.content_hash
    assert seg1.content_hash != seg3.content_hash
