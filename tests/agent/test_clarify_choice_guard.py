from agent.clarify_choice_guard import clarify_choice_guard_action


def test_nudges_markdown_next_action_choices_when_clarify_available():
    response = """작업 확인됨.

## 결과 요약
확인했습니다.

## 다음 추천 작업
- 그대로 두기: 지금 상태를 유지합니다.
- 소스 하드게이트 구현하기: 재발을 막습니다.
- 보류: 나중에 합니다.
"""

    action, payload = clarify_choice_guard_action(
        response,
        valid_tool_names={"clarify", "terminal"},
        platform="desktop",
        attempts=0,
    )

    assert action == "nudge"
    assert payload is not None
    assert "clarify" in payload
    assert "choices" in payload


def test_nudges_fenced_select_blocks_because_they_are_not_native_ui():
    response = """## 다음 추천 작업

```select
A. Continue
B. Stop
```
"""

    action, payload = clarify_choice_guard_action(
        response,
        valid_tool_names={"clarify"},
        platform="desktop",
        attempts=0,
    )

    assert action == "nudge"
    assert payload is not None
    assert "fenced" in payload
    assert "clarify" in payload


def test_nudges_approval_choice_headings_too():
    response = """## 승인 필요
- 승인 실행
- 수정
- 보류
- 금지
"""

    action, payload = clarify_choice_guard_action(
        response,
        valid_tool_names={"clarify"},
        platform="desktop",
        attempts=0,
    )

    assert action == "nudge"
    assert payload is not None
    assert "clarify" in payload


def test_nudges_english_next_action_choice_headings_too():
    response = """## Recommended next actions
A. Keep current local state
B. Apply runtime restart
"""

    action, payload = clarify_choice_guard_action(
        response,
        valid_tool_names={"clarify"},
        platform="desktop",
        attempts=0,
    )

    assert action == "nudge"
    assert payload is not None
    assert "clarify" in payload


def test_allows_result_bullets_when_next_action_section_has_no_choices():
    response = """## 결과 요약
- helper 복원
- focused tests 통과

## 다음 추천 작업
추천은 현재 상태를 보존하고 나중에 커밋 후보를 정리하는 것입니다.
"""

    assert clarify_choice_guard_action(
        response,
        valid_tool_names={"clarify"},
        platform="desktop",
        attempts=0,
    ) == ("allow", None)


def test_allows_plain_next_action_recommendation_without_choice_list():
    response = """## 결과 요약
확인했습니다.

## 다음 추천 작업
추천은 지금 상태를 유지하는 것입니다. 선택이 필요하지 않습니다.
"""

    assert clarify_choice_guard_action(
        response,
        valid_tool_names={"clarify"},
        platform="desktop",
        attempts=0,
    ) == ("allow", None)


def test_allows_when_clarify_tool_is_not_available_or_noninteractive():
    response = """## 다음 추천 작업
- A
- B
"""

    assert clarify_choice_guard_action(
        response,
        valid_tool_names={"terminal"},
        platform="desktop",
        attempts=0,
    ) == ("allow", None)
    assert clarify_choice_guard_action(
        response,
        valid_tool_names={"clarify"},
        platform="cron",
        attempts=0,
    ) == ("allow", None)


def test_blocks_after_retry_instead_of_showing_dead_markdown_choices():
    response = """## 다음 추천 작업
1. 승인 실행
2. 수정
3. 보류
4. 금지
"""

    action, payload = clarify_choice_guard_action(
        response,
        valid_tool_names={"clarify"},
        platform="desktop",
        attempts=1,
    )

    assert action == "block"
    assert payload is not None
    assert "clarify" in payload
    assert "Markdown-only" in payload
    assert "## 다음 추천 작업" not in payload
