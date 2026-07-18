from gateway.runtime_footer import build_footer_line, format_runtime_footer


def test_footer_exposes_role_full_session_and_root_thread():
    line = format_runtime_footer(
        model=None,
        context_tokens=0,
        context_length=None,
        profile="default",
        session_id="20260718_011205_4d2617",
        session_role="foreground",
        thread_id=None,
        fields=["role", "profile", "session", "thread"],
    )

    assert line == (
        "foreground · profile:default · "
        "session:20260718 · thread:root"
    )


def test_build_footer_exposes_background_topic():
    line = build_footer_line(
        user_config={
            "display": {
                "runtime_footer": {
                    "enabled": True,
                    "fields": ["role", "session", "thread"],
                }
            }
        },
        platform_key="telegram",
        model=None,
        context_tokens=0,
        context_length=None,
        session_id="session-2",
        session_role="background",
        thread_id="topic-77",
    )

    assert line == "background · session:session- · thread:topic-77"
