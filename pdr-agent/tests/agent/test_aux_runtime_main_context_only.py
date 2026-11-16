"""The auxiliary client's main runtime lives only in a context variable (Code Health PRV-09).

The process-global mirrors (``_RUNTIME_MAIN_PROVIDER`` and friends) were rewritten by every
session in the process, and a main-thread lookup with no context binding fell back to them. The
six fields were stored one at a time, unlocked for readers, so that fallback could return another
session's runtime, or one session's base_url paired with another's api_key. The
``auxiliary_is_nous`` flag was likewise written by every session and read by nothing.
"""

import threading

import agent.auxiliary_client as aux

_DELETED = (
    "auxiliary_is_nous",
    "get_auxiliary_extra_body",
    "_RUNTIME_MAIN_PROVIDER",
    "_RUNTIME_MAIN_MODEL",
    "_RUNTIME_MAIN_BASE_URL",
    "_RUNTIME_MAIN_API_KEY",
    "_RUNTIME_MAIN_API_MODE",
    "_RUNTIME_MAIN_AUTH_MODE",
    "_RUNTIME_MAIN_COMPAT_SNAPSHOT",
    "_RUNTIME_MAIN_COMPAT_LOCK",
    "_publish_runtime_main_mirrors",
    "_compat_runtime_main",
)


def test_no_process_global_runtime_state():
    assert [name for name in _DELETED if hasattr(aux, name)] == []


def test_a_binding_on_another_thread_never_reaches_this_one():
    aux.clear_runtime_main()
    bound = threading.Event()

    def other_session():
        aux.set_runtime_main(
            "custom", "other-model", base_url="https://other.example/v1", api_key="sk-other",
        )
        assert aux._runtime_main_value("base_url") == "https://other.example/v1"
        bound.set()

    worker = threading.Thread(target=other_session)
    worker.start()
    worker.join()

    assert bound.is_set()
    assert aux._normalize_main_runtime(None) == {}
    assert aux._runtime_main_value("base_url") == ""
    assert aux._runtime_main_value("api_key") == ""


def test_scoped_binding_is_visible_inside_and_gone_after():
    aux.clear_runtime_main()
    with aux.scoped_runtime_main({"base_url": "https://scoped.example/v1", "api_key": "sk-scoped"}):
        assert aux._runtime_main_value("base_url") == "https://scoped.example/v1"
    assert aux._runtime_main_value("base_url") == ""
