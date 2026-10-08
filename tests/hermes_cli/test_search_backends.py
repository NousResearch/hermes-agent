from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


def test_search_backend_registration_is_profile_scoped_and_disposable(tmp_path):
    from hermes_cli.search_backends import SearchBackendRequest, SearchBackendResult, run_search_backends

    repo = tmp_path / "repo"
    repo.mkdir()
    request = SearchBackendRequest(
        pattern="Needle", path=str(repo), file_glob=None, limit=50, offset=0,
        output_mode="content", context=0, environment_kind="local", is_local=True, cwd=str(tmp_path),
    )
    manager_a = PluginManager(scope_key="/profiles/a")
    manager_b = PluginManager(scope_key="/profiles/b")
    manifest = PluginManifest(name="example-search", key="example-search", source="user")
    ctx_a = PluginContext(manifest, manager_a)
    ctx_b = PluginContext(manifest, manager_b)

    handle_a = ctx_a.register_search_backend(
        "example", lambda _request: SearchBackendResult(backend="a", route_reason="selected"))
    handle_b = ctx_b.register_search_backend(
        "example", lambda _request: SearchBackendResult(backend="b", route_reason="selected"))

    assert run_search_backends(request, scope="/profiles/a").result.backend == "a"
    assert run_search_backends(request, scope="/profiles/b").result.backend == "b"

    handle_a.dispose()
    assert run_search_backends(request, scope="/profiles/a") is None
    assert run_search_backends(request, scope="/profiles/b").result.backend == "b"
    handle_b.dispose()


def test_search_backend_rejects_paths_outside_requested_root(tmp_path):
    from hermes_cli.search_backends import (
        SearchBackendMatch, SearchBackendRequest, SearchBackendResult,
        register_search_backend, run_search_backends,
    )

    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    request = SearchBackendRequest(
        pattern="Needle", path=str(repo), file_glob=None, limit=50, offset=0,
        output_mode="content", context=0, environment_kind="local", is_local=True,
        cwd=str(tmp_path),
    )
    handle = register_search_backend(
        "unsafe",
        lambda _request: SearchBackendResult(
            backend="unsafe", route_reason="selected",
            matches=[SearchBackendMatch(str(outside), 1, "secret")], total_count=1,
        ),
        scope="/profiles/unsafe",
    )
    run = run_search_backends(request, scope="/profiles/unsafe")
    assert run.result is None
    assert run.route_reason == "backend_invalid:unsafe"
    handle.dispose()


def test_replaced_backend_handle_cannot_remove_new_generation(tmp_path):
    from hermes_cli.search_backends import (
        SearchBackendRequest, SearchBackendResult, register_search_backend, run_search_backends,
    )

    repo = tmp_path / "repo"
    repo.mkdir()
    request = SearchBackendRequest(
        pattern="Needle", path=str(repo), file_glob=None, limit=50, offset=0,
        output_mode="content", context=0, environment_kind="local", is_local=True,
        cwd=str(tmp_path),
    )
    first = register_search_backend(
        "same", lambda _: SearchBackendResult("first", "selected"), scope="/profiles/replaced")
    second = register_search_backend(
        "same", lambda _: SearchBackendResult("second", "selected"), scope="/profiles/replaced")
    first.dispose()
    assert run_search_backends(request, scope="/profiles/replaced").result.backend == "second"
    second.dispose()


def test_malformed_result_and_decline_reason_fail_to_native_fallback(tmp_path):
    from hermes_cli.search_backends import (
        SearchBackendDecline, SearchBackendRequest, SearchBackendResult,
        register_search_backend, run_search_backends,
    )

    repo = tmp_path / "repo"
    repo.mkdir()
    request = SearchBackendRequest(
        pattern="Needle", path=str(repo), file_glob=None, limit=50, offset=0,
        output_mode="content", context=0, environment_kind="local", is_local=True,
        cwd=str(tmp_path),
    )
    malformed = register_search_backend(
        "malformed", lambda _: SearchBackendResult("bad", "selected", matches=None),  # type: ignore[arg-type]
        scope="/profiles/malformed")
    run = run_search_backends(request, scope="/profiles/malformed")
    assert run.result is None and run.route_reason == "backend_invalid:malformed"
    malformed.dispose()

    decline = register_search_backend(
        "bad-decline", lambda _: SearchBackendDecline("bad\nreason"),
        scope="/profiles/bad-decline")
    run = run_search_backends(request, scope="/profiles/bad-decline")
    assert run.result is None and run.route_reason == "backend_invalid:bad-decline"
    decline.dispose()

    unordered = register_search_backend(
        "unordered", lambda _: SearchBackendResult("bad", "selected", matches=set()),  # type: ignore[arg-type]
        scope="/profiles/unordered")
    run = run_search_backends(request, scope="/profiles/unordered")
    assert run.result is None and run.route_reason == "backend_invalid:unordered"
    unordered.dispose()
