# MUTATION_PROOF.md

Working tree was committed and clean before every mutation:

```
$ git status --short
$ git log --oneline -1
ed0dd5de76 gateway: fail loud when a port-binding platform cannot bind at startup
```

Transcripts are verbatim except that interleaved gateway log lines (`^INFO|WARNING|DEBUG|<timestamp>`) were filtered out of stdout; pytest's own output is untouched.

Each mutation below is applied to the committed code, the new test file is run (RED), the file is
restored with `git checkout -- <file>` and the same command is re-run (GREEN).

---

## (a) The fatal classification — `gateway/run_startup.py`

Mutation: delete the escalation in `_start_handle_no_connections`, so a non-retryable port-binder
failure next to a connected sibling falls through to the old `return False` (park + degrade).

```diff
                 if _parked:
                     logger.error(...)
-                if _port_binder_reasons:
-                    return self._startup_fail_unbound_port_binder(_port_binder_reasons)
             return False
```

### RED

```
$ .venv/bin/python -m pytest tests/gateway/test_startup_port_binder_fatal.py -q -p no:randomly
.F.....                                                                  [100%]
=================================== FAILURES ===================================
_______ test_unbound_port_binder_exits_78_even_with_a_connected_sibling ________

tmp_path = PosixPath('/private/var/folders/gq/0rs975rd2b9bj2h6zkymwx6r0000gn/T/pytest-of-engineer/pytest-653/test_unbound_port_binder_exits0')
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x10933b2d0>

    def test_unbound_port_binder_exits_78_even_with_a_connected_sibling(tmp_path, monkeypatch):
        runner = _runner(tmp_path, monkeypatch)
        message = (
            "Port 8642 already in use. Set platforms.api_server.port in config.yaml to a different "
            "value, then `/platform resume api_server`."
        )
    
        must_exit = runner._start_handle_no_connections(
            connected_count=1,
            enabled_platform_count=2,
            startup_retryable_errors=[],
            startup_nonretryable_errors=[f"api_server: {message}"],
            startup_nonretryable_details=[(Platform.API_SERVER, {"port": 8642}, message)],
        )
    
>       assert must_exit is True
E       assert False is True

tests/gateway/test_startup_port_binder_fatal.py:113: AssertionError
------------------------------ Captured log call -------------------------------
=============================== warnings summary ===============================
tests/gateway/test_startup_port_binder_fatal.py::test_real_port_conflict_is_fatal_and_non_retryable
tests/gateway/test_startup_port_binder_fatal.py::test_bind_retries_past_five_attempts_within_the_budget
tests/gateway/test_startup_port_binder_fatal.py::test_port_freed_after_the_old_window_still_binds
  /Users/engineer/workspace/hermes-worktrees/t_b241802d/gateway/platforms/api_server.py:4204: NotAppKeyWarning: It is recommended to use web.AppKey instances for keys.
  https://docs.aiohttp.org/en/stable/web_advanced.html#application-s-config
    self._app["api_server_adapter"] = self

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/gateway/test_startup_port_binder_fatal.py::test_unbound_port_binder_exits_78_even_with_a_connected_sibling
1 failed, 6 passed, 3 warnings in 4.31s
```

### GREEN (after `git checkout -- gateway/run_startup.py`)

```
$ git checkout -- gateway/run_startup.py
$ .venv/bin/python -m pytest tests/gateway/test_startup_port_binder_fatal.py -q -p no:randomly
.......                                                                  [100%]
=============================== warnings summary ===============================
tests/gateway/test_startup_port_binder_fatal.py::test_real_port_conflict_is_fatal_and_non_retryable
tests/gateway/test_startup_port_binder_fatal.py::test_bind_retries_past_five_attempts_within_the_budget
tests/gateway/test_startup_port_binder_fatal.py::test_port_freed_after_the_old_window_still_binds
  /Users/engineer/workspace/hermes-worktrees/t_b241802d/gateway/platforms/api_server.py:4204: NotAppKeyWarning: It is recommended to use web.AppKey instances for keys.
  https://docs.aiohttp.org/en/stable/web_advanced.html#application-s-config
    self._app["api_server_adapter"] = self

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
7 passed, 3 warnings in 4.34s
```

---

## (b) The derive-check — `gateway/run_startup.py`

Mutation: reintroduce exactly the hand-written literal the brief forbids, in
`_startup_fatal_port_binder_reasons`. Note that the fatal-path tests (a) still pass under this
mutation — only the derive-check catches it, which is the point.

```diff
-            if platform_binds_port(_platform.value, _extra)
+            if _platform.value in {"api_server", "webhook"}  # MUTATION: hand-written literal
```

### RED

```
$ .venv/bin/python -m pytest tests/gateway/test_startup_port_binder_fatal.py -q -p no:randomly
....F..                                                                  [100%]
=================================== FAILURES ===================================
____ test_fatal_classification_is_derived_from_port_binding_platform_values ____

    def test_fatal_classification_is_derived_from_port_binding_platform_values():
        """Every configured port binder must classify fatal, and non-binders must not.
    
        Computed by calling the production helper over the production set: a literal tuple in
        run_startup.py (``{"api_server", "webhook"}``) would pass a hand-written test and silently skip
        every platform added to PORT_BINDING_PLATFORM_VALUES afterwards.
        """
        classify = GatewayRunner._startup_fatal_port_binder_reasons
    
        for value in sorted(PORT_BINDING_PLATFORM_VALUES):
            platform = Platform(value)
            mode = PORT_BINDING_CONDITIONAL_MODES.get(value)
            extra = {"connection_mode": mode} if mode else {}
>           assert classify([(platform, extra, "boom")]) == [f"{value}: boom"], (
                f"{value} binds a port and must be fatal at startup"
            )
E           AssertionError: bluebubbles binds a port and must be fatal at startup
E           assert [] == ['bluebubbles: boom']
E             
E             Right contains one more item: 'bluebubbles: boom'
E             Use -v to get more diff

tests/gateway/test_startup_port_binder_fatal.py:186: AssertionError
=============================== warnings summary ===============================
tests/gateway/test_startup_port_binder_fatal.py::test_real_port_conflict_is_fatal_and_non_retryable
tests/gateway/test_startup_port_binder_fatal.py::test_bind_retries_past_five_attempts_within_the_budget
tests/gateway/test_startup_port_binder_fatal.py::test_port_freed_after_the_old_window_still_binds
  /Users/engineer/workspace/hermes-worktrees/t_b241802d/gateway/platforms/api_server.py:4204: NotAppKeyWarning: It is recommended to use web.AppKey instances for keys.
  https://docs.aiohttp.org/en/stable/web_advanced.html#application-s-config
    self._app["api_server_adapter"] = self

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/gateway/test_startup_port_binder_fatal.py::test_fatal_classification_is_derived_from_port_binding_platform_values
1 failed, 6 passed, 3 warnings in 4.30s
```

### GREEN (after `git checkout -- gateway/run_startup.py`)

```
$ git checkout -- gateway/run_startup.py
$ .venv/bin/python -m pytest tests/gateway/test_startup_port_binder_fatal.py -q -p no:randomly
.......                                                                  [100%]
=============================== warnings summary ===============================
tests/gateway/test_startup_port_binder_fatal.py::test_real_port_conflict_is_fatal_and_non_retryable
tests/gateway/test_startup_port_binder_fatal.py::test_bind_retries_past_five_attempts_within_the_budget
tests/gateway/test_startup_port_binder_fatal.py::test_port_freed_after_the_old_window_still_binds
  /Users/engineer/workspace/hermes-worktrees/t_b241802d/gateway/platforms/api_server.py:4204: NotAppKeyWarning: It is recommended to use web.AppKey instances for keys.
  https://docs.aiohttp.org/en/stable/web_advanced.html#application-s-config
    self._app["api_server_adapter"] = self

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
7 passed, 3 warnings in 4.17s
```

---

## (c) The widened bind budget — `gateway/platforms/api_server.py`

Mutation: put the old fixed 5-attempt cap back in front of the wall-clock deadline.

```diff
                 _bind_sleep = _BIND_RETRY_INITIAL_SLEEP
+                _attempts = 0
                 while True:
                     ...
                         _remaining = _bind_deadline - time.monotonic()
-                        if exc.errno != errno.EADDRINUSE or _remaining <= 0:
+                        _attempts += 1  # MUTATION: back to the fixed 5-attempt cap
+                        if exc.errno != errno.EADDRINUSE or _remaining <= 0 or _attempts >= 5:
                             raise
```

### RED

```
$ .venv/bin/python -m pytest tests/gateway/test_startup_port_binder_fatal.py -q -p no:randomly
.....FF                                                                  [100%]
=================================== FAILURES ===================================
____________ test_bind_retries_past_five_attempts_within_the_budget ____________

monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x104babd10>
fast_sleep = [0.2, 0.4, 0.8, 1.6]

    @pytest.mark.asyncio
    async def test_bind_retries_past_five_attempts_within_the_budget(monkeypatch, fast_sleep):
        """The old schedule gave up after 5 attempts / ~3s. The budget keeps trying until it expires."""
        monkeypatch.setattr(api_server_mod, "_BIND_RETRY_BUDGET_SECONDS", 3.0)
        squatter, port = _squat_port()
        adapter = _adapter(port)
        started = time.monotonic()
        try:
            assert await adapter.connect() is False
>           assert len(fast_sleep) > 5, (
                f"expected the retry loop to outlive the old 5-attempt cap, got {len(fast_sleep)}"
            )
E           AssertionError: expected the retry loop to outlive the old 5-attempt cap, got 4
E           assert 4 > 5
E            +  where 4 = len([0.2, 0.4, 0.8, 1.6])

tests/gateway/test_startup_port_binder_fatal.py:236: AssertionError
------------------------------ Captured log call -------------------------------
ERROR    gateway.platforms.api_server:api_server.py:4272 [Api_Server] Could not bind 127.0.0.1:59231: [Errno 48] error while attempting to bind on address ('127.0.0.1', 59231): address already in use. Set a different port in config.yaml: platforms.api_server.port
_______________ test_port_freed_after_the_old_window_still_binds _______________

monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x104869e90>
fast_sleep = [0.2, 0.4, 0.8, 1.6]

    @pytest.mark.asyncio
    async def test_port_freed_after_the_old_window_still_binds(monkeypatch, fast_sleep):
        """The losing side of a restart race self-heals instead of leaving the port dark forever."""
        monkeypatch.setattr(api_server_mod, "_BIND_RETRY_BUDGET_SECONDS", 5.0)
        squatter, port = _squat_port()
        real_start = api_server_mod.start_tcp_site
        attempts = {"n": 0}
    
        async def _counting_start(runner, host, prt, *, log_tag):
            attempts["n"] += 1
            # Release the port only after the old fixed 5-attempt schedule would have given up.
            if attempts["n"] == 7:
                squatter.close()
            if attempts["n"] < 7:
                raise OSError(errno.EADDRINUSE, "address already in use")
            return await real_start(runner, host, prt, log_tag=log_tag)
    
        monkeypatch.setattr(api_server_mod, "start_tcp_site", _counting_start)
        adapter = _adapter(port)
        try:
>           assert await adapter.connect() is True
E           assert False is True

tests/gateway/test_startup_port_binder_fatal.py:267: AssertionError
------------------------------ Captured log call -------------------------------
ERROR    gateway.platforms.api_server:api_server.py:4272 [Api_Server] Could not bind 127.0.0.1:59237: [Errno 48] address already in use. Set a different port in config.yaml: platforms.api_server.port
=============================== warnings summary ===============================
tests/gateway/test_startup_port_binder_fatal.py::test_real_port_conflict_is_fatal_and_non_retryable
tests/gateway/test_startup_port_binder_fatal.py::test_bind_retries_past_five_attempts_within_the_budget
tests/gateway/test_startup_port_binder_fatal.py::test_port_freed_after_the_old_window_still_binds
  /Users/engineer/workspace/hermes-worktrees/t_b241802d/gateway/platforms/api_server.py:4204: NotAppKeyWarning: It is recommended to use web.AppKey instances for keys.
  https://docs.aiohttp.org/en/stable/web_advanced.html#application-s-config
    self._app["api_server_adapter"] = self

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/gateway/test_startup_port_binder_fatal.py::test_bind_retries_past_five_attempts_within_the_budget
FAILED tests/gateway/test_startup_port_binder_fatal.py::test_port_freed_after_the_old_window_still_binds
2 failed, 5 passed, 3 warnings in 1.18s
```

### GREEN (after `git checkout -- gateway/platforms/api_server.py`)

```
$ git checkout -- gateway/platforms/api_server.py
$ .venv/bin/python -m pytest tests/gateway/test_startup_port_binder_fatal.py -q -p no:randomly
.......                                                                  [100%]
=============================== warnings summary ===============================
tests/gateway/test_startup_port_binder_fatal.py::test_real_port_conflict_is_fatal_and_non_retryable
tests/gateway/test_startup_port_binder_fatal.py::test_bind_retries_past_five_attempts_within_the_budget
tests/gateway/test_startup_port_binder_fatal.py::test_port_freed_after_the_old_window_still_binds
  /Users/engineer/workspace/hermes-worktrees/t_b241802d/gateway/platforms/api_server.py:4204: NotAppKeyWarning: It is recommended to use web.AppKey instances for keys.
  https://docs.aiohttp.org/en/stable/web_advanced.html#application-s-config
    self._app["api_server_adapter"] = self

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
7 passed, 3 warnings in 4.27s
```

---

## Supporting suites (brief §5)

```
$ .venv/bin/python -m pytest tests/gateway/test_runner_startup_failures.py tests/gateway/test_api_server_bind_guard.py tests/gateway/test_api_server.py tests/gateway/test_startup_port_binder_fatal.py tests/gateway/test_startup_failed_platform_reports_degraded.py -q -p no:randomly
........................................................................ [ 48%]
........................................................................ [ 96%]
.....                                                                    [100%]
=============================== warnings summary ===============================
tests/gateway/test_api_server_bind_guard.py::TestBindMechanics::test_immediate_rebind_after_disconnect
tests/gateway/test_api_server_bind_guard.py::TestBindMechanics::test_rebind_over_time_wait
tests/gateway/test_api_server_bind_guard.py::TestBindMechanics::test_port_conflict_sets_non_retryable_fatal_error
tests/gateway/test_startup_port_binder_fatal.py::test_real_port_conflict_is_fatal_and_non_retryable
tests/gateway/test_startup_port_binder_fatal.py::test_bind_retries_past_five_attempts_within_the_budget
tests/gateway/test_startup_port_binder_fatal.py::test_port_freed_after_the_old_window_still_binds
  /Users/engineer/workspace/hermes-worktrees/t_b241802d/gateway/platforms/api_server.py:4204: NotAppKeyWarning: It is recommended to use web.AppKey instances for keys.
  https://docs.aiohttp.org/en/stable/web_advanced.html#application-s-config
    self._app["api_server_adapter"] = self

tests/gateway/test_api_server.py: 59 warnings
  /Users/engineer/workspace/hermes-worktrees/t_b241802d/tests/gateway/test_api_server.py:351: NotAppKeyWarning: It is recommended to use web.AppKey instances for keys.
  https://docs.aiohttp.org/en/stable/web_advanced.html#application-s-config
    app["api_server_adapter"] = adapter

tests/gateway/test_api_server.py::TestPlatformEventCallbackEndpoint::test_rejects_invalid_google_chat_auth
  /Users/engineer/workspace/hermes-worktrees/t_b241802d/tests/gateway/test_api_server.py:1919: NotAppKeyWarning: It is recommended to use web.AppKey instances for keys.
  https://docs.aiohttp.org/en/stable/web_advanced.html#application-s-config
    app["platform_event_adapters"] = {

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
149 passed, 66 warnings in 33.11s
```

149 passed, 0 failed. (The brief's §5 list plus the new file; I added
`test_startup_failed_platform_reports_degraded.py` because it is the one existing test the rule
touches — see the note below.)

---

## Note: the one existing test whose expectation this change reverses

`tests/gateway/test_startup_failed_platform_reports_degraded.py::test_parked_platform_is_logged_at_error_and_the_run_is_degraded`
asserted exactly the scenario this brief overturns — api_server EADDRINUSE + a healthy Telegram
sibling => stay alive, `gateway_state == "degraded"`. I did not treat this as the §7 escape hatch
because the brief measured that precise scenario in §0 and §1 decides, with reasoning, that it is
wrong for port binders specifically. The test was therefore split:

- `test_parked_port_binder_takes_the_gateway_down_loudly` — the api_server scenario, now asserting
  exit 78 / `gateway_state == "startup_failed"`.
- `test_parked_non_port_platform_is_logged_at_error_and_the_run_is_degraded` — the ORIGINAL
  degraded-and-alive contract, carried by an unpaired WhatsApp adapter (binds no port), so the
  behaviour the old comment protects still has a live test.

`tests/gateway/test_api_server_bind_guard.py::test_port_conflict_sets_non_retryable_fatal_error` was
not changed in meaning: it only monkeypatches `_BIND_RETRY_BUDGET_SECONDS` down to 0.3s so CI does
not wait out the new 30s budget for a conflict that is permanent by construction.
