"""hermes_cli test suite — read this before trusting a green run.

Green on one file is not green on the suite. Before 2026-11-14 that was a real
trap here: the package selection reported 71 failures while most of the same
files passed when run alone. Two independent mechanisms caused it, and both are
about the ENVIRONMENT a test inherits rather than the code under test:

1. Module identity. Two fixtures in this directory purge every ``hermes_cli*``,
   ``hermes_state*`` and ``hermes_constants`` entry from ``sys.modules`` to
   simulate a cold import, and left the purge in place for whatever ran next. A
   later test then patched the pre-purge copy while production's function-local
   ``from hermes_cli.x import y`` re-imported a fresh module, so the patch was
   invisible and the real code ran. ``conftest.py`` now restores module identity
   between tests, including the parent-package attribute
   (``sys.modules`` alone is not the only registry).

2. INVOCATION_ID. ``tools/process_registry.py`` early-returns when
   ``INVOCATION_ID`` is unset, and systemd sets it for every unit — so a suite run
   by a kanban worker takes a different code path than the same suite run from a
   human shell. A test that asserts on spawn behaviour must pin the value it
   assumes, not inherit it.

What to do when a failure appears only in a package run:

- Re-run the single file alone. If it passes, suspect the two mechanisms above
  before suspecting the product.
- Do not "fix" it with ``-p no:randomly``, deselection or a marker: masking the
  order dependence leaves the next person with a suite that answers a different
  question depending on how it was invoked, which is exactly the failure these
  notes exist to prevent.
- If a test genuinely cannot be isolated, say so HERE, by name, with the reason.

Known stale tests (not product bugs), all building a claimed card with
``current_run_id=None`` and driving ``_default_spawn`` directly — a state dispatch
cannot produce, because claiming always sets a run id:
``test_kanban_db.py::TestSharedBoardPaths::test_dispatcher_spawn_injects_kanban_paths_without_stale_session``,
``test_kanban_db.py::test_default_spawn_pins_repo_root_on_module_worker_pythonpath``,
``test_kanban_boards.py::TestWorkerSpawnEnv::test_default_spawn_sets_env_vars``,
and the equivalent cases in ``test_kanban_worker_terminal_scope.py``,
``test_kanban_worker_session_source.py`` and ``test_kanban_core_functionality.py``.
"""
