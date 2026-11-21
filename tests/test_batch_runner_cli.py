"""batch_runner's CLI values survive fire's argument parsing.

fire literal-evaluates every flag, so the documented ``--providers_allowed=anthropic,openai``
arrives as a tuple and a date-stamped ``--run_name=20260926`` as an int; both used to abort
the run with a "Fatal error" before any prompt ran.
"""

import fire

import batch_runner


def test_cli_values_reach_the_runner_as_documented(monkeypatch):
    captured = {}

    class _Runner:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self, resume=False):
            pass

    monkeypatch.setattr(batch_runner, "BatchRunner", _Runner)

    fire.Fire(batch_runner.main, command=[
        "--dataset_file=d.jsonl", "--batch_size=1", "--run_name=20260926",
        "--providers_allowed=anthropic,openai", "--providers_ignored=together, deepinfra",
        "--providers_order=anthropic, ,openai",
    ])

    assert captured["providers_allowed"] == ["anthropic", "openai"]
    assert captured["providers_ignored"] == ["together", "deepinfra"]
    assert captured["providers_order"] == ["anthropic", "openai"]
    assert captured["run_name"] == "20260926"
