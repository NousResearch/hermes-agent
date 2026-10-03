"""Non-interactive runner helpers must not ship retired model ids as defaults (#127422).

Both helpers talk to OpenRouter by default (``batch_runner`` pins its ``base_url``
there; ``mini_swe_runner`` falls back to the provider router), so their ``model``
defaults must stay current-line OpenRouter slugs instead of date-stamped Anthropic
snapshot ids that get retired first.
"""
import inspect

# Module-level on purpose: batch_runner's import chain probes the checkout's git
# dir, which the home I/O guard would refuse once a test is running.
import batch_runner
from mini_swe_runner import MiniSWERunner, main as mini_swe_main


def _default(callable_, name="model"):
    return inspect.signature(callable_).parameters[name].default


def test_mini_swe_main_default_matches_runner_class_default():
    """``main()`` forwards its ``model`` default straight into ``MiniSWERunner``.

    When the two defaults drift, a bare invocation silently starts runs on a
    superseded model even though the class default was updated.
    """
    assert _default(mini_swe_main) == _default(MiniSWERunner.__init__)


def test_runner_defaults_are_current_openrouter_slugs():
    """Defaults follow the current model line, not retired snapshot ids."""
    assert _default(batch_runner.BatchRunner.__init__) == "anthropic/claude-opus-5.5"
    assert _default(mini_swe_main) == "anthropic/claude-sonnet-4.6"
